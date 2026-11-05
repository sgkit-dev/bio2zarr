import dataclasses
import logging
import os
import pathlib
import tempfile

import numpy as np
import pandas as pd
import zarr

from bio2zarr import constants, core, vcz, zarr_utils
from bio2zarr.zarr_utils import STRING_DTYPE_NAME

logger = logging.getLogger(__name__)


FAM_FIELDS = [
    ("family_id", str, "U"),
    ("individual_id", str, "U"),
    ("paternal_id", str, "U"),
    ("maternal_id", str, "U"),
    ("sex", str, "int8"),
    ("phenotype", str, "int8"),
]
FAM_DF_DTYPE = {f[0]: f[1] for f in FAM_FIELDS}
FAM_ARRAY_DTYPE = {f[0]: f[2] for f in FAM_FIELDS}

BIM_FIELDS = [
    ("contig", str, "U"),
    ("variant_id", str, "U"),
    ("cm_position", "float32", "float32"),
    ("position", "int32", "int32"),
    ("allele_1", str, "S"),
    ("allele_2", str, "S"),
]
BIM_DF_DTYPE = {f[0]: f[1] for f in BIM_FIELDS}
BIM_ARRAY_DTYPE = {f[0]: f[2] for f in BIM_FIELDS}


# See https://github.com/sgkit-dev/bio2zarr/issues/409 for discussion
# on the parameters to Pandas here.
def read_fam(path):
    # See: https://www.cog-genomics.org/plink/1.9/formats#fam
    names = [f[0] for f in FAM_FIELDS]
    df = pd.read_csv(path, sep=None, names=names, dtype=FAM_DF_DTYPE, engine="python")
    return df


def read_bim(path):
    # See: https://www.cog-genomics.org/plink/1.9/formats#bim
    names = [f[0] for f in BIM_FIELDS]
    df = pd.read_csv(path, sep=None, names=names, dtype=BIM_DF_DTYPE, engine="python")
    return df


@dataclasses.dataclass
class PlinkPaths:
    bed_path: str
    bim_path: str
    fam_path: str


class BedReader:
    def __init__(self, path, num_variants, num_samples):
        self.num_variants = num_variants
        self.num_samples = num_samples
        self.path = path
        # bytes per variant: 1 byte per 4 samples, rounded up
        self.bytes_per_variant = (self.num_samples + 3) // 4

        # TODO open this as a persistent file and support reading from a
        # stream
        with open(self.path, "rb") as f:
            magic = f.read(3)
            if magic != b"\x6c\x1b\x01":
                raise ValueError("Invalid BED file magic bytes")

        # We could check the size of the bed file here, but that would
        # mean we can't work with streams.

        # Initialize the lookup table with shape (256, 4, 2)
        # 256 possible byte values, 4 samples per byte, 2 alleles per sample
        lookup = np.zeros((256, 4, 2), dtype=np.int8)

        # For each possible byte value (0-255)
        for byte in range(256):
            # For each of the 4 samples encoded in this byte
            for sample in range(4):
                # Extract the 2 bits for this sample
                bits = (byte >> (sample * 2)) & 0b11
                # Convert PLINK's bit encoding to genotype values
                if bits == 0b00:
                    lookup[byte, sample] = [1, 1]
                elif bits == 0b01:
                    lookup[byte, sample] = [-1, -1]
                elif bits == 0b10:
                    lookup[byte, sample] = [0, 1]
                elif bits == 0b11:
                    lookup[byte, sample] = [0, 0]

        self.byte_lookup = lookup

    def iter_decode(self, start, stop, buffer_size=None):
        """
        Iterate of over the variants in the specified window
        with the specified approximate buffer size in bytes (default=10MiB).
        """
        if buffer_size is None:
            buffer_size = 10 * 1024 * 1024
        variants_per_read = max(1, int(buffer_size / self.bytes_per_variant))
        for off in range(start, stop, variants_per_read):
            genotypes = self.decode(off, min(off + variants_per_read, stop))
            yield from genotypes

    def decode(self, start, stop):
        chunk_size = stop - start

        # Calculate file offsets for the required data
        # 3 bytes for the magic number at the beginning of the file
        start_offset = 3 + (start * self.bytes_per_variant)
        bytes_to_read = chunk_size * self.bytes_per_variant

        logger.debug(
            f"Reading {chunk_size} variants ({bytes_to_read} bytes) from {self.path}"
        )

        # TODO make it possible to read sequentially from the same file handle,
        # seeking only when necessary.
        with open(self.path, "rb") as f:
            f.seek(start_offset)
            chunk_data = f.read(bytes_to_read)

        data_bytes = np.frombuffer(chunk_data, dtype=np.uint8)
        data_matrix = data_bytes.reshape(chunk_size, self.bytes_per_variant)

        # Apply lookup table to get genotypes
        # Shape becomes: (chunk_size, bytes_per_variant, 4, 2)
        all_genotypes = self.byte_lookup[data_matrix]

        # Reshape to get all samples in one dimension
        # (chunk_size, bytes_per_variant*4, 2)
        samples_padded = self.bytes_per_variant * 4
        genotypes_reshaped = all_genotypes.reshape(chunk_size, samples_padded, 2)

        return genotypes_reshaped[:, : self.num_samples]


class PlinkFormat(vcz.Source):
    def __init__(self, prefix):
        # TODO we will need support multiple chromosomes here to join
        # plinks into on big zarr. So, these will require multiple
        # bed and bim files, but should share a .fam
        self.prefix = str(prefix)
        self.paths = PlinkPaths(
            self.prefix + ".bed",
            self.prefix + ".bim",
            self.prefix + ".fam",
        )
        self.bim = read_bim(self.paths.bim_path)
        self.fam = read_fam(self.paths.fam_path)
        self._num_records = self.bim.shape[0]
        self._num_samples = self.fam.shape[0]
        self.bed_reader = BedReader(
            self.paths.bed_path, self.num_records, self.num_samples
        )

    @property
    def path(self):
        return self.prefix

    @property
    def num_records(self):
        return self._num_records

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def samples(self):
        return [vcz.Sample(id=iid) for iid in self.fam.individual_id]

    @property
    def contigs(self):
        return [vcz.Contig(id=str(chrom)) for chrom in self.bim.contig.unique()]

    def iter_contig(self, start, stop):
        chrom_to_contig_index = {contig.id: i for i, contig in enumerate(self.contigs)}
        for chrom in self.bim.contig[start:stop]:
            yield chrom_to_contig_index[str(chrom)]

    def iter_field(self, field_name, shape, dtype, start, stop):
        assert field_name == "position"  # Only position field is supported from plink
        yield from self.bim.position[start:stop]

    def iter_id(self, start, stop):
        yield from self.bim.variant_id[start:stop]

    def iter_alleles_and_genotypes(self, start, stop, shape, num_alleles):
        alt_iter = self.bim.allele_1.values[start:stop]
        ref_iter = self.bim.allele_2.values[start:stop]
        gt_iter = self.bed_reader.iter_decode(start, stop)
        for alt, ref, gt in zip(alt_iter, ref_iter, gt_iter):
            alleles = np.full(num_alleles, constants.STR_FILL, dtype=STRING_DTYPE_NAME)
            alleles[0] = ref
            alleles[1 : 1 + len(alt)] = alt
            phased = np.zeros(gt.shape[0], dtype=bool)
            # rlen is the length of the REF in PLINK as there's no END annotations
            yield vcz.VariantData(len(alleles[0]), alleles, gt, phased)

    def generate_schema(
        self,
        variants_chunk_size=None,
        samples_chunk_size=None,
    ):
        n = self.num_samples
        m = self.num_records
        logging.info(f"Scanned plink with {n} samples and {m} variants")
        dimensions = vcz.standard_dimensions(
            variants_size=m,
            variants_chunk_size=variants_chunk_size,
            samples_size=n,
            samples_chunk_size=samples_chunk_size,
            ploidy_size=2,
            alleles_size=2,
        )
        schema_instance = vcz.VcfZarrSchema(
            format_version=vcz.ZARR_SCHEMA_FORMAT_VERSION,
            dimensions=dimensions,
            fields=[],
        )

        logger.info(
            "Generating schema with chunks="
            f"variants={dimensions['variants'].chunk_size}, "
            f"samples={dimensions['samples'].chunk_size}"
        )
        # If we don't have SVLEN or END annotations, the rlen field is defined
        # as the length of the REF
        # Explicitly cast to fixed size array to support pandas 2.x and 3.x
        allele_2_array = self.bim.allele_2.values.astype("S")
        max_len = allele_2_array.itemsize
        array_specs = [
            vcz.ZarrArraySpec(
                source="position",
                name="variant_position",
                dtype="i4",
                dimensions=["variants"],
                description=None,
            ),
            vcz.ZarrArraySpec(
                name="variant_allele",
                dtype=STRING_DTYPE_NAME,
                dimensions=["variants", "alleles"],
                description=None,
            ),
            vcz.ZarrArraySpec(
                name="variant_id",
                dtype=STRING_DTYPE_NAME,
                dimensions=["variants"],
                description=None,
            ),
            vcz.ZarrArraySpec(
                name="variant_id_mask",
                dtype="bool",
                dimensions=["variants"],
                description=None,
            ),
            vcz.ZarrArraySpec(
                source=None,
                name="variant_length",
                dtype=core.min_int_dtype(0, max_len),
                dimensions=["variants"],
                description="Length of each variant",
            ),
            vcz.ZarrArraySpec(
                name="variant_contig",
                dtype=core.min_int_dtype(0, len(np.unique(self.bim.contig))),
                dimensions=["variants"],
                description="Contig/chromosome index for each variant",
            ),
            vcz.ZarrArraySpec(
                name="call_genotype_phased",
                dtype="bool",
                dimensions=["variants", "samples"],
                description=None,
                compressor=dict(zarr_utils.DEFAULT_COMPRESSOR_BOOL_CONFIG),
            ),
            vcz.ZarrArraySpec(
                name="call_genotype",
                dtype="i1",
                dimensions=["variants", "samples", "ploidy"],
                description=None,
                compressor=dict(zarr_utils.DEFAULT_COMPRESSOR_GENOTYPES_CONFIG),
            ),
            vcz.ZarrArraySpec(
                name="call_genotype_mask",
                dtype="bool",
                dimensions=["variants", "samples", "ploidy"],
                description=None,
                compressor=dict(zarr_utils.DEFAULT_COMPRESSOR_BOOL_CONFIG),
            ),
        ]
        schema_instance.fields = array_specs
        return schema_instance


def convert(
    prefix,
    out=None,
    *,
    mode="r",
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=core.DEFAULT_WORKER_PROCESSES,
    show_progress=False,
    zarr_format=None,
    input_format="auto",
    ploidy_mode="metadata",
    genome_build=None,
    dosages=True,
):
    """
    Convert a PLINK 1 or PLINK 2 fileset to VCF Zarr format.

    Parameters
    ----------
    prefix : str or Path
        Path prefix for the PLINK fileset (i.e. the shared prefix of the
        .bed/.bim/.fam or .pgen/.pvar[.zst]/.psam files).
    out : str, Path, or None
        Output path for the Zarr store. The output format depends on the value:

        - **None**: write to a temporary directory and return an in-memory
          :class:`zarr.storage.MemoryStore`-backed group.
        - **Ends with .zip**: write to a directory, then package as a zip
          archive readable via :class:`zarr.storage.ZipStore`. The
          intermediate directory is removed.
        - **Otherwise**: write directly to the given directory path.
    mode : str
        Mode in which the returned :class:`zarr.Group` is opened.
        Use ``"r"`` (default) for read-only access or ``"r+"`` for
        read-write access.
    variants_chunk_size : int, optional
        Number of variants per chunk. If None, a default is used.
    samples_chunk_size : int, optional
        Number of samples per chunk. If None, a default is used.
    worker_processes : int
        Number of worker processes for parallel encoding. 0 (the default)
        means use the main process only.
    show_progress : bool
        If True, display a progress bar during conversion.
    zarr_format : int, optional
        Zarr format version of output (default: 2, or value of
        BIO2ZARR_DEFAULT_ZARR_FORMAT env var)
    input_format : {"auto", "bed", "pgen"}
        Select the input fileset. Auto detects the genotype file and rejects
        ambiguous prefixes with both BED and PGEN files.
    ploidy_mode : {"metadata", "diploid"}
        PGEN only: reconstruct human chromosome ploidy using PSAM sex and PAR
        metadata (default), or retain pgenlib's duplicated diploid encoding.
    genome_build : {"hg19", "hg38"}, optional
        PAR boundaries for unsplit male X calls. Not needed with explicit
        PAR1/PAR2 contigs. Unknown sex on X/Y is rejected in metadata mode.
    dosages : bool
        PGEN only: export call_DS and call_HDS (default True). Native stored
        dosages are retained; pgenlib supplies hardcall-derived values when
        explicit dosages are absent. False selects hardcall-only conversion.

    Notes
    -----
    PGEN conversion requires ``bio2zarr[plink2]``. Phenotype, pedigree, and
    arbitrary PVAR annotations are not exported. PGEN does not preserve the
    original phase flag of homozygous or missing calls.

    Returns
    -------
    zarr.Group
        The root group of the Zarr store containing the converted data.
    """
    if input_format not in ("auto", "bed", "pgen"):
        raise ValueError("input_format must be 'auto', 'bed', or 'pgen'")
    if input_format == "auto":
        has_pgen = pathlib.Path(str(prefix) + ".pgen").exists()
        has_bed = pathlib.Path(str(prefix) + ".bed").exists()
        if has_pgen and has_bed:
            raise ValueError(
                "Both BED and PGEN files exist; choose input_format='bed' or 'pgen'"
            )
        input_format = "pgen" if has_pgen else "bed"
    if input_format == "pgen":
        return _convert_pgen(
            prefix,
            out,
            mode=mode,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            worker_processes=worker_processes,
            show_progress=show_progress,
            zarr_format=zarr_format,
            ploidy_mode=ploidy_mode,
            genome_build=genome_build,
            dosages=dosages,
        )
    if genome_build is not None or ploidy_mode != "metadata" or not dosages:
        raise ValueError("Dosage and ploidy/PAR options require PGEN input")
    plink_format = PlinkFormat(prefix)
    schema = plink_format.generate_schema(
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
    )
    return vcz.encode(
        plink_format,
        schema,
        out,
        mode=mode,
        worker_processes=worker_processes,
        show_progress=show_progress,
        zarr_format=zarr_format,
    )


# Inclusive, one-based PAR intervals from PLINK 2 --split-par.
PAR_INTERVALS = {
    "hg19": ((60001, 2699520), (154931044, 155260560)),
    "hg38": ((10001, 2781479), (155701383, 156030895)),
}


def read_psam(path):
    with open(path) as f:
        for line in f:
            if not line.startswith("##"):
                header = line.split()
                break
        else:
            raise ValueError("PSAM file is empty")
        if not header or header[0] not in ("#FID", "#IID"):
            raise ValueError("PSAM requires a #FID or #IID header")
        header[0] = header[0][1:]
        if "IID" not in header or len(set(header)) != len(header):
            raise ValueError("PSAM requires unique column names including IID")
        df = pd.read_csv(f, sep=r"\s+", names=header, dtype=str, keep_default_na=False)
    if len(df) == 0 or df.IID.isin(["", "0", "."]).any():
        raise ValueError("PSAM requires nonempty sample IDs")
    if df.IID.duplicated().any():
        raise ValueError("PSAM IID values must be unique for VCZ sample IDs")
    if (
        not isinstance(df.index, pd.RangeIndex)
        or df.isna().any().any()
        or (df == "").any().any()
    ):
        raise ValueError("PSAM rows must match the header")
    return df


class PgenFormat(PlinkFormat):
    @core.requires_optional_dependency("pgenlib", "plink2")
    def __init__(
        self, prefix, *, ploidy_mode="metadata", genome_build=None, dosages=True
    ):
        import pgenlib  # noqa: PLC0415

        if ploidy_mode not in ("metadata", "diploid"):
            raise ValueError("ploidy_mode must be 'metadata' or 'diploid'")
        if genome_build not in (None, *PAR_INTERVALS):
            raise ValueError("genome_build must be 'hg19' or 'hg38'")
        if not isinstance(dosages, bool):
            raise ValueError("dosages must be a boolean")
        self.dosages = dosages
        self.prefix = str(prefix)
        self.pgen_path = self.prefix + ".pgen"
        self.pvar_path = self.prefix + ".pvar"
        if not pathlib.Path(self.pvar_path).exists():
            self.pvar_path += ".zst"
        self.psam = read_psam(self.prefix + ".psam")
        self._num_samples = len(self.psam)
        self.ploidy_mode = ploidy_mode
        self.genome_build = genome_build
        with pgenlib.PvarReader(os.fsencode(self.pvar_path)) as reader:
            self._num_records = reader.get_variant_ct()
            if self.num_records == 0:
                raise ValueError("PVAR must contain at least one variant")
            self.allele_counts = np.array(
                [reader.get_allele_ct(j) for j in range(self.num_records)],
                dtype=np.uint32,
            )
            self.allele_offsets = reader.get_allele_idx_offsets()
            self.alleles = [
                [
                    reader.get_allele_code(j, k).decode()
                    for k in range(self.allele_counts[j])
                ]
                for j in range(self.num_records)
            ]
            for j, alleles in enumerate(self.alleles):
                if alleles[1:] == ["."]:
                    self.alleles[j] = alleles[:1]
                    self.allele_counts[j] = 1
            self.bim = pd.DataFrame(
                {
                    "contig": [
                        reader.get_variant_chrom(j).decode()
                        for j in range(self.num_records)
                    ],
                    "variant_id": [
                        reader.get_variant_id(j).decode()
                        for j in range(self.num_records)
                    ],
                    "position": [
                        reader.get_variant_pos(j) for j in range(self.num_records)
                    ],
                    "allele_2": [a[0] for a in self.alleles],
                }
            )
        self.max_alleles = int(self.allele_counts.max())
        self._validate_variants()
        self.contig_codes, self.contig_ids = pd.factorize(self.bim.contig)
        self.chromosomes = self.bim.contig.str.upper().str.removeprefix("CHR")
        self.is_x = self.chromosomes.isin(["X", "23"]).to_numpy()
        self.is_y = self.chromosomes.isin(["Y", "24"]).to_numpy()
        self.is_mt = self.chromosomes.isin(["M", "MT", "26"]).to_numpy()
        self.is_par = self.chromosomes.isin(["PAR1", "PAR2", "XY", "25"]).to_numpy()
        self._plan_ploidy()
        with self.open_reader() as reader:
            self.has_phase = reader.hardcall_phase_present()
        logger.info(
            "Scanned PGEN metadata: %s variants, %s samples, ploidy=%s, phased=%s",
            self.num_records,
            self.num_samples,
            self.max_ploidy,
            self.has_phase,
        )

    def _validate_variants(self):
        if (self.bim.position <= 0).any():
            raise ValueError("PVAR positions must be positive")
        seen = set()
        last_contig = None
        last_position = 0
        for chrom, pos in zip(self.bim.contig, self.bim.position):
            if chrom != last_contig:
                if chrom in seen:
                    raise ValueError("PVAR contigs must be contiguous")
                seen.add(chrom)
                last_contig, last_position = chrom, 0
            if pos < last_position:
                raise ValueError("PVAR positions must be sorted within each contig")
            last_position = pos
        for alleles in self.alleles:
            if any(a in ("", ".") for a in alleles):
                raise ValueError("PVAR REF/ALT alleles must not be missing")
            if len(set(alleles)) != len(alleles):
                raise ValueError("PVAR alleles must be distinct")

    def _plan_ploidy(self):
        self.male = np.zeros(self.num_samples, dtype=bool)
        self.female = np.zeros(self.num_samples, dtype=bool)
        if self.ploidy_mode == "metadata":
            if np.any(self.is_x | self.is_y):
                sex = self.psam.get("SEX")
                if sex is None or not sex.isin(["1", "2"]).all():
                    raise ValueError(
                        "X/Y ploidy requires PSAM SEX=1 or 2 for every sample; "
                        "use ploidy_mode='diploid' to preserve PGEN encoding"
                    )
                self.male = (sex == "1").to_numpy()
                self.female = (sex == "2").to_numpy()
            if np.any(self.is_x) and np.any(self.male):
                if self.genome_build is None:
                    # --split-par makes every remaining X position non-PAR.
                    if not self.chromosomes.isin(["PAR1", "PAR2"]).any():
                        raise ValueError(
                            "Male X ploidy requires genome_build='hg19'/'hg38' "
                            "or explicit PAR1/PAR2 contigs from --split-par; "
                            "alternatively use ploidy_mode='diploid'"
                        )
                else:
                    for start, stop in PAR_INTERVALS[self.genome_build]:
                        self.is_par = self.is_par | (
                            self.is_x
                            & self.bim.position.between(start, stop).to_numpy()
                        )
        # Compute the maximum from O(variants + samples) metadata, not a dense
        # variants-by-samples matrix or a genotype scan.
        if self.ploidy_mode == "diploid":
            self.max_ploidy = 2
        else:
            diploid = ~(self.is_x | self.is_y | self.is_mt) | self.is_par
            self.max_ploidy = (
                2
                if (np.any(diploid) or (np.any(self.is_x) and np.any(self.female)))
                else 1
            )

    def ploidies(self, start, stop, sample_start, sample_stop):
        ploidy = np.full((stop - start, sample_stop - sample_start), 2, dtype=np.int8)
        if self.ploidy_mode == "metadata":
            male = self.male[sample_start:sample_stop]
            female = self.female[sample_start:sample_stop]
            nonpar_x = self.is_x[start:stop] & ~self.is_par[start:stop]
            ploidy[np.ix_(nonpar_x, male)] = 1
            ploidy[self.is_y[start:stop]] = 1
            ploidy[np.ix_(self.is_y[start:stop], female)] = 0
            ploidy[self.is_mt[start:stop]] = 1
        return ploidy

    def open_reader(self, sample_subset=None):
        import pgenlib  # noqa: PLC0415

        return pgenlib.PgenReader(
            os.fsencode(self.pgen_path),
            raw_sample_ct=self.num_samples,
            variant_ct=self.num_records,
            allele_idx_offsets=(self.allele_offsets if self.max_alleles > 2 else None),
            sample_subset=sample_subset,
        )

    @property
    def samples(self):
        return [vcz.Sample(id=iid) for iid in self.psam.IID]

    @property
    def contigs(self):
        return [vcz.Contig(id=chrom) for chrom in self.contig_ids]

    @property
    def root_attrs(self):
        return {
            "plink2_ploidy_mode": self.ploidy_mode,
            "plink2_genome_build": self.genome_build,
            "plink2_hardcall_phase_present": bool(self.has_phase),
            "plink2_dosages": self.dosages,
        }

    def generate_schema(self, variants_chunk_size=None, samples_chunk_size=None):
        schema = super().generate_schema(variants_chunk_size, samples_chunk_size)
        schema.dimensions["ploidy"] = vcz.VcfZarrDimension.unchunked(self.max_ploidy)
        num_alleles = self.max_alleles
        schema.dimensions["alleles"] = vcz.VcfZarrDimension.unchunked(num_alleles)
        if num_alleles > 1:
            schema.dimensions["alt_alleles"] = vcz.VcfZarrDimension.unchunked(
                num_alleles - 1
            )
        else:
            schema.dimensions.pop("alt_alleles", None)
        for field in schema.fields:
            if field.name == "call_genotype":
                field.dtype = core.min_int_dtype(-2, num_alleles - 1)
            elif field.name == "variant_position":
                field.dtype = core.min_int_dtype(0, int(self.bim.position.max()))
        if self.dosages and num_alleles > 1:
            ds_dimensions = ["variants", "samples"]
            if num_alleles > 2:
                ds_dimensions.append("alt_alleles")
                schema.dimensions["alt_alleles"].chunk_size = 1
            schema.dimensions["FORMAT_HDS_dim"] = vcz.VcfZarrDimension(
                self.max_ploidy * (num_alleles - 1), 1
            )
            schema.fields.extend(
                [
                    vcz.ZarrArraySpec(
                        name="call_DS",
                        dtype="f4",
                        dimensions=ds_dimensions,
                        description="Expected alternate allele counts",
                        compressor=dict(zarr_utils.DEFAULT_COMPRESSOR_DOSAGES_CONFIG),
                    ),
                    vcz.ZarrArraySpec(
                        name="call_HDS",
                        dtype="f4",
                        dimensions=["variants", "samples", "FORMAT_HDS_dim"],
                        description=(
                            "Alternate allele dosages in haplotype-major order; "
                            "unphased dosages follow pgenlib's even split"
                        ),
                        compressor=dict(zarr_utils.DEFAULT_COMPRESSOR_DOSAGES_CONFIG),
                    ),
                ]
            )
        return schema

    def iter_alleles_and_genotypes(self, start, stop, shape, num_alleles):
        raise NotImplementedError("PGEN requires the chunked PgenZarrWriter")


class PgenZarrWriter(vcz.VcfZarrWriter):
    """Encode native tiles through the shared VCZ writer lifecycle."""

    def prepare_partition(self, partition_index):
        self.wip_partition_path(partition_index).mkdir(exist_ok=True)
        for field in self.schema.fields:
            self.open_partition_array(partition_index, field.name)

    def open_partition_arrays(self, partition_index, *, calls):
        return {
            field.name: zarr.open_array(
                self.wip_partition_array_path(partition_index, field.name), mode="a"
            )
            for field in self.schema.fields
            if field.name.startswith("call_") == calls
        }

    def encode_metadata(self, partition_index):
        arrays = self.open_partition_arrays(partition_index, calls=False)
        partition = self.metadata.partitions[partition_index]
        chunk_size = self.schema.dimensions["variants"].chunk_size
        for start in range(partition.start, partition.stop, chunk_size):
            stop = min(start + chunk_size, partition.stop)
            alleles = np.full(
                (stop - start, self.schema.dimensions["alleles"].size),
                constants.STR_FILL,
                dtype=zarr_utils.STRING_DTYPE_NAME,
            )
            for j, row in enumerate(self.source.alleles[start:stop]):
                alleles[j, : len(row)] = row
            ids = self.source.bim.variant_id.to_numpy()[start:stop]
            values = {
                "variant_allele": alleles,
                "variant_length": np.array([len(a[0]) for a in alleles]),
                "variant_position": self.source.bim.position.to_numpy()[start:stop],
                "variant_contig": self.source.contig_codes[start:stop],
                "variant_id": ids,
                "variant_id_mask": ids == ".",
            }
            for name, data in values.items():
                arrays[name][start:stop] = data
                core.update_progress(data.nbytes)

    def encode_tile(self, partition_index, sample_start, sample_stop):
        arrays = self.open_partition_arrays(partition_index, calls=True)
        partition = self.metadata.partitions[partition_index]
        chunk_size = min(
            self.schema.dimensions["variants"].chunk_size,
            partition.stop - partition.start,
        )
        samples = sample_stop - sample_start
        alleles = np.empty((chunk_size, 2 * samples), dtype=np.int32)
        phase = np.zeros((chunk_size, samples), dtype=bool)
        subset = np.arange(sample_start, sample_stop, dtype=np.uint32)
        with self.source.open_reader(subset) as reader:
            for start in range(partition.start, partition.stop, chunk_size):
                stop = min(start + chunk_size, partition.stop)
                n = stop - start
                if self.source.has_phase:
                    reader.read_alleles_and_phasepresent_range(
                        start, stop, alleles[:n], phase[:n]
                    )
                else:
                    reader.read_alleles_range(start, stop, alleles[:n])
                gt = alleles[:n].reshape(n, samples, 2)
                counts = self.source.allele_counts[start:stop, None, None]
                if np.any((gt != -9) & ((gt < 0) | (gt >= counts))):
                    raise ValueError(
                        f"Invalid PGEN allele code at variants {start}:{stop}"
                    )
                gt[gt == -9] = constants.INT_MISSING
                ploidy = self.source.ploidies(start, stop, sample_start, sample_stop)
                haploid = ploidy == 1
                if np.any(haploid & (gt[:, :, 0] != gt[:, :, 1])):
                    raise ValueError(
                        f"Heterozygous haploid call at variants {start}:{stop}; "
                        "check sex/PAR metadata or use ploidy_mode='diploid'"
                    )
                if np.any((ploidy == 0) & np.any(gt >= 0, axis=2)):
                    raise ValueError(
                        f"Nonmissing female Y call at variants {start}:{stop}; "
                        "check sex metadata or use ploidy_mode='diploid'"
                    )
                gt[:, :, 1][ploidy < 2] = constants.INT_FILL
                gt[:, :, 0][ploidy == 0] = constants.INT_FILL
                gt = gt[:, :, : self.source.max_ploidy].astype(
                    arrays["call_genotype"].dtype
                )
                phased = phase[:n] & (ploidy == 2)
                data = {
                    "call_genotype": gt,
                    "call_genotype_mask": gt < 0,
                    "call_genotype_phased": phased,
                }
                for name, value in data.items():
                    arrays[name][start:stop, sample_start:sample_stop] = value
                    core.update_progress(value.nbytes)
                if "call_DS" in arrays:
                    self.encode_dosages(
                        reader, arrays, start, stop, sample_start, sample_stop, ploidy
                    )

    def encode_dosages(
        self, reader, arrays, start, stop, sample_start, sample_stop, ploidy
    ):
        n = stop - start
        samples = sample_stop - sample_start
        counts = self.source.allele_counts[start:stop]
        max_alt = self.schema.dimensions["alleles"].size - 1
        raw = np.empty((n, 2 * samples), dtype=np.float32)
        ds = np.empty((n, samples), dtype=np.float32)
        totals = np.zeros((n, samples), dtype=np.float32)
        sample_slice = slice(sample_start, sample_stop)
        # Stream one alternate allele at a time: a highly multiallelic variant
        # must not multiply per-worker memory by the global allele dimension.
        for allele in range(1, max_alt + 1):
            rows = np.flatnonzero(counts > allele)
            ds.fill(constants.FLOAT32_FILL)
            if len(rows):
                if len(rows) == n:
                    reader.read_phased_dosages_range(
                        start, stop, raw, allele_idx=allele
                    )
                else:
                    reader.read_phased_dosages_list(
                        np.asarray(rows + start, dtype=np.uint32),
                        raw[: len(rows)],
                        allele_idx=allele,
                    )
                hds = raw[: len(rows)].reshape(len(rows), samples, 2)
                missing = hds == -9
                invalid = ~missing & (~np.isfinite(hds) | (hds < 0) | (hds > 1))
                if np.any(invalid) or np.any(missing[:, :, 0] != missing[:, :, 1]):
                    raise ValueError(f"Invalid PGEN dosage at variants {start}:{stop}")
                local_ploidy = ploidy[rows]
                present = ~missing[:, :, 0]
                if np.any((local_ploidy == 0) & present):
                    raise ValueError(
                        f"Nonmissing female Y dosage at variants {start}:{stop}; "
                        "check sex metadata or use ploidy_mode='diploid'"
                    )
                if np.any(
                    (local_ploidy == 1) & present & (hds[:, :, 0] != hds[:, :, 1])
                ):
                    raise ValueError(
                        "Unequal haploid haplotype dosages at "
                        f"variants {start}:{stop}; "
                        "check ploidy/PAR metadata or use ploidy_mode='diploid'"
                    )
                values = hds.sum(axis=2)
                values[local_ploidy == 1] /= 2
                totals[rows] += np.where(present, values, 0)
                values[~present] = constants.FLOAT32_MISSING
                values[local_ploidy == 0] = constants.FLOAT32_FILL
                ds[rows] = values
                # HDS is a variable-length VCF vector: pack each variant's
                # actual alternate alleles, then pad the remaining vector.
                for count in np.unique(counts[rows]):
                    group = counts[rows] == count
                    variant_rows = rows[group]
                    hds_group = hds[group]
                    group_ploidy = local_ploidy[group]
                    group_missing = ~present[group]
                    for hap in range(self.source.max_ploidy):
                        value = hds_group[:, :, hap].copy()
                        value[group_missing] = constants.FLOAT32_MISSING
                        value[group_ploidy <= hap] = constants.FLOAT32_FILL
                        column = hap * (int(count) - 1) + allele - 1
                        arrays["call_HDS"].oindex[
                            variant_rows + start, sample_slice, column
                        ] = value
                        core.update_progress(value.nbytes)
            if max_alt == 1:
                arrays["call_DS"][start:stop, sample_slice] = ds
            else:
                arrays["call_DS"][start:stop, sample_slice, allele - 1] = ds
            core.update_progress(ds.nbytes)
        # Empty HDS slots need explicit VCZ fill NaNs, not Zarr's zero fill.
        width = self.schema.dimensions["FORMAT_HDS_dim"].size
        fill = np.full((n, samples), constants.FLOAT32_FILL, dtype=np.float32)
        for column in range(width):
            rows = np.flatnonzero(column >= (counts - 1) * self.source.max_ploidy)
            if len(rows):
                arrays["call_HDS"].oindex[rows + start, sample_slice, column] = fill[
                    : len(rows)
                ]
                core.update_progress(fill[: len(rows)].nbytes)
        if np.any(totals > ploidy + 1e-6):
            raise ValueError(
                f"Alternate dosages exceed ploidy at variants {start}:{stop}"
            )

    def encode_partition_fields(self, partition_index):
        self.prepare_partition(partition_index)
        self.encode_metadata(partition_index)
        chunk = self.schema.dimensions["samples"].chunk_size
        for start in range(0, self.source.num_samples, chunk):
            self.encode_tile(
                partition_index, start, min(start + chunk, self.source.num_samples)
            )


def _convert_pgen(
    prefix,
    out=None,
    *,
    mode="r",
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=0,
    show_progress=False,
    zarr_format=None,
    ploidy_mode="metadata",
    genome_build=None,
    dosages=True,
):
    for name, value, minimum in (
        ("worker_processes", worker_processes, 0),
        ("variants_chunk_size", variants_chunk_size, 1),
        ("samples_chunk_size", samples_chunk_size, 1),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    source = PgenFormat(
        prefix, ploidy_mode=ploidy_mode, genome_build=genome_build, dosages=dosages
    )
    schema = source.generate_schema(variants_chunk_size, samples_chunk_size)
    schema.validate()
    options = dict(
        mode=mode,
        worker_processes=worker_processes,
        show_progress=show_progress,
        zarr_format=zarr_format,
        writer_class=PgenZarrWriter,
    )
    if out is None:
        return vcz.encode(source, schema, **options)
    destination = pathlib.Path(out)
    if destination.exists():
        raise ValueError("Zarr path already exists")
    # Publish only after every worker, finalisation and indexing has succeeded.
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temp:
        staged = pathlib.Path(temp) / destination.name
        root = vcz.encode(source, schema, staged, **options)
        if isinstance(root.store, zarr.storage.ZipStore):
            root.store.close()
        os.rename(staged, destination)
    store = destination
    if destination.suffix == ".zip":
        store = zarr.storage.ZipStore(destination, mode="r")
    return zarr.open(store, mode=mode)
