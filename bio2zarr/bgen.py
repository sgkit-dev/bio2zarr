"""BGEN conversion using the bgen-limix distribution (Python namespace cbgen)."""

import importlib.metadata
import logging
import pathlib
import struct
import tempfile

import numpy as np

from bio2zarr import constants, core, vcz
from bio2zarr.zarr_utils import STRING_DTYPE_NAME

logger = logging.getLogger(__name__)


def check_header(path):
    """Check the supported encoding before entering the native reader."""
    with open(path, "rb") as f:
        header = f.read(20)
        if len(header) != 20:
            raise ValueError("Truncated BGEN header")
        offset, length, variants, samples, magic = struct.unpack("<IIII4s", header)
        if magic not in (b"bgen", b"\0" * 4) or length < 20 or offset < length:
            raise ValueError("Invalid BGEN header")
        f.seek(length)
        flags = f.read(4)
        if len(flags) != 4:
            raise ValueError("Truncated BGEN flags")
        flags = int.from_bytes(flags, "little")
    compression = flags & 3
    layout = (flags >> 2) & 15
    if layout != 2:
        raise ValueError(
            f"Unsupported BGEN layout {layout}; bgen-limix 4.11 indexing "
            "requires layout 2"
        )
    if compression == 3:
        raise ValueError(
            f"Unsupported BGEN compression {compression} for layout {layout}"
        )
    # bgen-limix 4.11 reads a fixed 6*N bytes for this header, rather than
    # the variable-length block. Do not pass these files to the native reader.
    if layout == 2 and compression == 0:
        raise ValueError(
            "Uncompressed BGEN layout 2 is not supported by bgen-limix 4.11; "
            "use zlib or zstd compression"
        )
    if variants == 0 or samples == 0:
        raise ValueError("BGEN conversion requires at least one variant and sample")


class BgenFormat(vcz.Source):
    @core.requires_optional_dependency("cbgen", "bgen")
    def __init__(self, path, metafile_path, sample_ids=None):
        import cbgen  # noqa: PLC0415

        try:
            importlib.metadata.distribution("bgen-limix")
        except importlib.metadata.PackageNotFoundError:
            raise ImportError(
                "BGEN conversion requires bgen-limix, not the legacy cbgen "
                "distribution. Install bio2zarr[bgen]."
            ) from None
        try:
            importlib.metadata.distribution("cbgen")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise ImportError(
                "The legacy cbgen distribution conflicts with bgen-limix. "
                "Use an environment containing only bio2zarr[bgen]."
            )
        self._path = str(pathlib.Path(path).resolve())
        self.metafile_path = str(metafile_path)
        check_header(self._path)
        with cbgen.BgenFile(self.path) as source:
            self._num_records = source.nvariants
            self._num_samples = source.nsamples
            embedded = source.read_samples() if source.contain_samples else None
            if sample_ids is None:
                sample_ids = embedded
            if sample_ids is None:
                sample_ids = [f"sample_{j}" for j in range(self.num_samples)]
            if isinstance(sample_ids, str):
                raise ValueError("sample_ids must be a sequence of identifiers")
            sample_ids = list(sample_ids)
            if len(sample_ids) != self.num_samples:
                raise ValueError("sample_ids must match the number of BGEN samples")
            if any(not isinstance(s, str) or not s or s == "." for s in sample_ids):
                raise ValueError("sample_ids must be nonempty strings other than '.'")
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError("sample_ids must be unique")
            self._samples = [vcz.Sample(id=s) for s in sample_ids]
            source.create_metafile(self.metafile_path)

        self._contigs = []
        last_position = {}
        self.max_position = 0
        self.max_ref_length = 0
        with cbgen.BgenFile(self.path) as source:
            for j, variant in enumerate(self.iter_variants(0, self.num_records)):
                if variant.nalleles != 2:
                    raise ValueError(
                        f"BGEN variant {j}: only biallelic variants are supported"
                    )
                if not variant.chrom or variant.chrom == ".":
                    raise ValueError(f"BGEN variant {j}: missing chromosome")
                if variant.position < 1:
                    raise ValueError(f"BGEN variant {j}: position must be positive")
                if variant.position < last_position.get(variant.chrom, 0):
                    raise ValueError(
                        f"BGEN variant {j}: positions must be nondecreasing "
                        "within each chromosome"
                    )
                last_position[variant.chrom] = variant.position
                if any(not a or a == "." for a in variant.allele_ids):
                    raise ValueError(f"BGEN variant {j}: missing allele")
                if variant.chrom not in self._contigs:
                    self._contigs.append(variant.chrom)
                self.max_position = max(self.max_position, variant.position)
                self.max_ref_length = max(
                    self.max_ref_length, len(variant.allele_ids[0])
                )
                self.read_genotype(source, variant)
        logger.info(
            "Scanned BGEN with %s samples and %s variants",
            self.num_samples,
            self.num_records,
        )

    @property
    def path(self):
        return self._path

    @property
    def num_records(self):
        return self._num_records

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def samples(self):
        return self._samples

    @property
    def contigs(self):
        return [vcz.Contig(id=c) for c in self._contigs]

    def iter_variants(self, start, stop):
        import cbgen  # noqa: PLC0415

        with cbgen.BgenMetafile(self.metafile_path) as index:
            size = index.partition_size
            for partition in range(start // size, (stop + size - 1) // size):
                records = index.read_partition(partition)
                left = max(0, start - records.offset)
                right = min(len(records.variants), stop - records.offset)
                yield from records.variants[left:right]

    def read_genotype(self, source, variant):
        genotype = source.read_genotype(variant.offset)
        label = f"{variant.chrom}:{variant.position}"
        if genotype.phased:
            raise ValueError(
                f"BGEN variant {label}: phased haplotype probabilities are not "
                "supported; they cannot be stored as unphased FORMAT/GP"
            )
        if np.any(genotype.ploidy != 2):
            raise ValueError(
                f"BGEN variant {label}: only diploid samples are supported"
            )
        probability = genotype.probability
        if probability.shape != (self.num_samples, 3):
            raise ValueError(f"BGEN variant {label}: expected three probabilities")
        called = probability[~genotype.missing]
        if (
            np.any(~np.isfinite(called))
            or np.any(called < 0)
            or np.any(called > 1)
            or not np.allclose(called.sum(axis=1), 1, rtol=0, atol=1e-6)
        ):
            raise ValueError(f"BGEN variant {label}: invalid genotype probabilities")
        return genotype

    def iter_contig(self, start, stop):
        lookup = {c: j for j, c in enumerate(self._contigs)}
        for variant in self.iter_variants(start, stop):
            yield lookup[variant.chrom]

    def iter_id(self, start, stop):
        for variant in self.iter_variants(start, stop):
            yield variant.rsid if variant.rsid not in ("", ".") else None

    def iter_alleles_and_genotypes(self, start, stop, shape, num_alleles):
        for variant in self.iter_variants(start, stop):
            alleles = np.asarray(variant.allele_ids, dtype=STRING_DTYPE_NAME)
            yield vcz.VariantData(len(alleles[0]), alleles, None, None)

    def iter_field(self, field_name, shape, dtype, start, stop):
        import cbgen  # noqa: PLC0415

        if field_name not in ("position", "id", "GP", "DS"):
            raise ValueError(f"Unknown BGEN field {field_name}")
        with cbgen.BgenFile(self.path) as source:
            for variant in self.iter_variants(start, stop):
                if field_name == "position":
                    yield variant.position
                elif field_name == "id":
                    yield variant.id or constants.STR_MISSING
                else:
                    genotype = self.read_genotype(source, variant)
                    probability = genotype.probability
                    if field_name == "GP":
                        value = probability.copy()
                    else:
                        value = probability[:, 1] + 2 * probability[:, 2]
                    value[genotype.missing] = constants.FLOAT64_MISSING
                    yield value

    def generate_schema(self, variants_chunk_size=None, samples_chunk_size=None):
        for name, size in (
            ("variants_chunk_size", variants_chunk_size),
            ("samples_chunk_size", samples_chunk_size),
        ):
            if size is not None and (not isinstance(size, int) or size < 1):
                raise ValueError(f"{name} must be a positive integer")
        dimensions = vcz.standard_dimensions(
            variants_size=self.num_records,
            samples_size=self.num_samples,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            alleles_size=2,
            genotypes_size=3,
        )
        # The shared region index uses the position dtype for record counts too.
        # Keep at least int32, and allow reference spans beyond int32 positions.
        position_dtype = core.min_int_dtype(
            0,
            max(
                np.iinfo(np.int32).max,
                self.max_position + self.max_ref_length - 1,
            ),
        )
        definitions = [
            (
                "variant_position",
                position_dtype,
                ["variants"],
                "position",
                None,
            ),
            (
                "variant_contig",
                core.min_int_dtype(0, len(self.contigs)),
                ["variants"],
                None,
                None,
            ),
            ("variant_allele", STRING_DTYPE_NAME, ["variants", "alleles"], None, None),
            (
                "variant_length",
                core.min_int_dtype(0, self.max_ref_length),
                ["variants"],
                None,
                None,
            ),
            ("variant_id", STRING_DTYPE_NAME, ["variants"], None, None),
            ("variant_id_mask", "bool", ["variants"], None, None),
            (
                "variant_bgen_id",
                STRING_DTYPE_NAME,
                ["variants"],
                "id",
                "BGEN variant identifier (distinct from the rsID)",
            ),
            (
                "call_GP",
                "f8",
                ["variants", "samples", "genotypes"],
                "GP",
                "Unphased genotype probabilities in VCF order (0/0, 0/1, 1/1)",
            ),
            (
                "call_DS",
                "f8",
                ["variants", "samples"],
                "DS",
                "Expected alternate allele count, P(0/1) + 2*P(1/1)",
            ),
        ]
        fields = [
            vcz.ZarrArraySpec(
                name=name,
                dtype=dtype,
                dimensions=dims,
                source=source,
                description=description,
            )
            for name, dtype, dims, source, description in definitions
        ]
        return vcz.VcfZarrSchema(
            format_version=vcz.ZARR_SCHEMA_FORMAT_VERSION,
            dimensions=dimensions,
            fields=fields,
        )


def convert(
    bgen_path,
    out=None,
    *,
    sample_ids=None,
    mode="r",
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=core.DEFAULT_WORKER_PROCESSES,
    show_progress=False,
    zarr_format=None,
):
    """Convert unphased, diploid, biallelic BGEN data to VCF Zarr.

    Parameters
    ----------
    bgen_path : str or Path
        Local layout 2 BGEN file, compressed with zlib or zstd.
        Other layouts, uncompressed data, phased, non-diploid and multiallelic
        data raise ValueError.
    out : str, Path, or None
        Directory, .zip archive, or None for a MemoryStore-backed group.
    sample_ids : sequence of str, optional
        Sample identifiers in file order, overriding embedded identifiers.
        Without either, identifiers sample_0, sample_1, ... are generated.
    mode : str
        Returned group access mode, "r" (default) or "r+". Zip output is
        read-only and requires "r".
    variants_chunk_size, samples_chunk_size : int, optional
        Output chunk sizes. Defaults follow the other converters.
    worker_processes : int
        Parallel encoder processes; 0 (default) runs in the main process.
    show_progress : bool
        Display encoding progress.
    zarr_format : int, optional
        Output format, 2 (default) or 3. BIO2ZARR_DEFAULT_ZARR_FORMAT overrides
        the default.

    Returns
    -------
    zarr.Group
        VCZ data, including call_GP and derived call_DS. No hard genotypes
        are inferred. The first BGEN allele is treated as reference.

    Notes
    -----
    Metadata is indexed in a temporary directory, never alongside the input.
    Reads decode one variant (all samples) at a time. Output buffers contain
    one variant chunk, as in the other converters. Probabilities use float64
    to retain the backend's precision, including 32-bit BGEN probabilities.
    """
    if mode not in ("r", "r+"):
        raise ValueError("mode must be 'r' or 'r+'")
    if out is not None and str(out).endswith(".zip") and mode != "r":
        raise ValueError("Zip output is read-only; use mode='r'")
    with tempfile.TemporaryDirectory(prefix="bio2zarr-bgen-") as tmp:
        source = BgenFormat(
            bgen_path, pathlib.Path(tmp) / "index.metafile", sample_ids=sample_ids
        )
        schema = source.generate_schema(variants_chunk_size, samples_chunk_size)
        return vcz.encode(
            source,
            schema,
            out,
            mode=mode,
            worker_processes=worker_processes,
            show_progress=show_progress,
            zarr_format=zarr_format,
        )
