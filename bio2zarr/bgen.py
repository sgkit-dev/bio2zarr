"""BGEN conversion using the bgen-limix distribution (Python namespace cbgen)."""

import importlib.metadata
import logging
import math
import pathlib
import tempfile
from collections import Counter

import numpy as np

from bio2zarr import constants, core, vcz, zarr_utils
from bio2zarr.zarr_utils import STRING_DTYPE_NAME

logger = logging.getLogger(__name__)


def genotype_order(num_alleles, ploidy):
    """Yield allele tuples in the colexicographic order shared by BGEN and VCF."""
    if ploidy == 0:
        yield ()
    else:
        for last in range(num_alleles):
            for prefix in genotype_order(last + 1, ploidy - 1):
                yield (*prefix, last)


def probability_lengths(genotype, num_alleles):
    if np.any(genotype.ploidy > 63):
        raise ValueError("Invalid BGEN ploidy; expected values from 0 to 63")
    if genotype.phased:
        return genotype.ploidy.astype(np.int64) * num_alleles
    lookup = np.empty(64, dtype=np.int64)
    for p in np.unique(genotype.ploidy):
        lookup[p] = math.comb(int(p) + num_alleles - 1, num_alleles - 1)
    return lookup[genotype.ploidy]


def compute_dosage(genotype, num_alleles):
    probability = genotype.probability
    result = np.zeros((len(probability), num_alleles - 1))
    if genotype.phased:
        values = np.nan_to_num(probability, nan=0).reshape(
            len(probability), -1, num_alleles
        )
        result[:] = values.sum(axis=1)[:, 1:]
    else:
        for ploidy in np.unique(genotype.ploidy):
            p = int(ploidy)
            selected = (genotype.ploidy == p) & ~genotype.missing
            if not np.any(selected) or p == 0:
                continue
            if num_alleles == 2:
                result[selected, 0] = probability[selected, : p + 1] @ np.arange(p + 1)
            elif p == 1:
                result[selected] = probability[selected, 1:num_alleles]
            else:
                # Avoid a potentially enormous dense genotypes-by-alleles matrix.
                indices = np.flatnonzero(selected)
                for j, alleles in enumerate(genotype_order(num_alleles, p)):
                    for allele, count in Counter(alleles).items():
                        if allele:
                            result[indices, allele - 1] += (
                                count * probability[indices, j]
                            )
    result[genotype.missing] = constants.FLOAT64_MISSING
    return result


class BgenFormat(vcz.Source):
    @core.requires_optional_dependency("cbgen", "bgen")
    def __init__(
        self,
        path,
        metafile_path,
        sample_ids=None,
        *,
        probabilities=True,
        dosage=True,
        float_dtype="float64",
        compressor="zstd",
        compression_level=7,
    ):
        import cbgen  # noqa: PLC0415

        validate_storage_options(
            probabilities, dosage, float_dtype, compressor, compression_level
        )
        self.storage_options = {
            "probabilities": probabilities,
            "dosage": dosage,
            "float_dtype": float_dtype,
            "compressor": compressor,
            "compression_level": compression_level,
        }
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
        if not hasattr(cbgen.BgenFile, "layout"):
            raise ImportError(
                "BGEN conversion requires bgen-limix>=4.12.0. "
                "Upgrade with bio2zarr[bgen]."
            )
        self._path = str(pathlib.Path(path).resolve())
        self.metafile_path = str(metafile_path)
        with cbgen.BgenFile(self.path) as source:
            self.layout = source.layout
            self._num_records = source.nvariants
            self._num_samples = source.nsamples
            if self.num_records == 0 or self.num_samples == 0:
                raise ValueError(
                    "BGEN conversion requires at least one variant and sample"
                )
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
        self.max_alleles = 0
        self.max_genotypes = 0
        self.max_haplotype_probabilities = 0
        self.has_phased = False
        self.has_unphased = False
        with self.open_reader() as source:
            for j, variant in enumerate(self.iter_variants(0, self.num_records)):
                if variant.nalleles < 2:
                    raise ValueError(f"BGEN variant {j}: expected at least two alleles")
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
                genotype = self.read_genotype(source, variant)
                self.max_alleles = max(self.max_alleles, variant.nalleles)
                length = int(max(probability_lengths(genotype, variant.nalleles)))
                if genotype.phased:
                    self.has_phased = True
                    self.max_haplotype_probabilities = max(
                        self.max_haplotype_probabilities, length
                    )
                else:
                    self.has_unphased = True
                    self.max_genotypes = max(self.max_genotypes, length)
        logger.info(
            "Scanned BGEN with %s samples and %s variants",
            self.num_samples,
            self.num_records,
        )

    @property
    def root_attrs(self):
        return {"bio2zarr_bgen": dict(self.storage_options)}

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

    def open_reader(self):
        import cbgen  # noqa: PLC0415

        return cbgen.BgenFile(self.path)

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
        probability = genotype.probability
        lengths = probability_lengths(genotype, variant.nalleles)
        if (
            probability.shape[0] != self.num_samples
            or probability.shape[1] < max(lengths)
            or np.any(genotype.ploidy > 63)
        ):
            raise ValueError(f"BGEN variant {label}: invalid ploidy/probability shape")
        active = np.arange(probability.shape[1])[None, :] < lengths[:, None]
        called = probability[active & ~genotype.missing[:, None]]
        if np.any(~np.isfinite(called)) or np.any(called < 0) or np.any(called > 1):
            raise ValueError(f"BGEN variant {label}: invalid genotype probabilities")
        tolerance = 3 / 32768 if self.layout == 1 else 1e-6
        for p in np.unique(genotype.ploidy):
            selected = (genotype.ploidy == p) & ~genotype.missing
            values = probability[selected, : int(lengths[genotype.ploidy == p][0])]
            if genotype.phased:
                values = values.reshape(-1, variant.nalleles)
            if not np.allclose(values.sum(axis=1), 1, rtol=0, atol=tolerance):
                raise ValueError(
                    f"BGEN variant {label}: invalid genotype probabilities"
                )
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
            alleles = np.full(num_alleles, constants.STR_FILL, dtype=STRING_DTYPE_NAME)
            alleles[: variant.nalleles] = variant.allele_ids
            yield vcz.VariantData(len(alleles[0]), alleles, None, None)

    def iter_field(self, field_name, shape, dtype, start, stop):
        if field_name not in (
            "position",
            "id",
            "GP",
            "HP",
            "DS",
            "PLOIDY",
            "BGEN_MISSING",
            "phased",
        ):
            raise ValueError(f"Unknown BGEN field {field_name}")
        with self.open_reader() as source:
            for variant in self.iter_variants(start, stop):
                if field_name == "position":
                    yield variant.position
                elif field_name == "id":
                    yield variant.id or constants.STR_MISSING
                else:
                    genotype = self.read_genotype(source, variant)
                    if field_name == "PLOIDY":
                        yield genotype.ploidy
                    elif field_name == "BGEN_MISSING":
                        yield genotype.missing.astype(np.int8)
                    elif field_name == "phased":
                        yield genotype.phased
                    elif field_name == "DS":
                        missing, fill = constants.FLOAT_MISSING_FILL[np.dtype(dtype)]
                        value = compute_dosage(genotype, variant.nalleles)
                        value[genotype.missing] = 0
                        value = value.astype(dtype)
                        value[genotype.missing] = missing
                        if self.max_alleles == 2:
                            yield value[:, 0]
                        else:
                            result = np.full(shape, fill, dtype=dtype)
                            result[:, : variant.nalleles - 1] = value
                            yield result
                    else:
                        missing, fill = constants.FLOAT_MISSING_FILL[np.dtype(dtype)]
                        result = np.full(shape, fill, dtype=dtype)
                        if genotype.phased != (field_name == "HP"):
                            result[:, 0] = missing
                        else:
                            lengths = probability_lengths(genotype, variant.nalleles)
                            active = np.arange(shape[1])[None, :] < lengths[:, None]
                            width = min(shape[1], genotype.probability.shape[1])
                            result[:, :width] = genotype.probability[:, :width]
                            result[~active] = fill
                            result[active & genotype.missing[:, None]] = missing
                        yield result

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
            alleles_size=self.max_alleles,
            genotypes_size=self.max_genotypes
            if self.has_unphased and self.storage_options["probabilities"]
            else None,
        )
        if self.has_phased and self.storage_options["probabilities"]:
            dimensions["FORMAT_HP_dim"] = vcz.VcfZarrDimension.unchunked(
                max(1, self.max_haplotype_probabilities)
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
                "call_DS",
                "f8",
                ["variants", "samples"]
                + (["alt_alleles"] if self.max_alleles > 2 else []),
                "DS",
                "Expected count for each alternate allele",
            ),
            (
                "call_PLOIDY",
                "i1",
                ["variants", "samples"],
                "PLOIDY",
                "BGEN sample ploidy, including for missing probability vectors",
            ),
            (
                "call_BGEN_MISSING",
                "i1",
                ["variants", "samples"],
                "BGEN_MISSING",
                "Original BGEN missingness indicator (1=missing, 0=present)",
            ),
            (
                "variant_bgen_phased",
                "bool",
                ["variants"],
                "phased",
                "Whether the BGEN record stores phased haplotype probabilities",
            ),
        ]
        if self.has_unphased:
            definitions.append(
                (
                    "call_GP",
                    "f8",
                    ["variants", "samples", "genotypes"],
                    "GP",
                    "Unphased genotype probabilities in VCF/BGEN colexicographic order",
                )
            )
        if self.has_phased:
            definitions.append(
                (
                    "call_HP",
                    "f8",
                    ["variants", "samples", "FORMAT_HP_dim"],
                    "HP",
                    "Phased haplotype probabilities: haplotype-major, then alleles",
                )
            )
        fields = [
            vcz.ZarrArraySpec(
                name=name,
                dtype=self.storage_options["float_dtype"] if dtype == "f8" else dtype,
                dimensions=dims,
                source=source,
                description=description,
            )
            for name, dtype, dims, source, description in definitions
            if (source not in ("GP", "HP") or self.storage_options["probabilities"])
            and (source != "DS" or self.storage_options["dosage"])
        ]
        compressor = dict(zarr_utils.DEFAULT_COMPRESSOR_CONFIG)
        compressor.update(
            cname=self.storage_options["compressor"],
            clevel=self.storage_options["compression_level"],
        )
        return vcz.VcfZarrSchema(
            format_version=vcz.ZARR_SCHEMA_FORMAT_VERSION,
            dimensions=dimensions,
            fields=fields,
            defaults={"compressor": compressor},
        )


def validate_storage_options(
    probabilities, dosage, float_dtype, compressor, compression_level
):
    if not isinstance(probabilities, bool) or not isinstance(dosage, bool):
        raise ValueError("probabilities and dosage must be booleans")
    if not probabilities and not dosage:
        raise ValueError("At least one of probabilities or dosage must be enabled")
    if float_dtype not in ("float64", "float32"):
        raise ValueError("float_dtype must be 'float64' or 'float32'")
    if compressor not in ("zstd", "lz4"):
        raise ValueError("compressor must be 'zstd' or 'lz4'")
    if (
        not isinstance(compression_level, int)
        or isinstance(compression_level, bool)
        or not 0 <= compression_level <= 9
    ):
        raise ValueError("compression_level must be an integer from 0 to 9")


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
    probabilities=True,
    dosage=True,
    float_dtype="float64",
    compressor="zstd",
    compression_level=7,
):
    """Convert BGEN genotype or phased haplotype probabilities to VCF Zarr.

    Parameters
    ----------
    bgen_path : str or Path
        Local BGEN layout 1 (raw/zlib) or layout 2 (raw/zlib/zstd) file.
        Layout 2 supports phased/unphased, multiallelic and variable-ploidy data.
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
    probabilities, dosage : bool
        Retain GP/HP and/or derived DS (both default True). At least one is
        required. Dosage alone discards uncertainty and haplotype information.
    float_dtype : {"float64", "float32"}
        GP/HP/DS storage precision, default float64. Float32 rounds values;
        dosage is always computed in float64 before the final cast.
    compressor : {"zstd", "lz4"}
        Blosc codec for schema-defined output arrays, default zstd.
    compression_level : int
        Blosc compression level from 0 to 9, default 7; byte shuffle is retained.

    Returns
    -------
    zarr.Group
        VCZ data, including call_GP and/or call_HP, derived call_DS, sample
        ploidy and original missingness. No hard genotypes are inferred.
        The first BGEN allele is treated as reference.

    Notes
    -----
    Metadata is indexed in a temporary directory, never alongside the input.
    Every supported encoding is read natively from the original file, without
    staging a rewritten copy. Reads decode one variant (all samples) at a time.
    Output buffers contain one variant chunk, as in the other converters.
    Probabilities default to float64 to retain the backend's precision,
    including independently rounded layout-1 values and 32-bit layout-2
    probabilities. Explicit float32 output rounds values without renormalizing.
    """
    validate_storage_options(
        probabilities, dosage, float_dtype, compressor, compression_level
    )
    logger.info(
        "BGEN storage: probabilities=%s dosage=%s float_dtype=%s "
        "compressor=%s compression_level=%s",
        probabilities,
        dosage,
        float_dtype,
        compressor,
        compression_level,
    )
    if mode not in ("r", "r+"):
        raise ValueError("mode must be 'r' or 'r+'")
    if out is not None and str(out).endswith(".zip") and mode != "r":
        raise ValueError("Zip output is read-only; use mode='r'")
    with tempfile.TemporaryDirectory(prefix="bio2zarr-bgen-") as tmp:
        source = BgenFormat(
            bgen_path,
            pathlib.Path(tmp) / "index.metafile",
            sample_ids=sample_ids,
            probabilities=probabilities,
            dosage=dosage,
            float_dtype=float_dtype,
            compressor=compressor,
            compression_level=compression_level,
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
