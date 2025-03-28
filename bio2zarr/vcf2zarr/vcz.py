import contextlib
import dataclasses
import logging
import pathlib
import tempfile

import numpy as np

from bio2zarr import core, schema, writer

from . import icf

logger = logging.getLogger(__name__)


def inspect(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError(f"Path not found: {path}")
    if (path / "metadata.json").exists():
        obj = icf.IntermediateColumnarFormat(path)
    # NOTE: this is too strict, we should support more general Zarrs, see #276
    elif (path / ".zmetadata").exists():
        obj = writer.VcfZarr(path)
    else:
        raise ValueError(f"{path} not in ICF or VCF Zarr format")
    return obj.summary_table()


def convert_local_allele_field_types(fields):
    """
    Update the specified list of fields to include the LAA field, and to convert
    any supported localisable fields to the L* counterpart.

    Note that we currently support only two ALT alleles per sample, and so the
    dimensions of these fields are fixed by that requirement. Later versions may
    use summary data storted in the ICF to make different choices, if information
    about subsequent alleles (not in the actual genotype calls) should also be
    stored.
    """
    fields_by_name = {field.name: field for field in fields}
    gt = fields_by_name["call_genotype"]
    if gt.shape[-1] != 2:
        raise ValueError("Local alleles only supported on diploid data")

    # TODO check if LA is already in here

    shape = gt.shape[:-1]
    chunks = gt.chunks[:-1]
    dimensions = gt.dimensions[:-1]

    la = schema.ZarrArraySpec.new(
        vcf_field=None,
        name="call_LA",
        dtype="i1",
        shape=gt.shape,
        chunks=gt.chunks,
        dimensions=(*dimensions, "local_alleles"),
        description=(
            "0-based indices into REF+ALT, indicating which alleles"
            " are relevant (local) for the current sample"
        ),
    )
    ad = fields_by_name.get("call_AD", None)
    if ad is not None:
        # TODO check if call_LAD is in the list already
        ad.name = "call_LAD"
        ad.vcf_field = None
        ad.shape = (*shape, 2)
        ad.chunks = (*chunks, 2)
        ad.dimensions = (*dimensions, "local_alleles")
        ad.description += " (local-alleles)"

    pl = fields_by_name.get("call_PL", None)
    if pl is not None:
        # TODO check if call_LPL is in the list already
        pl.name = "call_LPL"
        pl.vcf_field = None
        pl.shape = (*shape, 3)
        pl.chunks = (*chunks, 3)
        pl.description += " (local-alleles)"
        pl.dimensions = (*dimensions, "local_" + pl.dimensions[-1])
    return [*fields, la]


def generate_schema(
    icf, variants_chunk_size=None, samples_chunk_size=None, local_alleles=None
):
    # Import schema here to avoid circular import
    from bio2zarr import schema

    m = icf.num_records
    n = icf.num_samples
    if samples_chunk_size is None:
        samples_chunk_size = 10_000
    if variants_chunk_size is None:
        variants_chunk_size = 1000
    if local_alleles is None:
        local_alleles = False
    logger.info(
        f"Generating schema with chunks={variants_chunk_size, samples_chunk_size}"
    )

    def spec_from_field(field, array_name=None):
        return schema.ZarrArraySpec.from_field(
            field,
            num_samples=n,
            num_variants=m,
            samples_chunk_size=samples_chunk_size,
            variants_chunk_size=variants_chunk_size,
            array_name=array_name,
        )

    def fixed_field_spec(
        name,
        dtype,
        vcf_field=None,
        shape=(m,),
        dimensions=("variants",),
        chunks=None,
    ):
        return schema.ZarrArraySpec.new(
            vcf_field=vcf_field,
            name=name,
            dtype=dtype,
            shape=shape,
            description="",
            dimensions=dimensions,
            chunks=chunks or [variants_chunk_size],
        )

    alt_field = icf.fields["ALT"]
    max_alleles = alt_field.vcf_field.summary.max_number + 1

    array_specs = [
        fixed_field_spec(
            name="variant_contig",
            dtype=core.min_int_dtype(0, icf.metadata.num_contigs),
        ),
        fixed_field_spec(
            name="variant_filter",
            dtype="bool",
            shape=(m, icf.metadata.num_filters),
            dimensions=["variants", "filters"],
            chunks=(variants_chunk_size, icf.metadata.num_filters),
        ),
        fixed_field_spec(
            name="variant_allele",
            dtype="O",
            shape=(m, max_alleles),
            dimensions=["variants", "alleles"],
            chunks=(variants_chunk_size, max_alleles),
        ),
        fixed_field_spec(
            name="variant_id",
            dtype="O",
        ),
        fixed_field_spec(
            name="variant_id_mask",
            dtype="bool",
        ),
    ]
    name_map = {field.full_name: field for field in icf.metadata.fields}

    # Only three of the fixed fields have a direct one-to-one mapping.
    array_specs.extend(
        [
            spec_from_field(name_map["QUAL"], array_name="variant_quality"),
            spec_from_field(name_map["POS"], array_name="variant_position"),
            spec_from_field(name_map["rlen"], array_name="variant_length"),
        ]
    )
    array_specs.extend([spec_from_field(field) for field in icf.metadata.info_fields])

    gt_field = None
    for field in icf.metadata.format_fields:
        if field.name == "GT":
            gt_field = field
            continue
        array_specs.append(spec_from_field(field))

    if gt_field is not None and n > 0:
        ploidy = max(gt_field.summary.max_number - 1, 1)
        shape = [m, n]
        chunks = [variants_chunk_size, samples_chunk_size]
        dimensions = ["variants", "samples"]
        array_specs.append(
            schema.ZarrArraySpec.new(
                vcf_field=None,
                name="call_genotype_phased",
                dtype="bool",
                shape=list(shape),
                chunks=list(chunks),
                dimensions=list(dimensions),
                description="",
            )
        )
        shape += [ploidy]
        chunks += [ploidy]
        dimensions += ["ploidy"]
        array_specs.append(
            schema.ZarrArraySpec.new(
                vcf_field=None,
                name="call_genotype",
                dtype=gt_field.smallest_dtype(),
                shape=list(shape),
                chunks=list(chunks),
                dimensions=list(dimensions),
                description="",
            )
        )
        array_specs.append(
            schema.ZarrArraySpec.new(
                vcf_field=None,
                name="call_genotype_mask",
                dtype="bool",
                shape=list(shape),
                chunks=list(chunks),
                dimensions=list(dimensions),
                description="",
            )
        )

    if local_alleles:
        array_specs = convert_local_allele_field_types(array_specs)

    return schema.VcfZarrSchema(
        format_version=schema.ZARR_SCHEMA_FORMAT_VERSION,
        samples_chunk_size=samples_chunk_size,
        variants_chunk_size=variants_chunk_size,
        fields=array_specs,
        samples=icf.metadata.samples,
        contigs=icf.metadata.contigs,
        filters=icf.metadata.filters,
    )


def compute_la_field(genotypes):
    """
    Computes the value of the LA field for each sample given the genotypes
    for a variant. The LA field lists the unique alleles observed for
    each sample, including the REF.
    """
    v = 2**31 - 1
    if np.any(genotypes >= v):
        raise ValueError("Extreme allele value not supported")
    G = genotypes.astype(np.int32)
    if len(G) > 0:
        # Anything < 0 gets mapped to -2 (pad) in the output, which comes last.
        # So, to get this sorting correctly, we remap to the largest value for
        # sorting, then map back. We promote the genotypes up to 32 bit for convenience
        # here, assuming that we'll never have a allele of 2**31 - 1.
        assert np.all(G != v)
        G[G < 0] = v
        G.sort(axis=1)
        G[G[:, 0] == G[:, 1], 1] = -2
        # Equal values result in padding also
        G[G == v] = -2
    return G.astype(genotypes.dtype)


def compute_lad_field(ad, la):
    assert ad.shape[0] == la.shape[0]
    assert la.shape[1] == 2
    lad = np.full((ad.shape[0], 2), -2, dtype=ad.dtype)
    homs = np.where((la[:, 0] != -2) & (la[:, 1] == -2))
    lad[homs, 0] = ad[homs, la[homs, 0]]
    hets = np.where(la[:, 1] != -2)
    lad[hets, 0] = ad[hets, la[hets, 0]]
    lad[hets, 1] = ad[hets, la[hets, 1]]
    return lad


def pl_index(a, b):
    """
    Returns the PL index for alleles a and b.
    """
    return b * (b + 1) // 2 + a


def compute_lpl_field(pl, la):
    lpl = np.full((pl.shape[0], 3), -2, dtype=pl.dtype)

    homs = np.where((la[:, 0] != -2) & (la[:, 1] == -2))
    a = la[homs, 0]
    lpl[homs, 0] = pl[homs, pl_index(a, a)]

    hets = np.where(la[:, 1] != -2)[0]
    a = la[hets, 0]
    b = la[hets, 1]
    lpl[hets, 0] = pl[hets, pl_index(a, a)]
    lpl[hets, 1] = pl[hets, pl_index(a, b)]
    lpl[hets, 2] = pl[hets, pl_index(b, b)]

    return lpl


@dataclasses.dataclass
class LocalisableFieldDescriptor:
    array_name: str
    vcf_field: str
    sanitise: callable
    convert: callable


localisable_fields = [
    LocalisableFieldDescriptor(
        "call_LAD", "FORMAT/AD", icf.sanitise_int_array, compute_lad_field
    ),
    LocalisableFieldDescriptor(
        "call_LPL", "FORMAT/PL", icf.sanitise_int_array, compute_lpl_field
    ),
]


def mkschema(
    if_path,
    out,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    local_alleles=None,
):
    store = icf.IntermediateColumnarFormat(if_path)
    spec = generate_schema(
        store,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        local_alleles=local_alleles,
    )
    out.write(spec.asjson())


def encode(
    if_path,
    zarr_path,
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    max_variant_chunks=None,
    dimension_separator=None,
    max_memory=None,
    local_alleles=None,
    worker_processes=1,
    show_progress=False,
):
    # Rough heuristic to split work up enough to keep utilisation high
    target_num_partitions = max(1, worker_processes * 4)
    encode_init(
        if_path,
        zarr_path,
        target_num_partitions,
        schema_path=schema_path,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        local_alleles=local_alleles,
        max_variant_chunks=max_variant_chunks,
        dimension_separator=dimension_separator,
    )
    vzw = writer.VcfZarrWriter(zarr_path)
    vzw.encode_all_partitions(
        worker_processes=worker_processes,
        show_progress=show_progress,
        max_memory=max_memory,
    )
    vzw.finalise(show_progress)
    vzw.create_index()


def encode_init(
    icf_path,
    zarr_path,
    target_num_partitions,
    *,
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    local_alleles=None,
    max_variant_chunks=None,
    dimension_separator=None,
    max_memory=None,
    worker_processes=1,
    show_progress=False,
):
    icf_store = icf.IntermediateColumnarFormat(icf_path)
    if schema_path is None:
        schema_instance = generate_schema(
            icf_store,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            local_alleles=local_alleles,
        )
    else:
        logger.info(f"Reading schema from {schema_path}")
        if variants_chunk_size is not None or samples_chunk_size is not None:
            raise ValueError(
                "Cannot specify schema along with chunk sizes"
            )  # NEEDS TEST
        with open(schema_path) as f:
            schema_instance = schema.VcfZarrSchema.fromjson(f.read())
    zarr_path = pathlib.Path(zarr_path)
    vzw = writer.VcfZarrWriter(zarr_path)
    return vzw.init(
        icf_store,
        target_num_partitions=target_num_partitions,
        schema=schema_instance,
        dimension_separator=dimension_separator,
        max_variant_chunks=max_variant_chunks,
    )


def encode_partition(zarr_path, partition):
    writer_instance = writer.VcfZarrWriter(zarr_path)
    writer_instance.encode_partition(partition)


def encode_finalise(zarr_path, show_progress=False):
    writer_instance = writer.VcfZarrWriter(zarr_path)
    writer_instance.finalise(show_progress=show_progress)


def convert(
    vcfs,
    out_path,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=1,
    local_alleles=None,
    show_progress=False,
    icf_path=None,
):
    """
    Convert VCF files to zarr format using the shared writer infrastructure.
    """
    if icf_path is None:
        cm = temp_icf_path(prefix="vcf2zarr")
    else:
        cm = contextlib.nullcontext(icf_path)

    with cm as icf_path:
        icf.explode(
            icf_path,
            vcfs,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )

        # Create ICF store
        icf_store = icf.IntermediateColumnarFormat(icf_path)

        # Generate schema
        schema_instance = generate_schema(
            icf_store,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            local_alleles=local_alleles,
        )

        # Create a VCF data adapter
        vcf_adapter = VcfDataAdapter(icf_store)

        # Use the generic writer
        from bio2zarr.writer import GenericZarrWriter

        writer_instance = GenericZarrWriter(out_path)
        writer_instance.init_from_schema(schema_instance)

        # Encode data using the writer
        logger.info(f"Converting VCF data to zarr at {out_path}")
        writer_instance.encode_data(
            vcf_adapter,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )

        # Finalize and index the zarr store
        writer_instance.finalise(show_progress)

        # Create an index if needed
        index_creator = VcfZarrIndexer(out_path)
        index_creator.create_index()


class VcfDataAdapter:
    """
    Adapter class to provide VCF data from an ICF store to the generic writer.
    """

    def __init__(self, icf_store):
        self.icf = icf_store
        self.n_samples = self.icf.num_samples
        self.n_variants = self.icf.num_records
        self.field_lookup = {
            field.full_name: field for field in self.icf.metadata.fields
        }

    def get_sample_ids(self):
        return [sample.id for sample in self.icf.metadata.samples]

    def get_variant_positions(self):
        position_field = self.icf.fields["POS"]
        positions = np.empty(self.n_variants, dtype=np.int32)

        # Read all positions
        for i, value in enumerate(position_field.iter_values(0, self.n_variants)):
            positions[i] = value[0]

        return positions

    def get_variant_alleles(self):
        ref_field = self.icf.fields["REF"]
        alt_field = self.icf.fields["ALT"]

        max_alleles = alt_field.vcf_field.summary.max_number + 1
        alleles = np.empty((self.n_variants, max_alleles), dtype=object)

        # Read all alleles
        for i, (ref, alt) in enumerate(
            zip(
                ref_field.iter_values(0, self.n_variants),
                alt_field.iter_values(0, self.n_variants),
            )
        ):
            alleles[i, 0] = ref[0]
            for j, a in enumerate(alt):
                if j < max_alleles - 1:
                    alleles[i, j + 1] = a

        return alleles

    def get_genotypes_slice(self, start, stop):
        """
        Read a slice of genotypes from the VCF data.
        Returns a dictionary with the genotype arrays for this slice.
        """
        result = {}

        # Get genotype data
        if "FORMAT/GT" in self.icf.fields:
            gt_field = self.icf.fields["FORMAT/GT"]
            n_variants = stop - start

            # Create return arrays
            gt = np.zeros((n_variants, self.n_samples, 2), dtype=np.int8)
            gt_phased = np.zeros((n_variants, self.n_samples), dtype=bool)
            gt_mask = np.zeros((n_variants, self.n_samples, 2), dtype=bool)

            # Fill arrays
            for i, value in enumerate(gt_field.iter_values(start, stop)):
                if value is not None:
                    gt[i] = value[:, :-1]
                    gt_phased[i] = value[:, -1]
                    gt_mask[i] = gt[i] < 0

            result["call_genotype"] = gt
            result["call_genotype_phased"] = gt_phased
            result["call_genotype_mask"] = gt_mask

            # Handle local alleles if needed
            if hasattr(self, "handle_local_alleles") and self.handle_local_alleles:
                la = compute_la_field(gt)
                result["call_LA"] = la

                # Handle other localisable fields
                for descriptor in localisable_fields:
                    if descriptor.vcf_field in self.icf.fields:
                        source_field = self.icf.fields[descriptor.vcf_field]
                        values = []
                        for value in source_field.iter_values(start, stop):
                            values.append(descriptor.sanitise(value, 2, value.dtype))
                        values = np.array(values)

                        # Convert to local allele representation
                        local_values = descriptor.convert(values, la)
                        result[descriptor.array_name] = local_values

        return result

    def close(self):
        # Clean up resources if needed
        pass


class VcfZarrIndexer:
    """
    Creates an index for efficient region queries in a VCF Zarr dataset.
    """

    def __init__(self, path):
        self.path = pathlib.Path(path)

    def create_index(self):
        """Create an index to support efficient region queries."""
        root = zarr.open_group(store=self.path, mode="r+")

        if (
            "variant_contig" not in root
            or "variant_position" not in root
            or "variant_length" not in root
        ):
            logger.warning("Cannot create index: required arrays not found")
            return

        contig = root["variant_contig"]
        pos = root["variant_position"]
        length = root["variant_length"]

        assert contig.cdata_shape == pos.cdata_shape

        index = []

        logger.info("Creating region index")
        for v_chunk in range(pos.cdata_shape[0]):
            c = contig.blocks[v_chunk]
            p = pos.blocks[v_chunk]
            e = p + length.blocks[v_chunk] - 1

            # create a row for each contig in the chunk
            d = np.diff(c, append=-1)
            c_start_idx = 0
            for c_end_idx in np.nonzero(d)[0]:
                assert c[c_start_idx] == c[c_end_idx]
                index.append(
                    (
                        v_chunk,  # chunk index
                        c[c_start_idx],  # contig ID
                        p[c_start_idx],  # start
                        p[c_end_idx],  # end
                        np.max(e[c_start_idx : c_end_idx + 1]),  # max end
                        c_end_idx - c_start_idx + 1,  # num records
                    )
                )
                c_start_idx = c_end_idx + 1

        index = np.array(index, dtype=pos.dtype)
        kwargs = {}
        if not zarr_utils.zarr_v3():
            kwargs["dimension_separator"] = "/"
        array = root.array(
            "region_index",
            data=index,
            shape=index.shape,
            chunks=index.shape,
            dtype=index.dtype,
            compressor=numcodecs.Blosc("zstd", clevel=9, shuffle=0),
            fill_value=None,
            **kwargs,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = [
            "region_index_values",
            "region_index_fields",
        ]

        logger.info("Consolidating Zarr metadata")
        zarr.consolidate_metadata(self.path)


@contextlib.contextmanager
def temp_icf_path(prefix=None):
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        yield pathlib.Path(tmp) / "icf"
