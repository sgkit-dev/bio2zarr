import contextlib
import logging
import pathlib
import tempfile

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
    icf_path,
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
        icf_path,
        zarr_path,
        target_num_partitions,
        schema_path=schema_path,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        local_alleles=local_alleles,
        max_variant_chunks=max_variant_chunks,
        dimension_separator=dimension_separator,
    )
    vzw = writer.VcfZarrWriter("icf", zarr_path)
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
    vzw = writer.VcfZarrWriter("icf", zarr_path)
    return vzw.init(
        icf_store,
        target_num_partitions=target_num_partitions,
        schema=schema_instance,
        dimension_separator=dimension_separator,
        max_variant_chunks=max_variant_chunks,
    )


def encode_partition(zarr_path, partition):
    writer_instance = writer.VcfZarrWriter("icf", zarr_path)
    writer_instance.encode_partition(partition)


def encode_finalise(zarr_path, show_progress=False):
    writer_instance = writer.VcfZarrWriter("icf", zarr_path)
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
        encode(
            icf_path,
            out_path,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            worker_processes=worker_processes,
            show_progress=show_progress,
            local_alleles=local_alleles,
        )


@contextlib.contextmanager
def temp_icf_path(prefix=None):
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        yield pathlib.Path(tmp) / "icf"
