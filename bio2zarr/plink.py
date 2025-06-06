import dataclasses
import logging
import pathlib

import numpy as np
import pandas as pd

from bio2zarr import constants, core, vcz

logger = logging.getLogger(__name__)


FAM_FIELDS = [
    ("family_id", str, "U"),
    ("individual_id", str, "U"),
    ("paternal_id", str, "U"),
    ("maternal_id", str, "U"),
    ("sex", str, "int8"),
    ("phenotype", str, "int8"),
]
FAM_DF_DTYPE = dict([(f[0], f[1]) for f in FAM_FIELDS])
FAM_ARRAY_DTYPE = dict([(f[0], f[2]) for f in FAM_FIELDS])

BIM_FIELDS = [
    ("contig", str, "U"),
    ("variant_id", str, "U"),
    ("cm_position", "float32", "float32"),
    ("position", "int32", "int32"),
    ("allele_1", str, "S"),
    ("allele_2", str, "S"),
]
BIM_DF_DTYPE = dict([(f[0], f[1]) for f in BIM_FIELDS])
BIM_ARRAY_DTYPE = dict([(f[0], f[2]) for f in BIM_FIELDS])


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
            f"Reading {chunk_size} variants ({bytes_to_read} bytes) "
            f"from {self.path}"
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

    def iter_field(self, field_name, shape, start, stop):
        assert field_name == "position"  # Only position field is supported from plink
        yield from self.bim.position[start:stop]

    def iter_id(self, start, stop):
        yield from self.bim.variant_id[start:stop]

    def iter_alleles_and_genotypes(self, start, stop, shape, num_alleles):
        alt_iter = self.bim.allele_1.values[start:stop]
        ref_iter = self.bim.allele_2.values[start:stop]
        gt_iter = self.bed_reader.iter_decode(start, stop)
        for alt, ref, gt in zip(alt_iter, ref_iter, gt_iter):
            alleles = np.full(num_alleles, constants.STR_FILL, dtype="O")
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
        max_len = self.bim.allele_2.values.itemsize

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
                dtype="O",
                dimensions=["variants", "alleles"],
                description=None,
            ),
            vcz.ZarrArraySpec(
                name="variant_id",
                dtype="O",
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
                compressor=vcz.DEFAULT_ZARR_COMPRESSOR_BOOL.get_config(),
            ),
            vcz.ZarrArraySpec(
                name="call_genotype",
                dtype="i1",
                dimensions=["variants", "samples", "ploidy"],
                description=None,
                compressor=vcz.DEFAULT_ZARR_COMPRESSOR_GENOTYPES.get_config(),
            ),
            vcz.ZarrArraySpec(
                name="call_genotype_mask",
                dtype="bool",
                dimensions=["variants", "samples", "ploidy"],
                description=None,
                compressor=vcz.DEFAULT_ZARR_COMPRESSOR_BOOL.get_config(),
            ),
        ]
        schema_instance.fields = array_specs
        return schema_instance


def convert(
    prefix,
    out,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=core.DEFAULT_WORKER_PROCESSES,
    show_progress=False,
):
    plink_format = PlinkFormat(prefix)
    schema_instance = plink_format.generate_schema(
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
    )
    zarr_path = pathlib.Path(out)
    vzw = vcz.VcfZarrWriter(PlinkFormat, zarr_path)
    # Rough heuristic to split work up enough to keep utilisation high
    target_num_partitions = max(1, worker_processes * 4)
    vzw.init(
        plink_format,
        target_num_partitions=target_num_partitions,
        schema=schema_instance,
    )
    vzw.encode_all_partitions(
        worker_processes=worker_processes,
        show_progress=show_progress,
    )
    vzw.finalise(show_progress)
    vzw.create_index()
