import dataclasses
import logging
import os
import pathlib
import warnings

import numpy as np

from bio2zarr import constants, core, vcz

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PlinkPaths:
    bed_path: str
    bim_path: str
    fam_path: str


@dataclasses.dataclass
class FamData:
    sid: np.ndarray
    sid_count: int


@dataclasses.dataclass
class BimData:
    chromosome: np.ndarray
    vid: np.ndarray
    bp_position: np.ndarray
    allele_1: np.ndarray
    allele_2: np.ndarray
    vid_count: int


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

        # Read sample information from .fam file
        samples = []
        with open(self.paths.fam_path) as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) >= 2:  # At minimum, we need FID and IID
                    samples.append(fields[1])
        self.fam = FamData(sid=np.array(samples), sid_count=len(samples))
        self.n_samples = len(samples)

        # Read variant information from .bim file
        chromosomes = []
        vids = []
        positions = []
        allele1 = []
        allele2 = []

        with open(self.paths.bim_path) as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) >= 6:
                    chrom, vid, _, pos, a1, a2 = (
                        fields[0],
                        fields[1],
                        fields[2],
                        fields[3],
                        fields[4],
                        fields[5],
                    )
                    chromosomes.append(chrom)
                    vids.append(vid)
                    positions.append(int(pos))
                    allele1.append(a1)
                    allele2.append(a2)

        self.bim = BimData(
            chromosome=np.array(chromosomes),
            vid=np.array(vids),
            bp_position=np.array(positions),
            allele_1=np.array(allele1),
            allele_2=np.array(allele2),
            vid_count=len(vids),
        )
        self.n_variants = len(vids)

        # Calculate bytes per SNP: 1 byte per 4 samples, rounded up
        self.bytes_per_snp = (self.n_samples + 3) // 4

        # Verify BED file has correct magic bytes
        with open(self.paths.bed_path, "rb") as f:
            magic = f.read(3)
            assert magic == b"\x6c\x1b\x01", "Invalid BED file format"

        expected_size = self.n_variants * self.bytes_per_snp + 3  # +3 for magic bytes
        actual_size = os.path.getsize(self.paths.bed_path)
        if actual_size < expected_size:
            raise ValueError(
                f"BED file is truncated: expected at least {expected_size} bytes, "
                f"but only found {actual_size} bytes. "
                f"Check that .bed, .bim, and .fam files match."
            )
        elif actual_size > expected_size:
            # Warn if there's extra data (might indicate file mismatch)
            warnings.warn(
                f"BED file contains {actual_size} bytes but only expected "
                f"{expected_size}. "
                f"Using first {expected_size} bytes only.",
                stacklevel=1,
            )

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

    @property
    def path(self):
        return self.prefix

    @property
    def num_records(self):
        return self.bim.vid_count

    @property
    def samples(self):
        return [vcz.Sample(id=sample) for sample in self.fam.sid]

    @property
    def contigs(self):
        return [vcz.Contig(id=str(chrom)) for chrom in np.unique(self.bim.chromosome)]

    @property
    def num_samples(self):
        return len(self.samples)

    def iter_contig(self, start, stop):
        chrom_to_contig_index = {contig.id: i for i, contig in enumerate(self.contigs)}
        for chrom in self.bim.chromosome[start:stop]:
            yield chrom_to_contig_index[str(chrom)]

    def iter_field(self, field_name, shape, start, stop):
        assert field_name == "position"  # Only position field is supported from plink
        yield from self.bim.bp_position[start:stop]

    def iter_id(self, start, stop):
        yield from self.bim.vid[start:stop]

    def iter_alleles_and_genotypes(self, start, stop, shape, num_alleles):
        alt_field = self.bim.allele_1
        ref_field = self.bim.allele_2

        chunk_size = stop - start

        # Calculate file offsets for the required data
        # 3 bytes for the magic number at the beginning of the file
        start_offset = 3 + (start * self.bytes_per_snp)
        bytes_to_read = chunk_size * self.bytes_per_snp

        # Read only the needed portion of the BED file
        with open(self.paths.bed_path, "rb") as f:
            f.seek(start_offset)
            chunk_data = f.read(bytes_to_read)

        data_bytes = np.frombuffer(chunk_data, dtype=np.uint8)
        data_matrix = data_bytes.reshape(chunk_size, self.bytes_per_snp)

        # Apply lookup table to get genotypes
        # Shape becomes: (chunk_size, bytes_per_snp, 4, 2)
        all_genotypes = self.byte_lookup[data_matrix]

        # Reshape to get all samples in one dimension
        # (chunk_size, bytes_per_snp*4, 2)
        samples_padded = self.bytes_per_snp * 4
        genotypes_reshaped = all_genotypes.reshape(chunk_size, samples_padded, 2)

        gt = genotypes_reshaped[:, : self.n_samples]

        phased = np.zeros((chunk_size, self.n_samples), dtype=bool)

        for i, (ref, alt) in enumerate(
            zip(ref_field[start:stop], alt_field[start:stop])
        ):
            alleles = np.full(num_alleles, constants.STR_FILL, dtype="O")
            alleles[0] = ref
            alleles[1 : 1 + len(alt)] = alt

            # rlen is the length of the REF in PLINK as there's no END annotations
            yield vcz.VariantData(len(alleles[0]), alleles, gt[i], phased[i])

    def generate_schema(
        self,
        variants_chunk_size=None,
        samples_chunk_size=None,
    ):
        n = self.fam.sid_count
        m = self.bim.vid_count
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
        max_len = self.bim.allele_2.itemsize

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
                dtype=core.min_int_dtype(0, len(np.unique(self.bim.chromosome))),
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
    worker_processes=1,
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
