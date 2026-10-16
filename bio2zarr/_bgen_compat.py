"""Compatibility paths for encodings not exposed safely by bgen-limix's bindings."""

import contextlib
import dataclasses
import json
import os
import sqlite3
import struct
import types
import zlib

import numcodecs
import numpy as np


def read_exact(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Truncated BGEN record")
    return value


@dataclasses.dataclass
class Header:
    offset: int
    length: int
    variants: int
    samples: int
    flags: int

    @property
    def layout(self):
        return (self.flags >> 2) & 15

    @property
    def compression(self):
        return self.flags & 3


def read_header(path):
    with open(path, "rb") as f:
        value = f.read(20)
        if len(value) != 20:
            raise ValueError("Truncated BGEN header")
        offset, length, variants, samples, magic = struct.unpack("<IIII4s", value)
        if magic not in (b"bgen", b"\0" * 4) or length < 20 or offset < length:
            raise ValueError("Invalid BGEN header")
        f.seek(length)
        flags = f.read(4)
        if len(flags) != 4:
            raise ValueError("Truncated BGEN flags")
    header = Header(offset, length, variants, samples, int.from_bytes(flags, "little"))
    if header.layout not in (1, 2):
        raise ValueError(f"Unsupported BGEN layout {header.layout}")
    if header.compression == 3 or (header.layout == 1 and header.compression == 2):
        raise ValueError(
            f"Unsupported BGEN compression {header.compression} "
            f"for layout {header.layout}"
        )
    if variants == 0 or samples == 0:
        raise ValueError("BGEN conversion requires at least one variant and sample")
    if max(variants, samples) > np.iinfo(np.int32).max:
        raise ValueError("bgen-limix supports at most 2**31-1 variants and samples")
    return header


def read_string(stream, width):
    size = int.from_bytes(read_exact(stream, width), "little")
    return read_exact(stream, size).decode("utf-8")


def read_variant(stream, header):
    if header.layout == 1:
        samples = int.from_bytes(read_exact(stream, 4), "little")
        if samples != header.samples:
            raise ValueError("BGEN layout 1 variant sample count differs from header")
    id = read_string(stream, 2)
    rsid = read_string(stream, 2)
    chrom = read_string(stream, 2)
    position = int.from_bytes(read_exact(stream, 4), "little")
    nalleles = (
        2 if header.layout == 1 else int.from_bytes(read_exact(stream, 2), "little")
    )
    if nalleles < 2:
        raise ValueError("BGEN records must have at least two alleles")
    alleles = [read_string(stream, 4) for _ in range(nalleles)]
    return types.SimpleNamespace(
        id=id,
        rsid=rsid,
        chrom=chrom,
        position=position,
        nalleles=nalleles,
        allele_ids=alleles,
        offset=stream.tell(),
    )


def copy_exact(source, dest, size):
    while size:
        block = read_exact(source, min(size, 1024 * 1024))
        dest.write(block)
        size -= len(block)


def compress_layout2(path, dest, header):
    """Losslessly frame raw layout-2 blocks for the backend's compressed reader."""
    with open(path, "rb") as source, open(dest, "w+b") as out:
        copy_exact(source, out, header.offset + 4)
        out.seek(header.length)
        out.write(struct.pack("<I", header.flags | 1))
        out.seek(0, os.SEEK_END)
        for _ in range(header.variants):
            start = source.tell()
            variant = read_variant(source, header)
            source.seek(start)
            copy_exact(source, out, variant.offset - start)
            size = int.from_bytes(read_exact(source, 4), "little")
            block = read_exact(source, size)
            compressed = zlib.compress(block)
            out.write(struct.pack("<II", len(compressed) + 4, size))
            out.write(compressed)


def write_string(stream, value, width):
    encoded = value.encode("utf-8")
    stream.write(len(encoded).to_bytes(width, "little"))
    stream.write(encoded)


def prepare_layout1(path, dest, index_path, header):
    """Transport layout-1 uint16 probabilities through the safe layout-2 reader.

    Three independently rounded q/32768 probabilities cannot be represented as
    three probabilities with denominator 2**B-1 without losing precision.
    Instead carry the original three integers as a four-state haploid vector
    with denominator 65535. The fourth state is discarded after native decode,
    and the original integers recovered exactly before dividing by 32768.
    """
    with (
        open(path, "rb") as source,
        open(dest, "w+b") as out,
        contextlib.closing(sqlite3.connect(index_path)) as db,
    ):
        db.execute("CREATE TABLE variants (ordinal INTEGER PRIMARY KEY, data TEXT)")
        copy_exact(source, out, header.offset + 4)
        out.seek(header.length)
        out.write(struct.pack("<I", (header.flags & ~63) | (2 << 2) | 1))
        out.seek(0, os.SEEK_END)
        for ordinal in range(header.variants):
            variant = read_variant(source, header)
            size = 6 * header.samples
            if header.compression == 0:
                raw = read_exact(source, size)
            else:
                compressed_size = int.from_bytes(read_exact(source, 4), "little")
                compressed = read_exact(source, compressed_size)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(compressed, size + 1)
                if not decoder.eof or decoder.unused_data or len(raw) != size:
                    raise ValueError("Invalid BGEN layout 1 probability block")
            values = np.frombuffer(raw, dtype="<u2").reshape(header.samples, 3)
            sums = values.sum(axis=1, dtype=np.int64)
            missing = sums == 0
            if np.any(values > 32768) or np.any(~missing & (np.abs(sums - 32768) > 3)):
                raise ValueError("Invalid BGEN layout 1 genotype probabilities")
            for value in [variant.id, variant.rsid, variant.chrom]:
                write_string(out, value, 2)
            out.write(struct.pack("<IH", variant.position, 4))
            for allele in ["0", "1", "2", "3"]:
                write_string(out, allele, 4)
            variant.offset = out.tell()
            db.execute(
                "INSERT INTO variants VALUES (?, ?)",
                (ordinal, json.dumps(vars(variant))),
            )
            block = struct.pack("<IHBB", header.samples, 4, 1, 1)
            block += (1 | (missing.astype(np.uint8) << 7)).tobytes()
            block += bytes([0, 16]) + raw
            compressed = zlib.compress(block)
            out.write(struct.pack("<II", len(compressed) + 4, len(block)))
            out.write(compressed)
        db.commit()


def iter_layout1_variants(index_path, start, stop):
    with contextlib.closing(sqlite3.connect(index_path)) as db:
        for (value,) in db.execute(
            "SELECT data FROM variants WHERE ordinal >= ? AND ordinal < ? "
            "ORDER BY ordinal",
            (start, stop),
        ):
            yield types.SimpleNamespace(**json.loads(value))


@contextlib.contextmanager
def open_layout2_reader(path, compression, num_samples):
    import cbgen  # noqa: PLC0415

    with open(path, "rb") as stream, cbgen.BgenFile(path) as source:
        yield Layout2Reader(source, stream, compression, num_samples)


class Layout2Reader:
    def __init__(self, source, stream, compression, num_samples):
        self.source = source
        self.stream = stream
        self.compression = compression
        self.num_samples = num_samples

    def read_genotype(self, offset):
        # The native reader rejects phased records with no haplotypes. Such a
        # record has no probabilities to decode, only ploidy/missingness flags.
        self.stream.seek(offset)
        size, decoded_size = struct.unpack("<II", read_exact(self.stream, 8))
        if decoded_size == self.num_samples + 10:
            if size < 4:
                raise ValueError("Invalid BGEN probability block size")
            encoded = read_exact(self.stream, size - 4)
            if self.compression == 1:
                decoder = zlib.decompressobj()
                block = decoder.decompress(encoded, decoded_size + 1)
                if not decoder.eof or decoder.unused_data:
                    raise ValueError("Invalid BGEN probability block")
            else:
                block = numcodecs.Zstd().decode(encoded, out=bytearray(decoded_size))
            if len(block) != decoded_size:
                raise ValueError("Invalid BGEN probability block size")
            n, k, minimum, maximum = struct.unpack_from("<IHBB", block)
            ploidy = np.frombuffer(
                block, dtype=np.uint8, count=self.num_samples, offset=8
            )
            if (
                n == self.num_samples
                and k >= 2
                and minimum == maximum == 0
                and np.all((ploidy & 127) == 0)
                and block[-2] == 1
                and 1 <= block[-1] <= 32
            ):
                return types.SimpleNamespace(
                    probability=np.empty((n, 0)),
                    phased=True,
                    ploidy=np.zeros(n, dtype=np.uint8),
                    missing=(ploidy >> 7).astype(bool),
                )
        return self.source.read_genotype(offset)


class Layout1Reader:
    """Decode staged integer transport using the public bgen-limix Python API."""

    def __init__(self, path, num_samples):
        import cbgen  # noqa: PLC0415

        self.source = cbgen.BgenFile(path)
        self.num_samples = num_samples

    def read_genotype(self, offset):
        genotype = self.source.read_genotype(offset)
        probability = np.rint(genotype.probability[:, :3] * 65535) / 32768
        return types.SimpleNamespace(
            probability=probability,
            phased=False,
            ploidy=np.full(self.num_samples, 2, dtype=np.uint8),
            missing=genotype.missing,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.source.close()
