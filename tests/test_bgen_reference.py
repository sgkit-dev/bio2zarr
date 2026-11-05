"""Independent Oxford BGEN reference files; provenance in data/bgen/README.md."""

import gzip
import hashlib
import pathlib
import struct

import cbgen
import numpy as np
import numpy.testing as nt
import pytest

from bio2zarr import bgen
from bio2zarr.vcz_verification import assert_all_fill_float, assert_all_missing_float

DATA = pathlib.Path(__file__).parent / "data" / "bgen"


def test_fixture_checksums():
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest


def example_prefix(tmp_path):
    """Select original records 0..97 without re-encoding their probability bytes.

    Oxford's full example is not sorted (record 98 goes from 99000 to 10000).
    Retain its first sorted run for conversion; test full-file rejection too.
    """
    path = DATA / "example.8bits.bgen"
    index_path = tmp_path / "example.metafile"
    with cbgen.BgenFile(path) as source:
        source.create_metafile(str(index_path))
    with cbgen.BgenMetafile(str(index_path)) as index:
        offset = index.read_partition(0).variants[97].offset
    data = bytearray(path.read_bytes())
    block_size = struct.unpack_from("<I", data, offset)[0]
    data = data[: offset + 4 + block_size]
    struct.pack_into("<I", data, 8, 98)
    out = tmp_path / "example-prefix.bgen"
    out.write_bytes(data)
    return out


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_oxford_probabilities_against_gen(tmp_path, zarr_format):
    path = example_prefix(tmp_path)
    with gzip.open(DATA / "example.first98.gen.gz", "rt") as stream:
        rows = [line.split() for line in stream][:98]
    expected = np.array(
        [np.array(row[6:], dtype=float).reshape(500, 3) for row in rows]
    )
    missing = expected.sum(axis=2) == 0
    root = bgen.convert(
        path,
        variants_chunk_size=41,
        samples_chunk_size=113,
        zarr_format=zarr_format,
    )
    assert root["call_GP"].shape == (98, 500, 3)
    nt.assert_array_equal(
        root["sample_id"][:], (DATA / "example.sample").read_text().splitlines()[2:]
    )
    nt.assert_array_equal(root["contig_id"][:], ["01"])
    nt.assert_array_equal(root["variant_position"][:], [int(row[3]) for row in rows])
    nt.assert_array_equal(root["variant_bgen_id"][:], [row[1] for row in rows])
    nt.assert_array_equal(root["variant_id"][:], [row[2] for row in rows])
    nt.assert_array_equal(root["variant_allele"][:], [row[4:6] for row in rows])
    nt.assert_array_equal(root["call_BGEN_MISSING"][:], missing)
    assert missing.sum() == 2
    nt.assert_array_equal(root["call_PLOIDY"][:], np.full((98, 500), 2))
    nt.assert_array_equal(root["variant_bgen_phased"][:], np.zeros(98, dtype=bool))
    actual = root["call_GP"][:]
    assert_all_missing_float(actual[missing])
    # Companion GEN and 8-bit BGEN are independently quantized representations.
    nt.assert_allclose(actual[~missing], expected[~missing], rtol=0, atol=1 / 255)
    nt.assert_allclose(actual[~missing].sum(axis=1), 1, rtol=0, atol=1e-14)
    ds = root["call_DS"][:]
    assert_all_missing_float(ds[missing])
    expected_ds = expected[:, :, 1] + 2 * expected[:, :, 2]
    expected_ds[missing] = np.nan
    nt.assert_allclose(ds[~missing], expected_ds[~missing], rtol=0, atol=3 / 255)
    # Independent per-variant aggregate dosage and frequency catch orientation
    # and missing-denominator errors in addition to full vector comparisons.
    nt.assert_allclose(
        np.nansum(ds, axis=1),
        np.nansum(expected_ds, axis=1),
        rtol=0,
        atol=500 * 3 / 255,
    )
    denominator = 2 * (~missing).sum(axis=1)
    nt.assert_allclose(
        np.nansum(ds, axis=1) / denominator,
        np.nansum(expected_ds, axis=1) / denominator,
        rtol=0,
        atol=3 / 510,
    )


def test_oxford_full_example_is_unsorted(tmp_path):
    out = tmp_path / "out.vcz"
    with pytest.raises(ValueError, match="positions must be nondecreasing"):
        bgen.convert(DATA / "example.8bits.bgen", out)
    assert not out.exists()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_oxford_complex_against_vcf(tmp_path, zarr_format):
    lines = (DATA / "complex.vcf").read_text().splitlines()
    header = next(line for line in lines if line.startswith("#CHROM")).split("\t")
    rows = [line.split("\t") for line in lines if not line.startswith("#")]
    root = bgen.convert(
        DATA / "complex.23bits.bgen",
        variants_chunk_size=3,
        samples_chunk_size=3,
        zarr_format=zarr_format,
    )
    nt.assert_array_equal(root["sample_id"][:], header[9:])
    nt.assert_array_equal(root["variant_position"][:], [int(row[1]) for row in rows])
    nt.assert_array_equal(root["contig_id"][:], ["01"])
    assert root["call_GP"].shape[0:2] == (10, 4)
    assert not root["call_BGEN_MISSING"][:].any()
    # Upstream encodes all-haploid records as phased although its companion VCF
    # labels their identical probability vectors GP rather than HP.
    expected_phase = [False, True, True, False, True, True, True, True, False, False]
    nt.assert_array_equal(root["variant_bgen_phased"][:], expected_phase)
    for j, row in enumerate(rows):
        alleles = [row[3], *row[4].split(",")]
        nt.assert_array_equal(root["variant_allele"][j, : len(alleles)], alleles)
        assert list(root["variant_allele"][j, len(alleles) :]) == [""] * (
            8 - len(alleles)
        )
        ids = row[2].split(",")
        assert root["variant_id"][j] == ids[0]
        assert root["variant_bgen_id"][j] == (ids[1] if len(ids) > 1 else ".")
        phased = expected_phase[j]
        field = "call_HP" if phased else "call_GP"
        if row[8].endswith("GP") and phased:
            assert all("/" not in call.split(":")[0] for call in row[9:])
        for s, call in enumerate(row[9:]):
            gt, probability = call.split(":")
            genotype = [int(a) for a in gt.replace("|", "/").split("/")]
            expected = np.array(probability.split(","), dtype=float)
            assert root["call_PLOIDY"][j, s] == len(genotype)
            nt.assert_allclose(root[field][j, s, : len(expected)], expected, atol=1e-7)
            assert_all_fill_float(root[field][j, s, len(expected) :])
            other = "call_GP" if phased else "call_HP"
            assert_all_missing_float(root[other][j, s, :1])
            assert_all_fill_float(root[other][j, s, 1:])
            dosage = [genotype.count(a) for a in range(1, len(alleles))]
            nt.assert_array_equal(root["call_DS"][j, s, : len(dosage)], dosage)
            assert_all_fill_float(root["call_DS"][j, s, len(dosage) :])


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_oxford_haplotypes_against_haps(tmp_path, zarr_format):
    rows = [
        line.split() for line in (DATA / "haplotypes.haps").read_text().splitlines()
    ]
    haps = np.array([row[6:] for row in rows], dtype=int).reshape(4, 4, 2)
    root = bgen.convert(
        DATA / "haplotypes.bgen",
        variants_chunk_size=3,
        samples_chunk_size=3,
        zarr_format=zarr_format,
    )
    assert "call_GP" not in root
    assert root["variant_bgen_phased"][:].all()
    assert not root["call_BGEN_MISSING"][:].any()
    nt.assert_array_equal(root["sample_id"][:], [f"sample_{j}" for j in range(4)])
    nt.assert_array_equal(root["variant_bgen_id"][:], [row[1] for row in rows])
    nt.assert_array_equal(root["variant_id"][:], [row[2] for row in rows])
    nt.assert_array_equal(root["variant_position"][:], [int(row[3]) for row in rows])
    nt.assert_array_equal(root["variant_allele"][:], [row[4:6] for row in rows])
    nt.assert_array_equal(root["call_PLOIDY"][:], np.full((4, 4), 2))
    hp = np.eye(2)[haps].reshape(4, 4, 4)
    nt.assert_array_equal(root["call_HP"][:], hp)
    nt.assert_array_equal(root["call_DS"][:], haps.sum(axis=2))
