import math
import pathlib
import struct
import sys
import types
import zlib
from unittest import mock

import cbgen
import numpy as np
import numpy.testing as nt
import pytest
import zstandard

from bio2zarr import bgen, constants, vcf
from bio2zarr.vcz_verification import (
    assert_all_fill_float,
    assert_all_missing_float,
    float_int_view,
)
from tests.test_bgen import encode_string, write_bgen


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phased": True},
        {"ploidy": (1, 2, 3, 2)},
        {"alleles": ("A", "C", "G")},
        {"compression": 0},
    ],
)
def test_general_conversion(tmp_path, kwargs):
    path = tmp_path / "general.bgen"
    write_bgen(path, **kwargs)
    root = bgen.convert(path, variants_chunk_size=2, samples_chunk_size=3)
    assert "call_DS" in root
    assert "call_genotype" not in root
    if kwargs.get("phased"):
        nt.assert_array_equal(root["call_HP"][0, 0], [0, 1, 0, 1])
        assert root["call_DS"][0, 0] == 2
    elif "ploidy" in kwargs:
        nt.assert_array_equal(root["call_PLOIDY"][0], [1, 2, 3, 2])
        nt.assert_array_equal(root["call_DS"][0, :3], [1, 2, 3])
    elif "alleles" in kwargs:
        nt.assert_array_equal(root["call_GP"][0, 0], [0, 0, 0, 0, 0, 1])
        nt.assert_array_equal(root["call_DS"][0, 0], [0, 2])
    else:
        nt.assert_array_equal(root["call_GP"][0, 0], [1, 0, 0])
    assert np.isnan(root["call_DS"][2]).all()


def write_layout1(path, compression, probabilities=None, nvariants=3):
    if probabilities is None:
        probabilities = np.array(
            [
                [32768, 0, 0],
                [0, 32768, 0],
                [6553, 10922, 15292],
                [0, 0, 0],
            ],
            dtype="<u2",
        )
    nsamples = len(probabilities)
    with open(path, "wb") as out:
        out.write(
            struct.pack(
                "<IIII4sI", 20, 20, nvariants, nsamples, b"bgen", (1 << 2) | compression
            )
        )
        for j in range(nvariants):
            out.write(struct.pack("<I", nsamples))
            for value in [f"snp_{j}", f"rs{j}", "1"]:
                out.write(encode_string(value))
            out.write(struct.pack("<I", j + 1))
            out.write(encode_string("A", 4) + encode_string("T", 4))
            block = (
                probabilities if j != 1 else np.zeros_like(probabilities)
            ).tobytes()
            if compression:
                block = zlib.compress(block)
                out.write(struct.pack("<I", len(block)))
            out.write(block)
    expected = probabilities.astype(float) / 32768
    expected[probabilities.sum(axis=1) == 0] = np.nan
    return expected


@pytest.mark.parametrize("compression", [0, 1])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("workers", [0, 2])
def test_layout1_exact_probabilities(tmp_path, compression, zarr_format, workers):
    path = tmp_path / "layout1.bgen"
    expected = write_layout1(path, compression)
    root = bgen.convert(
        path,
        tmp_path / "out.vcz",
        variants_chunk_size=2,
        zarr_format=zarr_format,
        worker_processes=workers,
    )
    nt.assert_array_equal(root["call_GP"][0], expected)
    nt.assert_array_equal(root["call_GP"][2], expected)
    nt.assert_array_equal(root["call_DS"][0], expected[:, 1] + 2 * expected[:, 2])
    # Layout 1 stores three independently rounded probabilities; do not renormalize.
    assert root["call_GP"][0, 2].sum() == 32767 / 32768
    assert_all_missing_float(root["call_GP"][1])
    nt.assert_array_equal(root["call_PLOIDY"][:], np.full((3, 4), 2))


def test_layout1_exhaustive_probability_precision(tmp_path):
    q = np.arange(32769)
    original = np.column_stack((q, 32768 - q, np.zeros_like(q))).astype("<u2")
    path = tmp_path / "all-probabilities.bgen"
    expected = write_layout1(path, 1, probabilities=original, nvariants=1)
    root = bgen.convert(path, variants_chunk_size=1, samples_chunk_size=4096)
    nt.assert_array_equal(root["call_GP"][0], expected)


def write_records(path, records, compression, bits=8):
    nsamples = len(records[0]["ploidy"])
    flags = (2 << 2) | compression
    with open(path, "wb") as out:
        out.write(
            struct.pack("<IIII4sI", 20, 20, len(records), nsamples, b"bgen", flags)
        )
        for j, record in enumerate(records):
            alleles = record["alleles"]
            ploidy = record["ploidy"]
            missing = record.get("missing", [False] * nsamples)
            phased = record.get("phased", False)
            for value in [f"snp_{j}", f"rs{j}", "1"]:
                out.write(encode_string(value))
            out.write(struct.pack("<IH", j + 1, len(alleles)))
            for allele in alleles:
                out.write(encode_string(allele, 4))
            block = struct.pack(
                "<IHBB",
                nsamples,
                len(alleles),
                min(ploidy),
                record.get("max_ploidy", max(ploidy)),
            )
            block += bytes(p | (128 if m else 0) for p, m in zip(ploidy, missing))
            block += bytes([phased, bits])
            values = []
            scale = (2**bits - 1) // 255
            for p, probability in zip(ploidy, record["probability"]):
                if phased:
                    assert len(probability) == p * len(alleles)
                    for h in range(p):
                        values.extend(
                            scale * v
                            for v in probability[
                                h * len(alleles) : (h + 1) * len(alleles) - 1
                            ]
                        )
                else:
                    assert len(probability) == math.comb(p + len(alleles) - 1, p)
                    values.extend(scale * v for v in probability[:-1])
            packed = sum(int(v) << (bits * k) for k, v in enumerate(values))
            block += packed.to_bytes((bits * len(values) + 7) // 8, "little")
            if compression == 0:
                encoded = block
            else:
                compressed = (
                    zlib.compress(block)
                    if compression == 1
                    else zstandard.compress(block)
                )
                encoded = struct.pack("<I", len(block)) + compressed
            out.write(struct.pack("<I", len(encoded)) + encoded)


@pytest.fixture
def mixed_records():
    return [
        {
            "alleles": ("A", "C", "G"),
            "ploidy": (1, 2, 3, 2),
            "missing": (False, False, False, True),
            "probability": [
                [51, 102, 102],
                [0, 51, 0, 102, 102, 0],
                [0, 0, 0, 0, 51, 51, 51, 51, 51, 0],
                [0] * 6,
            ],
        },
        {
            "alleles": ("A", "C", "G"),
            "ploidy": (2, 1, 0, 3),
            "phased": True,
            "missing": (False, False, False, True),
            "probability": [
                [51, 102, 102, 0, 51, 204],
                [102, 0, 153],
                [],
                [0] * 9,
            ],
        },
        {
            "alleles": ("A", "T"),
            "ploidy": (0, 1, 2, 3),
            "probability": [
                [255],
                [51, 204],
                [51, 102, 102],
                [0, 51, 102, 102],
            ],
        },
        {
            "alleles": ("A", "T"),
            "ploidy": (1, 2, 2, 1),
            "phased": True,
            "probability": [
                [255, 0],
                [0, 255, 255, 0],
                [51, 204, 102, 153],
                [0, 255],
            ],
        },
    ]


@pytest.mark.parametrize("compression", [0, 1, 2])
@pytest.mark.parametrize("bits", [8, 32])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_mixed_models(tmp_path, mixed_records, compression, bits, zarr_format):
    path = tmp_path / "mixed.bgen"
    write_records(path, mixed_records, compression, bits)
    root = bgen.convert(
        path, variants_chunk_size=3, samples_chunk_size=3, zarr_format=zarr_format
    )
    assert root["call_GP"].shape == (4, 4, 10)
    assert root["call_HP"].shape == (4, 4, 9)
    assert root["call_DS"].shape == (4, 4, 2)
    dimensions = (
        root["call_HP"].attrs["_ARRAY_DIMENSIONS"]
        if zarr_format == 2
        else list(root["call_HP"].metadata.dimension_names)
    )
    assert dimensions == [
        "variants",
        "samples",
        "FORMAT_HP_dim",
    ]
    nt.assert_array_equal(root["variant_bgen_phased"][:], [False, True, False, True])
    for j, record in enumerate(mixed_records):
        nt.assert_array_equal(root["call_PLOIDY"][j], record["ploidy"])
        nt.assert_array_equal(
            root["call_BGEN_MISSING"][j], record.get("missing", [False] * 4)
        )
        field = "call_HP" if record.get("phased") else "call_GP"
        for sample in range(4):
            length = len(record["probability"][sample])
            actual = root[field][j, sample]
            if record.get("missing", [False] * 4)[sample]:
                assert_all_missing_float(actual[:length])
            else:
                nt.assert_array_equal(
                    actual[:length], np.array(record["probability"][sample]) / 255
                )
            assert_all_fill_float(actual[length:])
        absent_field = "call_GP" if field == "call_HP" else "call_HP"
        assert_all_missing_float(root[absent_field][j, :, 0])
        assert_all_fill_float(root[absent_field][j, :, 1:])
    nt.assert_allclose(root["call_DS"][0, :3], [[0.4, 0.4], [0.6, 0.8], [0.8, 1.4]])
    assert_all_missing_float(root["call_DS"][0, 3])
    nt.assert_allclose(root["call_DS"][1, :3], [[0.6, 1.2], [0, 0.6], [0, 0]])
    nt.assert_allclose(root["call_DS"][2, :, 0], [0, 0.8, 1.2, 2.2])
    assert_all_fill_float(root["call_DS"][2, :, 1])
    nt.assert_allclose(root["call_DS"][3, :, 0], [0, 1, 1.4, 1])
    assert "call_genotype" not in root


@pytest.mark.parametrize("compression", [0, 1, 2])
@pytest.mark.parametrize("workers", [1, 2])
def test_mixed_model_workers(tmp_path, mixed_records, compression, workers):
    path = tmp_path / "mixed.bgen"
    write_records(path, mixed_records, compression)
    expected = bgen.convert(
        path, variants_chunk_size=2, samples_chunk_size=3, zarr_format=3
    )
    root = bgen.convert(
        path,
        tmp_path / "out.vcz",
        worker_processes=workers,
        variants_chunk_size=2,
        samples_chunk_size=3,
        zarr_format=3,
    )
    for name in expected.array_keys():
        if name.startswith("call_") and root[name].dtype.kind == "f":
            nt.assert_array_equal(
                root[name][:].view("u8"), expected[name][:].view("u8")
            )
        else:
            nt.assert_array_equal(root[name][:], expected[name][:])


def test_maximum_ploidy(tmp_path):
    records = [
        {
            "alleles": ("A", "T"),
            "ploidy": (63, 0),
            "probability": [[0] * 63 + [255], [255]],
        },
        {
            "alleles": ("A", "C", "G"),
            "ploidy": (63, 0),
            "phased": True,
            "probability": [[0, 0, 255] * 63, []],
        },
    ]
    path = tmp_path / "ploidy.bgen"
    write_records(path, records, 2)
    root = bgen.convert(path)
    nt.assert_array_equal(root["call_PLOIDY"][:], [[63, 0], [63, 0]])
    nt.assert_array_equal(root["call_DS"][0, :, 0], [63, 0])
    nt.assert_array_equal(root["call_DS"][1], [[0, 63], [0, 0]])
    assert_all_fill_float(root["call_HP"][1, 1])


@pytest.mark.parametrize("compression", [0, 1, 2])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("phased", [False, True])
def test_all_zero_ploidy(tmp_path, compression, zarr_format, phased):
    record = {
        "alleles": ("A", "T"),
        "ploidy": (0, 0),
        "phased": phased,
        "missing": (False, True),
        "probability": [[], []] if phased else [[255], [255]],
    }
    path = tmp_path / "zero-ploidy.bgen"
    write_records(path, [record], compression)
    root = bgen.convert(path, zarr_format=zarr_format)
    nt.assert_array_equal(root["call_PLOIDY"][:], [[0, 0]])
    assert root["call_DS"][0, 0] == 0
    assert_all_missing_float(root["call_DS"][0, 1:])
    nt.assert_array_equal(root["call_BGEN_MISSING"][:], [[0, 1]])
    if phased:
        assert "call_GP" not in root
        assert_all_fill_float(root["call_HP"][:])
    else:
        assert "call_HP" not in root
        nt.assert_array_equal(root["call_GP"][0, 0], [1])
        assert_all_missing_float(root["call_GP"][0, 1])


@pytest.mark.parametrize("layout", [1, 2])
@pytest.mark.parametrize("fail", [False, True])
def test_native_reader_lifecycle(tmp_path, layout, fail):
    path = tmp_path / "input.bgen"
    if layout == 1:
        write_layout1(path, 1)
    else:
        write_bgen(path, compression=0)
    original = path.read_bytes()
    native = cbgen.BgenFile
    readers = []
    indexes = []
    generate = bgen.BgenFormat.generate_schema

    def open_reader(reader_path):
        readers.append(pathlib.Path(reader_path))
        assert readers[-1] == path.resolve()
        return native(reader_path)

    def generate_schema(source, *args, **kwargs):
        index = pathlib.Path(source.metafile_path)
        indexes.append(index)
        assert not list(index.parent.glob("*.bgen"))
        if fail:
            raise RuntimeError("schema")
        return generate(source, *args, **kwargs)

    out = tmp_path / "out.vcz"
    with (
        mock.patch.object(cbgen, "BgenFile", side_effect=open_reader),
        mock.patch.object(bgen.BgenFormat, "generate_schema", generate_schema),
    ):
        if fail:
            with pytest.raises(RuntimeError, match="schema"):
                bgen.convert(path, out)
            assert not out.exists()
        else:
            root = bgen.convert(path, out)
            if layout == 2:
                nt.assert_array_equal(
                    root["sample_id"][:], [f"person_{j}" for j in range(4)]
                )
    assert readers
    assert indexes
    assert all(not p.exists() for p in indexes)
    assert path.read_bytes() == original


def test_haploid_dosage_avoids_dense_allele_counts():
    probability = np.arange(1024, dtype=float)[None, :]
    probability /= probability.sum()
    genotype = types.SimpleNamespace(
        probability=probability,
        phased=False,
        ploidy=np.array([1], dtype=np.uint8),
        missing=np.array([False]),
    )
    with mock.patch.object(bgen, "genotype_order", side_effect=AssertionError):
        nt.assert_array_equal(bgen.compute_dosage(genotype, 1024), probability[:, 1:])


@pytest.mark.parametrize(
    "values",
    [
        [32769, 0, 0],
        [0, 0, 1],
        [32768, 32768, 0],
    ],
)
def test_layout1_invalid_probabilities(tmp_path, values):
    path = tmp_path / "invalid.bgen"
    write_layout1(path, 1, probabilities=np.array([values], dtype="<u2"))
    out = tmp_path / "out.vcz"
    with pytest.raises(ValueError, match="invalid genotype probabilities"):
        bgen.convert(path, out)
    assert not out.exists()


def test_layout1_rounding_and_missing_first_sample(tmp_path):
    values = np.array(
        [[0, 0, 0], [32767, 0, 0], [1, 0, 32768], [0, 32768, 0]], dtype="<u2"
    )
    path = tmp_path / "rounded.bgen"
    expected = write_layout1(path, 1, probabilities=values, nvariants=1)
    root = bgen.convert(path)
    nt.assert_array_equal(root["call_GP"][0], expected)


@pytest.mark.parametrize("compression", [0, 1, 2])
def test_missing_before_present_samples(tmp_path, compression):
    records = [
        {
            "alleles": ("A", "C", "G"),
            "ploidy": (2, 2, 1, 1),
            "missing": (True, False, True, False),
            "probability": [
                [0] * 6,
                [0, 51, 0, 102, 102, 0],
                [0] * 3,
                [51, 102, 102],
            ],
        },
        {
            "alleles": ("A", "C", "G"),
            "ploidy": (1, 2, 0, 1),
            "phased": True,
            "missing": (True, False, True, False),
            "probability": [
                [0, 0, 0],
                [51, 102, 102, 0, 51, 204],
                [],
                [102, 0, 153],
            ],
        },
    ]
    path = tmp_path / "missing.bgen"
    write_records(path, records, compression)
    root = bgen.convert(path, variants_chunk_size=1)
    nt.assert_allclose(root["call_DS"][0, [1, 3]], [[0.6, 0.8], [0.4, 0.4]])
    nt.assert_allclose(root["call_DS"][1, [1, 3]], [[0.6, 1.2], [0, 0.6]])
    assert_all_missing_float(root["call_DS"][:, [0, 2]])
    for j, field in enumerate(["call_GP", "call_HP"]):
        for s in [1, 3]:
            expected = np.array(records[j]["probability"][s]) / 255
            nt.assert_array_equal(root[field][j, s, : len(expected)], expected)


@pytest.mark.parametrize("layout", [1, 2])
def test_truncated_native_input(tmp_path, layout, capfd):
    path = tmp_path / "input.bgen"
    if layout == 1:
        write_layout1(path, 0)
    else:
        write_bgen(path, compression=0)
    path.write_bytes(path.read_bytes()[:-1])
    out = tmp_path / "out.vcz"
    with pytest.raises(RuntimeError, match="creating metafile"):
        bgen.convert(path, out)
    assert "truncated genotype block" in capfd.readouterr().err
    assert not out.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="VCF support requires cyvcf2")
def test_general_against_vcf(tmp_path, mixed_records):
    import pysam  # noqa: PLC0415

    mixed_records[1]["ploidy"] = (2, 1, 1, 3)
    mixed_records[1]["probability"][2] = [0, 0, 255]
    mixed_records[2]["ploidy"] = (1, 1, 2, 3)
    mixed_records[2]["probability"][0] = [255, 0]
    path = tmp_path / "mixed.bgen"
    write_records(path, mixed_records, 2)
    root = bgen.convert(path)
    expected_ds = [
        [[0.4, 0.4], [0.6, 0.8], [0.8, 1.4], None],
        [[0.6, 1.2], [0, 0.6], [0, 1], None],
        [[0], [0.8], [1.2], [2.2]],
        [[0], [1], [1.4], [1]],
    ]
    lines = [
        "##fileformat=VCFv4.3",
        "##contig=<ID=1>",
        '##INFO=<ID=BGEN_PHASED,Number=0,Type=Flag,Description="Phased probabilities">',
        '##FORMAT=<ID=GP,Number=G,Type=Float,Description="Genotype probabilities">',
        '##FORMAT=<ID=HP,Number=.,Type=Float,Description="Haplotype probabilities">',
        '##FORMAT=<ID=DS,Number=A,Type=Float,Description="Alternate allele dosages">',
        '##FORMAT=<ID=PLOIDY,Number=1,Type=Integer,Description="Ploidy">',
        '##FORMAT=<ID=BGEN_MISSING,Number=1,Type=Integer,Description="Missing flag">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(f"sample_{j}" for j in range(4)),
    ]
    for j, record in enumerate(mixed_records):
        calls = []
        phased = record.get("phased", False)
        missing = record.get("missing", [False] * 4)
        for s, p in enumerate(record["ploidy"]):
            length = len(record["probability"][s])
            probability = (
                ",".join(["."] * length)
                if missing[s]
                else ",".join(str(q / 255) for q in record["probability"][s])
            )
            if missing[s]:
                ds = ",".join(["."] * (len(record["alleles"]) - 1))
            else:
                ds = ",".join(map(str, expected_ds[j][s]))
            gp, hp = (".", probability) if phased else (probability, ".")
            calls.append(f"{gp}:{hp}:{ds}:{p}:{int(missing[s])}")
        lines.append(
            f"1\t{j + 1}\trs{j}\t{record['alleles'][0]}\t"
            + ",".join(record["alleles"][1:])
            + f"\t.\t.\t{'BGEN_PHASED' if phased else '.'}"
            + "\tGP:HP:DS:PLOIDY:BGEN_MISSING\t"
            + "\t".join(calls)
        )
    vcf_path = tmp_path / "expected.vcf"
    vcf_path.write_text("\n".join(lines) + "\n")
    compressed = str(vcf_path) + ".gz"
    pysam.tabix_compress(str(vcf_path), compressed)
    pysam.tabix_index(compressed, preset="vcf")
    expected = vcf.convert([compressed])
    for name in (
        "variant_position",
        "variant_allele",
        "sample_id",
        "call_PLOIDY",
        "call_BGEN_MISSING",
    ):
        nt.assert_array_equal(root[name][:], expected[name][:])
    nt.assert_array_equal(
        root["variant_bgen_phased"][:], expected["variant_BGEN_PHASED"][:]
    )
    for name in ("call_GP", "call_HP", "call_DS"):
        a, e = root[name][:], expected[name][:]
        nt.assert_allclose(a, e, rtol=1e-6)
        active = np.ones(a.shape, dtype=bool)
        if name != "call_DS":
            # cyvcf2 can report missing rather than fill for shorter GP rows.
            # Compare sentinel identity only where VCF encodes an actual value.
            active[:] = False
            for j, record in enumerate(mixed_records):
                if record.get("phased", False) != (name == "call_HP"):
                    active[j, :, 0] = True
                else:
                    for s, probability in enumerate(record["probability"]):
                        active[j, s, : len(probability)] = True
        for sentinel in range(2):
            nt.assert_array_equal(
                float_int_view(a[active])
                == float_int_view(constants.FLOAT_MISSING_FILL[a.dtype][sentinel]),
                float_int_view(e[active])
                == float_int_view(constants.FLOAT_MISSING_FILL[e.dtype][sentinel]),
            )
