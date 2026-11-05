import os
import sys
from unittest import mock

import click.testing
import numcodecs
import numpy as np
import numpy.testing as nt
import pytest
import zarr

from bio2zarr import cli, constants, pgen, plink, vcf, vcz

pgenlib = pytest.importorskip("pgenlib")


def write_fileset(
    tmp_path,
    genotypes,
    *,
    chroms=None,
    positions=None,
    sexes=None,
    alleles=None,
    phase=None,
):
    gt = np.asarray(genotypes, dtype=np.int32)
    m, n, _ = gt.shape
    prefix = tmp_path / "input"
    chroms = ["1"] * m if chroms is None else chroms
    positions = list(range(1, m + 1)) if positions is None else positions
    alleles = [["A", "C"]] * m if alleles is None else alleles
    counts = np.array([len(a) for a in alleles], dtype=np.uint32)
    with pgenlib.PgenWriter(
        os.fsencode(str(prefix) + ".pgen"),
        n,
        variant_ct=m,
        allele_ct_limit=int(counts.max()),
        hardcall_phase_present=phase is not None,
    ) as writer:
        for j, row in enumerate(gt):
            if phase is None:
                writer.append_alleles(row.ravel(), allele_ct=int(counts[j]))
            else:
                writer.append_partially_phased(
                    row.ravel(),
                    np.asarray(phase[j], dtype=np.uint8),
                    allele_ct=int(counts[j]),
                )
    with open(str(prefix) + ".pvar", "w") as f:
        f.write("##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\n")
        for j, (chrom, pos, a) in enumerate(zip(chroms, positions, alleles)):
            variant_id = "." if j == 0 else f"v{j}"
            f.write(f"{chrom}\t{pos}\t{variant_id}\t{a[0]}\t{','.join(a[1:])}\n")
    with open(str(prefix) + ".psam", "w") as f:
        f.write("#IID" + ("\tSEX" if sexes is not None else "") + "\n")
        for j in range(n):
            f.write(f"s{j:03}" + (f"\t{sexes[j]}" if sexes is not None else "") + "\n")
    return prefix


@pytest.fixture
def fileset(tmp_path):
    gt = np.array(
        [
            [[0, 0], [0, 1], [1, 1], [-9, -9], [0, 1]],
            [[1, 0], [0, 1], [-9, -9], [1, 1], [0, 0]],
            [[0, 0], [1, 1], [0, 1], [1, 0], [-9, -9]],
            [[1, 1], [-9, -9], [0, 0], [0, 1], [1, 0]],
            [[0, 1], [0, 0], [1, 1], [-9, -9], [1, 0]],
        ],
        dtype=np.int32,
    )
    phase = np.array(
        [
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    return write_fileset(tmp_path, gt, phase=phase), gt, phase


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "workers",
    [dict(), dict(worker_processes=1), dict(worker_processes=2)],
)
@pytest.mark.parametrize("chunks", [(2, 2), (1, 1), (20, 20)])
def test_roundtrip(fileset, tmp_path, zarr_format, workers, chunks):
    prefix, _, _ = fileset
    v, s = chunks
    out = tmp_path / "result.vcz"
    root = plink.convert(
        prefix,
        out,
        variants_chunk_size=v,
        samples_chunk_size=s,
        zarr_format=zarr_format,
        **workers,
    )
    assert root["call_genotype"].chunks == (v, s, 2)
    with pgenlib.PgenReader(os.fsencode(str(prefix) + ".pgen")) as reader:
        expected = np.empty((5, 10), dtype=np.int32)
        phase = np.empty((5, 5), dtype=bool)
        reader.read_alleles_and_phasepresent_range(0, 5, expected, phase)
    expected[expected == -9] = -1
    nt.assert_array_equal(root["call_genotype"][:], expected.reshape(5, 5, 2))
    nt.assert_array_equal(root["call_genotype_phased"][:], phase)
    nt.assert_array_equal(root["call_genotype_mask"][:], expected.reshape(5, 5, 2) < 0)
    nt.assert_array_equal(root["sample_id"][:], [f"s{j:03}" for j in range(5)])
    nt.assert_array_equal(
        root["variant_id_mask"][:], [True, False, False, False, False]
    )
    nt.assert_array_equal(root["variant_position"][:], np.arange(1, 6))
    nt.assert_array_equal(root["variant_length"][:], np.ones(5))
    nt.assert_array_equal(root["variant_allele"][:], [["A", "C"]] * 5)
    assert "region_index" in root
    assert not (out / "wip").exists()


def test_metadata_only_planning(fileset):
    prefix, _, _ = fileset
    with mock.patch.object(
        pgen.PgenFormat,
        "iter_alleles_and_genotypes",
        side_effect=AssertionError("No decoding in planning"),
    ):
        source = pgen.PgenFormat(prefix)
        schema = source.generate_schema(2, 3)
    assert source.has_phase
    assert schema.get_shape(["variants", "samples", "ploidy"]) == [5, 5, 2]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_shared_writer_interface(fileset, tmp_path, monkeypatch, zarr_format):
    assert pgen.PgenZarrWriter.__init__ is vcz.VcfZarrWriter.__init__
    assert pgen.PgenZarrWriter.init is vcz.VcfZarrWriter.init
    assert (
        pgen.PgenZarrWriter.encode_all_partitions
        is vcz.VcfZarrWriter.encode_all_partitions
    )
    prefix, _, _ = fileset
    source = pgen.PgenFormat(prefix)
    schema = source.generate_schema(2, 2)
    opened = []
    open_array = vcz.VcfZarrWriter.open_partition_array

    def observed_open(writer, partition_index, name):
        opened.append((partition_index, name))
        return open_array(writer, partition_index, name)

    monkeypatch.setattr(vcz.VcfZarrWriter, "open_partition_array", observed_open)
    out = tmp_path / "output"
    root = vcz.encode(
        source,
        schema,
        out,
        zarr_format=zarr_format,
        writer_class=pgen.PgenZarrWriter,
    )
    assert sorted(opened) == [(0, name) for name in sorted(schema.field_map())]
    assert root["call_genotype"].chunks == (2, 2, 2)
    assert root["call_genotype"].shape == (5, 5, 2)
    assert "region_index" in root
    assert not (out / "wip").exists()


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("output", ["memory", "directory", "zip"])
def test_outputs_and_modes(fileset, tmp_path, output, zarr_format):
    prefix, _, _ = fileset
    out = (
        None
        if output == "memory"
        else tmp_path / ("output.vcz.zip" if output == "zip" else "output.vcz")
    )
    root = plink.convert(prefix, out, zarr_format=zarr_format)
    assert root.read_only
    assert root["call_genotype"].shape == (5, 5, 2)
    if output == "zip":
        assert isinstance(root.store, zarr.storage.ZipStore)
        root.store.close()
        assert not (tmp_path / "output.vcz").exists()
    assert not list(tmp_path.glob(".output.vcz*"))


def test_writable(fileset, tmp_path):
    prefix, _, _ = fileset
    root = plink.convert(prefix, tmp_path / "output", mode="r+")
    assert not root.read_only
    root.create_array("extra", data=np.arange(3))


def test_unphased(tmp_path):
    prefix = write_fileset(tmp_path, [[[0, 1], [1, 1], [-9, -9]]])
    root = plink.convert(prefix)
    assert not root["call_genotype_phased"][:].any()
    assert not root.attrs["plink2_hardcall_phase_present"]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_multiallelic(tmp_path, zarr_format):
    gt = [[[0, 2], [2, 1]], [[1, 0], [-9, -9]], [[128, 128], [0, 129]]]
    alleles = [["AA", "C", "GG"], ["G", "T"], [f"<A{j}>" for j in range(130)]]
    prefix = write_fileset(
        tmp_path, gt, alleles=alleles, phase=[[1, 1], [1, 0], [0, 1]]
    )
    root = plink.convert(
        prefix,
        worker_processes=2,
        variants_chunk_size=2,
        samples_chunk_size=1,
        zarr_format=zarr_format,
    )
    expected = np.array(gt)
    expected[expected == -9] = -1
    nt.assert_array_equal(root["call_genotype"][:], expected)
    assert root["call_genotype"].dtype == np.dtype("i2")
    assert root["variant_allele"].shape == (3, 130)
    nt.assert_array_equal(root["variant_allele"][0, :4], ["AA", "C", "GG", ""])
    nt.assert_array_equal(root["variant_length"][:], [2, 1, 4])


@pytest.mark.parametrize("build", ["hg19", "hg38"])
@pytest.mark.parametrize("worker_processes", [0, 2])
def test_ploidy_and_par(tmp_path, build, worker_processes):
    (par_start, par_end), _ = pgen.PAR_INTERVALS[build]
    chroms = ["1", "X", "X", "X", "X", "Y", "MT"]
    positions = [1, par_start - 1, par_start, par_end, par_end + 1, 1, 1]
    gt = [
        [[0, 1], [0, 1]],
        [[1, 1], [0, 1]],
        [[0, 1], [0, 1]],
        [[0, 1], [0, 1]],
        [[-9, -9], [-9, -9]],
        [[1, 1], [-9, -9]],
        [[-9, -9], [0, 0]],
    ]
    prefix = write_fileset(
        tmp_path, gt, chroms=chroms, positions=positions, sexes=[1, 2]
    )
    root = plink.convert(
        prefix,
        genome_build=build,
        worker_processes=worker_processes,
        variants_chunk_size=3,
        samples_chunk_size=1,
    )
    expected = np.array(gt)
    expected[expected == -9] = -1
    expected[[1, 4, 5], 0, 1] = -2
    expected[5, 1] = -2
    expected[6, :, 1] = -2
    nt.assert_array_equal(root["call_genotype"][:], expected)
    nt.assert_array_equal(root["call_genotype_mask"][:], expected < 0)


@pytest.mark.parametrize("chrom", ["MT", "chrM", "26", "X", "chrX", "23", "Y"])
def test_haploid_only(tmp_path, chrom):
    prefix = write_fileset(
        tmp_path,
        [[[1, 1], [-9, -9]]],
        chroms=[chrom],
        sexes=[1, 1],
        positions=[5_000_000],
    )
    root = plink.convert(prefix, genome_build="hg38")
    assert root["call_genotype"].shape == (1, 2, 1)
    nt.assert_array_equal(root["call_genotype"][:], [[[1], [-1]]])


def test_split_par(tmp_path):
    prefix = write_fileset(
        tmp_path,
        [[[0, 0]], [[0, 1]], [[0, 1]]],
        chroms=["X", "PAR1", "PAR2"],
        sexes=[1],
    )
    root = plink.convert(prefix)
    nt.assert_array_equal(root["call_genotype"][:], [[[0, -2]], [[0, 1]], [[0, 1]]])


@pytest.mark.parametrize("sexes", [None, [0], ["NA"]])
def test_unknown_sex(tmp_path, sexes):
    prefix = write_fileset(tmp_path, [[[0, 0]]], chroms=["X"], sexes=sexes)
    with pytest.raises(ValueError, match="SEX=1 or 2"):
        plink.convert(prefix)
    root = plink.convert(prefix, ploidy_mode="diploid")
    nt.assert_array_equal(root["call_genotype"][:], [[[0, 0]]])


def test_missing_par_build(tmp_path):
    prefix = write_fileset(tmp_path, [[[0, 0]]], chroms=["X"], sexes=[1])
    with pytest.raises(ValueError, match="genome_build"):
        plink.convert(prefix)


@pytest.mark.parametrize("worker_processes", [0, 2])
@pytest.mark.parametrize("output", ["directory", "zip"])
def test_late_failure_leaves_no_output(tmp_path, worker_processes, output):
    prefix = write_fileset(
        tmp_path,
        [[[0, 0]], [[1, 1]], [[0, 1]]],
        chroms=["MT"] * 3,
    )
    out = tmp_path / ("output.zip" if output == "zip" else "output.vcz")
    with pytest.raises(ValueError, match="Heterozygous haploid"):
        plink.convert(
            prefix, out, variants_chunk_size=1, worker_processes=worker_processes
        )
    assert not out.exists()
    assert not list(tmp_path.glob(".output*"))
    # Readers have closed even when the traceback is retained by pytest.
    os.rename(str(prefix) + ".pgen", str(prefix) + ".moved")


def test_female_y_nonmissing(tmp_path):
    prefix = write_fileset(tmp_path, [[[0, 0]]], chroms=["Y"], sexes=[2])
    with pytest.raises(ValueError, match="Nonmissing female Y"):
        plink.convert(prefix)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("#IID\ns0\ns0\n", "unique"),
        ("#FID\nf0\n", "including IID"),
        ("IID\ns0\n", "header"),
        ("", "empty"),
        ("#IID\n", "nonempty"),
        ("#IID\n0\n", "nonempty"),
        ("#IID SEX\na\n", "match the header"),
        ("#IID SEX\na 1 extra\n", "match the header"),
    ],
)
def test_bad_psam(tmp_path, text, match):
    path = tmp_path / "input.psam"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        pgen.read_psam(path)


def test_psam_columns_and_ids(tmp_path):
    path = tmp_path / "input.psam"
    path.write_text("##comment\n#FID IID SEX PHENO1\n000 001 1 NA\n000 002 2 -9\n")
    df = pgen.read_psam(path)
    assert list(df.IID) == ["001", "002"]


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (dict(chroms=["1", "2", "1"]), "contiguous"),
        (dict(positions=[2, 1, 3]), "sorted"),
        (dict(positions=[0, 1, 2]), "positive"),
        (dict(alleles=[["A", "A"]] * 3), "distinct"),
        (dict(alleles=[[".", "C"]] * 3), "missing"),
    ],
)
def test_invalid_variant_metadata(tmp_path, metadata, match):
    prefix = write_fileset(tmp_path, [[[0, 0]]] * 3, **metadata)
    with pytest.raises(ValueError, match=match):
        plink.convert(prefix, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_pvar_zstd(fileset):
    prefix, _, _ = fileset
    path = prefix.with_suffix(".pvar")
    path.with_suffix(".pvar.zst").write_bytes(
        numcodecs.Zstd().encode(path.read_bytes())
    )
    path.unlink()
    root = plink.convert(prefix)
    assert root["call_genotype"].shape == (5, 5, 2)


def test_sample_count_mismatch(fileset):
    prefix, _, _ = fileset
    prefix.with_suffix(".psam").write_text("#IID\ns0\n")
    with pytest.raises(RuntimeError, match="sample"):
        plink.convert(prefix)


def test_variant_count_mismatch(fileset):
    prefix, _, _ = fileset
    path = prefix.with_suffix(".pvar")
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(RuntimeError, match="variant"):
        plink.convert(prefix)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (dict(input_format="bad"), "input_format"),
        (dict(ploidy_mode="bad"), "ploidy_mode"),
        (dict(genome_build="bad"), "genome_build"),
        (dict(worker_processes=-1), "worker_processes"),
        (dict(variants_chunk_size=0), "chunk"),
        (dict(dosages="yes"), "dosages must be a boolean"),
    ],
)
def test_invalid_options(fileset, options, match):
    prefix, _, _ = fileset
    with pytest.raises(ValueError, match=match):
        plink.convert(prefix, **options)


def test_ambiguous_prefix(fileset):
    prefix, _, _ = fileset
    prefix.with_suffix(".bed").write_bytes(b"")
    with pytest.raises(ValueError, match="Both BED and PGEN"):
        plink.convert(prefix)
    assert plink.convert(prefix, input_format="pgen")["call_genotype"].shape == (
        5,
        5,
        2,
    )


def test_existing_output(fileset, tmp_path):
    prefix, _, _ = fileset
    out = tmp_path / "output"
    out.mkdir()
    marker = out / "marker"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        plink.convert(prefix, out)
    assert marker.read_text() == "keep"


def test_optional_dependency(fileset):
    prefix, _, _ = fileset
    with mock.patch.dict(sys.modules, {"pgenlib": None}):
        with pytest.raises(ImportError, match=r"bio2zarr\[plink2\]"):
            plink.convert(prefix)


def test_cli(fileset, tmp_path):
    prefix, _, _ = fileset
    out = tmp_path / "output.vcz"
    result = click.testing.CliRunner().invoke(
        cli.plink2zarr_main,
        [
            "convert",
            str(prefix),
            str(out),
            "--worker-processes",
            "2",
            "--variants-chunk-size",
            "2",
            "--samples-chunk-size",
            "2",
            "-Q",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert zarr.open(out)["call_genotype"].shape == (5, 5, 2)


@pytest.mark.skipif(sys.platform == "win32", reason="Requires cyvcf2")
def test_against_vcf(tmp_path):
    gt = [[[0, 0], [0, 1]], [[1, 0], [-9, -9]], [[2, 1], [1, 1]]]
    prefix = write_fileset(
        tmp_path,
        gt,
        alleles=[["A", "C"], ["G", "T"], ["A", "C", "G"]],
        phase=[[0, 0], [1, 0], [1, 0]],
    )
    path = tmp_path / "input.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=1>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts000\ts001\n"
        "1\t1\t.\tA\tC\t.\t.\t.\tGT\t0/0\t0/1\n"
        "1\t2\tv1\tG\tT\t.\t.\t.\tGT\t1|0\t./.\n"
        "1\t3\tv2\tA\tC,G\t.\t.\t.\tGT\t2|1\t1/1\n"
    )
    actual = plink.convert(prefix)
    expected = vcf.convert([path])
    for name in (
        "call_genotype",
        "call_genotype_mask",
        "variant_allele",
        "variant_contig",
        "variant_position",
        "variant_id",
        "sample_id",
    ):
        nt.assert_array_equal(actual[name][:], expected[name][:])
    # Homozygous phase flags are not retained in PGEN; heterozygous flags are.
    het = np.array([[False, True], [True, False], [True, False]])
    nt.assert_array_equal(
        actual["call_genotype_phased"][:][het], expected["call_genotype_phased"][:][het]
    )


def test_reference_only_variant(tmp_path):
    prefix = write_fileset(tmp_path, [[[0, 0], [-9, -9]]], alleles=[["A", "."]])
    root = plink.convert(prefix)
    nt.assert_array_equal(root["variant_allele"][:], [["A"]])
    nt.assert_array_equal(root["call_genotype"][:], [[[0, 0], [-1, -1]]])


@pytest.mark.parametrize("phased", [False, True])
def test_decode_is_chunk_bounded(tmp_path, monkeypatch, phased):
    prefix = write_fileset(
        tmp_path,
        [[[0, 1]] * 5] * 5,
        phase=[[1] * 5] * 5 if phased else None,
    )
    native_open = pgen.PgenFormat.open_reader
    reads = []
    dosage_reads = []
    opened = []
    closed = []

    class Reader:
        def __init__(self, source, subset=None):
            self.reader = native_open(source, subset)
            self.samples = source.num_samples if subset is None else len(subset)
            opened.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.reader.close()
            closed.append(self)

        def hardcall_phase_present(self):
            return self.reader.hardcall_phase_present()

        def read_alleles_range(self, start, stop, out):
            reads.append((start, stop, out.shape, self.samples))
            self.reader.read_alleles_range(start, stop, out)

        def read_alleles_and_phasepresent_range(self, start, stop, out, phase):
            reads.append((start, stop, out.shape, self.samples))
            self.reader.read_alleles_and_phasepresent_range(start, stop, out, phase)

        def read_phased_dosages_range(self, start, stop, out, allele_idx):
            dosage_reads.append((start, stop, out.shape, self.samples))
            self.reader.read_phased_dosages_range(
                start, stop, out, allele_idx=allele_idx
            )

    monkeypatch.setattr(
        pgen.PgenFormat,
        "open_reader",
        lambda source, subset=None: Reader(source, subset),
    )
    source = pgen.PgenFormat(prefix)
    source.generate_schema(2, 2)
    assert not reads
    plink.convert(prefix, variants_chunk_size=2, samples_chunk_size=2)
    assert len(reads) == 9
    assert dosage_reads == reads or sorted(dosage_reads) == sorted(reads)
    assert all(start % 2 == 0 and stop - start <= 2 for start, stop, _, _ in reads)
    assert all(
        shape == (stop - start, 2 * samples) for start, stop, shape, samples in reads
    )
    assert all(samples <= 2 for _, _, _, samples in reads)
    assert len(opened) == len(closed)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "workers", [{}, {"worker_processes": 1}, {"worker_processes": 2}]
)
def test_stored_phased_dosages(tmp_path, zarr_format, workers):
    # Generated with PLINK 2 (15 Oct 2026):
    # plink2 --vcf phased_dosages.vcf dosage=HDS --make-pgen --out phased_dosages
    root = plink.convert(
        "tests/data/pgen/phased_dosages",
        tmp_path / "out",
        variants_chunk_size=2,
        samples_chunk_size=2,
        zarr_format=zarr_format,
        **workers,
    )
    expected = np.array(
        [
            [[0.125, 0.875], [0.125, 0.125], [np.nan, np.nan]],
            [[0.25, 0.5], [0.875, 0.125], [0.75, 0.875]],
            [[0, 0], [0.25, 0.75], [1, 1]],
        ],
        dtype=np.float32,
    )
    nt.assert_array_equal(root["call_HDS"][:], expected)
    nt.assert_array_equal(root["call_DS"][:], expected.sum(axis=2))
    assert root["call_DS"].dtype == np.dtype("f4")
    assert root["call_HDS"].chunks == (2, 2, 1)
    missing_bits = np.asarray(constants.FLOAT32_MISSING).view(np.uint32)
    assert root["call_DS"][:].view(np.uint32)[0, 2] == missing_bits
    # Stored dosage is preserved even when no hardcall could be made.
    nt.assert_array_equal(root["call_genotype"][1, 0], [-1, -1])
    assert root["call_DS"][1, 0] == 0.75


def write_dosage_fileset(tmp_path, dosages, **metadata):
    dosages = np.asarray(dosages, dtype=np.float32)
    m, n = dosages.shape
    prefix = write_fileset(tmp_path, np.zeros((m, n, 2), dtype=np.int32), **metadata)
    with pgenlib.PgenWriter(
        os.fsencode(str(prefix) + ".pgen"), n, variant_ct=m, dosage_present=True
    ) as writer:
        writer.append_dosages_batch(dosages)
    return prefix


@pytest.mark.parametrize(("chrom", "sexes"), [("MT", None), ("X", [1, 1, 1])])
def test_haploid_dosage_scaling(tmp_path, chrom, sexes):
    prefix = write_dosage_fileset(
        tmp_path,
        [[0.5, 1.5, -9]],
        chroms=[chrom],
        sexes=sexes,
        positions=[5_000_000],
    )
    root = plink.convert(prefix, genome_build="hg38")
    nt.assert_array_equal(root["call_DS"][:], [[0.25, 0.75, np.nan]])
    nt.assert_array_equal(root["call_HDS"][:], [[[0.25], [0.75], [np.nan]]])
    nt.assert_array_equal(root["call_genotype"][:], [[[-1], [-1], [-1]]])


def test_female_y_dosage_without_hardcall(tmp_path):
    prefix = write_dosage_fileset(tmp_path, [[0.5]], chroms=["Y"], sexes=[2])
    with pytest.raises(ValueError, match="Nonmissing female Y dosage"):
        plink.convert(prefix)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_dosage_missing_and_padding(tmp_path, zarr_format):
    prefix = write_fileset(
        tmp_path,
        [[[0, 2], [-9, -9]], [[1, 1], [-9, -9]], [[0, 0], [-9, -9]]],
        alleles=[["A", "C", "G"], ["A", "C"], ["A", "."]],
    )
    root = plink.convert(
        prefix,
        worker_processes=2,
        variants_chunk_size=2,
        samples_chunk_size=1,
        zarr_format=zarr_format,
    )
    missing, fill = constants.FLOAT32_MISSING, constants.FLOAT32_FILL
    ds = np.array(
        [
            [[0, 1], [missing, missing]],
            [[2, fill], [missing, fill]],
            [[fill, fill], [fill, fill]],
        ],
        dtype=np.float32,
    )
    hds = np.array(
        [
            [[0, 0.5, 0, 0.5], [missing] * 4],
            [[1, 1, fill, fill], [missing, missing, fill, fill]],
            [[fill] * 4, [fill] * 4],
        ],
        dtype=np.float32,
    )
    nt.assert_array_equal(root["call_DS"][:].view(np.uint32), ds.view(np.uint32))
    nt.assert_array_equal(root["call_HDS"][:].view(np.uint32), hds.view(np.uint32))
    assert root["call_DS"].chunks == (2, 1, 1)


def test_disable_dosages(fileset):
    prefix, _, _ = fileset
    root = plink.convert(prefix, dosages=False)
    assert "call_DS" not in root
    assert "call_HDS" not in root
    assert not root.attrs["plink2_dosages"]


def test_cli_disable_dosages(fileset, tmp_path):
    prefix, _, _ = fileset
    result = click.testing.CliRunner().invoke(
        cli.plink2zarr_main,
        ["convert", str(prefix), str(tmp_path / "output"), "--no-dosages", "-Q"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "call_DS" not in zarr.open(tmp_path / "output")


def test_unphased_dosages(tmp_path):
    prefix = write_dosage_fileset(tmp_path, [[0.5, 1.5, 0.75, 0, -9]])
    root = plink.convert(prefix)
    expected = np.array([[0.5, 1.5, 0.75, 0, np.nan]], dtype=np.float32)
    nt.assert_array_equal(root["call_DS"][:], expected)
    nt.assert_array_equal(
        root["call_HDS"][:], np.repeat(expected[:, :, None] / 2, 2, axis=2)
    )
    assert not root["call_genotype_phased"][:].any()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1, 1.5])
def test_invalid_native_dosage(fileset, tmp_path, monkeypatch, bad_value):
    prefix, _, _ = fileset
    native_open = pgen.PgenFormat.open_reader

    class Reader:
        def __init__(self, source, subset=None):
            self.reader = native_open(source, subset)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.reader.close()

        def __getattr__(self, name):
            return getattr(self.reader, name)

        def read_phased_dosages_range(self, start, stop, out, allele_idx):
            self.reader.read_phased_dosages_range(
                start, stop, out, allele_idx=allele_idx
            )
            out[0, 0] = bad_value

    monkeypatch.setattr(
        pgen.PgenFormat,
        "open_reader",
        lambda source, subset=None: Reader(source, subset),
    )
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="Invalid PGEN dosage"):
        plink.convert(prefix, out, variants_chunk_size=2)
    assert not out.exists()
