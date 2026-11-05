import json
import multiprocessing
import pathlib
import queue
import random
import sys

import boto3
import click.testing
import pytest
from moto.server import ThreadedMotoServer

from bio2zarr import cli, storage, vcf_utils
from bio2zarr import vcf as vcf_mod
from tests.test_vcf_streaming import assert_zarr_equal

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="VCF support requires cyvcf2"
)
pysam = pytest.importorskip("pysam")
DATA = pathlib.Path(__file__).parent / "data" / "vcf"


def serve_s3(connection, requests):
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    _, port = server.get_host_and_port()
    app = server._server.app

    def record_requests(environ, start_response):
        requests.put(
            (
                environ.get("PATH_INFO"),
                environ.get("HTTP_RANGE"),
                environ.get("QUERY_STRING"),
                environ.get("HTTP_AUTHORIZATION"),
                environ.get("HTTP_USER_AGENT"),
            )
        )
        return app(environ, start_response)

    server._server.app = record_requests
    connection.send(port)
    connection.recv()
    server.stop()


@pytest.fixture
def s3(monkeypatch, tmp_path):
    # Native htslib holds the GIL during HTTP reads: the server must be external.
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    requests = ctx.Queue()
    process = ctx.Process(target=serve_s3, args=(child, requests))
    process.start()
    assert parent.poll(30), "S3 test server failed to start"
    endpoint = f"http://127.0.0.1:{parent.recv()}"
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    monkeypatch.setenv("HTS_S3_HOST", endpoint.removeprefix("http://"))
    monkeypatch.setenv("HTS_S3_ADDRESS_STYLE", "path")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    client = boto3.client("s3", endpoint_url=endpoint)
    client.create_bucket(Bucket="bio2zarr-test")
    try:
        yield client, requests
        assert not list(cwd.iterdir()), "HTSlib cached an index in cwd"
    finally:
        parent.send(None)
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
        child.close()
        requests.close()


def upload(s3, filename):
    client, _ = s3
    path = DATA / filename
    client.upload_file(str(path), "bio2zarr-test", f"input/{filename}")
    for suffix in [".tbi", ".csi"]:
        index = pathlib.Path(str(path) + suffix)
        if index.exists():
            client.upload_file(str(index), "bio2zarr-test", f"input/{filename}{suffix}")
    return f"s3+http://bio2zarr-test/input/{filename}"


def drain_requests(request_queue):
    requests = []
    while True:
        try:
            requests.append(request_queue.get_nowait())
        except queue.Empty:
            return requests


@pytest.mark.parametrize(
    "filename",
    ["sample.vcf.gz", "sample.bcf", "empty.vcf.gz", "sample_extra_contig.vcf.gz"],
)
def test_s3_input(s3, filename):
    uri = upload(s3, filename)
    expected = vcf_mod.convert([DATA / filename], infer_shape=True)
    actual = vcf_mod.convert([uri], infer_shape=True)
    assert_zarr_equal(expected, actual)
    _, request_queue = s3
    requests = drain_requests(request_queue)
    assert any(name == f"/bio2zarr-test/input/{filename}" for name, *_ in requests)
    assert all(
        "Signature" not in query and "X-Amz" not in query
        for _, _, query, _, _ in requests
    )
    assert any(
        name == f"/bio2zarr-test/input/{filename}" and auth
        for name, _, _, auth, _ in requests
    )


@pytest.mark.parametrize(
    ("filename", "index_suffix"),
    [
        ("sample.vcf.gz", ".tbi"),
        ("sample_extra_contig.vcf.gz", ".csi"),
        ("sample.bcf", ".csi"),
    ],
)
def test_s3_custom_index_isolation(s3, filename, index_suffix):
    client, request_queue = s3
    uri = upload(s3, filename)
    index = pathlib.Path(str(DATA / filename) + index_suffix)
    for prefix in ["one", "two"]:
        client.upload_file(
            str(index), "bio2zarr-test", f"{prefix}/shared{index_suffix}"
        )
    client.delete_object(Bucket="bio2zarr-test", Key=f"input/{filename}{index_suffix}")
    with (
        vcf_utils.VcfFile(
            uri, index_path=f"s3+http://bio2zarr-test/one/shared{index_suffix}"
        ) as first,
        vcf_utils.VcfFile(
            uri, index_path=f"s3+http://bio2zarr-test/two/shared{index_suffix}"
        ) as second,
    ):
        expected = [(v.CHROM, v.POS) for v in first.vcf]
        assert expected == [(v.CHROM, v.POS) for v in second.vcf]
        region = vcf_utils.Region(expected[0][0], expected[0][1], expected[0][1])
        assert first.count_variants(region) == second.count_variants(region) == 1
        if filename.endswith(".bcf"):
            assert first._temporary_index is second._temporary_index is None
        else:
            assert first._temporary_index.name != second._temporary_index.name
            temp_paths = [first._temporary_index.name, second._temporary_index.name]
            assert all(pathlib.Path(p, "index.tbi").exists() for p in temp_paths)
    if not filename.endswith(".bcf"):
        assert all(not pathlib.Path(p).exists() for p in temp_paths)
    requests = drain_requests(request_queue)
    if filename.endswith(".bcf"):
        # BCF index reads use HTSlib, not the s3fs statistics reader alone.
        assert any(
            name == f"/bio2zarr-test/one/shared{index_suffix}"
            and auth
            and "htslib" in agent.lower()
            for name, _, _, auth, agent in requests
        )
    assert all(
        "Signature" not in query and "X-Amz" not in query
        for _, _, query, _, _ in requests
    )


def test_s3_variant_range_read(s3, tmp_path):
    source = tmp_path / "large.vcf"
    rng = random.Random(42)
    with source.open("w") as file:
        file.write(
            "##fileformat=VCFv4.3\n##contig=<ID=1>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        )
        for j in range(10000):
            alt = "".join(rng.choices("ACGT", k=128))
            file.write(f"1\t{j * 1000 + 1}\t.\tA\t{alt}\t.\t.\t.\n")
    compressed = str(source) + ".gz"
    pysam.tabix_compress(str(source), compressed)
    pysam.tabix_index(compressed, preset="vcf")
    assert pathlib.Path(compressed).stat().st_size > 65536
    client, request_queue = s3
    for suffix in ["", ".tbi"]:
        client.upload_file(
            compressed + suffix, "bio2zarr-test", "input/large.vcf.gz" + suffix
        )
    with vcf_utils.VcfFile("s3+http://bio2zarr-test/input/large.vcf.gz") as file:
        assert file.count_variants(vcf_utils.Region("1", 9999001, 9999001)) == 1
    requests = drain_requests(request_queue)
    assert any(
        name == "/bio2zarr-test/input/large.vcf.gz"
        and byte_range
        and int(byte_range.removeprefix("bytes=").split("-")[0]) > 65536
        and "htslib" in agent.lower()
        for name, byte_range, _, _, agent in requests
    )


def test_s3_icf_lifecycle(s3, tmp_path):
    uri = upload(s3, "sample.vcf.gz")
    icf_path = "s3://bio2zarr-test/intermediate/run"
    summary = vcf_mod.explode_init(icf_path, [uri], target_num_partitions=3)
    with pytest.raises(FileNotFoundError, match="Partition metadata not found"):
        vcf_mod.explode_finalise(icf_path)
    for j in range(summary.num_partitions):
        vcf_mod.explode_partition(icf_path, j)
    vcf_mod.explode_partition(icf_path, 0)
    vcf_mod.explode_finalise(icf_path)
    assert not storage.path(icf_path + "/wip").exists()
    icf = vcf_mod.IntermediateColumnarFormat(icf_path)
    assert icf.num_records == 9
    assert vcf_mod.inspect(icf_path) == icf.summary_table()
    expected = vcf_mod.convert([DATA / "sample.vcf.gz"])
    out = tmp_path / "vcz"
    summary = vcf_mod.encode_init(icf_path, out, 3, batch_size=2, variants_chunk_size=3)
    for j in range(summary.num_partitions):
        vcf_mod.encode_partition(out, j)
    vcf_mod.encode_finalise(out)
    vcf_mod.create_index(out)
    same_chunks = vcf_mod.encode(icf_path, variants_chunk_size=3, batch_size=2)
    assert_zarr_equal(same_chunks, vcf_mod.zarr_utils.open_vcf_zarr(out))
    assert_zarr_equal(expected, vcf_mod.encode(icf_path, batch_size=2))
    metadata = json.loads(storage.path(icf_path + "/metadata.json").open().read())
    assert metadata["partitions"][0]["vcf_path"] == uri
    assert "testing" not in json.dumps(metadata)
    assert "X-Amz" not in json.dumps(metadata)


def test_temporary_s3_icf_cleanup(s3):
    client, _ = s3
    neighbour = "temp/not-our-run"
    client.put_object(Bucket="bio2zarr-test", Key=neighbour, Body=b"keep")
    expected = vcf_mod.convert([DATA / "sample.vcf.gz"])
    actual = vcf_mod.convert(
        [DATA / "sample.vcf.gz"],
        temp_icf_dir="s3://bio2zarr-test/temp",
    )
    assert_zarr_equal(expected, actual)
    objects = client.list_objects_v2(Bucket="bio2zarr-test", Prefix="temp/")
    assert [o["Key"] for o in objects["Contents"]] == [neighbour]

    def fail_after_writing():
        with vcf_mod.temp_icf_path(dir="s3://bio2zarr-test/temp") as tmp:
            with (tmp / "written").open("wb") as file:
                file.write(b"data")
            raise RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        fail_after_writing()
    objects = client.list_objects_v2(Bucket="bio2zarr-test", Prefix="temp/")
    assert [o["Key"] for o in objects["Contents"]] == [neighbour]


def test_s3_index_errors(s3):
    uri = upload(s3, "sample.vcf.gz")
    with pytest.raises(FileNotFoundError, match="Specified index path"):
        vcf_utils.VcfFile(uri, index_path=uri + ".missing.csi")
    with pytest.raises(FileNotFoundError):
        vcf_utils.VcfFile(uri + ".absent")
    s3[0].delete_object(Bucket="bio2zarr-test", Key="input/sample.vcf.gz.tbi")
    with vcf_utils.VcfFile(uri) as file:
        with pytest.raises(ValueError, match="Multi-contig VCFs must be indexed"):
            list(file.variants())


def test_s3_native_error_propagates(s3, monkeypatch):
    import cyvcf2  # noqa: PLC0415

    uri = upload(s3, "sample.vcf.gz")

    def unsupported(path):
        assert path == uri
        raise OSError("native S3 protocol unsupported")

    monkeypatch.setattr(cyvcf2, "VCF", unsupported)
    with pytest.raises(OSError, match="native S3 protocol unsupported"):
        vcf_utils.VcfFile(uri)


def test_s3_cli(s3, tmp_path):
    uri = upload(s3, "sample.vcf.gz")
    result = click.testing.CliRunner().invoke(
        cli.vcf2zarr_main,
        ["convert", uri, str(tmp_path / "out"), "--max-ploidy", "2", "-Q"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


@pytest.mark.parametrize("filename", ["sample.vcf.gz", "sample.bcf"])
def test_s3_worker_processes(s3, filename):
    uri = upload(s3, filename)
    actual = vcf_mod.convert(
        [uri], max_ploidy=2, worker_processes=2, variants_chunk_size=3
    )
    expected = vcf_mod.convert([DATA / filename], max_ploidy=2, variants_chunk_size=3)
    assert_zarr_equal(expected, actual)


def test_s3_prefix_validation():
    with pytest.raises(ValueError, match="bucket and an object"):
        storage.path("s3://bucket")
    with pytest.raises(ValueError, match="Unsupported storage protocol"):
        storage.path("gs://bucket/input")
