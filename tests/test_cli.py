from unittest import mock

import pytest
import click.testing as ct

from bio2zarr import cli
from bio2zarr import __main__ as main
from bio2zarr import provenance


class TestWithMocks:
    @mock.patch("bio2zarr.vcf.explode")
    def test_vcf_explode(self, mocked):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, ["explode", "source", "dest"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            ("source",),
            "dest",
            column_chunk_size=64,
            worker_processes=1,
            show_progress=True,
        )

    def test_vcf_dexplode_init(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.vcf.explode_init", return_value=5) as mocked:
            result = runner.invoke(
                cli.vcf2zarr, ["dexplode-init", "source", "dest", "5"], catch_exceptions=False
            )
            assert result.exit_code == 0
            assert len(result.stderr) == 0
            assert result.stdout == "5\n"
            mocked.assert_called_once_with(
                ("source",),
                "dest",
                target_num_partitions=5,
                worker_processes=1,
                show_progress=True,
            )

    def test_vcf_dexplode_slice(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.vcf.explode_slice") as mocked:
            result = runner.invoke(
                cli.vcf2zarr, ["dexplode-slice", "path", "1", "2"], catch_exceptions=False
            )
            assert result.exit_code == 0
            assert len(result.stdout) == 0
            assert len(result.stderr) == 0
            mocked.assert_called_once_with(
                "path",
                1,
                2,
                column_chunk_size=64,
                worker_processes=1,
                show_progress=True,
            )

    def test_vcf_dexplode_finalise(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.vcf.explode_finalise") as mocked:
            result = runner.invoke(
                cli.vcf2zarr, ["dexplode-finalise", "path"], catch_exceptions=False
            )
            assert result.exit_code == 0
            assert len(result.stdout) == 0
            assert len(result.stderr) == 0
            mocked.assert_called_once_with("path")

    @mock.patch("bio2zarr.vcf.inspect")
    def test_inspect(self, mocked):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, ["inspect", "path"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert result.stdout == "\n"
        assert len(result.stderr) == 0
        mocked.assert_called_once_with("path")

    @mock.patch("bio2zarr.vcf.mkschema")
    def test_mkschema(self, mocked):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, ["mkschema", "path"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        # TODO figure out how to test that we call it with stdout from
        # the CliRunner
        # mocked.assert_called_once_with("path", stdout)
        mocked.assert_called_once()

    @mock.patch("bio2zarr.vcf.encode")
    def test_encode(self, mocked):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, ["encode", "if_path", "zarr_path"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            "if_path",
            "zarr_path",
            None,
            variants_chunk_size=None,
            samples_chunk_size=None,
            max_v_chunks=None,
            worker_processes=1,
            max_memory=None,
            show_progress=True,
        )

    @mock.patch("bio2zarr.vcf.convert")
    def test_convert_vcf(self, mocked):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            ["convert", "vcf_path", "zarr_path"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            ("vcf_path",),
            "zarr_path",
            variants_chunk_size=None,
            samples_chunk_size=None,
            worker_processes=1,
            show_progress=True,
        )

    @mock.patch("bio2zarr.vcf.validate")
    def test_validate(self, mocked):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            ["validate", "vcf_path", "zarr_path"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            "vcf_path",
            "zarr_path",
            show_progress=True,
        )

    @mock.patch("bio2zarr.plink.convert")
    def test_convert_plink(self, mocked):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.plink2zarr, ["convert", "in", "out"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            "in",
            "out",
            worker_processes=1,
            samples_chunk_size=None,
            variants_chunk_size=None,
            show_progress=True,
        )


class TestVcfPartition:
    def test_num_parts(self):
        path = "tests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz"

        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf_partition, [path, "-n", "5"], catch_exceptions=False
        )
        assert list(result.stdout.splitlines()) == [
            "20:1-278528",
            "20:278529-442368",
            "20:442369-638976",
            "20:638977-819200",
            "20:819201-",
        ]


@pytest.mark.parametrize(
    "cmd", [main.bio2zarr, cli.vcf2zarr, cli.plink2zarr, cli.vcf_partition]
)
def test_version(cmd):
    runner = ct.CliRunner(mix_stderr=False)
    result = runner.invoke(cmd, ["--version"], catch_exceptions=False)
    s = f"version {provenance.__version__}\n"
    assert result.stdout.endswith(s)
