from unittest import mock

import pytest
import click.testing as ct

from bio2zarr import cli
from bio2zarr import __main__ as main
from bio2zarr import provenance


class TestWithMocks:
    def test_vcf_explode(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.vcf.explode") as mocked:
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

    def test_inspect(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.vcf.inspect", return_value={}) as mocked:
            result = runner.invoke(
                cli.vcf2zarr, ["inspect", "path"], catch_exceptions=False
            )
            assert result.exit_code == 0
            assert result.stdout == "\n"
            assert len(result.stderr) == 0
            mocked.assert_called_once_with("path")

    def test_mkschema(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.vcf.mkschema") as mocked:
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

    def test_encode(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.vcf.encode") as mocked:
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
                worker_processes=1,
                show_progress=True,
            )

    def test_convert_vcf(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.vcf.convert") as mocked:
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
                worker_processes=1,
                show_progress=True,
            )

    def test_validate(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.vcf.validate") as mocked:
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

    def test_convert_plink(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.plink.convert") as mocked:
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
                chunk_width=None,
                chunk_length=None,
                show_progress=True,
            )


@pytest.mark.parametrize("cmd", [main.bio2zarr, cli.vcf2zarr, cli.plink2zarr])
def test_version(cmd):
    runner = ct.CliRunner(mix_stderr=False)
    result = runner.invoke(cmd, ["--version"], catch_exceptions=False)
    s = f"version {provenance.__version__}\n"
    assert result.stdout.endswith(s)
