from unittest import mock

import click.testing as ct

from bio2zarr import cli

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

    def test_summarise(self):
        runner = ct.CliRunner(mix_stderr=False)
        with mock.patch("bio2zarr.vcf.summarise", return_value={}) as mocked:
            result = runner.invoke(
                cli.vcf2zarr, ["summarise", "path"], catch_exceptions=False

            )
            assert result.exit_code == 0
            assert result.stdout == "\n"
            assert len(result.stderr) == 0
            mocked.assert_called_once_with("path")

