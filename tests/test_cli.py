from unittest import mock

import click.testing as ct
import numcodecs
import pytest

from bio2zarr import __main__ as main
from bio2zarr import cli, provenance

DEFAULT_EXPLODE_ARGS = dict(
    column_chunk_size=64,
    compressor=None,
    worker_processes=1,
    show_progress=True,
)

DEFAULT_DEXPLODE_PARTITION_ARGS = dict(show_progress=False)

DEFAULT_DEXPLODE_INIT_ARGS = dict(
    worker_processes=1,
    column_chunk_size=64,
    compressor=None,
    show_progress=True,
)

DEFAULT_ENCODE_ARGS = dict(
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    max_variant_chunks=None,
    worker_processes=1,
    max_memory=None,
    show_progress=True,
)

DEFAULT_DENCODE_INIT_ARGS = dict(
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    max_variant_chunks=None,
    show_progress=True,
)

DEFAULT_DENCODE_PARTITION_ARGS = dict()

DEFAULT_DENCODE_FINALISE_ARGS = dict(show_progress=True)


class TestWithMocks:
    vcf_path = "tests/data/vcf/sample.vcf.gz"

    @mock.patch("bio2zarr.vcf.explode")
    def test_vcf_explode(self, mocked, tmp_path):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, f"explode {self.vcf_path} {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path), (self.vcf_path,), **DEFAULT_EXPLODE_ARGS
        )

    @pytest.mark.parametrize("compressor", ["lz4", "zstd"])
    @mock.patch("bio2zarr.vcf.explode")
    def test_vcf_explode_compressor(self, mocked, tmp_path, compressor):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"explode {self.vcf_path} {icf_path} -C {compressor}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        kwargs = dict(DEFAULT_EXPLODE_ARGS)
        kwargs["compressor"] = numcodecs.Blosc(
            compressor, clevel=7, shuffle=numcodecs.Blosc.NOSHUFFLE
        )
        mocked.assert_called_once_with(
            str(icf_path),
            (self.vcf_path,),
            **kwargs,
        )

    @pytest.mark.parametrize("compressor", ["lz4", "zstd"])
    @mock.patch("bio2zarr.vcf.explode_init")
    def test_vcf_dexplode_init_compressor(self, mocked, tmp_path, compressor):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"dexplode-init {self.vcf_path} {icf_path} 1 -C {compressor}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) > 0
        assert len(result.stderr) == 0
        kwargs = dict(DEFAULT_EXPLODE_ARGS)
        kwargs["compressor"] = numcodecs.Blosc(
            compressor, clevel=7, shuffle=numcodecs.Blosc.NOSHUFFLE
        )
        mocked.assert_called_once_with(
            str(icf_path),
            (self.vcf_path,),
            target_num_partitions=1,
            **kwargs,
        )

    @pytest.mark.parametrize("compressor", ["LZ4", "asdf"])
    @mock.patch("bio2zarr.vcf.explode")
    def test_vcf_explode_bad_compressor(self, mocked, tmp_path, compressor):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        result = runner.invoke(
            cli.vcf2zarr,
            f"explode {self.vcf_path} {icf_path} --compressor {compressor}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert "Invalid value for '-C'" in result.stderr
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf.explode")
    def test_vcf_explode_multiple_vcfs(self, mocked, tmp_path):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"explode {self.vcf_path} {self.vcf_path} {icf_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path), (self.vcf_path, self.vcf_path), **DEFAULT_EXPLODE_ARGS
        )

    @pytest.mark.parametrize("response", ["y", "Y", "yes"])
    @mock.patch("bio2zarr.vcf.explode")
    def test_vcf_explode_overwrite_icf_confirm_yes(self, mocked, tmp_path, response):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"explode {self.vcf_path} {icf_path}",
            catch_exceptions=False,
            input=response,
        )
        assert result.exit_code == 0
        assert f"Do you want to overwrite {icf_path}" in result.stdout
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path), (self.vcf_path,), **DEFAULT_EXPLODE_ARGS
        )

    @pytest.mark.parametrize("response", ["y", "Y", "yes"])
    @mock.patch("bio2zarr.vcf.encode")
    def test_vcf_encode_overwrite_zarr_confirm_yes(self, mocked, tmp_path, response):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        zarr_path = tmp_path / "zarr"
        zarr_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"encode {icf_path} {zarr_path}",
            catch_exceptions=False,
            input=response,
        )
        assert result.exit_code == 0
        assert f"Do you want to overwrite {zarr_path}" in result.stdout
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path), str(zarr_path), **DEFAULT_ENCODE_ARGS
        )

    @pytest.mark.parametrize("force_arg", ["-f", "--force"])
    @mock.patch("bio2zarr.vcf.explode")
    def test_vcf_explode_overwrite_icf_force(self, mocked, tmp_path, force_arg):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"explode {self.vcf_path} {icf_path} {force_arg}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path), (self.vcf_path,), **DEFAULT_EXPLODE_ARGS
        )

    @pytest.mark.parametrize("force_arg", ["-f", "--force"])
    @mock.patch("bio2zarr.vcf.encode")
    def test_vcf_encode_overwrite_icf_force(self, mocked, tmp_path, force_arg):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        zarr_path = tmp_path / "zarr"
        zarr_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"encode {icf_path} {zarr_path} {force_arg}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path),
            str(zarr_path),
            **DEFAULT_ENCODE_ARGS,
        )

    @mock.patch("bio2zarr.vcf.explode")
    def test_vcf_explode_missing_vcf(self, mocked, tmp_path):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, f"explode no_such_file {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 2
        assert len(result.stdout) == 0
        assert "'no_such_file' does not exist" in result.stderr
        mocked.assert_not_called()

    @pytest.mark.parametrize("response", ["n", "N", "No"])
    @mock.patch("bio2zarr.vcf.explode")
    def test_vcf_explode_overwrite_icf_confirm_no(self, mocked, tmp_path, response):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"explode {self.vcf_path} {icf_path}",
            catch_exceptions=False,
            input=response,
        )
        assert result.exit_code == 1
        assert "Aborted" in result.stderr
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf.explode")
    def test_vcf_explode_missing_and_existing_vcf(self, mocked, tmp_path):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"explode {self.vcf_path} no_such_file {icf_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert len(result.stdout) == 0
        assert "'no_such_file' does not exist" in result.stderr
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf.explode_init", return_value=5)
    def test_vcf_dexplode_init(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        result = runner.invoke(
            cli.vcf2zarr,
            f"dexplode-init {self.vcf_path} {icf_path} 5",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stderr) == 0
        assert result.stdout == "5\n"
        mocked.assert_called_once_with(
            str(icf_path),
            (self.vcf_path,),
            target_num_partitions=5,
            **DEFAULT_DEXPLODE_INIT_ARGS,
        )

    @pytest.mark.parametrize("num_partitions", ["-- -1", "0", "asdf", "1.112"])
    @mock.patch("bio2zarr.vcf.explode_init", return_value=5)
    def test_vcf_dexplode_init_bad_num_partitions(
        self, mocked, tmp_path, num_partitions
    ):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        result = runner.invoke(
            cli.vcf2zarr,
            f"dexplode-init {self.vcf_path} {icf_path} {num_partitions}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert "Invalid value for 'NUM_PARTITIONS'" in result.stderr
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf.explode_partition")
    def test_vcf_dexplode_partition(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        result = runner.invoke(
            cli.vcf2zarr,
            f"dexplode-partition {icf_path} 1",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path), 1, **DEFAULT_DEXPLODE_PARTITION_ARGS
        )

    @mock.patch("bio2zarr.vcf.explode_partition")
    def test_vcf_dexplode_partition_missing_dir(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        result = runner.invoke(
            cli.vcf2zarr,
            f"dexplode-partition {icf_path} 1",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert len(result.stdout) == 0
        assert f"'{icf_path}' does not exist" in result.stderr
        mocked.assert_not_called()

    @pytest.mark.parametrize("partition", ["-- -1", "asdf", "1.112"])
    @mock.patch("bio2zarr.vcf.explode_partition")
    def test_vcf_dexplode_partition_bad_partition(self, mocked, tmp_path, partition):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        result = runner.invoke(
            cli.vcf2zarr,
            f"dexplode-partition {icf_path} {partition}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert "Invalid value for 'PARTITION'" in result.stderr
        assert len(result.stdout) == 0
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf.explode_finalise")
    def test_vcf_dexplode_finalise(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, f"dexplode-finalise {tmp_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(str(tmp_path))

    @mock.patch("bio2zarr.vcf.inspect")
    def test_inspect(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, f"inspect {tmp_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        assert result.stdout == "\n"
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(str(tmp_path))

    @mock.patch("bio2zarr.vcf.mkschema")
    def test_mkschema(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, f"mkschema {tmp_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        # TODO figure out how to test that we call it with stdout from
        # the CliRunner
        # mocked.assert_called_once_with("path", stdout)
        mocked.assert_called_once()

    @mock.patch("bio2zarr.vcf.encode")
    def test_encode(self, mocked, tmp_path):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        zarr_path = tmp_path / "zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, f"encode {icf_path} {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path),
            str(zarr_path),
            **DEFAULT_ENCODE_ARGS,
        )

    @mock.patch("bio2zarr.vcf.encode_init", return_value=(10, 1024))
    def test_dencode_init(self, mocked, tmp_path):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        zarr_path = tmp_path / "zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"dencode-init {icf_path} {zarr_path} 10",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert result.stdout == "10\t1 KiB\n"
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path),
            str(zarr_path),
            target_num_partitions=10,
            **DEFAULT_DENCODE_INIT_ARGS,
        )

    @mock.patch("bio2zarr.vcf.encode_partition")
    def test_vcf_dencode_partition(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        zarr_path = tmp_path / "zarr"
        zarr_path.mkdir()
        result = runner.invoke(
            cli.vcf2zarr,
            f"dencode-partition {zarr_path} 1",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(zarr_path), 1, **DEFAULT_DENCODE_PARTITION_ARGS
        )

    @mock.patch("bio2zarr.vcf.encode_finalise")
    def test_vcf_dencode_finalise(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, f"dencode-finalise {tmp_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(str(tmp_path), **DEFAULT_DENCODE_FINALISE_ARGS)

    @mock.patch("bio2zarr.vcf.convert")
    def test_convert_vcf(self, mocked):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"convert {self.vcf_path} zarr_path",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            (self.vcf_path,),
            "zarr_path",
            variants_chunk_size=None,
            samples_chunk_size=None,
            worker_processes=1,
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


class TestVcfEndToEnd:
    vcf_path = "tests/data/vcf/sample.vcf.gz"

    def test_dexplode(self, tmp_path):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"dexplode-init {self.vcf_path} {icf_path} 5",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "3"

        for j in range(3):
            result = runner.invoke(
                cli.vcf2zarr,
                f"dexplode-partition {icf_path} {j}",
                catch_exceptions=False,
            )
            assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr, f"dexplode-finalise {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 0

        result = runner.invoke(
            cli.vcf2zarr, f"inspect {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        # Arbitrary check
        assert "CHROM" in result.stdout

    def test_explode(self, tmp_path):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, f"explode {self.vcf_path} {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr, f"inspect {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        # Arbitrary check
        assert "CHROM" in result.stdout

    def test_encode(self, tmp_path):
        icf_path = tmp_path / "icf"
        zarr_path = tmp_path / "zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, f"explode {self.vcf_path} {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr, f"encode {icf_path} {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr, f"inspect {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        # Arbitrary check
        assert "variant_position" in result.stdout

    def test_dencode(self, tmp_path):
        icf_path = tmp_path / "icf"
        zarr_path = tmp_path / "zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr, f"explode {self.vcf_path} {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr,
            f"dencode-init {icf_path} {zarr_path} 5 --variants-chunk-size=3",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert result.stdout.split()[0] == "3"

        for j in range(3):
            result = runner.invoke(
                cli.vcf2zarr,
                f"dencode-partition {zarr_path} {j}",
                catch_exceptions=False,
            )
        assert result.exit_code == 0

        result = runner.invoke(
            cli.vcf2zarr, f"dencode-finalise {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0

        result = runner.invoke(
            cli.vcf2zarr, f"inspect {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        # Arbitrary check
        assert "variant_position" in result.stdout

    def test_convert(self, tmp_path):
        zarr_path = tmp_path / "zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr,
            f"convert {self.vcf_path} {zarr_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr, f"inspect {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        # Arbitrary check
        assert "variant_position" in result.stdout


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
