import dataclasses
import json
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

DEFAULT_DEXPLODE_PARTITION_ARGS = dict()

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

DEFAULT_CONVERT_ARGS = dict(
    variants_chunk_size=None,
    samples_chunk_size=None,
    show_progress=True,
    worker_processes=1,
)


@dataclasses.dataclass
class FakeWorkSummary:
    num_partitions: int

    def asdict(self):
        return dataclasses.asdict(self)

    def asjson(self):
        return json.dumps(self.asdict())


class TestWithMocks:
    vcf_path = "tests/data/vcf/sample.vcf.gz"

    @pytest.mark.parametrize(("progress", "flag"), [(True, "-P"), (False, "-Q")])
    @mock.patch("bio2zarr.vcf2zarr.explode")
    def test_vcf_explode(self, mocked, tmp_path, progress, flag):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"explode {self.vcf_path} {icf_path} {flag}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        args = dict(DEFAULT_EXPLODE_ARGS)
        args["show_progress"] = progress
        mocked.assert_called_once_with(str(icf_path), (self.vcf_path,), **args)

    @pytest.mark.parametrize("compressor", ["lz4", "zstd"])
    @mock.patch("bio2zarr.vcf2zarr.explode")
    def test_vcf_explode_compressor(self, mocked, tmp_path, compressor):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
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
    @mock.patch("bio2zarr.vcf2zarr.explode_init")
    def test_vcf_dexplode_init_compressor(self, mocked, tmp_path, compressor):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dexplode-init {self.vcf_path} {icf_path} -n 1 -C {compressor}",
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
    @mock.patch("bio2zarr.vcf2zarr.explode")
    def test_vcf_explode_bad_compressor(self, mocked, tmp_path, compressor):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"explode {self.vcf_path} {icf_path} --compressor {compressor}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert "Invalid value for '-C'" in result.stderr
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf2zarr.explode")
    def test_vcf_explode_multiple_vcfs(self, mocked, tmp_path):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
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
    @mock.patch("bio2zarr.vcf2zarr.explode")
    def test_vcf_explode_overwrite_icf_confirm_yes(self, mocked, tmp_path, response):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
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
    @mock.patch("bio2zarr.vcf2zarr.encode")
    def test_vcf_encode_overwrite_zarr_confirm_yes(self, mocked, tmp_path, response):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        zarr_path = tmp_path / "zarr"
        zarr_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
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
    @mock.patch("bio2zarr.vcf2zarr.explode")
    def test_vcf_explode_overwrite_icf_force(self, mocked, tmp_path, force_arg):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
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
    @mock.patch("bio2zarr.vcf2zarr.encode")
    def test_vcf_encode_overwrite_icf_force(self, mocked, tmp_path, force_arg):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        zarr_path = tmp_path / "zarr"
        zarr_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
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

    @mock.patch("bio2zarr.vcf2zarr.explode")
    def test_vcf_explode_missing_vcf(self, mocked, tmp_path):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"explode no_such_file {icf_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert len(result.stdout) == 0
        assert "'no_such_file' does not exist" in result.stderr
        mocked.assert_not_called()

    @pytest.mark.parametrize("response", ["n", "N", "No"])
    @mock.patch("bio2zarr.vcf2zarr.explode")
    def test_vcf_explode_overwrite_icf_confirm_no(self, mocked, tmp_path, response):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"explode {self.vcf_path} {icf_path}",
            catch_exceptions=False,
            input=response,
        )
        assert result.exit_code == 1
        assert "Aborted" in result.stderr
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf2zarr.explode")
    def test_vcf_explode_missing_and_existing_vcf(self, mocked, tmp_path):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"explode {self.vcf_path} no_such_file {icf_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert len(result.stdout) == 0
        assert "'no_such_file' does not exist" in result.stderr
        mocked.assert_not_called()

    @pytest.mark.parametrize(("progress", "flag"), [(True, "-P"), (False, "-Q")])
    @mock.patch("bio2zarr.vcf2zarr.explode_init", return_value=FakeWorkSummary(5))
    def test_vcf_dexplode_init(self, mocked, tmp_path, progress, flag):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dexplode-init {self.vcf_path} {icf_path} -n 5 {flag}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stderr) == 0
        assert list(result.stdout.split()) == ["num_partitions", "5"]
        args = dict(DEFAULT_DEXPLODE_INIT_ARGS)
        args["show_progress"] = progress
        mocked.assert_called_once_with(
            str(icf_path),
            (self.vcf_path,),
            target_num_partitions=5,
            **args,
        )

    @pytest.mark.parametrize("num_partitions", ["-1", "0", "asdf", "1.112"])
    @mock.patch("bio2zarr.vcf2zarr.explode_init", return_value=5)
    def test_vcf_dexplode_init_bad_num_partitions(
        self, mocked, tmp_path, num_partitions
    ):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dexplode-init {self.vcf_path} {icf_path} -n {num_partitions}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert "Invalid value for '-n'" in result.stderr
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf2zarr.explode_init", return_value=5)
    def test_vcf_dexplode_init_no_partitions(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dexplode-init {self.vcf_path} {icf_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert "-n/--num-partitions must currently be specified" in result.stderr
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf2zarr.explode_partition")
    def test_vcf_dexplode_partition(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dexplode-partition {icf_path} 1",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path), 1, **DEFAULT_DEXPLODE_PARTITION_ARGS
        )

    @mock.patch("bio2zarr.vcf2zarr.explode_partition")
    def test_vcf_dexplode_partition_one_based(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dexplode-partition {icf_path} 1 --one-based",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(icf_path), 0, **DEFAULT_DEXPLODE_PARTITION_ARGS
        )

    @mock.patch("bio2zarr.vcf2zarr.explode_partition")
    def test_vcf_dexplode_partition_missing_dir(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dexplode-partition {icf_path} 1",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert len(result.stdout) == 0
        assert f"'{icf_path}' does not exist" in result.stderr
        mocked.assert_not_called()

    @pytest.mark.parametrize("partition", ["-- -1", "asdf", "1.112"])
    @mock.patch("bio2zarr.vcf2zarr.explode_partition")
    def test_vcf_dexplode_partition_bad_partition(self, mocked, tmp_path, partition):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dexplode-partition {icf_path} {partition}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert "Invalid value for 'PARTITION'" in result.stderr
        assert len(result.stdout) == 0
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf2zarr.explode_finalise")
    def test_vcf_dexplode_finalise(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main, f"dexplode-finalise {tmp_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(str(tmp_path))

    @mock.patch("bio2zarr.vcf2zarr.inspect")
    def test_inspect(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main, f"inspect {tmp_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        assert result.stdout == "\n"
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(str(tmp_path))

    @mock.patch("bio2zarr.vcf2zarr.mkschema")
    def test_mkschema(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main, f"mkschema {tmp_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        # TODO figure out how to test that we call it with stdout from
        # the CliRunner
        # mocked.assert_called_once_with("path", stdout)
        mocked.assert_called_once()

    @pytest.mark.parametrize(("progress", "flag"), [(True, "-P"), (False, "-Q")])
    @mock.patch("bio2zarr.vcf2zarr.encode")
    def test_encode(self, mocked, tmp_path, progress, flag):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        zarr_path = tmp_path / "zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"encode {icf_path} {zarr_path} {flag}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        args = DEFAULT_ENCODE_ARGS
        args["show_progress"] = progress
        mocked.assert_called_once_with(
            str(icf_path),
            str(zarr_path),
            **args,
        )

    @pytest.mark.parametrize(("progress", "flag"), [(True, "-P"), (False, "-Q")])
    @mock.patch("bio2zarr.vcf2zarr.encode_init", return_value=FakeWorkSummary(10))
    def test_dencode_init(self, mocked, tmp_path, progress, flag):
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        zarr_path = tmp_path / "zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dencode-init {icf_path} {zarr_path} -n 10 {flag}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert list(result.stdout.split()) == ["num_partitions", "10"]
        assert len(result.stderr) == 0
        args = DEFAULT_DENCODE_INIT_ARGS
        args["show_progress"] = progress
        mocked.assert_called_once_with(
            str(icf_path),
            str(zarr_path),
            target_num_partitions=10,
            **args,
        )

    @mock.patch("bio2zarr.vcf2zarr.encode_init", return_value=5)
    def test_vcf_dencode_init_no_partitions(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        icf_path = tmp_path / "icf"
        icf_path.mkdir()
        zarr_path = tmp_path / "zarr"
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dencode-init {icf_path} {zarr_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert "-n/--num-partitions must currently be specified" in result.stderr
        mocked.assert_not_called()

    @mock.patch("bio2zarr.vcf2zarr.encode_partition")
    def test_vcf_dencode_partition(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        zarr_path = tmp_path / "zarr"
        zarr_path.mkdir()
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dencode-partition {zarr_path} 1",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(zarr_path), 1, **DEFAULT_DENCODE_PARTITION_ARGS
        )

    @mock.patch("bio2zarr.vcf2zarr.encode_partition")
    def test_vcf_dencode_partition_one_based(self, mocked, tmp_path):
        runner = ct.CliRunner(mix_stderr=False)
        zarr_path = tmp_path / "zarr"
        zarr_path.mkdir()
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dencode-partition {zarr_path} 1 --one-based",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            str(zarr_path), 0, **DEFAULT_DENCODE_PARTITION_ARGS
        )

    @pytest.mark.parametrize(("progress", "flag"), [(True, "-P"), (False, "-Q")])
    @mock.patch("bio2zarr.vcf2zarr.encode_finalise")
    def test_vcf_dencode_finalise(self, mocked, tmp_path, progress, flag):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dencode-finalise {tmp_path} {flag}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        args = DEFAULT_DENCODE_FINALISE_ARGS
        args["show_progress"] = progress
        mocked.assert_called_once_with(str(tmp_path), **args)

    @pytest.mark.parametrize(("progress", "flag"), [(True, "-P"), (False, "-Q")])
    @mock.patch("bio2zarr.vcf2zarr.convert")
    def test_convert_vcf(self, mocked, progress, flag):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"convert {self.vcf_path} zarr_path {flag}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        args = dict(DEFAULT_CONVERT_ARGS)
        args["show_progress"] = progress
        mocked.assert_called_once_with(
            (self.vcf_path,),
            "zarr_path",
            **args,
        )

    @pytest.mark.parametrize("response", ["n", "N", "No"])
    @mock.patch("bio2zarr.vcf2zarr.convert")
    def test_vcf_convert_overwrite_zarr_confirm_no(self, mocked, tmp_path, response):
        zarr_path = tmp_path / "zarr"
        zarr_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"convert {self.vcf_path} {zarr_path}",
            catch_exceptions=False,
            input=response,
        )
        assert result.exit_code == 1
        assert "Aborted" in result.stderr
        mocked.assert_not_called()

    @pytest.mark.parametrize(("progress", "flag"), [(True, "-P"), (False, "-Q")])
    @mock.patch("bio2zarr.plink.convert")
    def test_convert_plink(self, mocked, progress, flag):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.plink2zarr, ["convert", "in", "out", flag], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        args = dict(DEFAULT_CONVERT_ARGS)
        args["show_progress"] = progress
        mocked.assert_called_once_with("in", "out", **args)

    @pytest.mark.parametrize("response", ["y", "Y", "yes"])
    @mock.patch("bio2zarr.vcf2zarr.convert")
    def test_vcf_convert_overwrite_zarr_confirm_yes(self, mocked, tmp_path, response):
        zarr_path = tmp_path / "zarr"
        zarr_path.mkdir()
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"convert {self.vcf_path} {zarr_path}",
            catch_exceptions=False,
            input=response,
        )
        assert result.exit_code == 0
        assert f"Do you want to overwrite {zarr_path}" in result.stdout
        assert len(result.stderr) == 0
        mocked.assert_called_once_with(
            (self.vcf_path,), str(zarr_path), **DEFAULT_CONVERT_ARGS
        )


class TestVcfEndToEnd:
    vcf_path = "tests/data/vcf/sample.vcf.gz"

    @pytest.mark.parametrize("one_based", [False, True])
    def test_dexplode(self, tmp_path, one_based):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dexplode-init {self.vcf_path} {icf_path} -n 5 --json -Q",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stderr) == 0
        assert json.loads(result.stdout)["num_partitions"] == 3

        for j in range(3):
            if one_based:
                cmd = f"dexplode-partition {icf_path} {j + 1} --one-based"
            else:
                cmd = f"dexplode-partition {icf_path} {j}"
            result = runner.invoke(cli.vcf2zarr_main, cmd, catch_exceptions=False)
            assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr_main, f"dexplode-finalise {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        assert len(result.stderr) == 0

        result = runner.invoke(
            cli.vcf2zarr_main, f"inspect {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        # Arbitrary check
        assert "CHROM" in result.stdout

    def test_explode(self, tmp_path):
        icf_path = tmp_path / "icf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"explode {self.vcf_path} {icf_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr_main, f"inspect {icf_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        # Arbitrary check
        assert "CHROM" in result.stdout

    def test_encode(self, tmp_path):
        icf_path = tmp_path / "icf"
        zarr_path = tmp_path / "zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"explode {self.vcf_path} {icf_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr_main, f"encode {icf_path} {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr_main, f"inspect {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        # Arbitrary check
        assert "variant_position" in result.stdout

    @pytest.mark.parametrize("one_based", [False, True])
    def test_dencode(self, tmp_path, one_based):
        icf_path = tmp_path / "icf"
        zarr_path = tmp_path / "zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"explode {self.vcf_path} {icf_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"dencode-init {icf_path} {zarr_path} -n 5 --variants-chunk-size=3 --json",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert json.loads(result.stdout)["num_partitions"] == 3

        for j in range(3):
            if one_based:
                cmd = f"dencode-partition {zarr_path} {j + 1} --one-based"
            else:
                cmd = f"dencode-partition {zarr_path} {j}"
            result = runner.invoke(cli.vcf2zarr_main, cmd, catch_exceptions=False)
            assert result.exit_code == 0

        result = runner.invoke(
            cli.vcf2zarr_main, f"dencode-finalise {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0

        result = runner.invoke(
            cli.vcf2zarr_main, f"inspect {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        # Arbitrary check
        assert "variant_position" in result.stdout

    def test_convert(self, tmp_path):
        zarr_path = tmp_path / "zarr"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcf2zarr_main,
            f"convert {self.vcf_path} {zarr_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli.vcf2zarr_main, f"inspect {zarr_path}", catch_exceptions=False
        )
        assert result.exit_code == 0
        # Arbitrary check
        assert "variant_position" in result.stdout


class TestVcfPartition:
    path = "tests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz"

    def test_num_parts(self):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcfpartition, [self.path, "-n", "5"], catch_exceptions=False
        )
        assert result.stderr == ""
        assert result.exit_code == 0
        assert list(result.stdout.splitlines()) == [
            "20:60001-278528\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:278529-442368\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:442381-638976\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:638982-819200\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:819201-\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
        ]

    def test_part_size(self):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcfpartition, [self.path, "-s", "512K"], catch_exceptions=False
        )
        assert result.stderr == ""
        assert result.exit_code == 0
        assert list(result.stdout.splitlines()) == [
            "20:60001-212992\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:213070-327680\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:327695-442368\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:442381-557056\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:557078-688128\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:688129-802816\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:802817-933888\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
            "20:933890-\ttests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz",
        ]

    def test_no_part_spec(self):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(cli.vcfpartition, [self.path], catch_exceptions=False)
        assert result.exit_code != 0
        assert result.stdout == ""
        assert len(result.stderr) > 0

    def test_no_args(self):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(cli.vcfpartition, [], catch_exceptions=False)
        assert result.exit_code != 0
        assert result.stdout == ""
        assert len(result.stderr) > 0


@pytest.mark.parametrize(
    "cmd", [main.bio2zarr, cli.vcf2zarr_main, cli.plink2zarr, cli.vcfpartition]
)
def test_version(cmd):
    runner = ct.CliRunner(mix_stderr=False)
    result = runner.invoke(cmd, ["--version"], catch_exceptions=False)
    s = f"version {provenance.__version__}\n"
    assert result.stdout.endswith(s)
