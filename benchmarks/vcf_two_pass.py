"""Bounded, synthetic ICF versus two-pass VCF conversion benchmark.

Run from the repository root with the VCF extra installed. This prototype uses
private writer APIs and is deliberately separate from production conversion.
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
import pathlib
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import cyvcf2
import numcodecs
import numpy as np
import zarr

from bio2zarr import constants, vcf, vcz


def disk_bytes(path):
    """Logical bytes, not filesystem allocation or cumulative bytes written."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def sampled_disk_bytes(path):
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            # The writer moves completed chunks while the parent samples.
            continue
    return total


def generate(path, variants, samples, seed, fields, ploidy, info_bytes=0):
    rng = np.random.default_rng(seed)
    raw_bytes = 0
    with gzip.open(path, "wt", compresslevel=6) as out:

        def write(text):
            nonlocal raw_bytes
            raw_bytes += len(text.encode("utf-8"))
            out.write(text)

        write("##fileformat=VCFv4.3\n")
        write(f"##contig=<ID=1,length={variants + 100}>\n")
        write('##FILTER=<ID=PASS,Description="All filters passed">\n')
        write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        if info_bytes:
            write(
                '##INFO=<ID=PAYLOAD,Number=1,Type=String,Description="Random text">\n'
            )
        if fields == "GT:DP:PL":
            write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n')
            write('##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Likelihoods">\n')
        write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(f"S{j}" for j in range(samples))
            + "\n"
        )
        for i in range(variants):
            alleles = rng.integers(0, 2, size=(samples, ploidy))
            missing = rng.random(samples) < 0.03
            phased = rng.random(samples) < 0.25
            depth = rng.integers(0, 101, size=samples)
            likelihoods = rng.integers(0, 256, size=(samples, ploidy + 1))
            row = []
            for j in range(samples):
                # Include shorter calls and a late maximum-ploidy record.
                n = 1 if (i + j) % 29 == 0 and i < variants - 1 else ploidy
                sep = "|" if phased[j] else "/"
                gt = sep.join("." if missing[j] else str(a) for a in alleles[j, :n])
                if fields != "GT":
                    gt += f":{depth[j]}:" + ",".join(map(str, likelihoods[j, : n + 1]))
                row.append(gt)
            info = "."
            if info_bytes:
                alphabet = np.frombuffer(
                    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                    dtype="S1",
                )
                info = "PAYLOAD=" + rng.choice(alphabet, info_bytes).tobytes().decode()
            write(
                f"1\t{i + 1}\t{'v' + str(i) if i % 7 else '.'}\tA\tC\t"
                f"{i % 100}\tPASS\t{info}\t{fields}\t" + "\t".join(row) + "\n"
            )
    return raw_bytes


def raw_values(variant, fields):
    values = {
        "CHROM": variant.CHROM,
        "POS": variant.POS,
        "QUAL": variant.QUAL,
        "ID": variant.ID,
        "FILTERS": variant.FILTERS,
        "REF": variant.REF,
        "ALT": variant.ALT,
        "rlen": variant.end - variant.start,
    }
    for field in fields:
        if field.category == "INFO":
            values[field.full_name] = variant.INFO.get(field.name, None)
        elif field.category == "FORMAT":
            if field.name == "GT":
                values[field.full_name] = (
                    variant.genotype.array() if "GT" in variant.FORMAT else None
                )
            else:
                values[field.full_name] = variant.format(field.name)
    return values


class MetadataSource(vcf.IntermediateColumnarFormat):
    """Reuse schema generation, but never create/read an intermediate store."""

    def __init__(self, path, metadata, header):
        self._path = path
        self.metadata = metadata
        self.vcf_header = header
        self.fields = {
            f.full_name: SimpleNamespace(vcf_field=f) for f in metadata.fields
        }
        self.gt_field = next(f for f in metadata.fields if f.full_name == "FORMAT/GT")


def infer_schema(path):
    metadata, header = vcf.scan_vcf(path, 1)
    transformers = {
        f.full_name: vcf.VcfValueTransformer.factory(f, metadata.num_samples)
        for f in metadata.fields
    }
    count = 0
    reader = cyvcf2.VCF(str(path))
    try:
        for variant in reader:
            for name, value in raw_values(variant, metadata.fields).items():
                transformers[name].transform_and_update_bounds(value)
            count += 1
    finally:
        reader.close()
    metadata.num_records = count
    return MetadataSource(path, metadata, header), transformers


def infer_ploidy(path):
    maximum = 0
    count = 0
    reader = cyvcf2.VCF(str(path))
    try:
        for variant in reader:
            if "GT" in variant.FORMAT:
                maximum = max(maximum, variant.genotype.array().shape[1] - 1)
            count += 1
    finally:
        reader.close()
    return {"max_ploidy": maximum, "num_records": count}


def two_pass(path, output, chunk, sample_chunk):
    start = time.perf_counter()
    source, transformers = infer_schema(path)
    scan_seconds = time.perf_counter() - start
    schema = source.generate_schema(chunk, sample_chunk, False)
    writer = vcz.VcfZarrWriter(MetadataSource, output, zarr_format=2)
    writer.init(source, target_num_partitions=1, schema=schema)
    writer.wip_partition_path(0).mkdir()
    buffers = {
        spec.name: writer.init_partition_array(0, spec.name) for spec in schema.fields
    }
    sanitisers = {}
    for spec in schema.fields:
        if spec.source is not None:
            field = SimpleNamespace(vcf_field=source.fields[spec.source].vcf_field)
            ba = buffers[spec.name]
            factory = vcf.IntermediateColumnarFormatField.sanitiser_factory
            sanitisers[spec.name] = factory(field, ba.buff.shape[1:], ba.buff.dtype)
    reader = cyvcf2.VCF(str(path))
    count = 0
    try:
        for variant in reader:
            values = {
                name: transformers[name].transform(value) if value is not None else None
                for name, value in raw_values(variant, source.metadata.fields).items()
            }
            gt_value = values["FORMAT/GT"]
            gt = vcf.sanitise_value_int_2d(
                buffers["call_genotype"].buff.shape[1:],
                None if gt_value is None else gt_value[:, :-1],
            )
            phased = vcf.sanitise_value_bool_1d(
                (source.num_samples,), None if gt_value is None else gt_value[:, -1]
            )
            if gt.shape[1] == 1:
                phased[:] = True
            alleles = np.full(
                schema.dimensions["alleles"].size, "", dtype=np.dtypes.StringDType()
            )
            alleles[0] = variant.REF
            alleles[1 : 1 + len(variant.ALT)] = variant.ALT
            row = {
                "variant_contig": 0,
                "variant_filter": [True],
                "variant_allele": alleles,
                "variant_length": variant.end - variant.start,
                "variant_id": variant.ID or constants.STR_MISSING,
                "variant_id_mask": variant.ID is None,
                "call_genotype": gt,
                "call_genotype_phased": phased,
                "call_genotype_mask": gt < 0,
            }
            for spec in schema.fields:
                if spec.source is not None:
                    row[spec.name] = sanitisers[spec.name](values[spec.source])
            for name, ba in buffers.items():
                ba.buff[ba.next_buffer_row()] = row[name]
            count += 1
    finally:
        reader.close()
    if count != source.num_records:
        raise ValueError("Input changed between scan and encode")
    for ba in buffers.values():
        writer.finalise_partition_array(0, ba)
    writer.wip_partition_path(0).rename(writer.partition_path(0))
    writer.finalise()
    writer.create_index()
    return {
        "schema_scan_seconds": scan_seconds,
        "max_ploidy": schema.dimensions["ploidy"].size,
        "buffer_bytes": sum(ba.buff.nbytes for ba in buffers.values()),
    }


def worker(args):
    numcodecs.blosc.set_nthreads(1)
    start = time.perf_counter()
    details = {}
    if args.method == "icf":
        stage = args.work / "icf"
        phase_start = time.perf_counter()
        vcf.explode(stage, [args.input], worker_processes=0)
        details["explode_seconds"] = time.perf_counter() - phase_start
        details["temporary_icf_bytes"] = disk_bytes(stage)
        vcf.encode(
            stage,
            args.work / "output",
            variants_chunk_size=args.chunk,
            samples_chunk_size=args.sample_chunk,
            worker_processes=0,
            zarr_format=2,
        )
    elif args.method == "two_pass":
        details = two_pass(
            args.input, args.work / "output", args.chunk, args.sample_chunk
        )
        details["temporary_icf_bytes"] = 0
    elif args.method == "ploidy_scan":
        details = infer_ploidy(args.input)
        details["temporary_icf_bytes"] = 0
    elif args.method == "production_direct":
        kwargs = {}
        if args.production_batch_size is not None:
            kwargs["batch_size"] = args.production_batch_size
        vcf.convert(
            [args.input],
            args.work / "output",
            infer_ploidy=True,
            variants_chunk_size=args.chunk,
            samples_chunk_size=args.sample_chunk,
            worker_processes=0,
            zarr_format=2,
            **kwargs,
        )
        api_default = inspect.signature(vcf.convert).parameters["batch_size"].default
        details["batch_size_requested"] = args.production_batch_size
        details["batch_size_api_default"] = api_default
        details["batch_size"] = args.production_batch_size or api_default or args.chunk
        details["temporary_icf_bytes"] = 0
    details["wall_seconds"] = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    details["peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    details["output_bytes"] = disk_bytes(args.work / "output")
    print(json.dumps(details))


def compare(left, right, chunk):
    a = zarr.open_group(left, mode="r")
    b = zarr.open_group(right, mode="r")
    if set(a.array_keys()) != set(b.array_keys()):
        raise AssertionError("Output array sets differ")
    if dict(a.attrs) != dict(b.attrs):
        raise AssertionError("Output root attributes differ")
    for name in a.array_keys():
        x, y = a[name], b[name]
        if (
            x.shape != y.shape
            or x.chunks != y.chunks
            or x.dtype != y.dtype
            or dict(x.attrs) != dict(y.attrs)
        ):
            raise AssertionError(f"Array schema differs: {name}")
        for start in range(0, x.shape[0], chunk):
            xv, yv = x[start : start + chunk], y[start : start + chunk]
            if np.issubdtype(x.dtype, np.floating):
                # Missing and fill NaN payloads are part of VCF Zarr semantics.
                xv = xv.view(f"u{x.dtype.itemsize}")
                yv = yv.view(f"u{y.dtype.itemsize}")
            np.testing.assert_array_equal(xv, yv, err_msg=name)
    return sorted(a.array_keys())


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=positive, default=2000)
    parser.add_argument("--samples", type=positive, default=500)
    parser.add_argument("--ploidy", type=positive, default=2)
    parser.add_argument("--fields", choices=["GT", "GT:DP:PL"], default="GT")
    parser.add_argument("--chunk", type=positive, default=128)
    parser.add_argument("--sample-chunk", type=positive, default=250)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--production-direct", action="store_true")
    parser.add_argument("--production-batch-size", type=positive)
    parser.add_argument("--skip-prototype", action="store_true")
    parser.add_argument("--info-bytes", type=int, default=0)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--result", type=pathlib.Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=pathlib.Path, help=argparse.SUPPRESS)
    parser.add_argument("--work", type=pathlib.Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.skip_prototype and not args.production_direct:
        parser.error("--skip-prototype requires --production-direct")
    if args.info_bytes < 0 or args.info_bytes > 2048:
        parser.error("INFO payload must be between zero and 2048 bytes")
    allele_cap = 100_000_000 if args.large else 40_000_000
    if args.variants * args.samples * args.ploidy > allele_cap:
        parser.error(f"Synthetic workload capped at {allele_cap:,} genotype alleles")
    if (
        args.samples > 2000
        or args.variants > (1_000_000 if args.large else 20000)
        or args.ploidy > 8
        or args.chunk > (4096 if args.large else 256)
        or args.repeats > (2 if args.large else 5)
    ):
        parser.error(
            "Shared-host caps exceeded (use --large for up to 1M variants, "
            "100M genotype alleles, chunk<=4096 and repeats<=2)"
        )
    if args.large and shutil.disk_usage(tempfile.gettempdir()).free < 8 * 2**30:
        parser.error("Large workloads require at least 8 GiB free temporary disk")
    results = {
        "configuration": {
            name: getattr(args, name)
            for name in (
                "variants",
                "samples",
                "ploidy",
                "fields",
                "chunk",
                "sample_chunk",
                "repeats",
                "seed",
                "info_bytes",
                "production_batch_size",
            )
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("bio2zarr", "cyvcf2", "numpy", "zarr", "numcodecs")
            },
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "compression_threads": 1,
            "worker_processes": 0,
            "benchmark_sha256": hashlib.sha256(
                pathlib.Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "runs": [],
    }
    methods = ["icf"]
    if not args.skip_prototype:
        methods.extend(["two_pass", "ploidy_scan"])
    if args.production_direct:
        methods.append("production_direct")
    with tempfile.TemporaryDirectory(prefix="bio2zarr-benchmark-") as tmp:
        base = pathlib.Path(tmp)
        path = base / "input.vcf.gz"
        generation_start = time.perf_counter()
        results["input_uncompressed_bytes"] = generate(
            path,
            args.variants,
            args.samples,
            args.seed,
            args.fields,
            args.ploidy,
            args.info_bytes,
        )
        results["generation_seconds"] = time.perf_counter() - generation_start
        results["input_bytes"] = path.stat().st_size
        for repeat in range(args.repeats):
            outputs = {}
            # Rotate ordering to reduce systematic warm-cache/order effects.
            order = methods[repeat % len(methods) :] + methods[: repeat % len(methods)]
            for method in order:
                work = base / f"{repeat}-{method}"
                work.mkdir()
                env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                extra_args = (
                    ["--production-batch-size", str(args.production_batch_size)]
                    if args.production_batch_size is not None
                    else []
                )
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(pathlib.Path(__file__).resolve()),
                        "--worker",
                        "--method",
                        method,
                        "--input",
                        str(path),
                        "--work",
                        str(work),
                        "--chunk",
                        str(args.chunk),
                        "--sample-chunk",
                        str(args.sample_chunk),
                        *extra_args,
                    ],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                peak_work_bytes = 0
                peak_staging_bytes = 0
                while True:
                    peak_work_bytes = max(peak_work_bytes, sampled_disk_bytes(work))
                    peak_staging_bytes = max(
                        peak_staging_bytes, sampled_disk_bytes(work / "output" / "wip")
                    )
                    try:
                        stdout, stderr = process.communicate(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if process.returncode:
                    raise RuntimeError(f"{method} worker failed:\n{stderr}")
                record = json.loads(stdout)
                if method == "ploidy_scan" and (
                    record["max_ploidy"] != args.ploidy
                    or record["num_records"] != args.variants
                ):
                    raise AssertionError(
                        "GT-only scan differs from generated dimensions"
                    )
                record["sampled_peak_work_bytes"] = max(
                    peak_work_bytes, disk_bytes(work)
                )
                record["sampled_peak_output_staging_bytes"] = peak_staging_bytes
                record.update(method=method, repeat=repeat)
                results["runs"].append(record)
                outputs[method] = work / "output"
            for method in methods:
                if method not in ("icf", "ploidy_scan"):
                    arrays = compare(outputs["icf"], outputs[method], args.chunk)
                    shape = zarr.open_group(outputs[method])["call_genotype"].shape
                    if shape != (args.variants, args.samples, args.ploidy):
                        raise AssertionError("Output dimensions differ from generator")
                    results.setdefault("correctness", {})[method] = arrays
            print(f"Repeat {repeat + 1}: all output arrays identical", file=sys.stderr)
            for method in methods:
                shutil.rmtree(base / f"{repeat}-{method}")
    text = json.dumps(results, indent=2) + "\n"
    if args.result:
        args.result.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
