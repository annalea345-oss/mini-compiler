"""Benchmark float32 tensor programs with verified results and reusable native buffers."""
import argparse
from datetime import datetime, timezone
import itertools
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

from tensor_compiler import CompileError, emit_c, emit_metal, evaluate, parse, validate_data


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def run(command):
    result = subprocess.run([str(part) for part in command], text=True, capture_output=True)
    if result.returncode:
        raise CompileError(f"Command failed: {' '.join(str(p) for p in command)}\n{result.stderr.strip()}")
    return result.stdout


def summarize(samples):
    if not samples or any(not math.isfinite(v) or v < 0 for v in samples):
        raise CompileError("Invalid timing samples")
    return {"min_ms": min(samples), "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples), "max_ms": max(samples)}


def benchmark_case(size, backends, iterations, warmups, seed, workdir, metal_kernels):
    directory = workdir / f"n{size}"
    directory.mkdir(parents=True, exist_ok=True)
    source = (f"input x: matrix[{size},{size}]\ninput weights: matrix[{size},{size}]\n"
              f"input bias: vector[{size}]\noutput = relu(x @ weights + bias)\n")
    (directory / "layer.tensor").write_text(source)
    rng = random.Random(seed + size)
    # Binary fractions make the correctness workload reproducible without decimal parsing ambiguity.
    def value():
        return rng.randint(-8, 8) / 8
    raw = {"x": [[value() for _ in range(size)] for _ in range(size)],
           "weights": [[value() for _ in range(size)] for _ in range(size)],
           "bias": [value() for _ in range(size)]}
    data_file = directory / "input.json"
    data_file.write_text(json.dumps(raw))
    inputs, root = parse(source)
    expected = evaluate(root, validate_data(inputs, raw, dtype="float32"), dtype="float32")
    variants = [(backend, fuse, kernel) for backend, fuse in itertools.product(backends, (False, True))
                for kernel in (metal_kernels if backend == "metal" else (None,))]
    random.Random(seed + size).shuffle(variants)
    rows = []
    for backend, fuse, kernel in variants:
        name = f"{backend}_{'fused' if fuse else 'unfused'}"
        if kernel:
            name += f"_{kernel}"
        source_file = directory / (name + (".m" if backend == "metal" else ".c"))
        binary = directory / name
        if backend == "metal":
            shader, host = emit_metal(inputs, root, fuse=fuse, metal_kernel=kernel)
            source_file.write_text(host)
            source_file.with_suffix(".metal").write_text(shader)
        else:
            source_file.write_text(emit_c(inputs, root, fuse=fuse, dtype="float32"))
        command = ["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", source_file, "-o", binary]
        if backend == "metal":
            command.extend(["-fobjc-arc", "-framework", "Foundation", "-framework", "Metal"])
        start = time.perf_counter()
        run(command)
        compile_ms = (time.perf_counter() - start) * 1000
        actual = [float(v) for v in run([binary, data_file]).splitlines()]
        if len(actual) != len(expected) or any(
            not math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-5) for a, b in zip(actual, expected)
        ):
            raise CompileError(f"Correctness comparison failed for {name}, size {size}")
        native = json.loads(run([binary, data_file, iterations, warmups]))
        if not math.isclose(native["checksum"], math.fsum(expected), rel_tol=2e-5, abs_tol=2e-5):
            raise CompileError(f"Benchmark checksum mismatch for {name}, size {size}")
        if len(native["wall_ms"]) != iterations:
            raise CompileError("Wrong number of timing samples")
        row = {"size": size, "backend": backend, "fused": fuse, "metal_kernel": kernel, "dtype": "float32",
               "verified": True, "compile_ms": compile_ms, **native,
               "wall_summary": summarize(native["wall_ms"]),
               "gpu_command_summary": (summarize(native["gpu_command_ms"])
                                       if native["gpu_command_ms"] is not None else None)}
        rows.append(row)
        print(f"{size:5d}  {backend:5s}  {'fused' if fuse else 'unfused':7s}  {kernel or '-':7s}  "
              f"{row['wall_summary']['median_ms']:12.6f}", flush=True)
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sizes", type=positive, nargs="+", default=[16, 64, 128])
    cli.add_argument("--backends", choices=["cpu", "metal"], nargs="+",
                     default=["cpu", "metal"] if sys.platform == "darwin" else ["cpu"])
    cli.add_argument("--iterations", type=positive, default=10)
    cli.add_argument("--metal-kernels", choices=["untiled", "tiled"], nargs="+", default=["untiled"],
                     help="Metal matrix kernels to compare (default: untiled)")
    cli.add_argument("--warmups", type=int, default=3)
    cli.add_argument("--seed", type=int, default=23)
    cli.add_argument("--workdir", type=Path, default=Path("build/benchmarks"))
    cli.add_argument("--out", type=Path, default=Path("build/benchmarks/report.json"))
    args = cli.parse_args()
    if args.warmups < 0 or max(args.warmups, args.iterations) > 1000000:
        cli.error("iterations must be 1..1000000 and warmups 0..1000000")
    if "metal" in args.backends and sys.platform != "darwin":
        cli.error("Metal requires macOS")
    try:
        args.workdir = args.workdir.resolve()
        print(" size  target  mode     kernel   median wall ms", flush=True)
        rows = []
        for size in args.sizes:
            rows.extend(benchmark_case(size, list(dict.fromkeys(args.backends)), args.iterations,
                                       args.warmups, args.seed, args.workdir, list(dict.fromkeys(args.metal_kernels))))
        repo = Path(__file__).parent
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
        report = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "platform": platform.platform(), "machine": platform.machine(),
                  "compiler": run(["clang", "--version"]).splitlines()[0],
                  "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                  "source_dirty": bool(dirty.stdout.strip()), "seed": args.seed,
                  "source_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                                    for name in ("tensor_compiler.py", "tensor_runtime.c", "metal_runtime.m",
                                                 "benchmark_runtime.c", "benchmark.py")},
                  "notes": ["CPU is single-threaded generated C; Metal kernel choice is recorded per variant.",
                            "Wall samples exclude allocation, JSON loading, shader/pipeline setup, checksum, and printing.",
                            "Metal wall samples include command encoding, submission, and waiting for completion.",
                            "GPU command samples use Metal command-buffer timestamps, including all passes.",
                            "Buffers and pipelines are reused; warmups are discarded; outputs are verified before timing.",
                            "Variants run in a seeded shuffled order per size; this is one local run, not a universal ranking."],
                  "results": rows}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Report: {args.out}")
        return 0
    except (CompileError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
