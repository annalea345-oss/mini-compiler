"""Benchmark float32 tensor programs with verified results and reusable native buffers."""
import argparse
from datetime import datetime, timezone
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
from library_baselines import emit_library_baseline, validate_shape


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def rectangular_shape(value):
    try:
        return validate_shape(tuple(int(d) for d in value.lower().split("x")))
    except (ValueError, CompileError) as error:
        raise argparse.ArgumentTypeError("use three positive dimensions, e.g. 32x128x256 (M x K x N)") from error


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


def benchmark_case(shape, backends, iterations, warmups, seed, workdir, metal_kernels):
    m, k, n = validate_shape(shape)
    size = m if m == k == n else None
    label = f"{m}x{k}x{n}"
    directory = workdir / (f"n{size}" if size is not None else label)
    directory.mkdir(parents=True, exist_ok=True)
    source = (f"input x: matrix[{m},{k}]\ninput weights: matrix[{k},{n}]\n"
              f"input bias: vector[{n}]\noutput = relu(x @ weights + bias)\n")
    (directory / "layer.tensor").write_text(source)
    case_seed = seed + (size if size is not None else m * 1000003 + k * 1009 + n)
    rng = random.Random(case_seed)
    # Binary fractions make the correctness workload reproducible without decimal parsing ambiguity.
    def value():
        return rng.randint(-8, 8) / 8
    raw = {"x": [[value() for _ in range(k)] for _ in range(m)],
           "weights": [[value() for _ in range(n)] for _ in range(k)],
           "bias": [value() for _ in range(n)]}
    data_file = directory / "input.json"
    data_file.write_text(json.dumps(raw))
    inputs, root = parse(source)
    expected = evaluate(root, validate_data(inputs, raw, dtype="float32"), dtype="float32")
    variants = [(backend, fuse, kernel) for backend in backends
                for fuse in ((None,) if backend in ("accelerate", "mps") else (False, True))
                for kernel in (metal_kernels if backend == "metal" else (None,))]
    random.Random(case_seed).shuffle(variants)
    rows = []
    for backend, fuse, kernel in variants:
        mode = "library" if fuse is None else ("fused" if fuse else "unfused")
        name = f"{backend}_{mode}"
        if kernel:
            name += f"_{kernel}"
        source_file = directory / (name + (".m" if backend in ("metal", "mps") else ".c"))
        binary = directory / name
        if backend in ("accelerate", "mps"):
            source_file.write_text(emit_library_baseline((m, k, n), backend))
        elif backend == "metal":
            shader, host = emit_metal(inputs, root, fuse=fuse, metal_kernel=kernel)
            source_file.write_text(host)
            source_file.with_suffix(".metal").write_text(shader)
        else:
            source_file.write_text(emit_c(inputs, root, fuse=fuse, dtype="float32"))
        command = ["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", source_file, "-o", binary]
        if backend in ("metal", "mps"):
            command.extend(["-fobjc-arc", "-framework", "Foundation", "-framework", "Metal"])
        if backend == "mps":
            command.extend(["-framework", "MetalPerformanceShaders"])
        if backend == "accelerate":
            command.extend(["-mmacosx-version-min=15.0", "-framework", "Accelerate"])
        start = time.perf_counter()
        run(command)
        compile_ms = (time.perf_counter() - start) * 1000
        actual = [float(v) for v in run([binary, data_file]).splitlines()]
        if len(actual) != len(expected) or any(
            not math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-5) for a, b in zip(actual, expected)
        ):
            raise CompileError(f"Correctness comparison failed for {name}, shape {label}")
        max_absolute_error = max(abs(a - b) for a, b in zip(actual, expected))
        native = json.loads(run([binary, data_file, iterations, warmups]))
        if not math.isclose(native["checksum"], math.fsum(expected), rel_tol=2e-5, abs_tol=2e-5):
            raise CompileError(f"Benchmark checksum mismatch for {name}, shape {label}")
        if len(native["wall_ms"]) != iterations:
            raise CompileError("Wrong number of timing samples")
        row = {"size": size, "shape": [m, k, n], "backend": backend, "fused": fuse,
               "metal_kernel": kernel, "dtype": "float32", "max_absolute_error": max_absolute_error,
               "cpu_threading": "single" if backend in ("cpu", "accelerate") else None,
               "implementation": {"cpu": "generated C", "metal": "generated MSL",
                                  "accelerate": "Accelerate SGEMM + CPU bias/ReLU",
                                  "mps": "MPS matrix multiplication + custom GPU bias/ReLU"}[backend],
               "verified": True, "compile_ms": compile_ms, **native,
               "wall_summary": summarize(native["wall_ms"]),
               "gpu_command_summary": (summarize(native["gpu_command_ms"])
                                       if native["gpu_command_ms"] is not None else None)}
        rows.append(row)
        print(f"{label:16s}  {backend:10s}  {mode:7s}  {kernel or '-':7s}  "
              f"{row['wall_summary']['median_ms']:12.6f}", flush=True)
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sizes", type=positive, nargs="+", help="square sizes (default: 16 64 128 if no shapes supplied)")
    cli.add_argument("--shapes", type=rectangular_shape, nargs="+", help="M x K x N, e.g. 32x128x256")
    cli.add_argument("--backends", choices=["cpu", "metal", "accelerate", "mps"], nargs="+",
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
    if any(b in args.backends for b in ("metal", "accelerate", "mps")) and sys.platform != "darwin":
        cli.error("Metal and Apple library baselines require macOS")
    try:
        args.workdir = args.workdir.resolve()
        print("shape (MxKxN)     target      mode     kernel   median wall ms", flush=True)
        rows = []
        sizes = args.sizes if args.sizes is not None else ([] if args.shapes else [16, 64, 128])
        shapes = list(dict.fromkeys([(s, s, s) for s in sizes] + (args.shapes or [])))
        for shape in shapes:
            rows.extend(benchmark_case(shape, list(dict.fromkeys(args.backends)), args.iterations,
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
                                                 "benchmark_runtime.c", "benchmark.py", "library_baselines.py",
                                                 "accelerate_baseline.c", "mps_baseline.m")},
                  "notes": ["CPU is single-threaded generated C; Metal kernel choice is recorded per variant.",
                            "Wall samples exclude allocation, JSON loading, shader/pipeline setup, checksum, and printing.",
                            "Metal wall samples include command encoding, submission, and waiting for completion.",
                            "GPU command samples use Metal command-buffer timestamps, including all passes.",
                            "All variants perform the full relu(x @ weights + bias) calculation.",
                            "Accelerate uses explicit single-threaded BLAS (macOS 15+); library reduction orders may differ.",
                            "MPS uses padded float32 matrices with packing/unpacking outside timed work and a custom bias/ReLU GPU pass.",
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
