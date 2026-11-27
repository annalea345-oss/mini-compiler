import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tensor_compiler import CompileError, emit_c, parse


HERE = Path(__file__).parent


class AllocationTests(unittest.TestCase):
    def test_unrepresentable_allocation_is_rejected(self):
        inputs, root = parse(f"input x: vector[{sys.maxsize}] output = x")
        with self.assertRaisesRegex(CompileError, "allocation"):
            emit_c(inputs, root)


@unittest.skipUnless(shutil.which("clang"), "Clang is required")
class BenchmarkTests(unittest.TestCase):
    def test_large_tensor_runs_with_small_stack(self):
        if sys.platform == "win32":
            self.skipTest("requires POSIX stack limits")
        import resource

        def small_stack():
            resource.setrlimit(resource.RLIMIT_STACK, (256 * 1024, resource.getrlimit(resource.RLIMIT_STACK)[1]))

        inputs, root = parse("input x: vector[300000] output = relu(x)")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "large.c"
            binary = source.with_suffix("")
            data = Path(directory) / "input.json"
            data.write_text(json.dumps({"x": [1, -2] * 150000}))
            source.write_text(emit_c(inputs, root))
            subprocess.run(["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                            str(source), "-o", str(binary)], check=True, capture_output=True)
            result = subprocess.run([str(binary), str(data), "3", "2"], preexec_fn=small_stack,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["checksum"], 150000)
            self.assertEqual(report["iterations"], 3)
            self.assertEqual(report["warmups"], 2)
            self.assertGreater(report["clock_resolution_ns"], 0)
            if sys.platform == "darwin":
                self.assertEqual(report["clock"], "monotonic_raw")
            self.assertEqual(len(report["wall_ms"]), 3)
            self.assertTrue(all(math.isfinite(v) and v >= 0 for v in report["wall_ms"]))
            self.assertIsNone(report["gpu_command_ms"])
            for arguments in (("0", "1"), ("1", "-1"), ("1.5", "0"), ("1000001", "0")):
                with self.subTest(arguments=arguments):
                    bad = subprocess.run([str(binary), str(data), *arguments], text=True, capture_output=True)
                    self.assertEqual(bad.returncode, 1)
                    self.assertIn("benchmark", bad.stderr)

    def test_cpu_benchmark_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            result = subprocess.run([sys.executable, str(HERE / "benchmark.py"), "--backends", "cpu",
                                     "--sizes", "4", "--iterations", "2", "--warmups", "1",
                                     "--workdir", str(Path(directory) / "work"), "--out", str(output)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads(output.read_text())["results"]
            self.assertEqual({row["fused"] for row in rows}, {False, True})
            for row in rows:
                self.assertTrue(row["verified"])
                self.assertEqual(len(row["wall_ms"]), 2)
                self.assertIsNone(row["gpu_command_summary"])
                self.assertEqual(row["backend"], "cpu")


if __name__ == "__main__":
    unittest.main()
