import argparse
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

from benchmark import rectangular_shape
from library_baselines import emit_library_baseline, validate_shape
from tensor_compiler import CompileError, evaluate, parse, validate_data
import test_metal


HERE = Path(__file__).parent
APPLE_LIBRARIES = (sys.platform == "darwin" and shutil.which("clang")
                   and int(platform.mac_ver()[0].split(".")[0]) >= 15)


class ShapeTests(unittest.TestCase):
    def test_rectangular_shape_validation(self):
        self.assertEqual(rectangular_shape("3x5x7"), (3, 5, 7))
        self.assertEqual(rectangular_shape("1X17X33"), (1, 17, 33))
        for text in ("2x3", "0x3x2", "-1x3x2", "2x3x1.5", "65536x65536x1"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                rectangular_shape(text)
        with self.assertRaises(CompileError):
            emit_library_baseline((2, 3, 4), "unknown")
        with self.assertRaises(CompileError):
            validate_shape((True, 2, 3))


def verify_library(test, library):
    rng = random.Random(97)
    shapes = [(4, 3, 2), (1, 17, 33), (33, 19, 1), (17, 33, 7)]
    with tempfile.TemporaryDirectory() as directory:
        source_file = Path(directory) / ("baseline.m" if library == "mps" else "baseline.c")
        binary = source_file.with_suffix("")
        data_file = Path(directory) / "input.json"
        for m, k, n in shapes:
            with test.subTest(library=library, shape=(m, k, n)):
                source_file.write_text(emit_library_baseline((m, k, n), library))
                command = ["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", str(source_file), "-o", str(binary)]
                if library == "mps":
                    command.extend(["-fobjc-arc", "-framework", "Foundation", "-framework", "Metal",
                                    "-framework", "MetalPerformanceShaders"])
                else:
                    command.extend(["-mmacosx-version-min=15.0", "-framework", "Accelerate"])
                built = subprocess.run(command, text=True, capture_output=True)
                test.assertEqual(built.returncode, 0, built.stderr)
                inputs, root = parse(f"input x: matrix[{m},{k}] input weights: matrix[{k},{n}] "
                                     f"input bias: vector[{n}] output = relu(x @ weights + bias)")
                for zero in (False, True):
                    def value():
                        return 0 if zero else rng.uniform(-2, 2)
                    raw = {"x": [[value() for _ in range(k)] for _ in range(m)],
                           "weights": [[value() for _ in range(n)] for _ in range(k)],
                           "bias": [value() for _ in range(n)]}
                    if (m, k, n) == (4, 3, 2) and not zero:
                        raw = json.loads((HERE / "examples/layer.json").read_text())
                    expected = evaluate(root, validate_data(inputs, raw, dtype="float32"), dtype="float32")
                    if (m, k, n) == (4, 3, 2) and not zero:
                        test.assertEqual(expected, [0, 3.5, 0, 0.5, 3, 0, 0, 0])
                    data_file.write_text(json.dumps(raw))
                    actual = [float(v) for v in subprocess.check_output([str(binary), str(data_file)], text=True).splitlines()]
                    test.assertEqual(len(actual), len(expected))
                    for a, b in zip(actual, expected):
                        test.assertTrue(math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-5), (a, b))
                    report = json.loads(subprocess.check_output([str(binary), str(data_file), "3", "1"], text=True))
                    test.assertTrue(math.isclose(report["checksum"], math.fsum(expected), rel_tol=2e-5, abs_tol=2e-5))
                    test.assertEqual(len(report["wall_ms"]), 3)
                    if library == "mps":
                        test.assertEqual(len(report["gpu_command_ms"]), 3)
                    else:
                        test.assertIsNone(report["gpu_command_ms"])
                data_file.write_text("{}")
                invalid = subprocess.run([str(binary), str(data_file)], text=True, capture_output=True)
                test.assertEqual(invalid.returncode, 1)
                test.assertEqual(invalid.stdout, "")
                test.assertIn("missing", invalid.stderr)


@unittest.skipUnless(APPLE_LIBRARIES, "Apple comparisons require macOS 15+ and Clang")
class AccelerateTests(unittest.TestCase):
    def test_accelerate_rectangles_and_epilogue(self):
        verify_library(self, "accelerate")


@unittest.skipUnless(APPLE_LIBRARIES, "Apple comparisons require macOS 15+ and Clang")
class MPSBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_metal.MetalExecutionTests.setUpClass()

    def test_mps_rectangles_and_row_padding(self):
        verify_library(self, "mps")

    def test_rectangular_comparison_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            result = subprocess.run([sys.executable, str(HERE / "benchmark.py"), "--backends", "accelerate", "mps",
                                     "--shapes", "3x5x7", "--iterations", "2", "--warmups", "1",
                                     "--workdir", str(Path(directory) / "work"), "--out", str(output)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads(output.read_text())["results"]
            self.assertEqual({row["backend"] for row in rows}, {"accelerate", "mps"})
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["shape"], [3, 5, 7])
                self.assertIsNone(row["fused"])
                self.assertTrue(row["verified"])
                self.assertLessEqual(row["max_absolute_error"], 2e-5)
                if row["backend"] == "accelerate":
                    self.assertEqual(row["cpu_threading"], "single")


if __name__ == "__main__":
    unittest.main()
