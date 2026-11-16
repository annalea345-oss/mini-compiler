import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from tensor_compiler import CompileError, emit_c, evaluate, parse, validate_data


class CompilerTests(unittest.TestCase):
    def test_example_has_expected_answer(self):
        here = Path(__file__).parent
        inputs, root = parse((here / "examples/layer.tensor").read_text())
        data = validate_data(inputs, json.loads((here / "examples/layer.json").read_text()))
        self.assertEqual(evaluate(root, data), [0, 3.5, 0, 0.5, 3, 0, 0, 0])

    def test_shape_errors_and_unknown_names(self):
        programs = [
            "input x: matrix[2,3] input w: matrix[4,2] output = x @ w",
            "input x: matrix[2,3] input b: vector[2] output = x + b",
            "input x: vector[2] output = relu(missing)",
            "input x: matrix[0,2] output = x",
            "input x: vector[2] input x: vector[2] output = x",
        ]
        for source in programs:
            with self.subTest(source=source), self.assertRaises(CompileError):
                parse(source)

    def test_bad_data(self):
        inputs, _ = parse("input x: matrix[2,2] output = x")
        for data in ({}, {"x": [[1, 2], [3]]}, {"x": [[1, 2], [True, 4]]},
                     {"x": [[1, 2], [float('inf'), 4]]}, {"x": [[1, 2], [3, 4]], "extra": []}):
            with self.subTest(data=data), self.assertRaises(CompileError):
                validate_data(inputs, data)

    @unittest.skipUnless(shutil.which("clang"), "clang is required for native parity tests")
    def test_native_matches_reference(self):
        rng = random.Random(42)
        programs = [
            "input x: matrix[3,4] input w: matrix[4,2] input b: vector[2] output = relu(x @ w + b)",
            "input x: matrix[3,4] input w: matrix[4,2] output = (relu(x) + x) @ w",
            "input x: vector[5] output = relu(x + x)",
            "input x: matrix[2,3] output = x",
        ]
        for source in programs:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                inputs, root = parse(source)

                def array(shape):
                    if not shape:
                        return rng.uniform(-3, 3)
                    return [array(shape[1:]) for _ in range(shape[0])]

                data = validate_data(inputs, {name: array(node.shape) for name, node in inputs.items()})
                c_file = Path(directory) / "test.c"
                binary = Path(directory) / "test"
                c_file.write_text(emit_c(inputs, root, data))
                subprocess.run(["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                                str(c_file), "-o", str(binary)], check=True, capture_output=True)
                actual = [float(line) for line in subprocess.check_output([str(binary)], text=True).splitlines()]
                expected = evaluate(root, data)
                self.assertEqual(len(actual), len(expected))
                for left, right in zip(actual, expected):
                    self.assertAlmostEqual(left, right, places=10)


if __name__ == "__main__":
    unittest.main()
