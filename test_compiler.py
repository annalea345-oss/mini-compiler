import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

from tensor_compiler import CompileError, emit_c, evaluate, explain, fuse_operations, parse, validate_data


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

    def test_fusion_preserves_source_graph(self):
        inputs, root = parse("input x: matrix[2,3] input w: matrix[3,4] "
                             "input b: vector[4] output = relu(x @ w + b)")
        before = explain(root)
        fused = fuse_operations(root)
        self.assertEqual(fused.op, "matmul_bias_relu")
        self.assertEqual(fused.shape, (2, 4))
        for child, name in zip(fused.args, ("x", "w", "b")):
            self.assertIs(child, inputs[name])
        self.assertEqual(explain(root), before)
        self.assertIs(fuse_operations(fused), fused)

        # There should be only one result array instead of three intermediates.
        optimized = emit_c(inputs, root)
        unoptimized = emit_c(inputs, root, fuse=False)
        self.assertEqual(optimized.count("  double t"), 4)
        self.assertEqual(unoptimized.count("  double t"), 6)

    def test_fusion_leaves_other_operations_unchanged(self):
        declarations = "input x: matrix[2,2] input w: matrix[2,2] input b: vector[2] "
        for expression in ("x @ w", "x @ w + b", "relu(x @ w)",
                           "relu(x @ w + x)", "relu(x + b)", "relu(b + b)"):
            with self.subTest(expression=expression):
                _, root = parse(declarations + "output = " + expression)
                self.assertIs(fuse_operations(root), root)

    def test_nested_fusion(self):
        _, root = parse("input x: matrix[2,2] input w: matrix[2,2] input b: vector[2] "
                        "output = relu(relu(x @ w + b) @ w + relu(b))")
        fused = fuse_operations(root)
        self.assertEqual(fused.op, "matmul_bias_relu")
        self.assertEqual(fused.args[0].op, "matmul_bias_relu")
        self.assertEqual(fused.args[2].op, "relu")

    @unittest.skipUnless(shutil.which("clang"), "clang is required for native parity tests")
    def test_native_matches_reference(self):
        rng = random.Random(42)
        programs = [
            "input x: matrix[3,4] input w: matrix[4,2] input b: vector[2] output = relu(x @ w + b)",
            "input x: matrix[3,4] input w: matrix[4,2] output = (relu(x) + x) @ w",
            "input x: vector[5] output = relu(x + x)",
            "input x: matrix[2,3] output = x",
            "input x: vector[2] input unused: vector[3] output = x",
            "input x: matrix[1,1] input w: matrix[1,1] input b: vector[1] output = relu(x @ w + b)",
            "input x: matrix[2,3] input w: matrix[3,4] input b: vector[4] "
            "input v: matrix[4,2] input c: vector[2] output = relu(relu(x @ w + b) @ v + c)",
            "input x: matrix[2,3] input w: matrix[3,2] input b: vector[2] "
            "output = relu(x @ w + relu(b + b)) + x @ w",
            "input x: matrix[2,2] input w: matrix[2,2] output = relu(x @ w + x)",
        ]
        for source in programs:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                inputs, root = parse(source)

                def array(shape):
                    if not shape:
                        return rng.uniform(-3, 3)
                    return [array(shape[1:]) for _ in range(shape[0])]

                c_file = Path(directory) / "test.c"
                binary = Path(directory) / "test"
                datasets = []
                for _ in range(3):
                    raw_data = {name: array(node.shape) for name, node in inputs.items()}
                    datasets.append(raw_data)
                for fuse in (False, True):
                    with self.subTest(fuse=fuse):
                        c_file.write_text(emit_c(inputs, root, fuse=fuse))
                        subprocess.run(["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                                        str(c_file), "-o", str(binary)], check=True, capture_output=True)
                        # Reuse each binary with the same three independent datasets.
                        for raw_data in datasets:
                            data = validate_data(inputs, raw_data)
                            data_file = Path(directory) / "input.json"
                            data_file.write_text(json.dumps(raw_data))
                            actual = [float(line) for line in subprocess.check_output(
                                [str(binary), str(data_file)], text=True).splitlines()]
                            expected = evaluate(root, data)
                            self.assertEqual(len(actual), len(expected))
                            for left, right in zip(actual, expected):
                                self.assertAlmostEqual(left, right, places=10)

    @unittest.skipUnless(shutil.which("clang"), "clang is required for fusion tests")
    def test_fusion_bias_order_and_relu_boundary(self):
        inputs, root = parse("input x: matrix[3,2] input w: matrix[2,1] "
                             "input b: vector[1] output = relu(x @ w + b)")
        # Starting the reduction with the bias would round away the 1 and give 0.
        raw_data = {"x": [[1e16, -1e16], [-1, 0], [-2, 0]],
                    "w": [[1], [1]], "b": [1]}
        self.assertEqual(evaluate(root, validate_data(inputs, raw_data)), [1, 0, 0])
        with tempfile.TemporaryDirectory() as directory:
            c_file = Path(directory) / "test.c"
            binary = Path(directory) / "test"
            data_file = Path(directory) / "input.json"
            data_file.write_text(json.dumps(raw_data))
            for fuse in (False, True):
                with self.subTest(fuse=fuse):
                    c_file.write_text(emit_c(inputs, root, fuse=fuse))
                    subprocess.run(["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                                    str(c_file), "-o", str(binary)], check=True, capture_output=True)
                    actual = subprocess.check_output([str(binary), str(data_file)], text=True)
                    self.assertEqual([float(v) for v in actual.splitlines()], [1, 0, 0])

    @unittest.skipUnless(shutil.which("clang"), "clang is required for runtime tests")
    def test_runtime_input_validation(self):
        inputs, root = parse("input x: matrix[2,2] input b: vector[2] output = x + b")
        with tempfile.TemporaryDirectory() as directory:
            c_file = Path(directory) / "test.c"
            binary = Path(directory) / "test"
            data_file = Path(directory) / "input.json"
            c_file.write_text(emit_c(inputs, root))
            subprocess.run(["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                            str(c_file), "-o", str(binary)], check=True, capture_output=True)

            valid = [
                '{"b":[1e1,-0.5],"x":[[1,2],[3,4]]}',
                r'{"\u0078":[[1,2],[3,4]],"b":[10,-0.5]}',
            ]
            for text in valid:
                with self.subTest(valid=text):
                    data_file.write_text(text)
                    result = subprocess.run([str(binary), str(data_file)], text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual([float(v) for v in result.stdout.splitlines()], [11, 1.5, 13, 3.5])

            invalid = [
                ('{}', 'missing'),
                ('[]', 'object'),
                ('{"z":[]}', 'unknown'),
                ('{"x":[[1,2],[3,4]],"x":[[1,2],[3,4]]}', 'duplicate'),
                ('{"x":[[1,2]]}', 'shape'),
                ('{"x":[[1,2],[3,4],[5,6]]}', 'shape'),
                ('{"x":[[1],[3,4]]}', 'shape'),
                ('{"x":[[1,2,3],[3,4]]}', 'shape'),
                ('{"x":[1,2,3,4]}', 'array'),
                ('{"x":[[true,2],[3,4]]}', 'number'),
                ('{"x":[[null,2],[3,4]]}', 'number'),
                ('{"x":[["1",2],[3,4]]}', 'number'),
                ('{"x":[[1e999,2],[3,4]]}', 'finite'),
                ('{"x":[[NaN,2],[3,4]]}', 'number'),
                ('{"x":[[Infinity,2],[3,4]]}', 'number'),
                ('{"x":[[+1,2],[3,4]]}', 'number'),
                ('{"x":[[01,2],[3,4]]}', 'number'),
                ('{"x":[[1.,2],[3,4]]}', 'digits'),
                ('{"x":[[1e,2],[3,4]]}', 'digits'),
                ('{"x":[[1,2],[3,4]],"b":[1,2],}', 'name'),
                ('{"x":[[1,2],[3,4]],"b":[1,2]} extra', 'after'),
                ('{"x":[[1,2],[3,4]],"b":[1,2]}\x00', 'NUL'),
                ('{"x":', 'array'),
                ('{"', 'name'),
                (r'{"\u0', 'escape'),
                ('', 'object'),
            ]
            for text, error in invalid:
                with self.subTest(invalid=text):
                    data_file.write_text(text)
                    result = subprocess.run([str(binary), str(data_file)], text=True, capture_output=True)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertEqual(result.stdout, '')
                    self.assertIn(error, result.stderr)

            for arguments in ([], [str(Path(directory) / "missing.json")],
                              [str(data_file), "extra"]):
                with self.subTest(arguments=arguments):
                    result = subprocess.run([str(binary), *arguments], text=True, capture_output=True)
                    self.assertEqual(result.returncode, 1)
                    self.assertTrue(result.stderr)
                    self.assertEqual(result.stdout, '')

    @unittest.skipUnless(shutil.which("clang"), "clang is required for CLI tests")
    def test_compile_cli_without_data(self):
        here = Path(__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            c_file = Path(directory) / "program.c"
            command = [sys.executable, str(here / "tensor_compiler.py"), "compile",
                       str(here / "examples/layer.tensor"), "--out", str(c_file)]
            for options in ([], ["--no-fuse"]):
                with self.subTest(options=options):
                    result = subprocess.run(command + options, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    generated = c_file.read_text()
                    self.assertEqual("/* matmul_bias_relu:" in generated, not options)
                    actual = subprocess.check_output([str(c_file.with_suffix("")),
                                                      str(here / "examples/layer.json")], text=True)
                    self.assertEqual([float(v) for v in actual.splitlines()], [0, 3.5, 0, 0.5, 3, 0, 0, 0])
            result = subprocess.run(command + ["--data", str(here / "examples/layer.json")],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("pass the JSON file", result.stderr)


if __name__ == "__main__":
    unittest.main()
