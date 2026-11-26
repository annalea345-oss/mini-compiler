"""Portable backend checks plus native integration tests when a Metal GPU is available."""
import json
import itertools
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

from tensor_compiler import CompileError, Node, emit_c, emit_metal, evaluate, parse, validate_data


HERE = Path(__file__).parent


def build_host(source, binary):
    subprocess.run(["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", "-fobjc-arc",
                    str(source), "-framework", "Foundation", "-framework", "Metal", "-o", str(binary)],
                   check=True, capture_output=True)


class MetalBackendTests(unittest.TestCase):
    def test_unsupported_operations_and_precision(self):
        inputs, root = parse("input x: vector[1] output = relu(x)")
        with self.assertRaisesRegex(CompileError, "unsupported_op"):
            emit_metal(inputs, Node("unsupported_op", (1,), (root,)))
        with self.assertRaisesRegex(CompileError, "float32 only"):
            emit_metal(inputs, root, dtype="float64")
        inputs, root = parse("input x: matrix[65536,65536] output = x")
        with self.assertRaisesRegex(CompileError, "elements"):
            emit_metal(inputs, root)

    def test_fusion_removes_gpu_buffers_and_dispatches(self):
        inputs, root = parse((HERE / "examples/layer.tensor").read_text())
        fused_shader, fused_host = emit_metal(inputs, root)
        plain_shader, plain_host = emit_metal(inputs, root, fuse=False)
        self.assertEqual(fused_shader.count("kernel void"), 1)
        self.assertEqual(plain_shader.count("kernel void"), 3)
        self.assertEqual(fused_host.count("= metal_buffer(device,"), 4)
        self.assertEqual(plain_host.count("= metal_buffer(device,"), 6)
        self.assertIn("// matmul_bias_relu:", fused_shader)
        self.assertEqual(root.op, "relu")

    def test_cli_capability_checks(self):
        cli = [sys.executable, str(HERE / "tensor_compiler.py")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rejected.m"
            result = subprocess.run(cli + ["compile", str(HERE / "examples/layer.tensor"),
                                           "--backend", "metal", "--dtype", "float64", "--out", str(output)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("float32 only", result.stderr)
            self.assertFalse(output.exists())
        result = subprocess.run(cli + ["check", str(HERE / "examples/layer.tensor"),
                                       "--backend", "metal"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("float32", result.stdout)
        self.assertIn("matmul_bias_relu", result.stdout)


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("clang"), "requires macOS and Clang")
class MetalExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A real GPU is required: headless CI or sandboxed processes may have none.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "probe.m"
            binary = Path(directory) / "probe"
            source.write_text('#import <Metal/Metal.h>\n'
                              'int main(void) { @autoreleasepool { return MTLCreateSystemDefaultDevice() ? 0 : 1; } }\n')
            build_host(source, binary)
            if subprocess.run([str(binary)], capture_output=True).returncode != 0:
                raise unittest.SkipTest("Metal GPU is not accessible in this process")

    def test_gpu_matches_cpu_and_interpreter(self):
        rng = random.Random(23)
        programs = [
            "input x: vector[1] output = relu(x)",
            "input x: vector[257] output = relu(x + x)",
            "input x: matrix[7,37] input b: vector[37] output = relu(x + b)",
            "input x: matrix[3,2] input y: matrix[3,2] output = relu(x) + relu(y + x)",
            "input x: vector[2] input unused: vector[3] output = x",
            "input x: matrix[1,1] input w: matrix[1,1] output = x @ w",
            "input x: matrix[7,65] input w: matrix[65,37] input b: vector[37] "
            "output = relu(x @ w + b)",
            "input x: matrix[2,3] input w: matrix[3,4] input b: vector[4] "
            "input v: matrix[4,2] input c: vector[2] output = relu(relu(x @ w + b) @ v + c)",
            "input x: matrix[3,2] input w: matrix[2,2] input b: vector[2] "
            "output = relu(x @ w + relu(b + b)) + x @ w",
            "input x: matrix[3,2] input w: matrix[2,2] output = relu(x @ w + x)",
        ]
        with tempfile.TemporaryDirectory() as directory:
            host_file = Path(directory) / "gpu.m"
            gpu = Path(directory) / "gpu"
            cpu_file = Path(directory) / "cpu.c"
            cpu = Path(directory) / "cpu"
            data_file = Path(directory) / "input.json"
            for source, fuse in itertools.product(programs, (False, True)):
                with self.subTest(source=source, fuse=fuse):
                    inputs, root = parse(source)
                    _, host = emit_metal(inputs, root, fuse=fuse)
                    host_file.write_text(host)
                    build_host(host_file, gpu)
                    cpu_file.write_text(emit_c(inputs, root, dtype="float32", fuse=fuse))
                    subprocess.run(["clang", "-std=c11", "-O2", str(cpu_file), "-o", str(cpu)],
                                   check=True, capture_output=True)

                    def array(shape, zero):
                        if not shape:
                            return 0 if zero else rng.uniform(-10, 10)
                        return [array(shape[1:], zero) for _ in range(shape[0])]

                    for zero in (False, True, False):
                        raw = {name: array(node.shape, zero) for name, node in inputs.items()}
                        data_file.write_text(json.dumps(raw))
                        data = validate_data(inputs, raw, dtype="float32")
                        expected = evaluate(root, data, dtype="float32")
                        for binary in (cpu, gpu):
                            output = subprocess.check_output([str(binary), str(data_file)], text=True)
                            actual = [float(value) for value in output.splitlines()]
                            self.assertEqual(len(actual), len(expected))
                            for value, reference in zip(actual, expected):
                                self.assertTrue(math.isclose(value, reference, rel_tol=2e-5, abs_tol=2e-5),
                                                (binary.name, value, reference))

    def test_gpu_rounding_and_bias_order(self):
        cases = [
            ("input x: matrix[1,3] input w: matrix[3,1] output = x @ w",
             {"x": [[16777216, 1, -16777216]], "w": [[1], [1], [1]]}, [0]),
            # Contracting the multiplication and addition would leave a nonzero residual.
            ("input x: matrix[1,2] input w: matrix[2,1] output = x @ w",
             {"x": [[-1 - 2**-22, 1 + 2**-23]], "w": [[1], [1 + 2**-23]]}, [0]),
            ("input x: matrix[3,2] input w: matrix[2,1] input b: vector[1] "
             "output = relu(x @ w + b)",
             {"x": [[16777216, -16777216], [-1, 0], [-2, 0]],
              "w": [[1], [1]], "b": [1]}, [1, 0, 0]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            host = Path(directory) / "gpu.m"
            binary = host.with_suffix("")
            data_file = Path(directory) / "input.json"
            for (source, raw, expected), fuse in itertools.product(cases, (False, True)):
                with self.subTest(source=source, fuse=fuse):
                    inputs, root = parse(source)
                    self.assertEqual(evaluate(root, validate_data(inputs, raw, dtype="float32"), dtype="float32"), expected)
                    _, generated = emit_metal(inputs, root, fuse=fuse)
                    host.write_text(generated)
                    build_host(host, binary)
                    data_file.write_text(json.dumps(raw))
                    output = subprocess.check_output([str(binary), str(data_file)], text=True)
                    self.assertEqual([float(v) for v in output.splitlines()], expected)

    def test_layer_cli_fused_and_unfused(self):
        cli = [sys.executable, str(HERE / "tensor_compiler.py")]
        with tempfile.TemporaryDirectory() as directory:
            host_file = Path(directory) / "layer.m"
            binary = host_file.with_suffix("")
            for options in ([], ["--no-fuse"]):
                with self.subTest(options=options):
                    subprocess.run(cli + ["compile", str(HERE / "examples/layer.tensor"),
                                          "--backend", "metal", "--out", str(host_file)] + options,
                                   check=True, capture_output=True)
                    self.assertEqual("// matmul_bias_relu:" in host_file.with_suffix(".metal").read_text(), not options)
                    for name, expected in (("layer.json", [0, 3.5, 0, 0.5, 3, 0, 0, 0]),
                                           ("layer-alt.json", [1, 2, 1, 2, 1, 2, 1, 2])):
                        output = subprocess.check_output([str(binary), str(HERE / "examples" / name)], text=True)
                        self.assertEqual([float(v) for v in output.splitlines()], expected)

    def test_gpu_cli_and_input_errors(self):
        cli = [sys.executable, str(HERE / "tensor_compiler.py")]
        with tempfile.TemporaryDirectory() as directory:
            host_file = Path(directory) / "activation.m"
            binary = host_file.with_suffix("")
            result = subprocess.run(cli + ["compile", str(HERE / "examples/activation.tensor"),
                                           "--backend", "metal", "--out", str(host_file)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(host_file.with_suffix(".metal").exists())
            # Runtime compilation uses embedded MSL, so the sidecar is not required.
            host_file.with_suffix(".metal").unlink()
            output = subprocess.check_output([str(binary), str(HERE / "examples/activation.json")], text=True)
            self.assertEqual([float(v) for v in output.splitlines()], [0, 0, 0.5, 0, 3.5, 3])
            data_file = Path(directory) / "bad.json"
            for raw in ({}, {"x": [[1, 2]], "bias": [0, 0]},
                        {"x": [[1, 2], [3, 4], [5, 6]], "bias": [1e39, 0]}):
                with self.subTest(raw=raw):
                    data_file.write_text(json.dumps(raw))
                    result = subprocess.run([str(binary), str(data_file)], text=True, capture_output=True)
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("error:", result.stderr)
            for arguments in ([], [str(Path(directory) / "missing.json")]):
                result = subprocess.run([str(binary), *arguments], text=True, capture_output=True)
                self.assertEqual(result.returncode, 1)
                self.assertTrue(result.stderr)


if __name__ == "__main__":
    unittest.main()
