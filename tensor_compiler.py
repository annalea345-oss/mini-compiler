"""A small tensor compiler: source -> typed graph -> C/Metal -> CPU/GPU."""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys


class CompileError(Exception):
    pass


@dataclass
class Node:
    op: str
    shape: tuple
    args: tuple = ()
    name: str = ""


class Parser:
    def __init__(self, source):
        self.tokens = []
        for line_number, line in enumerate(source.splitlines(), 1):
            line = line.split("#", 1)[0]
            pos = 0
            while pos < len(line):
                if line[pos].isspace():
                    pos += 1
                    continue
                match = re.match(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[][():,@+=]", line[pos:])
                if not match:
                    raise CompileError(f"Line {line_number}: unexpected character {line[pos]!r}")
                self.tokens.append((match.group(), line_number))
                pos += len(match.group())
        self.tokens.append(("<end>", len(source.splitlines()) + 1))
        self.pos = 0
        self.inputs = {}

    def peek(self):
        return self.tokens[self.pos][0]

    def take(self, expected=None):
        token, line = self.tokens[self.pos]
        if expected is not None and token != expected:
            raise CompileError(f"Line {line}: expected {expected!r}, got {token!r}")
        if token == "<end>":
            raise CompileError(f"Line {line}: unexpected end of program")
        self.pos += 1
        return token

    def parse(self):
        while self.peek() == "input":
            self.take("input")
            name = self.take()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) or name in {"input", "output", "relu"}:
                raise CompileError(f"Invalid input name: {name!r}")
            if name in self.inputs:
                raise CompileError(f"Input {name!r} is declared twice")
            self.take(":")
            kind = self.take()
            if kind not in ("matrix", "vector"):
                raise CompileError("Input type must be matrix or vector")
            self.take("[")
            dimensions = [self.dimension()]
            if kind == "matrix":
                self.take(",")
                dimensions.append(self.dimension())
            self.take("]")
            self.inputs[name] = Node("input", tuple(dimensions), name=name)
        self.take("output")
        self.take("=")
        result = self.expression()
        if self.peek() != "<end>":
            raise CompileError(f"Unexpected token after output: {self.peek()!r}")
        return self.inputs, result

    def dimension(self):
        token = self.take()
        if not token.isdecimal() or int(token) < 1:
            raise CompileError("Dimensions must be positive integers")
        return int(token)

    def expression(self):
        node = self.product()
        while self.peek() == "+":
            self.take("+")
            right = self.product()
            if node.shape != right.shape and not (
                len(node.shape) == 2 and right.shape == (node.shape[1],)
            ):
                raise CompileError(f"Cannot add shapes {node.shape} and {right.shape}. "
                                   "Use equal shapes, or a matrix + a vector matching its columns.")
            node = Node("add", node.shape, (node, right))
        return node

    def product(self):
        node = self.atom()
        while self.peek() == "@":
            self.take("@")
            right = self.atom()
            if len(node.shape) != 2 or len(right.shape) != 2 or node.shape[1] != right.shape[0]:
                raise CompileError(f"Cannot multiply shapes {node.shape} and {right.shape}. "
                                   "Use two matrices; left columns must match right rows.")
            node = Node("matmul", (node.shape[0], right.shape[1]), (node, right))
        return node

    def atom(self):
        name = self.take()
        if name == "relu":
            self.take("(")
            node = self.expression()
            self.take(")")
            return Node("relu", node.shape, (node,))
        if name == "(":
            node = self.expression()
            self.take(")")
            return node
        if name not in self.inputs:
            raise CompileError(f"Unknown input {name!r}")
        return self.inputs[name]


def parse(source):
    return Parser(source).parse()


def numeric_type(dtype):
    if dtype not in ("float32", "float64"):
        raise CompileError(f"Unsupported numeric type: {dtype!r}; use float32 or float64")
    return "float" if dtype == "float32" else "double"


def float32(value):
    """Round to IEEE binary32, including arithmetic overflow to infinity."""
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def validate_data(inputs, data, *, dtype="float64"):
    numeric_type(dtype)
    if not isinstance(data, dict):
        raise CompileError("Input JSON must be an object mapping names to arrays")

    def check(value, shape, path):
        if shape:
            if not isinstance(value, list) or len(value) != shape[0]:
                raise CompileError(f"{path}: expected an array of length {shape[0]}")
            result = []
            for i, item in enumerate(value):
                result.extend(check(item, shape[1:], f"{path}[{i}]"))
            return result
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompileError(f"{path}: expected a finite number")
        try:
            number = float(value)
        except OverflowError:
            raise CompileError(f"{path}: number is too large")
        if not math.isfinite(number):
            raise CompileError(f"{path}: expected a finite number")
        if dtype == "float32":
            number = float32(number)
            if not math.isfinite(number):
                raise CompileError(f"{path}: expected a finite float32 number")
        return [number]

    for name in inputs:
        if name not in data:
            raise CompileError(f"Missing input {name!r}")
    if set(data) - set(inputs):
        raise CompileError(f"Unknown inputs: {', '.join(sorted(set(data) - set(inputs)))}")
    return {name: check(data[name], node.shape, name) for name, node in inputs.items()}


def evaluate(node, data, *, dtype="float64"):
    """Simple, deliberately unoptimized CPU reference implementation."""
    numeric_type(dtype)
    rounded = float32 if dtype == "float32" else float

    def visit(current):
        if current.op == "input":
            return [rounded(value) for value in data[current.name]]
        left = visit(current.args[0])
        if current.op == "relu":
            return [value if value > 0.0 else 0.0 for value in left]
        right = visit(current.args[1])
        if current.op == "add":
            return [rounded(value + right[i % len(right)]) for i, value in enumerate(left)]
        rows, columns = current.shape
        inner = current.args[0].shape[1]
        result = []
        for i in range(rows):
            for j in range(columns):
                total = 0.0
                for k in range(inner):
                    product = rounded(left[i * inner + k] * right[k * columns + j])
                    total = rounded(total + product)
                result.append(total)
        return result

    return visit(node)


def fuse_operations(root):
    """Fuse relu(matmul + row bias), preserving the original graph and inputs."""
    rewritten = {}

    def visit(node):
        key = id(node)
        if key in rewritten:
            return rewritten[key]
        args = tuple(visit(child) for child in node.args)
        result = node if all(a is b for a, b in zip(args, node.args)) else Node(
            node.op, node.shape, args, node.name)
        if node.op == "relu" and args[0].op == "add":
            product, bias = args[0].args
            if product.op == "matmul" and bias.shape == (product.shape[1],):
                result = Node("matmul_bias_relu", node.shape, (*product.args, bias))
        rewritten[key] = result
        return result

    return visit(root)


def emit_c(inputs, root, *, fuse=True, dtype="float64"):
    """Emit fixed-shape loops with input values loaded from JSON at runtime."""
    ctype = numeric_type(dtype)
    zero = "0.0f" if dtype == "float32" else "0.0"
    precision = 9 if dtype == "float32" else 17
    if fuse:
        root = fuse_operations(root)
    runtime = Path(__file__).with_name("tensor_runtime.c").read_text()
    timing = Path(__file__).with_name("benchmark_runtime.c").read_text()
    prefix = ["#define _DARWIN_C_SOURCE 1", "#define _POSIX_C_SOURCE 200809L", f"/* Numeric type: {dtype} */", f"typedef {ctype} TensorValue;",
             f"#define TENSOR_PARSE {'strtof' if dtype == 'float32' else 'strtod'}",
             "#pragma STDC FP_CONTRACT OFF", runtime, timing]
    lines = ["int main(int argc, char **argv) {", "  RunOptions options = run_options(argc, argv);",
             "  double setup_start = monotonic_ms();"]
    body = []
    names = {}

    def allocate(node, name):
        size = math.prod(node.shape)
        if size > sys.maxsize // (4 if dtype == "float32" else 8):
            raise CompileError("Tensor allocation is too large for this platform")
        lines.append(f"  {ctype} *{name} = tensor_alloc({size}); /* {node.name or node.op}: {node.shape} */")

    for node in inputs.values():
        name = f"t{len(names)}"
        names[id(node)] = name
        allocate(node, name)
    lines.append("  TensorInput inputs[] = {")
    for node in inputs.values():
        shape = ", ".join(str(d) for d in node.shape)
        lines.append(f'    {{"{node.name}", {names[id(node)]}, {len(node.shape)}, {{{shape}}}, 0}},')
    lines.extend(["  };", f"  char input_name[{max(len(name) for name in inputs) + 1}];",
                  "  double load_start = monotonic_ms();",
                  f"  load_inputs(argv[1], inputs, {len(inputs)}, input_name, sizeof(input_name));",
                  "  double load_ms = monotonic_ms() - load_start;"])

    def emit(node):
        key = id(node)
        if key in names:
            return names[key]
        children = [emit(child) for child in node.args]
        name = f"t{len(names)}"
        names[key] = name
        size = math.prod(node.shape)
        if node.op != "input":
            allocate(node, name)
            if node.op in ("matmul", "matmul_bias_relu"):
                rows, columns = node.shape
                inner = node.args[0].shape[1]
                body.extend([
                    f"  for (size_t i = 0; i < {rows}; ++i)",
                    f"    for (size_t j = 0; j < {columns}; ++j) {{",
                    f"      {ctype} sum = {zero};",
                    f"      for (size_t k = 0; k < {inner}; ++k)",
                    f"        sum += {children[0]}[i*{inner}+k] * {children[1]}[k*{columns}+j];"])
                if node.op == "matmul_bias_relu":
                    # Add the bias after the reduction, as in the original graph.
                    body.extend([f"      sum += {children[2]}[j];",
                                  f"      {name}[i*{columns}+j] = sum > {zero} ? sum : {zero};"])
                else:
                    body.append(f"      {name}[i*{columns}+j] = sum;")
                body.append("    }")
            else:
                if node.op == "add":
                    expression = f"{children[0]}[i] + {children[1]}[i % {math.prod(node.args[1].shape)}]"
                else:
                    expression = f"{children[0]}[i] > {zero} ? {children[0]}[i] : {zero}"
                body.append(f"  for (size_t i = 0; i < {size}; ++i) {name}[i] = {expression};")
        return name

    result = emit(root)
    params = ", ".join(f"{ctype} *{name}" for name in names.values())
    compute = [f"__attribute__((noinline)) void compute({params}) {{"]
    compute.extend(f"  (void){name};" for name in names.values())
    compute.extend(body + ["}"])
    call = f"compute({', '.join(names.values())});"
    size = math.prod(root.shape)
    lines.extend(["  double setup_ms = monotonic_ms() - setup_start - load_ms;",
                  "  volatile double checksum = 0.0;",
                  "  for (size_t step = 0; step < options.warmups + options.iterations; ++step) {",
                  "    double start = monotonic_ms();", f"    {call}",
                  "    double elapsed = monotonic_ms() - start;",
                  "    if (options.benchmark) {",
                  f"      checksum = result_checksum({result}, {size});",
                  "      if (step >= options.warmups) options.wall[step - options.warmups] = elapsed;",
                  "    }", "  }",
                  '  if (options.benchmark) benchmark_report(options, setup_ms, load_ms, checksum, "CPU", 0);',
                  f'  else for (size_t i = 0; i < {size}; ++i) printf("%.{precision}g\\n", (double){result}[i]);'])
    lines.extend(f"  free({name});" for name in names.values())
    lines.extend(["  return 0;", "}"])
    return "\n".join(prefix + compute + lines) + "\n"


def metal_nodes(root, *, dtype="float32"):
    """Validate this backend's capabilities and return operations in dependency order."""
    if dtype != "float32":
        raise CompileError("Metal currently supports float32 only; use --dtype float32")
    seen, nodes = set(), []

    def visit(node):
        if id(node) in seen:
            return
        if node.op not in ("input", "add", "relu", "matmul", "matmul_bias_relu"):
            raise CompileError(f"Metal does not support {node.op!r}")
        if math.prod(node.shape) > 2**32 - 1:
            raise CompileError("Metal tensors must contain at most 2^32 - 1 elements")
        seen.add(id(node))
        for child in node.args:
            visit(child)
        if node.op != "input":
            nodes.append(node)

    visit(root)
    return nodes


def emit_metal(inputs, root, *, dtype="float32", fuse=True, metal_kernel="untiled"):
    """Generate MSL kernels and a standalone Objective-C host with embedded MSL."""
    if metal_kernel not in ("untiled", "tiled"):
        raise CompileError("Metal kernel must be untiled or tiled")
    metal_nodes(root, dtype=dtype)
    for node in inputs.values():
        metal_nodes(node, dtype=dtype)
    if fuse:
        root = fuse_operations(root)
    operations = metal_nodes(root, dtype=dtype)
    nodes = list(inputs.values()) + operations
    names = {id(node): f"t{i}" for i, node in enumerate(nodes)}
    kernels = ["#include <metal_stdlib>", "using namespace metal;",
               "#pragma STDC FP_CONTRACT OFF"]
    for node in operations:
        name = names[id(node)]
        tiled = metal_kernel == "tiled" and node.op in ("matmul", "matmul_bias_relu")
        args = [f"device const float *a{i} [[buffer({i})]]" for i in range(len(node.args))]
        args.append(f"device float *out [[buffer({len(node.args)})]]")
        if tiled:
            args.extend(["uint2 group [[threadgroup_position_in_grid]]",
                         "uint2 local [[thread_position_in_threadgroup]]"])
        else:
            args.append("uint i [[thread_position_in_grid]]")
        kernels.extend([f"// {node.op}: {node.shape}",
                        f"kernel void {name}({', '.join(args)}) {{"])
        if not tiled:
            kernels.append(f"  if (i >= {math.prod(node.shape)}u) return;")
        if node.op in ("matmul", "matmul_bias_relu"):
            rows = node.shape[0]
            columns = node.shape[1]
            inner = node.args[0].shape[1]
            if tiled:
                kernels.extend(["  uint row = group.y * 32u + local.y * 2u;",
                                "  uint column = group.x * 32u + local.x * 2u;",
                                "  threadgroup float left_tile[32][32];",
                                "  threadgroup float right_tile[32][32];",
                                "  float4 sum = float4(0.0f);",
                                f"  for (uint block = 0; block < {(inner + 31) // 32}u; ++block) {{",
                                "    uint base = block * 32u;",
                                "    for (uint dy = 0; dy < 2u; ++dy)",
                                "      for (uint dx = 0; dx < 2u; ++dx) {",
                                "        uint ty = local.y * 2u + dy, tx = local.x * 2u + dx;",
                                f"        left_tile[ty][tx] = row + dy < {rows}u && base + tx < {inner}u",
                                f"          ? a0[(row + dy) * {inner}u + base + tx] : 0.0f;",
                                f"        right_tile[ty][tx] = base + ty < {inner}u && column + dx < {columns}u",
                                f"          ? a1[(base + ty) * {columns}u + column + dx] : 0.0f;",
                                "      }",
                                "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                                f"    for (uint k = 0; k < min(32u, {inner}u - base); ++k) {{",
                                "      float l0 = left_tile[local.y * 2u][k], l1 = left_tile[local.y * 2u + 1u][k];",
                                "      float r0 = right_tile[k][local.x * 2u], r1 = right_tile[k][local.x * 2u + 1u];",
                                "      sum.x += l0 * r0; sum.y += l0 * r1;",
                                "      sum.z += l1 * r0; sum.w += l1 * r1;",
                                "    }",
                                "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                                "  }",
                                "  for (uint dy = 0; dy < 2u; ++dy)",
                                "    for (uint dx = 0; dx < 2u; ++dx) {",
                                f"      if (row + dy >= {rows}u || column + dx >= {columns}u) continue;",
                                "      float value = sum[dy * 2u + dx];"])
                if node.op == "matmul_bias_relu":
                    kernels.extend(["      value += a2[column + dx];",
                                    "      value = value > 0.0f ? value : 0.0f;"])
                kernels.extend([f"      out[(row + dy) * {columns}u + column + dx] = value;", "    }"])
            else:
                kernels.extend([f"  uint row = i / {columns}u;",
                            f"  uint column = i % {columns}u;",
                            "  float sum = 0.0f;",
                            f"  for (uint k = 0; k < {inner}u; ++k)",
                            f"    sum += a0[row * {inner}u + k] * a1[k * {columns}u + column];"])
            if node.op == "matmul_bias_relu" and not tiled:
                kernels.extend(["  sum += a2[column];",
                                "  out[i] = sum > 0.0f ? sum : 0.0f;"])
            elif not tiled:
                kernels.append("  out[i] = sum;")
        else:
            expression = (f"a0[i] + a1[i % {math.prod(node.args[1].shape)}u]" if node.op == "add"
                          else "a0[i] > 0.0f ? a0[i] : 0.0f")
            kernels.append(f"  out[i] = {expression};")
        kernels.append("}")
    shader = "\n".join(kernels) + "\n"
    runtime = Path(__file__).with_name("tensor_runtime.c").read_text()
    metal_runtime = Path(__file__).with_name("metal_runtime.m").read_text()
    timing = Path(__file__).with_name("benchmark_runtime.c").read_text()
    lines = ["#define _DARWIN_C_SOURCE 1", "#define _POSIX_C_SOURCE 200809L", "/* Metal backend: float32, runtime-compiled embedded MSL. */",
             "typedef float TensorValue;", "#define TENSOR_PARSE strtof", runtime, timing, metal_runtime,
             "int main(int argc, char **argv) {", "  @autoreleasepool {",
             "    RunOptions options = run_options(argc, argv);",
             "    double setup_start = monotonic_ms();",
             "    id<MTLDevice> device = MTLCreateSystemDefaultDevice();",
             '    if (!device) fail("Metal", "no GPU device available");']
    for node in nodes:
        lines.append(f"    __attribute__((objc_precise_lifetime)) id<MTLBuffer> {names[id(node)]} = metal_buffer(device, {math.prod(node.shape)});")
    lines.append("    TensorInput inputs[] = {")
    for node in inputs.values():
        shape = ", ".join(str(d) for d in node.shape)
        lines.append(f'      {{"{node.name}", (float *){names[id(node)]}.contents, {len(node.shape)}, {{{shape}}}, 0}},')
    lines.extend(["    };", f"    char input_name[{max(len(name) for name in inputs) + 1}];",
                  "    double load_start = monotonic_ms();",
                  f"    load_inputs(argv[1], inputs, {len(inputs)}, input_name, sizeof(input_name));",
                  "    double load_ms = monotonic_ms() - load_start;"])
    if operations:
        lines.extend([f"    id<MTLLibrary> library = metal_library(device, @{json.dumps(shader)});",
                      "    id<MTLCommandQueue> queue = [device newCommandQueue];",
                      '    if (!queue) fail("Metal", "could not create command queue");'])
        for node in operations:
            name = names[id(node)]
            lines.append(f'    id<MTLComputePipelineState> p_{name} = metal_pipeline(device, library, @"{name}");')
    lines.extend(["    double setup_ms = monotonic_ms() - setup_start - load_ms;",
                  f"    const float *result = (const float *){names[id(root)]}.contents;",
                  "    volatile double checksum = 0.0;",
                  "    for (size_t step = 0; step < options.warmups + options.iterations; ++step) {",
                  "      @autoreleasepool {",
                  "        double start = monotonic_ms();", "        double gpu_ms = 0.0;"])
    if operations:
        lines.extend(["        id<MTLCommandBuffer> command = [queue commandBuffer];",
                      '        if (!command) fail("Metal", "could not create command buffer");'])
        for node in operations:
            buffers = ", ".join(names[id(child)] for child in (*node.args, node))
            tiled = metal_kernel == "tiled" and node.op in ("matmul", "matmul_bias_relu")
            rows, columns = node.shape if len(node.shape) == 2 else (node.shape[0], 1)
            lines.append(f'        metal_encode(command, p_{names[id(node)]}, @[{buffers}], {rows}, {columns}, {"YES" if tiled else "NO"});')
        lines.append("        metal_finish(command);")
    lines.append("        double elapsed = monotonic_ms() - start;")
    if operations:
        lines.append("        gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;")
    lines.extend(["        if (options.benchmark) {",
                  f"          checksum = result_checksum(result, {math.prod(root.shape)});",
                  "          if (step >= options.warmups) {",
                  "            options.wall[step - options.warmups] = elapsed;",
                  "            options.gpu[step - options.warmups] = gpu_ms;",
                  "          }", "        }", "      }", "    }",
                  f"    if (options.benchmark) benchmark_report(options, setup_ms, load_ms, checksum, device.name.UTF8String, {1 if operations else 0});",
                  f'    else for (size_t i = 0; i < {math.prod(root.shape)}; ++i) printf("%.9g\\n", (double)result[i]);',
                  "  }", "  return 0;", "}"])
    return shader, "\n".join(lines) + "\n"


def explain(node, indent=0):
    label = f"input {node.name}" if node.op == "input" else node.op
    lines = ["  " * indent + f"{label}: {node.shape}"]
    for child in node.args:
        lines.extend(explain(child, indent + 1))
    return lines


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=["check", "run", "compile"])
    cli.add_argument("source", type=Path)
    cli.add_argument("--data", type=Path, help="JSON input arrays (for run only)")
    cli.add_argument("--backend", choices=["cpu", "metal"], default="cpu",
                     help="compilation target (default: cpu; run uses the reference interpreter)")
    cli.add_argument("--out", type=Path, help="generated host source (default: build/program.c or build/program.m)")
    cli.add_argument("--dtype", choices=["float32", "float64"],
                     help="numeric type (default: float64 for cpu, float32 for metal)")
    cli.add_argument("--no-fuse", action="store_true",
                     help="disable operation fusion for check/compile (run always uses the reference interpreter)")
    cli.add_argument("--metal-kernel", choices=["untiled", "tiled"], default="untiled",
                     help="Metal matrix multiplication implementation (default: untiled)")
    args = cli.parse_args()
    try:
        args.dtype = args.dtype or ("float32" if args.backend == "metal" else "float64")
        args.out = args.out or Path("build/program.m" if args.backend == "metal" else "build/program.c")
        inputs, root = parse(args.source.read_text())
        if args.backend == "metal":
            if args.command == "run":
                raise CompileError("run uses the CPU reference interpreter; compile --backend metal, then run the executable")
            metal_nodes(root, dtype=args.dtype)
        if args.command == "check":
            print("Valid program. Typed calculation tree:")
            print(f"Numeric type: {args.dtype}")
            if args.backend == "metal":
                print(f"Metal matrix kernel: {args.metal_kernel}")
            print("\n".join(explain(root)))
            if not args.no_fuse:
                optimized = fuse_operations(root)
                if optimized is not root:
                    print("\nAfter operation fusion:")
                    print("\n".join(explain(optimized)))
            return 0
        if args.command == "run":
            if args.data is None:
                raise CompileError("--data is required for run")
            data = validate_data(inputs, json.loads(args.data.read_text()), dtype=args.dtype)
            values = evaluate(root, data, dtype=args.dtype)
            result = values if len(root.shape) == 1 else [
                values[i:i + root.shape[1]] for i in range(0, len(values), root.shape[1])]
            print(json.dumps(result, indent=2))
            return 0
        if args.data is not None:
            raise CompileError("compile no longer takes --data; pass the JSON file to the generated executable")
        if args.backend == "metal":
            if sys.platform != "darwin":
                raise CompileError("Building the Metal backend requires macOS and Apple's Command Line Tools")
            if args.out.suffix != ".m":
                raise CompileError("Metal --out must end in .m (Objective-C host source)")
            shader, host = emit_metal(inputs, root, dtype=args.dtype, fuse=not args.no_fuse,
                                      metal_kernel=args.metal_kernel)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(host)
            args.out.with_suffix(".metal").write_text(shader)
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(emit_c(inputs, root, fuse=not args.no_fuse, dtype=args.dtype))
        binary = args.out.with_suffix("")
        if binary == args.out:
            binary = args.out.with_name(args.out.name + ".bin")
        command = ["clang", "-std=c11", "-O2", str(args.out), "-o", str(binary)]
        if args.backend == "metal":
            command.extend(["-fobjc-arc", "-framework", "Foundation", "-framework", "Metal"])
        subprocess.run(command, check=True)
        print(f"Generated {'Objective-C' if args.backend == 'metal' else 'C'}: {args.out}")
        if args.backend == "metal":
            print(f"Generated MSL: {args.out.with_suffix('.metal')}")
        print(f"Native executable: {binary}")
        return 0
    except (CompileError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
