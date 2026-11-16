"""A small tensor compiler: source -> typed graph -> portable C -> native code."""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
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


def validate_data(inputs, data):
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
        return [number]

    for name in inputs:
        if name not in data:
            raise CompileError(f"Missing input {name!r}")
    if set(data) - set(inputs):
        raise CompileError(f"Unknown inputs: {', '.join(sorted(set(data) - set(inputs)))}")
    return {name: check(data[name], node.shape, name) for name, node in inputs.items()}


def evaluate(node, data):
    """Simple, deliberately unoptimized CPU reference implementation."""
    if node.op == "input":
        return data[node.name]
    left = evaluate(node.args[0], data)
    if node.op == "relu":
        return [max(0.0, value) for value in left]
    right = evaluate(node.args[1], data)
    if node.op == "add":
        return [value + right[i % len(right)] for i, value in enumerate(left)]
    rows, columns = node.shape
    inner = node.args[0].shape[1]
    return [sum(left[i * inner + k] * right[k * columns + j] for k in range(inner))
            for i in range(rows) for j in range(columns)]


def emit_c(inputs, root, data):
    """Emit one loop per operation. Input values are embedded for this first version."""
    lines = ["#include <stdio.h>", "int main(void) {"]
    names = {}

    def emit(node):
        key = id(node)
        if key in names:
            return names[key]
        children = [emit(child) for child in node.args]
        name = f"t{len(names)}"
        names[key] = name
        size = math.prod(node.shape)
        if node.op == "input":
            values = ", ".join(repr(value) for value in data[node.name])
            lines.append(f"  double {name}[{size}] = {{{values}}}; /* {node.name} */")
        else:
            lines.append(f"  double {name}[{size}]; /* {node.op}: {node.shape} */")
            if node.op == "matmul":
                rows, columns = node.shape
                inner = node.args[0].shape[1]
                lines.extend([
                    f"  for (int i = 0; i < {rows}; ++i)",
                    f"    for (int j = 0; j < {columns}; ++j) {{",
                    "      double sum = 0.0;",
                    f"      for (int k = 0; k < {inner}; ++k)",
                    f"        sum += {children[0]}[i*{inner}+k] * {children[1]}[k*{columns}+j];",
                    f"      {name}[i*{columns}+j] = sum;", "    }"])
            else:
                if node.op == "add":
                    expression = f"{children[0]}[i] + {children[1]}[i % {math.prod(node.args[1].shape)}]"
                else:
                    expression = f"{children[0]}[i] > 0.0 ? {children[0]}[i] : 0.0"
                lines.append(f"  for (int i = 0; i < {size}; ++i) {name}[i] = {expression};")
        return name

    result = emit(root)
    lines.extend([f'  for (int i = 0; i < {math.prod(root.shape)}; ++i) printf("%.17g\\n", {result}[i]);',
                  "  return 0;", "}"])
    return "\n".join(lines) + "\n"


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
    cli.add_argument("--data", type=Path, help="JSON input arrays (required for run/compile)")
    cli.add_argument("--out", type=Path, default=Path("build/program.c"))
    args = cli.parse_args()
    try:
        inputs, root = parse(args.source.read_text())
        if args.command == "check":
            print("Valid program. Typed calculation tree:")
            print("\n".join(explain(root)))
            return 0
        if args.data is None:
            raise CompileError("--data is required for run and compile")
        data = validate_data(inputs, json.loads(args.data.read_text()))
        if args.command == "run":
            values = evaluate(root, data)
            result = values if len(root.shape) == 1 else [
                values[i:i + root.shape[1]] for i in range(0, len(values), root.shape[1])]
            print(json.dumps(result, indent=2))
            return 0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(emit_c(inputs, root, data))
        binary = args.out.with_suffix("")
        if binary == args.out:
            binary = args.out.with_name(args.out.name + ".bin")
        subprocess.run(["clang", "-std=c11", "-O2", str(args.out), "-o", str(binary)], check=True)
        print(f"Generated C: {args.out}")
        print(f"Native executable: {binary}")
        return 0
    except (CompileError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
