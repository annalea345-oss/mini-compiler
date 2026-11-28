"""Apple library comparison programs for relu(x @ weights + bias), not DSL backends."""
from pathlib import Path

from tensor_compiler import CompileError


def validate_shape(shape):
    if len(shape) != 3 or any(not isinstance(d, int) or isinstance(d, bool) or d < 1 or d > 2**31 - 1 for d in shape):
        raise CompileError("Shape must contain three positive 32-bit dimensions: MxKxN")
    m, k, n = shape
    if max(m * k, k * n, m * n) > 2**32 - 1:
        raise CompileError("Benchmark tensors must contain at most 2^32 - 1 elements")
    return m, k, n


def emit_library_baseline(shape, library):
    m, k, n = validate_shape(shape)
    if library not in ("accelerate", "mps"):
        raise CompileError("Library baseline must be accelerate or mps")
    here = Path(__file__).parent
    pieces = ["#define _DARWIN_C_SOURCE 1", "#define _POSIX_C_SOURCE 200809L",
              f"#define ROWS {m}", f"#define INNER {k}", f"#define COLUMNS {n}",
              "typedef float TensorValue;", "#define TENSOR_PARSE strtof"]
    pieces.extend((here / name).read_text() for name in ("tensor_runtime.c", "benchmark_runtime.c"))
    pieces.append('''
void load_layer(const char *path, float *x, float *weights, float *bias) {
  TensorInput inputs[] = {
    {"x", x, 2, {ROWS, INNER}, 0},
    {"weights", weights, 2, {INNER, COLUMNS}, 0},
    {"bias", bias, 1, {COLUMNS}, 0}
  };
  char name[8];
  load_inputs(path, inputs, 3, name, sizeof(name));
}
''')
    if library == "mps":
        pieces.append((here / "metal_runtime.m").read_text())
    pieces.append((here / ("mps_baseline.m" if library == "mps" else "accelerate_baseline.c")).read_text())
    return "\n".join(pieces)
