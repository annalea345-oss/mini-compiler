# Mini tensor compiler

A tiny language for calculations used in neural networks. It checks your program,
executes it with a simple reference interpreter, or compiles it for CPU or GPU
execution. On an Apple Silicon Mac, the C backend runs on the ARM CPU. The Metal
backend generates MSL kernels and an Objective-C host for the Apple GPU.

## Try it

Requires Python 3.9+ and Clang (provided by Apple's Command Line Tools). No Python
packages are required. Run these commands from this folder:

```sh
python3 tensor_compiler.py check examples/layer.tensor
python3 tensor_compiler.py run examples/layer.tensor --data examples/layer.json
python3 tensor_compiler.py compile examples/layer.tensor
./build/program examples/layer.json
./build/program examples/layer-alt.json
```

`check` displays the calculation tree and the size of every result, followed by
the optimized tree when fusion applies. `run` prints
the interpreter's result as a JSON array. `compile` writes readable C to
`build/program.c` and a native executable to `build/program`. The executable prints
one number per line in row order. Compilation needs only the calculation file;
pass a JSON data file to the executable each time you run it. The executable is
standalone and does not need Python or the original calculation file to run.

The interpreter produces the following table for `examples/layer.json` (the native
executable prints the same eight values, one per line):

```text
[[0, 3.5],
 [0, 0.5],
 [3, 0],
 [0, 0]]
```

The alternate file produces four rows of `[1, 2]`, using the same executable.
Change input values freely; recompile when changing the calculation or shapes.
If you used the old `compile --data ...` command, move the data path to the
executable command as shown above.

## The language

```text
input x: matrix[4, 3]
input weights: matrix[3, 2]
input bias: vector[2]

output = relu(x @ weights + bias)
```

- `matrix[rows, columns]` declares a rectangular table of numbers.
- `vector[length]` declares a single list of numbers.
- `@` multiplies two matrices. Left columns must match right rows.
- `+` adds equal-sized arrays, or adds a vector to every row of a matrix.
  For the latter, the matrix goes on the left and the vector length must match its columns.
- `relu(...)` replaces negative values with zero.
- Parentheses group calculations; `@` takes precedence over `+`.
- `#` starts a comment. Declare inputs first, followed by exactly one output expression.

Values live in a separate JSON file with keys matching the input names. Shapes
are fixed at compilation time. Numbers use double precision by default; use
`--dtype float32` for single precision. Every declared input
must be supplied exactly once, including unused inputs; key order does not matter.
The executable rejects missing, unknown, or duplicate inputs, incorrect array
shapes, malformed JSON, and values that are not finite numbers. Errors go to
standard error with a nonzero exit status.

## Numeric precision

Choose one numeric type for the whole program; no DSL or JSON changes are needed:

```sh
python3 tensor_compiler.py check examples/layer.tensor --dtype float32
python3 tensor_compiler.py run examples/layer.tensor --data examples/layer.json --dtype float32
python3 tensor_compiler.py compile examples/layer.tensor --dtype float32 --out build/layer32.c
./build/layer32 examples/layer.json
```

- `float64` (default) uses C `double`, with 8 bytes per tensor element.
- `float32` uses 4 bytes per element and rounds inputs and arithmetic to single
  precision. Both CPU and Metal backends support it; CPU is the default backend.

The numeric type is fixed when compiling. Recompile to change it. Both modes
support fusion and `--no-fuse`. The reference interpreter rounds each product and
addition to the selected precision. Generated C disables multiply-add contraction
so multiplication and addition round separately, matching the reference order.
This is separate from operation fusion, which removes intermediate arrays.

Float32 can give slightly different answers: `16777217` rounds to `16777216`,
for example. Inputs that overflow the selected type are rejected; tiny inputs
may round to subnormal values or zero. Arithmetic overflow during calculation is
not trapped and may produce infinity or NaN. Native output uses 9 significant
digits for float32 and 17 for float64.

## How the compiler works

1. **Tokenize:** split text into names, numbers, and punctuation.
2. **Parse:** arrange those pieces into a calculation tree.
3. **Type check:** verify the shapes while building that tree.
4. **Optimize:** combine eligible matrix multiplication, row bias, and ReLU operations.
5. **Generate C:** turn operations into arrays and loops, with a JSON input reader.
6. **Build:** ask Clang to turn the C into a native program.

`tensor_compiler.py` contains the whole pipeline. The tree of `Node` objects is
our first intermediate representation: a structured form between source text
and generated code.

`tensor_runtime.c` supplies the small input reader copied into each generated C
file. The executable loads input values at runtime, then runs the generated
calculation. The fusion pass described below combines eligible operations;
other operations each generate a separate array and loop. CPU tensors use checked
heap allocations and are freed after execution, so their size is no longer
limited by stack space. Allocation overflow and failure produce errors. Metal
uses shared GPU buffers. The benchmark harness described below measures
calculation time separately from JSON loading, setup, and output.

## Operation fusion

Compilation enables fusion by default. The compiler recognizes
`relu(matrix @ matrix + vector)` and replaces those three operations with one
`matmul_bias_relu` node. For each output element, the generated C computes the
dot product, adds the column's bias, applies ReLU, and writes the final value.
It keeps the original calculation order and eliminates the two intermediate
arrays that would hold the multiplication and addition results.

The pass also finds this pattern inside larger expressions. Matrix-shaped
additions and partial patterns such as `relu(x @ weights)` use the original
code generation. The source graph is preserved, and `run` always uses the
unoptimized reference interpreter.

To inspect or build the original version:

```sh
python3 tensor_compiler.py check examples/layer.tensor --no-fuse
python3 tensor_compiler.py compile examples/layer.tensor --no-fuse --out build/unfused.c
./build/unfused examples/layer.json
```

For the example's 4-by-2 output, fusion reduces generated result storage from
three arrays (24 doubles) to one array (8 doubles), excluding input storage.
Clang may also optimize either version; a runtime speedup has not been measured.

## Metal GPU backend

The GPU backend supports **float32 matrix multiplication, addition, row bias
broadcasting, and ReLU**, including composed expressions and direct input output.
It also fuses `relu(x @ weights + bias)` into one kernel by default. Unsupported
GPU operations produce a compile error rather than falling back to the CPU.

Requires macOS, Apple's Command Line Tools, and an accessible Metal-capable GPU.
The native integration tests were run on an Apple M3. For example:

```sh
python3 tensor_compiler.py check examples/activation.tensor --backend metal
python3 tensor_compiler.py compile examples/activation.tensor --backend metal --out build/activation.m
./build/activation examples/activation.json
```

The result is `[0, 0, 0.5, 0, 3.5, 3]`, printed one value per line.
To run the original matrix multiplication example on the GPU:

```sh
python3 tensor_compiler.py check examples/layer.tensor --backend metal
python3 tensor_compiler.py compile examples/layer.tensor --backend metal --out build/layer_gpu.m
./build/layer_gpu examples/layer.json
./build/layer_gpu examples/layer-alt.json
```

The outputs match the original example: `[0, 3.5, 0, 0.5, 3, 0, 0, 0]` and
`[1, 2, 1, 2, 1, 2, 1, 2]`, respectively. For an unfused GPU version:

```sh
python3 tensor_compiler.py compile examples/layer.tensor --backend metal --no-fuse --out build/layer_gpu_unfused.m
./build/layer_gpu_unfused examples/layer.json
```

`--backend metal` defaults to float32; explicit float64 is rejected. The CPU
backend continues to default to float64. To compare with the CPU interpreter:

```sh
python3 tensor_compiler.py run examples/activation.tensor --data examples/activation.json --dtype float32
```

Compilation generates three artifacts: an Objective-C `.m` host, readable `.metal`
kernel source, and an executable. The host embeds the same MSL text and compiles
it through Metal when the executable runs, so neither Xcode's standalone `metal`
compiler nor the generated source files are needed at execution time.

The host loads validated JSON directly into shared GPU buffers, builds a compute
pipeline per operation, encodes the operations in dependency order, waits for
completion, and prints the result. Elementwise and untiled matrix kernels handle
one output element per thread, with a bounds check for the final threadgroup.
The tiled matrix kernel handles four neighboring outputs per thread. For matrix
multiplication, each output is its row/column dot product. The fused kernel adds
the column bias after the dot product, applies ReLU, and writes only the final
result. It removes two intermediate buffers and reduces three dispatches to one
for that pattern, including when nested inside larger expressions.

Metal uses safe math settings and disables multiply-add contraction to keep
products and additions separately rounded, as in the CPU reference. All declared
inputs are still required. `check` shows the fused graph; `--no-fuse` disables the
pass for both inspection and compilation.

The original untiled matrix kernel is retained as the default baseline. Select
the new tiled implementation explicitly:

```sh
python3 tensor_compiler.py compile examples/layer.tensor --backend metal --metal-kernel tiled --out build/layer_tiled.m
./build/layer_tiled examples/layer.json
```

Use `--metal-kernel untiled` to force the baseline. Both options support fusion
and `--no-fuse`; addition and ReLU keep their elementwise kernels. The tiled
kernel launches 16-by-16 threadgroups, with each thread computing a 2-by-2 output
block. Threads cooperatively load two 32-by-32 input blocks into threadgroup
memory, synchronize, and reuse those blocks and register values while computing
their output dot products. Out-of-bounds loads are filled with zero,
but every thread still participates in the barriers. The final partial inner
block processes only real elements, preserving reduction order and rounding.
The tiled pipeline requires support for 256 threads per group and reports an
error if that limit is unavailable.

Shader compilation, pipeline creation,
and JSON loading occur once per process; pipeline states and buffers are reused
when benchmarking repeated calculations. GPU launch overhead occurs on every
iteration.
Processes without GPU access report an error. In particular, some sandboxes or
headless CI environments cannot access a Metal device.

## Performance benchmarks

Run the four float32 variants (CPU/GPU, fused/unfused) with the same deterministic
inputs and full-result correctness checks against the interpreter:

```sh
python3 benchmark.py --sizes 16 64 128 256 --iterations 20 --warmups 5
```

For CPU-only environments, add `--backends cpu`. GPU access is required when
Metal is requested; the harness reports an error rather than silently dropping
it. Default sizes are 16, 64, and 128. Use `--shapes MxKxN` for rectangular
calculations, where `x` is M-by-K, weights are K-by-N, and output is M-by-N.
Supplying shapes replaces default sizes; explicit `--sizes` and `--shapes` can
be combined, with duplicate shapes removed. Generated programs and
inputs are kept in `build/benchmarks/`. The default JSON report is
`build/benchmarks/report.json`; choose another path with `--out`.

To compare tiled and untiled GPU kernels in the same run:

```sh
python3 benchmark.py --backends metal --metal-kernels untiled tiled --sizes 16 64 128 256 --iterations 30 --warmups 5 --out build/benchmarks/tiling.json
```

This tests both fused and unfused versions of each selected kernel. Include
`cpu` in `--backends` to add CPU variants. The default benchmark kernel remains
`untiled`; each report row records the selected implementation.

A second M3 run comparing the two kernels, with 30 measured iterations and 5
warmups, produced these fused wall medians (milliseconds):

| Square size | Untiled | Tiled |
| --- | ---: | ---: |
| 16 | 0.392 | 0.363 |
| 64 | 0.196 | 0.381 |
| 128 | 0.376 | 0.382 |
| 256 | 0.542 | 0.467 |

At 256, the tiled fused variant reduced wall time by about 14%; its median GPU
command time fell from 0.258 to 0.112 ms. Smaller sizes were mixed, and launch
latency and system variation affect the wall samples. The untiled kernel remains
the default rather than assuming tiling always wins. Unfused variants, raw
samples, and verification results are in
[benchmarks/apple-m3-tiling.json](benchmarks/apple-m3-tiling.json). This is a
single-machine observation, not a general performance guarantee or an automatic
kernel-selection rule.

### Rectangular workloads and Apple library comparisons

Compare the generated CPU and GPU programs with Apple library implementations:

```sh
python3 benchmark.py --backends cpu metal accelerate mps --metal-kernels untiled tiled --shapes 1x256x128 32x128x256 128x64x256 33x65x97 --iterations 20 --warmups 5 --out build/benchmarks/apple-libraries.json
```

These synthetic workloads cover a single-input layer, batched layers with
different input/output widths, and irregular dimensions. They perform exactly
`relu(x @ weights + bias)` using the same runtime inputs. The comparison variants
are selected through `benchmark.py`; compilation of the DSL continues to use
the CPU or Metal backend.

- **Accelerate:** Apple's [single-precision SGEMM](https://developer.apple.com/documentation/accelerate/blas-library)
  followed by a CPU bias/ReLU pass. The baseline explicitly requests
  single-threaded BLAS, matching the generated CPU backend's thread count.
  This baseline requires macOS 15+ and a corresponding SDK.
- **MPS:** Apple's [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
  followed by a custom GPU bias/ReLU kernel, encoded into the same command
  buffer. Matrix descriptors use float32 and MPS-recommended padded row strides.
  This uses a matrix multiplication and a postprocessing pass, rather than the
  generated fused kernel's single dispatch.

All variants include bias and ReLU in calculation timing. Library variants
appear once per shape, with `fused: null` and an explicit implementation label.
Requested CPU threading, M/K/N, maximum observed absolute error, and verification
status are saved with each row. Library reduction orders can differ, so every
full output is compared with the float32 reference using relative and absolute
tolerances of `2e-5` before timing; iteration checksums are also checked.

Library objects, buffers, and pipeline setup are reused. MPS input row packing
is part of setup; output row unpacking is outside the timed calculation, as is
the checksum scan for every variant. Setup and JSON loading remain separate
metrics. Library initialization on the first calculation is handled by the
discarded warmups. These measurements compare buffer-resident calculation paths,
not end-to-end application latency or hardware limits.

An M3 run with 20 measured iterations and 5 warmups produced these median wall
times in milliseconds. Generated CPU and Metal columns use their fused variants;
both library columns include their postprocessing passes:

| M x K x N | CPU | Metal untiled | Metal tiled | Accelerate | MPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 x 256 x 128 | 0.0237 | 0.390 | 0.404 | 0.00075 | 0.362 |
| 32 x 128 x 256 | 0.564 | 0.371 | 0.386 | 0.00596 | 0.352 |
| 128 x 64 x 256 | 0.819 | 0.195 | 0.370 | 0.01925 | 0.367 |
| 33 x 65 x 97 | 0.110 | 0.360 | 0.389 | 0.00542 | 0.367 |

Accelerate was fastest on these four workloads, showing the gap between the
simple generated CPU loops and an optimized library. GPU wall times include
launch and completion overhead, which is significant at these sizes. This does
not establish a universal backend-selection threshold. Full fused/unfused
results, all output checks, measured errors, and raw timing samples are in
[benchmarks/apple-m3-rectangular-libraries.json](benchmarks/apple-m3-rectangular-libraries.json).

Each executable also supports benchmarks directly. For example:

```sh
python3 tensor_compiler.py compile examples/layer.tensor --dtype float32
./build/program examples/layer.json 20 5
```

The two extra arguments select measured iterations and discarded warmups. This
mode prints JSON timing samples and a result checksum instead of the array.
Without those arguments, normal output is unchanged. Iterations must be
1..1000000 and warmups 0..1000000.

Timing definitions:

- **Wall calculation time:** CPU computation, or Metal command encoding,
  submission, and waiting for completion. This is the primary CPU/GPU comparison.
- **GPU command time:** Metal's command-buffer end timestamp minus start timestamp,
  read after completion. It includes all GPU passes and excludes CPU submission
  overhead. It is additional context, not interchangeable with wall time.
- **Setup and loading:** Allocation/device/shader/pipeline setup and JSON loading
  are reported separately. Native host compilation is timed by the harness.

All allocations and pipelines are outside the measured loop. Result checksums
are consumed after each timed iteration to keep the CPU computation live; this
scan, warmups, and printing are excluded from samples. CPU computation is a
non-inlined function using the preallocated buffers. Timings use a monotonic
clock (raw when available). Very short calculations can approach clock resolution.

Reports retain raw samples, medians, means, ranges, verification status, device,
compiler/OS, seed, source revision, and source hashes. Variants use a seeded
shuffled order per size. Measurements are local observations subject to system
load, clock scaling, and caching. The CPU backend is single-threaded generated
C, not an optimized BLAS implementation; these are comparisons of this project's
backends, not hardware performance limits.

An Apple M3 run on macOS 15.7.2, using 20 measured iterations after 5 warmups,
produced these median wall calculation times (milliseconds):

| Square size | CPU unfused | CPU fused | GPU unfused | GPU fused |
| --- | ---: | ---: | ---: | ---: |
| 16 | 0.00029 | 0.00021 | 0.345 | 0.179 |
| 64 | 0.103 | 0.105 | 0.357 | 0.198 |
| 128 | 1.051 | 1.043 | 0.397 | 0.361 |
| 256 | 11.062 | 11.122 | 0.633 | 0.591 |

For this run, GPU launch overhead outweighed its advantage at small sizes; the
GPU was faster at 128 and 256. Fusion reduced GPU wall time in these samples,
but did not consistently improve CPU wall time. Full samples, setup/loading
times, GPU command times, source hashes, and verification results are saved in
[benchmarks/apple-m3.json](benchmarks/apple-m3.json). These are one machine's
observations, not guarantees for other shapes or workloads.

## Verify it

```sh
python3 -m unittest -v
```

Tests cover a known result, invalid programs, runtime JSON validation, the compile
CLI, and native-vs-interpreter agreement on deterministic random inputs. Native
tests reuse each executable with multiple datasets and require Clang. Both fused
and unfused code are compared with the interpreter, including nested fusion,
computed biases, single-element shapes, ReLU boundaries, and cancellation that
would expose an incorrectly reordered bias addition. Tests exercise both numeric
types, float32 overflow and underflow, and precision-sensitive reductions. Random
float32 results are also compared with float64 using relative and absolute
tolerances of `2e-5` for those datasets; this is not a universal error bound.

Metal tests check unsupported operations and types on all platforms. On macOS
with Clang and an accessible Metal GPU, they also compile the host with warnings
as errors, run generated kernels, and compare GPU results with float32 CPU and
interpreter results. They cover fused and unfused matrix calculations, rectangular
and single-element matrices, nested layers, computed biases, broadcasting,
identity output, zero inputs, partial threadgroups, rounding-sensitive reductions,
and input errors. Both tiled and untiled kernels are exercised, including dimensions
below, equal to, and above the tile size and incomplete blocks in all dimensions.
Random comparisons use relative and absolute tolerances of
`2e-5`; dedicated rounding and ReLU boundary cases require exact results. GPU execution tests are
explicitly skipped when a Metal device is unavailable.

Allocation/benchmark tests include a 300,000-element CPU tensor under a 256 KB
stack limit, oversized allocation rejection, iteration validation, CPU/GPU
benchmark reports, sample counts, checksums, and the command-line harness.
Apple library tests cover known answers, random and zero data, rectangular
matrices with padded rows, repeated execution without accumulation, complete
bias/ReLU behavior, input errors, and comparison report metadata. Apple-only
tests are skipped on unsupported platforms; MPS tests also require GPU access.

## Next milestones

1. Use rectangular and library comparisons to guide tile sizes, kernel selection,
   memory access patterns, and CPU lowering improvements.
2. Measure broader workloads and multiple runs before introducing automatic backend selection.
