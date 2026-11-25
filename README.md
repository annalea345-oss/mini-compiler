# Mini tensor compiler

A tiny language for calculations used in neural networks. It checks your program,
executes it with a simple reference interpreter, or translates it into C and uses
Clang to build a native executable. On an Apple Silicon Mac, that executable runs
on the ARM CPU.

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
- `float32` uses C `float`, with 4 bytes per element, and rounds inputs and
  arithmetic to single precision. It prepares the CPU backend for comparison
  with a future GPU backend; execution still runs on the CPU.

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
other operations each generate a separate array and loop. There is no GPU backend
yet. Arrays currently use stack
storage, so large tensors may exceed the available stack. Performance benchmarks
are still a later milestone and should measure calculation time separately from
JSON loading and output.

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

## Next milestones

1. Generate a simple Metal Shading Language kernel and add a Metal host runtime.
2. Support the full matrix multiplication, bias, and ReLU calculation on the Apple GPU.
3. Improve memory allocation for large tensors and benchmark CPU/GPU execution,
   checking numerical agreement with appropriate tolerances.
