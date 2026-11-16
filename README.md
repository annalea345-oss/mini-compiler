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
python3 tensor_compiler.py compile examples/layer.tensor --data examples/layer.json
./build/program
```

`check` displays the calculation tree and the size of every result. `run` prints
the interpreter's result as a JSON array. `compile` writes readable C to
`build/program.c` and a native executable to `build/program`. The executable prints
one number per line in row order.

The example produces:

```text
[[0, 3.5],
 [0, 0.5],
 [3, 0],
 [0, 0]]
```

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
are fixed at compilation time. Numbers use double precision.

## How the compiler works

1. **Tokenize:** split text into names, numbers, and punctuation.
2. **Parse:** arrange those pieces into a calculation tree.
3. **Type check:** verify the shapes while building that tree.
4. **Generate C:** turn operations into arrays and loops.
5. **Build:** ask Clang to turn the C into a native program.

`tensor_compiler.py` contains the whole pipeline. The tree of `Node` objects is
our first intermediate representation: a structured form between source text
and generated code.

This first version embeds the supplied input numbers in the executable. To use
different values, compile again. It generates a separate array and loop for each
operation; it has no custom optimization passes or GPU backend yet. Clang can
optimize the generated C, including calculations involving embedded constants,
so this version is for correctness and learning, not meaningful performance comparisons.

## Verify it

```sh
python3 -m unittest -v
```

Tests cover a known result, invalid programs and data, and native-vs-interpreter
agreement on deterministic random inputs.

## Next milestones

1. Read input data at runtime so one executable can process different values.
2. Add a compiler pass that combines matrix multiplication, bias, and ReLU to
   avoid intermediate arrays. Compare answers with the unoptimized version.
3. Benchmark larger inputs with runtime data.
4. Add a Metal backend to run the same language on the Apple GPU.
