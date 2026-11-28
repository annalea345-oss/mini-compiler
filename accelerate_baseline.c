/* Full layer baseline: single-threaded Accelerate SGEMM plus a bias/ReLU pass. */
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

__attribute__((noinline)) void accelerate_layer(float *x, float *weights, float *bias, float *out) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              ROWS, COLUMNS, INNER, 1.0f, x, INNER, weights, COLUMNS, 0.0f, out, COLUMNS);
  for (size_t i = 0; i < (size_t)ROWS * COLUMNS; ++i) {
    float value = out[i] + bias[i % COLUMNS];
    out[i] = value > 0.0f ? value : 0.0f;
  }
}

int main(int argc, char **argv) {
  RunOptions options = run_options(argc, argv);
  double setup_start = monotonic_ms();
  if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) != 0)
    fail("Accelerate", "could not request single-threaded BLAS");
  float *x = tensor_alloc((size_t)ROWS * INNER);
  float *weights = tensor_alloc((size_t)INNER * COLUMNS);
  float *bias = tensor_alloc(COLUMNS);
  float *out = tensor_alloc((size_t)ROWS * COLUMNS);
  double load_start = monotonic_ms();
  load_layer(argv[1], x, weights, bias);
  double load_ms = monotonic_ms() - load_start;
  double setup_ms = monotonic_ms() - setup_start - load_ms;
  volatile double checksum = 0.0;
  for (size_t step = 0; step < options.warmups + options.iterations; ++step) {
    double start = monotonic_ms();
    accelerate_layer(x, weights, bias, out);
    double elapsed = monotonic_ms() - start;
    if (options.benchmark) {
      checksum = result_checksum(out, (size_t)ROWS * COLUMNS);
      if (step >= options.warmups) options.wall[step - options.warmups] = elapsed;
    }
  }
  if (options.benchmark)
    benchmark_report(options, setup_ms, load_ms, checksum, "Accelerate CPU (single-threaded)", 0);
  else for (size_t i = 0; i < (size_t)ROWS * COLUMNS; ++i) printf("%.9g\n", (double)out[i]);
  free(x); free(weights); free(bias); free(out);
  return 0;
}
