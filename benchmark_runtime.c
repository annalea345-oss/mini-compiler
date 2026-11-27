/* Timing support shared by CPU and Metal executables. Normal output is unchanged. */
#include <time.h>

#ifdef CLOCK_MONOTONIC_RAW
#define BENCHMARK_CLOCK CLOCK_MONOTONIC_RAW
#define BENCHMARK_CLOCK_NAME "monotonic_raw"
#else
#define BENCHMARK_CLOCK CLOCK_MONOTONIC
#define BENCHMARK_CLOCK_NAME "monotonic"
#endif

typedef struct {
  int benchmark;
  size_t iterations, warmups;
  double *wall, *gpu;
} RunOptions;

double monotonic_ms(void) {
  struct timespec time;
  if (clock_gettime(BENCHMARK_CLOCK, &time)) fail("benchmark", "clock read failed");
  return time.tv_sec * 1000.0 + time.tv_nsec / 1000000.0;
}

size_t iteration_count(const char *text, int allow_zero) {
  size_t value = 0;
  if (!*text) fail("benchmark", "expected an iteration count");
  for (const char *p = text; *p; ++p) {
    if (*p < '0' || *p > '9' || value > 100000)
      fail("benchmark", "iteration counts must be integers from 0 to 1000000");
    value = value * 10 + (size_t)(*p - '0');
  }
  if (value > 1000000 || (!allow_zero && !value)) fail("benchmark", "invalid iteration count");
  return value;
}

RunOptions run_options(int argc, char **argv) {
  if (argc != 2 && argc != 4) {
    fprintf(stderr, "Usage: %s INPUT.json [ITERATIONS WARMUPS]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  RunOptions options = {argc == 4, 1, 0, NULL, NULL};
  if (options.benchmark) {
    options.iterations = iteration_count(argv[2], 0);
    options.warmups = iteration_count(argv[3], 1);
    options.wall = malloc(options.iterations * sizeof(double));
    options.gpu = malloc(options.iterations * sizeof(double));
    if (!options.wall || !options.gpu) fail("benchmark", "could not allocate timing samples");
  }
  return options;
}

double result_checksum(const TensorValue *values, size_t count) {
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) sum += (double)values[i];
  if (!isfinite(sum)) fail("benchmark", "nonfinite result checksum");
  return sum;
}

void json_string(const char *text) {
  putchar('"');
  for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
    if (*p == '"' || *p == '\\') printf("\\%c", *p);
    else if (*p < 32) printf("\\u%04x", (unsigned int)*p);
    else putchar(*p);
  }
  putchar('"');
}

void benchmark_report(RunOptions options, double setup_ms, double load_ms,
                      double checksum, const char *device, int has_gpu_timing) {
  printf("{\"iterations\":%zu,\"warmups\":%zu,\"setup_ms\":%.9g,\"load_ms\":%.9g,"
         "\"checksum\":%.17g,\"device\":", options.iterations, options.warmups, setup_ms, load_ms, checksum);
  json_string(device);
  struct timespec resolution;
  if (clock_getres(BENCHMARK_CLOCK, &resolution)) fail("benchmark", "clock resolution unavailable");
  printf(",\"clock\":\"%s\",\"clock_resolution_ns\":%.9g,\"wall_ms\":[",
         BENCHMARK_CLOCK_NAME, resolution.tv_sec * 1000000000.0 + resolution.tv_nsec);
  for (size_t i = 0; i < options.iterations; ++i) printf("%s%.9g", i ? "," : "", options.wall[i]);
  printf("],\"gpu_command_ms\":");
  if (has_gpu_timing) {
    putchar('[');
    for (size_t i = 0; i < options.iterations; ++i) printf("%s%.9g", i ? "," : "", options.gpu[i]);
    putchar(']');
  } else printf("null");
  puts("}");
  free(options.wall);
  free(options.gpu);
}
