/* Full layer baseline: MPS SGEMM followed by one custom GPU bias/ReLU pass. */
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdint.h>

MPSMatrix *baseline_matrix(id<MTLDevice> device, size_t rows, size_t columns) {
  size_t stride = [MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];
  if (!stride || rows > (size_t)-1 / stride) fail("MPS", "matrix size overflow");
  if (rows * stride / sizeof(float) > UINT32_MAX)
    fail("MPS", "padded matrix exceeds the postprocess kernel's index range");
  id<MTLBuffer> buffer = [device newBufferWithLength:rows * stride options:MTLResourceStorageModeShared];
  if (!buffer) fail("MPS", "could not allocate matrix buffer");
  memset(buffer.contents, 0, buffer.length);
  MPSMatrixDescriptor *descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns
                                                                      rowBytes:stride dataType:MPSDataTypeFloat32];
  MPSMatrix *matrix = [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
  if (!matrix) fail("MPS", "could not create matrix");
  return matrix;
}

void pack_matrix(MPSMatrix *matrix, const float *values) {
  for (size_t row = 0; row < matrix.rows; ++row)
    memcpy((char *)matrix.data.contents + row * matrix.rowBytes,
           values + row * matrix.columns, matrix.columns * sizeof(float));
}

int main(int argc, char **argv) {
  @autoreleasepool {
    RunOptions options = run_options(argc, argv);
    double setup_start = monotonic_ms();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || !MPSSupportsMTLDevice(device)) fail("MPS", "no supported GPU device available");
    float *x = tensor_alloc((size_t)ROWS * INNER);
    float *weights = tensor_alloc((size_t)INNER * COLUMNS);
    float *bias = tensor_alloc(COLUMNS);
    float *out = tensor_alloc((size_t)ROWS * COLUMNS);
    double load_start = monotonic_ms();
    load_layer(argv[1], x, weights, bias);
    double load_ms = monotonic_ms() - load_start;
    MPSMatrix *left = baseline_matrix(device, ROWS, INNER);
    MPSMatrix *right = baseline_matrix(device, INNER, COLUMNS);
    MPSMatrix *result = baseline_matrix(device, ROWS, COLUMNS);
    pack_matrix(left, x); pack_matrix(right, weights);
    __attribute__((objc_precise_lifetime)) id<MTLBuffer> bias_buffer = metal_buffer(device, COLUMNS);
    memcpy(bias_buffer.contents, bias, COLUMNS * sizeof(float));
    MPSMatrixMultiplication *multiply = [[MPSMatrixMultiplication alloc] initWithDevice:device
      transposeLeft:NO transposeRight:NO resultRows:ROWS resultColumns:COLUMNS interiorColumns:INNER alpha:1.0 beta:0.0];
    if (!multiply) fail("MPS", "could not create matrix multiplication");
    NSString *source = [NSString stringWithFormat:
      @"#include <metal_stdlib>\nusing namespace metal;\n"
       "kernel void bias_relu(device float *out [[buffer(0)]], device const float *bias [[buffer(1)]],"
       "uint i [[thread_position_in_grid]]) { if (i >= %uu) return; uint column = i %% %uu;"
       "uint index = (i / %uu) * %zuu + column; float value = out[index] + bias[column];"
       "out[index] = value > 0.0f ? value : 0.0f; }",
       (unsigned int)((size_t)ROWS * COLUMNS), (unsigned int)COLUMNS,
       (unsigned int)COLUMNS, result.rowBytes / sizeof(float)];
    id<MTLLibrary> library = metal_library(device, source);
    id<MTLComputePipelineState> postprocess = metal_pipeline(device, library, @"bias_relu");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) fail("MPS", "could not create command queue");
    double setup_ms = monotonic_ms() - setup_start - load_ms;
    volatile double checksum = 0.0;
    for (size_t step = 0; step < options.warmups + options.iterations; ++step) {
      @autoreleasepool {
        double start = monotonic_ms();
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (!command) fail("MPS", "could not create command buffer");
        [multiply encodeToCommandBuffer:command leftMatrix:left rightMatrix:right resultMatrix:result];
        metal_encode(command, postprocess, @[result.data, bias_buffer], ROWS, COLUMNS, NO);
        metal_finish(command);
        double elapsed = monotonic_ms() - start;
        // Row unpacking and checksums are outside the timed calculation.
        for (size_t row = 0; row < ROWS; ++row)
          memcpy(out + row * COLUMNS, (char *)result.data.contents + row * result.rowBytes, COLUMNS * sizeof(float));
        if (options.benchmark) {
          checksum = result_checksum(out, (size_t)ROWS * COLUMNS);
          if (step >= options.warmups) {
            options.wall[step - options.warmups] = elapsed;
            options.gpu[step - options.warmups] = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
          }
        }
      }
    }
    if (options.benchmark) benchmark_report(options, setup_ms, load_ms, checksum, device.name.UTF8String, 1);
    else for (size_t i = 0; i < (size_t)ROWS * COLUMNS; ++i) printf("%.9g\n", (double)out[i]);
    free(x); free(weights); free(bias); free(out);
  }
  return 0;
}
