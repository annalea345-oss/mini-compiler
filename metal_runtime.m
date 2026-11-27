/* Host helpers copied into generated Objective-C programs. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

id<MTLBuffer> metal_buffer(id<MTLDevice> device, size_t count) {
  if (count > (size_t)-1 / sizeof(float)) fail("Metal", "buffer size overflow");
  id<MTLBuffer> buffer = [device newBufferWithLength:count * sizeof(float)
                                           options:MTLResourceStorageModeShared];
  if (!buffer) fail("Metal", "could not allocate tensor buffer");
  return buffer;
}

id<MTLLibrary> metal_library(id<MTLDevice> device, NSString *source) {
  MTLCompileOptions *options = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeSafe;
  } else {
    // Retain safe arithmetic on older macOS versions as well.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
  if (!library) fail("Metal shader compilation", error.localizedDescription.UTF8String ?: "failed");
  return library;
}

id<MTLComputePipelineState> metal_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function) fail("Metal", "kernel function not found");
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) fail("Metal pipeline", error.localizedDescription.UTF8String ?: "failed");
  return pipeline;
}

void metal_encode(id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline,
                  NSArray<id<MTLBuffer>> *buffers, NSUInteger count) {
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (!encoder) fail("Metal", "could not create compute encoder");
  [encoder setComputePipelineState:pipeline];
  for (NSUInteger i = 0; i < buffers.count; ++i)
    [encoder setBuffer:buffers[i] offset:0 atIndex:i];
  NSUInteger width = MIN(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup);
  // Round up for devices without nonuniform threadgroup support. Kernels guard i < count.
  [encoder dispatchThreadgroups:MTLSizeMake((count + width - 1) / width, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder endEncoding];
}

void metal_finish(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    fail("Metal execution", command.error.localizedDescription.UTF8String ?: "failed");
}
