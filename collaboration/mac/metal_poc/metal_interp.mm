// 2026-09-01 22:24 CST (mac): Add an isolated Objective-C++/Metal proof of
// concept for FEW's cubic amplitude interpolation. This file is not linked by
// the production build and does not change the CPU or CUDA backends.

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

struct MetalContext {
  __strong id<MTLDevice> device;
  __strong id<MTLCommandQueue> queue;
  __strong id<MTLComputePipelineState> pipeline;
  // 2026-09-01 22:31 CST (mac): Add a second pipeline whose cubic basis is
  // prepared once in host FP64 instead of redundantly recomputed in GPU FP32.
  __strong id<MTLComputePipelineState> prepared_pipeline;
  // 2026-09-01 22:32 CST (mac): Test two-float arithmetic as a precision
  // bridge; this is not native Metal double support.
  __strong id<MTLComputePipelineState> double_single_pipeline;
};

struct MetalPlan {
  MetalContext *context;
  __strong id<MTLBuffer> tx;
  __strong id<MTLBuffer> ty;
  __strong id<MTLBuffer> coefficients;
  __strong id<MTLBuffer> coefficient_low_parts;
  __strong id<MTLBuffer> x;
  __strong id<MTLBuffer> y;
  __strong id<MTLBuffer> output;
  __strong id<MTLBuffer> prepared_indices;
  __strong id<MTLBuffer> prepared_weights;
  __strong id<MTLBuffer> prepared_weight_low_parts;
  __strong id<MTLBuffer> double_single_output;
  std::vector<double> host_tx;
  std::vector<double> host_ty;
  std::size_t point_capacity;
  std::uint32_t nx;
  std::uint32_t ny;
  std::uint32_t num_grids;
  std::uint32_t coefficients_per_grid;
};

struct KernelParameters {
  std::uint32_t nx;
  std::uint32_t ny;
  std::uint32_t point_count;
  std::uint32_t num_grids;
  std::uint32_t coefficients_per_grid;
};

const char *metal_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct KernelParameters {
  uint nx;
  uint ny;
  uint point_count;
  uint num_grids;
  uint coefficients_per_grid;
};

inline void cubic_basis(const device float *knots, float coordinate, int span,
                        thread float *basis) {
  float previous[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  basis[0] = 1.0f;
  basis[1] = 0.0f;
  basis[2] = 0.0f;
  basis[3] = 0.0f;

  for (int degree = 1; degree <= 3; ++degree) {
    for (int i = 0; i < 4; ++i) {
      previous[i] = basis[i];
    }
    basis[0] = 0.0f;
    for (int i = 0; i < degree; ++i) {
      const int upper = span + i;
      const int lower = upper - degree;
      const float fraction =
          previous[i] / (knots[upper] - knots[lower]);
      basis[i] += fraction * (knots[upper] - coordinate);
      basis[i + 1] = fraction * (coordinate - knots[lower]);
    }
  }
}

inline int find_span(const device float *knots, uint knot_count,
                     thread float &coordinate) {
  const int first = 4;
  const int final = int(knot_count) - 4;
  coordinate = clamp(coordinate, knots[3], knots[final]);
  int span = first;
  while (!(coordinate < knots[span] || span == final)) {
    ++span;
  }
  return span;
}

kernel void few_interp2d_f32(
    device float *result [[buffer(0)]],
    const device float *tx [[buffer(1)]],
    const device float *ty [[buffer(2)]],
    const device float *coefficients [[buffer(3)]],
    const device float *x [[buffer(4)]],
    const device float *y [[buffer(5)]],
    constant KernelParameters &parameters [[buffer(6)]],
    uint global_index [[thread_position_in_grid]]) {
  const uint total = parameters.num_grids * parameters.point_count;
  if (global_index >= total) {
    return;
  }

  const uint grid_index = global_index / parameters.point_count;
  const uint point_index = global_index - grid_index * parameters.point_count;
  float x_value = x[point_index];
  float y_value = y[point_index];
  const int x_span = find_span(tx, parameters.nx, x_value);
  const int y_span = find_span(ty, parameters.ny, y_value);

  float x_basis[4];
  float y_basis[4];
  cubic_basis(tx, x_value, x_span, x_basis);
  cubic_basis(ty, y_value, y_span, y_basis);

  const int y_coefficient_count = int(parameters.ny) - 4;
  int coefficient_index =
      (x_span - 4) * y_coefficient_count + (y_span - 4);
  const uint grid_offset = grid_index * parameters.coefficients_per_grid;
  float sum = 0.0f;
  for (int i = 0; i < 4; ++i) {
    int row_index = coefficient_index;
    for (int j = 0; j < 4; ++j) {
      sum += coefficients[grid_offset + uint(row_index++)] *
             x_basis[i] * y_basis[j];
    }
    coefficient_index += y_coefficient_count;
  }
  result[global_index] = sum;
}

kernel void few_interp2d_prepared_f32(
    device float *result [[buffer(0)]],
    const device float *coefficients [[buffer(1)]],
    const device uint2 *lower_indices [[buffer(2)]],
    const device float *weights [[buffer(3)]],
    constant KernelParameters &parameters [[buffer(4)]],
    uint global_index [[thread_position_in_grid]]) {
  const uint total = parameters.num_grids * parameters.point_count;
  if (global_index >= total) {
    return;
  }

  const uint grid_index = global_index / parameters.point_count;
  const uint point_index = global_index - grid_index * parameters.point_count;
  const uint2 lower = lower_indices[point_index];
  const device float *point_weights = weights + point_index * 8;
  const int y_coefficient_count = int(parameters.ny) - 4;
  int coefficient_index = int(lower.x) * y_coefficient_count + int(lower.y);
  const uint grid_offset = grid_index * parameters.coefficients_per_grid;
  float sum = 0.0f;
  for (int i = 0; i < 4; ++i) {
    int row_index = coefficient_index;
    for (int j = 0; j < 4; ++j) {
      sum += coefficients[grid_offset + uint(row_index++)] *
             point_weights[i] * point_weights[4 + j];
    }
    coefficient_index += y_coefficient_count;
  }
  result[global_index] = sum;
}

inline float2 ds_normalize(float high, float low) {
  const float sum = high + low;
  return float2(sum, low - (sum - high));
}

inline float2 ds_add(float2 left, float2 right) {
  const float sum = left.x + right.x;
  const float virtual_right = sum - left.x;
  const float error = (left.x - (sum - virtual_right)) +
                      (right.x - virtual_right) + left.y + right.y;
  return ds_normalize(sum, error);
}

inline float2 ds_multiply(float2 left, float2 right) {
  const float product = left.x * right.x;
  const float error = fma(left.x, right.x, -product) +
                      left.x * right.y + left.y * right.x;
  return ds_normalize(product, error);
}

kernel void few_interp2d_prepared_double_single(
    device float2 *result [[buffer(0)]],
    const device float *coefficient_high [[buffer(1)]],
    const device float *coefficient_low [[buffer(2)]],
    const device uint2 *lower_indices [[buffer(3)]],
    const device float *weight_high [[buffer(4)]],
    const device float *weight_low [[buffer(5)]],
    constant KernelParameters &parameters [[buffer(6)]],
    uint global_index [[thread_position_in_grid]]) {
  const uint total = parameters.num_grids * parameters.point_count;
  if (global_index >= total) {
    return;
  }

  const uint grid_index = global_index / parameters.point_count;
  const uint point_index = global_index - grid_index * parameters.point_count;
  const uint2 lower = lower_indices[point_index];
  const uint weight_offset = point_index * 8;
  const int y_coefficient_count = int(parameters.ny) - 4;
  int coefficient_index = int(lower.x) * y_coefficient_count + int(lower.y);
  const uint grid_offset = grid_index * parameters.coefficients_per_grid;
  float2 sum = float2(0.0f);
  for (int i = 0; i < 4; ++i) {
    int row_index = coefficient_index;
    const float2 x_weight =
        float2(weight_high[weight_offset + uint(i)],
               weight_low[weight_offset + uint(i)]);
    for (int j = 0; j < 4; ++j) {
      const uint coefficient_offset = grid_offset + uint(row_index++);
      const float2 coefficient =
          float2(coefficient_high[coefficient_offset],
                 coefficient_low[coefficient_offset]);
      const float2 y_weight =
          float2(weight_high[weight_offset + 4 + uint(j)],
                 weight_low[weight_offset + 4 + uint(j)]);
      sum = ds_add(sum, ds_multiply(ds_multiply(coefficient, x_weight),
                                    y_weight));
    }
    coefficient_index += y_coefficient_count;
  }
  result[global_index] = sum;
}
)METAL";

void set_error(const std::string &message) { last_error = message; }

void set_error(NSString *prefix, NSError *error) {
  NSString *description = error ? error.localizedDescription : @"unknown error";
  set_error(std::string(prefix.UTF8String) + ": " + description.UTF8String);
}

bool checked_float_bytes(std::size_t count, NSUInteger *bytes) {
  if (count > std::numeric_limits<NSUInteger>::max() / sizeof(float)) {
    set_error("Metal buffer size overflow");
    return false;
  }
  *bytes = static_cast<NSUInteger>(count * sizeof(float));
  return true;
}

id<MTLBuffer> make_float_buffer(id<MTLDevice> device, std::size_t count) {
  NSUInteger bytes = 0;
  if (!checked_float_bytes(count, &bytes)) {
    return nil;
  }
  // Metal disallows zero-length buffers. The caller may logically use zero
  // entries, but the proof of concept always allocates at least one float.
  bytes = std::max<NSUInteger>(bytes, sizeof(float));
  id<MTLBuffer> buffer =
      [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!buffer) {
    set_error("Metal failed to allocate a shared buffer");
  }
  return buffer;
}

id<MTLBuffer> make_byte_buffer(id<MTLDevice> device, std::size_t bytes) {
  bytes = std::max<std::size_t>(bytes, sizeof(float));
  if (bytes > std::numeric_limits<NSUInteger>::max()) {
    set_error("Metal byte buffer size overflow");
    return nil;
  }
  id<MTLBuffer> buffer = [device
      newBufferWithLength:static_cast<NSUInteger>(bytes)
                  options:MTLResourceStorageModeShared];
  if (!buffer) {
    set_error("Metal failed to allocate a shared byte buffer");
  }
  return buffer;
}

void convert_to_float(const double *source, id<MTLBuffer> destination,
                      std::size_t count) {
  vDSP_vdpsp(source, 1, static_cast<float *>(destination.contents), 1,
             static_cast<vDSP_Length>(count));
}

bool reserve_points(MetalPlan *plan, std::size_t point_count) {
  if (point_count <= plan->point_capacity) {
    return true;
  }
  const std::size_t output_count =
      point_count * static_cast<std::size_t>(plan->num_grids);
  if (plan->num_grids != 0 && output_count / plan->num_grids != point_count) {
    set_error("Metal output element count overflow");
    return false;
  }

  id<MTLBuffer> x = make_float_buffer(plan->context->device, point_count);
  id<MTLBuffer> y = make_float_buffer(plan->context->device, point_count);
  id<MTLBuffer> output =
      make_float_buffer(plan->context->device, output_count);
  id<MTLBuffer> prepared_indices = make_byte_buffer(
      plan->context->device, point_count * 2 * sizeof(std::uint32_t));
  id<MTLBuffer> prepared_weights =
      make_float_buffer(plan->context->device, point_count * 8);
  id<MTLBuffer> prepared_weight_low_parts =
      make_float_buffer(plan->context->device, point_count * 8);
  id<MTLBuffer> double_single_output = make_byte_buffer(
      plan->context->device, output_count * 2 * sizeof(float));
  if (!x || !y || !output || !prepared_indices || !prepared_weights ||
      !prepared_weight_low_parts || !double_single_output) {
    return false;
  }
  plan->x = x;
  plan->y = y;
  plan->output = output;
  plan->prepared_indices = prepared_indices;
  plan->prepared_weights = prepared_weights;
  plan->prepared_weight_low_parts = prepared_weight_low_parts;
  plan->double_single_output = double_single_output;
  plan->point_capacity = point_count;
  return true;
}

int find_span_host(const std::vector<double> &knots, double *coordinate) {
  const int final = static_cast<int>(knots.size()) - 4;
  *coordinate = std::clamp(*coordinate, knots[3], knots[final]);
  int span = 4;
  while (!(*coordinate < knots[span] || span == final)) {
    ++span;
  }
  return span;
}

void cubic_basis_host(const std::vector<double> &knots, double coordinate,
                      int span, float *float_basis, float *float_low_parts) {
  double basis[4] = {1.0, 0.0, 0.0, 0.0};
  double previous[4] = {0.0, 0.0, 0.0, 0.0};
  for (int degree = 1; degree <= 3; ++degree) {
    std::copy(std::begin(basis), std::end(basis), std::begin(previous));
    basis[0] = 0.0;
    for (int i = 0; i < degree; ++i) {
      const int upper = span + i;
      const int lower = upper - degree;
      const double fraction =
          previous[i] / (knots[upper] - knots[lower]);
      basis[i] += fraction * (knots[upper] - coordinate);
      basis[i + 1] = fraction * (coordinate - knots[lower]);
    }
  }
  for (int i = 0; i < 4; ++i) {
    float_basis[i] = static_cast<float>(basis[i]);
    float_low_parts[i] =
        static_cast<float>(basis[i] - static_cast<double>(float_basis[i]));
  }
}

void prepare_points(MetalPlan *plan, const double *x, const double *y,
                    std::size_t point_count) {
  auto *indices =
      static_cast<std::uint32_t *>(plan->prepared_indices.contents);
  auto *weights = static_cast<float *>(plan->prepared_weights.contents);
  auto *weight_low_parts =
      static_cast<float *>(plan->prepared_weight_low_parts.contents);
  for (std::size_t point = 0; point < point_count; ++point) {
    double x_value = x[point];
    double y_value = y[point];
    const int x_span = find_span_host(plan->host_tx, &x_value);
    const int y_span = find_span_host(plan->host_ty, &y_value);
    indices[point * 2] = static_cast<std::uint32_t>(x_span - 4);
    indices[point * 2 + 1] = static_cast<std::uint32_t>(y_span - 4);
    cubic_basis_host(plan->host_tx, x_value, x_span, weights + point * 8,
                     weight_low_parts + point * 8);
    cubic_basis_host(plan->host_ty, y_value, y_span, weights + point * 8 + 4,
                     weight_low_parts + point * 8 + 4);
  }
}

bool encode_and_wait(MetalPlan *plan, id<MTLComputePipelineState> pipeline,
                     const KernelParameters &parameters, int variant,
                     double *gpu_seconds) {
  id<MTLCommandBuffer> command_buffer = [plan->context->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  if (!command_buffer || !encoder) {
    set_error("Failed to create Metal command objects");
    return false;
  }
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:variant == 2 ? plan->double_single_output : plan->output
                offset:0
               atIndex:0];
  if (variant == 2) {
    [encoder setBuffer:plan->coefficients offset:0 atIndex:1];
    [encoder setBuffer:plan->coefficient_low_parts offset:0 atIndex:2];
    [encoder setBuffer:plan->prepared_indices offset:0 atIndex:3];
    [encoder setBuffer:plan->prepared_weights offset:0 atIndex:4];
    [encoder setBuffer:plan->prepared_weight_low_parts offset:0 atIndex:5];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:6];
  } else if (variant == 1) {
    [encoder setBuffer:plan->coefficients offset:0 atIndex:1];
    [encoder setBuffer:plan->prepared_indices offset:0 atIndex:2];
    [encoder setBuffer:plan->prepared_weights offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
  } else {
    [encoder setBuffer:plan->tx offset:0 atIndex:1];
    [encoder setBuffer:plan->ty offset:0 atIndex:2];
    [encoder setBuffer:plan->coefficients offset:0 atIndex:3];
    [encoder setBuffer:plan->x offset:0 atIndex:4];
    [encoder setBuffer:plan->y offset:0 atIndex:5];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:6];
  }

  const std::size_t total =
      static_cast<std::size_t>(plan->num_grids) * parameters.point_count;
  const NSUInteger width = pipeline.threadExecutionWidth;
  const NSUInteger maximum = pipeline.maxTotalThreadsPerThreadgroup;
  const NSUInteger desired = std::min<NSUInteger>(256, maximum);
  const NSUInteger threads = std::max<NSUInteger>(
      width, desired - (desired % std::max<NSUInteger>(width, 1)));
  [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  if (command_buffer.status == MTLCommandBufferStatusError) {
    set_error(@"Metal command buffer failed", command_buffer.error);
    return false;
  }
  const double start_time = command_buffer.GPUStartTime;
  const double end_time = command_buffer.GPUEndTime;
  if (gpu_seconds) {
    *gpu_seconds =
        end_time >= start_time && start_time > 0.0 ? end_time - start_time : 0.0;
  }
  return true;
}

} // namespace

extern "C" {

const char *few_metal_last_error() { return last_error.c_str(); }

void *few_metal_context_create() {
  @autoreleasepool {
    last_error.clear();
    auto *context = new (std::nothrow) MetalContext();
    if (!context) {
      set_error("Failed to allocate Metal context");
      return nullptr;
    }
    context->device = MTLCreateSystemDefaultDevice();
    if (!context->device) {
      set_error("No Metal device is available");
      delete context;
      return nullptr;
    }
    context->queue = [context->device newCommandQueue];
    if (!context->queue) {
      set_error("Failed to create Metal command queue");
      delete context;
      return nullptr;
    }

    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:metal_source];
    // 2026-09-01 22:35 CST (mac): Error-recovery arithmetic requires strict
    // operation boundaries; Metal fast-math may algebraically erase the FMA
    // residual used by the double-single product.
    MTLCompileOptions *compile_options = [MTLCompileOptions new];
    compile_options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library =
        [context->device newLibraryWithSource:source
                                      options:compile_options
                                        error:&error];
    if (!library) {
      set_error(@"Metal runtime compilation failed", error);
      delete context;
      return nullptr;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"few_interp2d_f32"];
    if (!function) {
      set_error("Metal library does not contain few_interp2d_f32");
      delete context;
      return nullptr;
    }
    context->pipeline =
        [context->device newComputePipelineStateWithFunction:function error:&error];
    if (!context->pipeline) {
      set_error(@"Metal pipeline creation failed", error);
      delete context;
      return nullptr;
    }
    id<MTLFunction> prepared_function =
        [library newFunctionWithName:@"few_interp2d_prepared_f32"];
    if (!prepared_function) {
      set_error("Metal library does not contain few_interp2d_prepared_f32");
      delete context;
      return nullptr;
    }
    context->prepared_pipeline = [context->device
        newComputePipelineStateWithFunction:prepared_function
                                      error:&error];
    if (!context->prepared_pipeline) {
      set_error(@"Prepared Metal pipeline creation failed", error);
      delete context;
      return nullptr;
    }
    id<MTLFunction> double_single_function =
        [library newFunctionWithName:@"few_interp2d_prepared_double_single"];
    if (!double_single_function) {
      set_error(
          "Metal library does not contain few_interp2d_prepared_double_single");
      delete context;
      return nullptr;
    }
    context->double_single_pipeline = [context->device
        newComputePipelineStateWithFunction:double_single_function
                                      error:&error];
    if (!context->double_single_pipeline) {
      set_error(@"Double-single Metal pipeline creation failed", error);
      delete context;
      return nullptr;
    }
    return context;
  }
}

void few_metal_context_destroy(void *opaque_context) {
  @autoreleasepool { delete static_cast<MetalContext *>(opaque_context); }
}

int few_metal_device_name(void *opaque_context, char *destination,
                          std::size_t destination_size) {
  @autoreleasepool {
    if (!opaque_context || !destination || destination_size == 0) {
      set_error("Invalid argument to few_metal_device_name");
      return -1;
    }
    auto *context = static_cast<MetalContext *>(opaque_context);
    const char *name = context->device.name.UTF8String;
    std::strncpy(destination, name, destination_size - 1);
    destination[destination_size - 1] = '\0';
    return 0;
  }
}

std::size_t few_metal_thread_execution_width(void *opaque_context) {
  auto *context = static_cast<MetalContext *>(opaque_context);
  return context ? context->pipeline.threadExecutionWidth : 0;
}

std::size_t few_metal_max_threads(void *opaque_context) {
  auto *context = static_cast<MetalContext *>(opaque_context);
  return context ? context->pipeline.maxTotalThreadsPerThreadgroup : 0;
}

void *few_metal_plan_create(void *opaque_context, const double *tx,
                            std::uint32_t nx, const double *ty,
                            std::uint32_t ny, const double *coefficients,
                            std::uint32_t num_grids,
                            std::uint32_t coefficients_per_grid) {
  @autoreleasepool {
    last_error.clear();
    if (!opaque_context || !tx || !ty || !coefficients || nx < 8 || ny < 8 ||
        num_grids == 0 || coefficients_per_grid == 0) {
      set_error("Invalid Metal interpolation plan arguments");
      return nullptr;
    }
    const std::size_t coefficient_count =
        static_cast<std::size_t>(num_grids) * coefficients_per_grid;
    if (coefficient_count / num_grids != coefficients_per_grid) {
      set_error("Metal coefficient element count overflow");
      return nullptr;
    }

    auto *plan = new (std::nothrow) MetalPlan();
    if (!plan) {
      set_error("Failed to allocate Metal interpolation plan");
      return nullptr;
    }
    plan->context = static_cast<MetalContext *>(opaque_context);
    plan->point_capacity = 0;
    plan->nx = nx;
    plan->ny = ny;
    plan->num_grids = num_grids;
    plan->coefficients_per_grid = coefficients_per_grid;
    plan->host_tx.assign(tx, tx + nx);
    plan->host_ty.assign(ty, ty + ny);
    plan->tx = make_float_buffer(plan->context->device, nx);
    plan->ty = make_float_buffer(plan->context->device, ny);
    plan->coefficients =
        make_float_buffer(plan->context->device, coefficient_count);
    plan->coefficient_low_parts =
        make_float_buffer(plan->context->device, coefficient_count);
    if (!plan->tx || !plan->ty || !plan->coefficients ||
        !plan->coefficient_low_parts) {
      delete plan;
      return nullptr;
    }
    convert_to_float(tx, plan->tx, nx);
    convert_to_float(ty, plan->ty, ny);
    convert_to_float(coefficients, plan->coefficients, coefficient_count);
    auto *coefficient_high =
        static_cast<float *>(plan->coefficients.contents);
    auto *coefficient_low =
        static_cast<float *>(plan->coefficient_low_parts.contents);
    for (std::size_t index = 0; index < coefficient_count; ++index) {
      coefficient_low[index] = static_cast<float>(
          coefficients[index] - static_cast<double>(coefficient_high[index]));
    }
    return plan;
  }
}

void few_metal_plan_destroy(void *opaque_plan) {
  @autoreleasepool { delete static_cast<MetalPlan *>(opaque_plan); }
}

int few_metal_plan_evaluate(void *opaque_plan, const double *x,
                            const double *y, std::uint32_t point_count,
                            double *output, double *gpu_seconds) {
  @autoreleasepool {
    last_error.clear();
    if (!opaque_plan || !x || !y || !output || point_count == 0) {
      set_error("Invalid Metal interpolation evaluation arguments");
      return -1;
    }
    auto *plan = static_cast<MetalPlan *>(opaque_plan);
    if (!reserve_points(plan, point_count)) {
      return -1;
    }
    convert_to_float(x, plan->x, point_count);
    convert_to_float(y, plan->y, point_count);

    KernelParameters parameters{plan->nx, plan->ny, point_count,
                                plan->num_grids,
                                plan->coefficients_per_grid};
    if (!encode_and_wait(plan, plan->context->pipeline, parameters, 0,
                         gpu_seconds)) {
      return -1;
    }
    const std::size_t total =
        static_cast<std::size_t>(plan->num_grids) * point_count;
    vDSP_vspdp(static_cast<float *>(plan->output.contents), 1, output, 1,
               static_cast<vDSP_Length>(total));
    return 0;
  }
}

int few_metal_plan_evaluate_prepared(void *opaque_plan, const double *x,
                                     const double *y,
                                     std::uint32_t point_count, double *output,
                                     double *gpu_seconds) {
  @autoreleasepool {
    last_error.clear();
    if (!opaque_plan || !x || !y || !output || point_count == 0) {
      set_error("Invalid prepared Metal interpolation evaluation arguments");
      return -1;
    }
    auto *plan = static_cast<MetalPlan *>(opaque_plan);
    if (!reserve_points(plan, point_count)) {
      return -1;
    }
    prepare_points(plan, x, y, point_count);
    KernelParameters parameters{plan->nx, plan->ny, point_count,
                                plan->num_grids,
                                plan->coefficients_per_grid};
    if (!encode_and_wait(plan, plan->context->prepared_pipeline, parameters, 1,
                         gpu_seconds)) {
      return -1;
    }
    const std::size_t total =
        static_cast<std::size_t>(plan->num_grids) * point_count;
    vDSP_vspdp(static_cast<float *>(plan->output.contents), 1, output, 1,
               static_cast<vDSP_Length>(total));
    return 0;
  }
}

int few_metal_plan_evaluate_double_single(
    void *opaque_plan, const double *x, const double *y,
    std::uint32_t point_count, double *output, double *gpu_seconds) {
  @autoreleasepool {
    last_error.clear();
    if (!opaque_plan || !x || !y || !output || point_count == 0) {
      set_error("Invalid double-single Metal interpolation arguments");
      return -1;
    }
    auto *plan = static_cast<MetalPlan *>(opaque_plan);
    if (!reserve_points(plan, point_count)) {
      return -1;
    }
    prepare_points(plan, x, y, point_count);
    KernelParameters parameters{plan->nx, plan->ny, point_count,
                                plan->num_grids,
                                plan->coefficients_per_grid};
    if (!encode_and_wait(plan, plan->context->double_single_pipeline,
                         parameters, 2, gpu_seconds)) {
      return -1;
    }
    const std::size_t total =
        static_cast<std::size_t>(plan->num_grids) * point_count;
    const auto *parts =
        static_cast<const float *>(plan->double_single_output.contents);
    for (std::size_t index = 0; index < total; ++index) {
      output[index] = static_cast<double>(parts[index * 2]) +
                      static_cast<double>(parts[index * 2 + 1]);
    }
    return 0;
  }
}

} // extern "C"
