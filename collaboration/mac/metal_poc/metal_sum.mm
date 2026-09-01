// 2026-09-01 22:49 CST (mac): Add an isolated Metal proof of concept for the
// time-domain mode-summation hotspot. CPU FP64 evaluates and range-reduces the
// three phase splines; the GPU receives high/low FP32 phase parts. This file is
// never linked into FEW's production CPU/CUDA modules.

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

struct SumContext {
  __strong id<MTLDevice> device;
  __strong id<MTLCommandQueue> queue;
  __strong id<MTLComputePipelineState> pipeline;
};

struct SumParameters {
  std::uint32_t init_length;
  std::uint32_t output_length;
  std::uint32_t mode_count;
  std::uint32_t padding;
};

const char *metal_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct SumParameters {
  uint init_length;
  uint output_length;
  uint mode_count;
  uint padding;
};

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

inline float2 ds_scale(float2 value, float factor) {
  return ds_normalize(value.x * factor, value.y * factor);
}

inline float2 reduced_mode_phase(float4 high, float4 low,
                                 int m, int k, int n) {
  float2 phase = ds_scale(float2(high.x, low.x), float(m));
  phase = ds_add(phase, ds_scale(float2(high.y, low.y), float(k)));
  phase = ds_add(phase, ds_scale(float2(high.z, low.z), float(n)));

  // The host has already reduced each fundamental phase. The mode indices in
  // this model keep this combined angle small enough for an exact integer q.
  constexpr float two_pi_high = 6.283185482025146484375f;
  constexpr float two_pi_low = -1.7484555314695172e-7f;
  constexpr float inverse_two_pi = 0.15915494309189535f;
  const float quotient = rint((phase.x + phase.y) * inverse_two_pi);
  return ds_add(phase,
                ds_scale(float2(two_pi_high, two_pi_low), -quotient));
}

kernel void few_mode_sum_f32(
    device float4 *waveform [[buffer(0)]],
    const device float *interpolation [[buffer(1)]],
    const device int *m_values [[buffer(2)]],
    const device int *k_values [[buffer(3)]],
    const device int *n_values [[buffer(4)]],
    const device float2 *ylms [[buffer(5)]],
    const device uint *segments [[buffer(6)]],
    const device float *local_times [[buffer(7)]],
    const device float4 *phase_high [[buffer(8)]],
    const device float4 *phase_low [[buffer(9)]],
    constant SumParameters &parameters [[buffer(10)]],
    uint sample [[thread_position_in_grid]]) {
  if (sample >= parameters.output_length) {
    return;
  }

  const uint segment = segments[sample];
  const float x = local_times[sample];
  const float x2 = x * x;
  const float x3 = x2 * x;
  const uint mode_count = parameters.mode_count;
  const uint base_count = parameters.init_length * 2 * mode_count;
  const uint segment_offset = segment * 2 * mode_count;
  float2 real_sum = float2(0.0f);
  float2 imag_sum = float2(0.0f);

  for (uint mode = 0; mode < mode_count; ++mode) {
    const uint real_index = segment_offset + mode;
    const uint imag_index = real_index + mode_count;
    const float amplitude_real =
        interpolation[real_index] + interpolation[base_count + real_index] * x +
        interpolation[2 * base_count + real_index] * x2 +
        interpolation[3 * base_count + real_index] * x3;
    const float amplitude_imag =
        interpolation[imag_index] + interpolation[base_count + imag_index] * x +
        interpolation[2 * base_count + imag_index] * x2 +
        interpolation[3 * base_count + imag_index] * x3;

    const float2 phase = reduced_mode_phase(
        phase_high[sample], phase_low[sample], m_values[mode], k_values[mode],
        n_values[mode]);
    float cosine = 0.0f;
    const float sine_high = sincos(phase.x, cosine);
    const float sine = fma(cosine, phase.y, sine_high);
    cosine = fma(-sine_high, phase.y, cosine);

    // exp(-i phase) * amplitude
    const float partial_real =
        fma(cosine, amplitude_real, sine * amplitude_imag);
    const float partial_imag =
        fma(cosine, amplitude_imag, -sine * amplitude_real);
    const float2 plus_ylm = ylms[mode];
    const float plus_real =
        fma(partial_real, plus_ylm.x, -partial_imag * plus_ylm.y);
    const float plus_imag =
        fma(partial_real, plus_ylm.y, partial_imag * plus_ylm.x);
    real_sum = ds_add(real_sum, float2(plus_real, 0.0f));
    imag_sum = ds_add(imag_sum, float2(plus_imag, 0.0f));

    if (m_values[mode] != 0) {
      const float2 minus_ylm = ylms[mode_count + mode];
      // conj(partial) * Y_minus
      const float minus_real =
          fma(partial_real, minus_ylm.x, partial_imag * minus_ylm.y);
      const float minus_imag =
          fma(partial_real, minus_ylm.y, -partial_imag * minus_ylm.x);
      real_sum = ds_add(real_sum, float2(minus_real, 0.0f));
      imag_sum = ds_add(imag_sum, float2(minus_imag, 0.0f));
    }
  }
  waveform[sample] = float4(real_sum.x, real_sum.y, imag_sum.x, imag_sum.y);
}
)METAL";

void set_error(const std::string &message) { last_error = message; }

void set_error(NSString *prefix, NSError *error) {
  NSString *description = error ? error.localizedDescription : @"unknown error";
  set_error(std::string(prefix.UTF8String) + ": " + description.UTF8String);
}

id<MTLBuffer> make_buffer(id<MTLDevice> device, std::size_t bytes) {
  bytes = std::max<std::size_t>(bytes, sizeof(float));
  if (bytes > std::numeric_limits<NSUInteger>::max()) {
    set_error("Metal summation buffer size overflow");
    return nil;
  }
  id<MTLBuffer> buffer = [device
      newBufferWithLength:static_cast<NSUInteger>(bytes)
                  options:MTLResourceStorageModeShared];
  if (!buffer) {
    set_error("Metal failed to allocate a summation buffer");
  }
  return buffer;
}

void convert_to_float(const double *source, id<MTLBuffer> destination,
                      std::size_t count) {
  vDSP_vdpsp(source, 1, static_cast<float *>(destination.contents), 1,
             static_cast<vDSP_Length>(count));
}

double evaluate_phase(const double *coefficients, int segment_count,
                      int segment, int phase, double s) {
  const int index = segment * 3 + phase;
  const double s1 = 1.0 - s;
  const double c0 = coefficients[index];
  const double c1 = coefficients[segment_count + index];
  const double c2 = coefficients[2 * segment_count + index];
  const double c3 = coefficients[3 * segment_count + index];
  const double c4 = coefficients[4 * segment_count + index];
  const double c5 = coefficients[5 * segment_count + index];
  const double c6 = coefficients[6 * segment_count + index];
  const double c7 = coefficients[7 * segment_count + index];
  return c0 + s * (c1 + s1 * (c2 + s * (c3 + s1 * (c4 + s * (c5 + s1 *
         (c6 + s * c7))))));
}

bool prepare_samples(const double *phase_times, const double *phase_coefficients,
                     int init_length, int output_length, double delta_t,
                     const double *trajectory_times, id<MTLBuffer> segments_buffer,
                     id<MTLBuffer> local_times_buffer,
                     id<MTLBuffer> phase_high_buffer,
                     id<MTLBuffer> phase_low_buffer) {
  auto *segments = static_cast<std::uint32_t *>(segments_buffer.contents);
  auto *local_times = static_cast<float *>(local_times_buffer.contents);
  auto *phase_high = static_cast<float *>(phase_high_buffer.contents);
  auto *phase_low = static_cast<float *>(phase_low_buffer.contents);
  const int phase_segment_count = (init_length - 1) * 3;
  int segment = 0;
  for (int sample = 0; sample < output_length; ++sample) {
    while (segment + 1 < init_length - 1 &&
           sample >= static_cast<int>(std::ceil(trajectory_times[segment + 1] /
                                                delta_t))) {
      ++segment;
    }
    if (segment + 1 >= init_length) {
      set_error("Metal summation sample exceeds the sparse trajectory");
      return false;
    }
    const double time = delta_t * sample;
    segments[sample] = static_cast<std::uint32_t>(segment);
    local_times[sample] =
        static_cast<float>(time - trajectory_times[segment]);
    const double width = phase_times[segment + 1] - phase_times[segment];
    const double s = (time - phase_times[segment]) / width;
    for (int phase = 0; phase < 3; ++phase) {
      const double value = std::remainder(
          evaluate_phase(phase_coefficients, phase_segment_count, segment,
                         phase, s),
          2.0 * M_PI);
      const float high = static_cast<float>(value);
      phase_high[sample * 4 + phase] = high;
      phase_low[sample * 4 + phase] =
          static_cast<float>(value - static_cast<double>(high));
    }
    phase_high[sample * 4 + 3] = 0.0f;
    phase_low[sample * 4 + 3] = 0.0f;
  }
  return true;
}

} // namespace

extern "C" {

const char *few_metal_sum_last_error() { return last_error.c_str(); }

void *few_metal_sum_context_create() {
  @autoreleasepool {
    last_error.clear();
    auto *context = new (std::nothrow) SumContext();
    if (!context) {
      set_error("Failed to allocate Metal summation context");
      return nullptr;
    }
    context->device = MTLCreateSystemDefaultDevice();
    context->queue = [context->device newCommandQueue];
    if (!context->device || !context->queue) {
      set_error("Metal device or summation command queue is unavailable");
      delete context;
      return nullptr;
    }
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    NSError *error = nil;
    id<MTLLibrary> library = [context->device
        newLibraryWithSource:[NSString stringWithUTF8String:metal_source]
                     options:options
                       error:&error];
    if (!library) {
      set_error(@"Metal summation runtime compilation failed", error);
      delete context;
      return nullptr;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"few_mode_sum_f32"];
    context->pipeline =
        [context->device newComputePipelineStateWithFunction:function error:&error];
    if (!function || !context->pipeline) {
      set_error(@"Metal summation pipeline creation failed", error);
      delete context;
      return nullptr;
    }
    return context;
  }
}

void few_metal_sum_context_destroy(void *opaque_context) {
  @autoreleasepool { delete static_cast<SumContext *>(opaque_context); }
}

int few_metal_sum_evaluate(
    void *opaque_context, double *waveform, const double *interpolation,
    const double *phase_times, const double *phase_coefficients,
    const int *m_values, const int *k_values, const int *n_values,
    int init_length, int output_length, int mode_count, const double *ylms,
    double delta_t, const double *trajectory_times, double *gpu_seconds) {
  @autoreleasepool {
    last_error.clear();
    if (!opaque_context || !waveform || !interpolation || !phase_times ||
        !phase_coefficients || !m_values || !k_values || !n_values || !ylms ||
        !trajectory_times || init_length < 2 || output_length < 1 ||
        mode_count < 1) {
      set_error("Invalid Metal summation arguments");
      return -1;
    }
    auto *context = static_cast<SumContext *>(opaque_context);
    const std::size_t interpolation_count =
        4ULL * init_length * 2ULL * mode_count;
    const std::size_t ylm_float_count = 4ULL * mode_count;
    id<MTLBuffer> output_buffer =
        make_buffer(context->device, output_length * 4ULL * sizeof(float));
    id<MTLBuffer> interpolation_buffer =
        make_buffer(context->device, interpolation_count * sizeof(float));
    id<MTLBuffer> m_buffer =
        make_buffer(context->device, mode_count * sizeof(int));
    id<MTLBuffer> k_buffer =
        make_buffer(context->device, mode_count * sizeof(int));
    id<MTLBuffer> n_buffer =
        make_buffer(context->device, mode_count * sizeof(int));
    id<MTLBuffer> ylm_buffer =
        make_buffer(context->device, ylm_float_count * sizeof(float));
    id<MTLBuffer> segments_buffer = make_buffer(
        context->device, output_length * sizeof(std::uint32_t));
    id<MTLBuffer> local_times_buffer =
        make_buffer(context->device, output_length * sizeof(float));
    id<MTLBuffer> phase_high_buffer =
        make_buffer(context->device, output_length * 4ULL * sizeof(float));
    id<MTLBuffer> phase_low_buffer =
        make_buffer(context->device, output_length * 4ULL * sizeof(float));
    if (!output_buffer || !interpolation_buffer || !m_buffer || !k_buffer ||
        !n_buffer || !ylm_buffer || !segments_buffer || !local_times_buffer ||
        !phase_high_buffer || !phase_low_buffer) {
      return -1;
    }

    convert_to_float(interpolation, interpolation_buffer, interpolation_count);
    convert_to_float(ylms, ylm_buffer, ylm_float_count);
    std::memcpy(m_buffer.contents, m_values, mode_count * sizeof(int));
    std::memcpy(k_buffer.contents, k_values, mode_count * sizeof(int));
    std::memcpy(n_buffer.contents, n_values, mode_count * sizeof(int));
    if (!prepare_samples(phase_times, phase_coefficients, init_length,
                         output_length, delta_t, trajectory_times,
                         segments_buffer, local_times_buffer, phase_high_buffer,
                         phase_low_buffer)) {
      return -1;
    }

    SumParameters parameters{static_cast<std::uint32_t>(init_length),
                             static_cast<std::uint32_t>(output_length),
                             static_cast<std::uint32_t>(mode_count), 0};
    id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context->pipeline];
    [encoder setBuffer:output_buffer offset:0 atIndex:0];
    [encoder setBuffer:interpolation_buffer offset:0 atIndex:1];
    [encoder setBuffer:m_buffer offset:0 atIndex:2];
    [encoder setBuffer:k_buffer offset:0 atIndex:3];
    [encoder setBuffer:n_buffer offset:0 atIndex:4];
    [encoder setBuffer:ylm_buffer offset:0 atIndex:5];
    [encoder setBuffer:segments_buffer offset:0 atIndex:6];
    [encoder setBuffer:local_times_buffer offset:0 atIndex:7];
    [encoder setBuffer:phase_high_buffer offset:0 atIndex:8];
    [encoder setBuffer:phase_low_buffer offset:0 atIndex:9];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:10];
    const NSUInteger width = context->pipeline.threadExecutionWidth;
    const NSUInteger maximum = context->pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger desired = std::min<NSUInteger>(256, maximum);
    const NSUInteger threads = std::max<NSUInteger>(
        width, desired - desired % std::max<NSUInteger>(width, 1));
    [encoder dispatchThreads:MTLSizeMake(output_length, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
      set_error(@"Metal summation command failed", command_buffer.error);
      return -1;
    }
    const double start_time = command_buffer.GPUStartTime;
    const double end_time = command_buffer.GPUEndTime;
    if (gpu_seconds) {
      *gpu_seconds = end_time >= start_time && start_time > 0.0
                         ? end_time - start_time
                         : 0.0;
    }
    const auto *parts = static_cast<const float *>(output_buffer.contents);
    for (int sample = 0; sample < output_length; ++sample) {
      waveform[sample * 2] = static_cast<double>(parts[sample * 4]) +
                             static_cast<double>(parts[sample * 4 + 1]);
      waveform[sample * 2 + 1] = static_cast<double>(parts[sample * 4 + 2]) +
                                 static_cast<double>(parts[sample * 4 + 3]);
    }
    return 0;
  }
}

} // extern "C"
