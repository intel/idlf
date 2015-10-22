/*
Copyright (c) 2014, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of Intel Corporation nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <cstdint>
#include <limits>

#include "device/cpu/api_internal/cpu_device_internal.h"
#include "device/common/nn_layer_parameters.h"

namespace convolution
{

const uint64_t OUTPUT_ACCEPTED_MOD = 4;
const uint64_t INPUT_ACCEPTED_MOD = 16;

namespace forward
{

template <typename T_Name>
using ValueU64=Value<T_Name, uint64_t>;

struct StackWithData;
struct CompiledKernelConvolution;
void release(CompiledKernelConvolution*);

CompiledKernelConvolution* compileConvolution(
    uint64_t num_of_threads,
    KernelInfo kernel_info,
    InputDimensions input_dimensions,
    InputDimensions full_input_dimensions,
    OutputDimensions output_dimensions,
    OutputDimensions full_output_dimensions,
    StrideInfo stride,
    ValueU64<InputFeatsStart> infeats_window_start,
    ValueU64<OutputFeatsStart> outfeats_window_start,
    uint64_t batch_iterations,
    float* input,
    float* output,
    float* weights,
    float* bias,
    bool apply_relu = false);

void convolve_fst_threaded_batch(
    nn_thread_worker_pool& thread_pool,
    CompiledKernelConvolution* convInternal,
    float* input_ref, //aligned and with batch
    float* output_ref, //aligned and with batch
    float* kernel_ref, //aligned
    float* bias_ref,
    ValueU64<Batch> batch,
    KernelInfo kernel_info,
    InputDimensions input_dimensions,
    OutputDimensions output_dimensions,
    StrideInfo stride,
    ValueU64<InputFeatsStart> infeats_window_start,
    ValueU64<OutputFeatsStart> outfeats_window_start);

void convolve_fst_single_threaded(
    CompiledKernelConvolution* convInternal,
    float* input_ref, //aligned and with batch
    float* output_ref, //aligned and with batch
    float* kernel_ref, //aligned
    float* bias_ref,
    ValueU64<Batch> batch,
    KernelInfo kernel_info,
    InputDimensions input_dimensions,
    OutputDimensions output_dimensions,
    StrideInfo stride,
    ValueU64<InputFeatsStart> infeats_window_start,
    ValueU64<OutputFeatsStart> outfeats_window_start);

} //namespace forward
} //namespace convolution

