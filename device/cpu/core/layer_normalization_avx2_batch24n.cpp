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

#include "device/common/nn_workload_data.h"
#include "device/common/nn_allocate.h"
#include "device/common/nn_asmjit_compilation.h"
#include "device/common/nn_layer_parameters.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "layer_normalization_avx2_batch24n.h"
#include "helper_zxyn_f32.h"
#include "nn_asmjit_power.h"
#include "nn_intrinsic_power.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <tuple>
#include <iostream>
#include "device/cpu/api_internal/data_helper.h"
#include "tester/g_ult/unit_tests/cpu/naive_implementations.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <asmjit/asmjit.h>

namespace layer
{
namespace
{

const uint64_t BATCH_SHIFT = 8;
const uint64_t ZERO = 0;
const uint64_t BATCH_BLOCKS = BATCH_ACCEPTED_BLOCK / BATCH_SHIFT;

struct InputBuffer {};
struct OutputBuffer {};
struct KParam {};
struct NParam {};
struct AlphaParam {};
struct BetaParam {};
struct FeatsParam {};
struct BatchSize {};
struct Width {};

struct InputData
{
    Value<InputBuffer, float*> input;
    Value<OutputBuffer, float*> output;

    Value<KParam, float> k;
    Value<AlphaParam, float> alpha;
    Value<BetaParam, float> beta;
    Value<FeatsParam, uint64_t> feats;
    Value<NParam, uint64_t> n;
    Value<Width, uint64_t> width;

    InputData(Value<InputBuffer, float*> input = make<InputBuffer>(nullptr),
              Value<OutputBuffer, float*> output = make<OutputBuffer>(nullptr),
              Value<KParam, float> k = make<KParam>(0),
              Value<AlphaParam, float> alpha = make<AlphaParam>(0),
              Value<BetaParam, float> beta = make<BetaParam>(0),
              Value<FeatsParam, uint64_t> feats = make<FeatsParam>(0),
              Value<NParam, uint64_t> n = make<NParam>(0),
              Value<Width, uint64_t> width = make<Width>(0))
        : input(input)
        , output(output)
        , k(k)
        , alpha(alpha)
        , beta(beta)
        , feats(feats)
        , n(n)
        , width(width)
    {}
};

//for beta == 0.75f
void intrinsic_version_for_0_75(InputData* args)
{
    float* input = args->input;
    float* output = args->output;
    uint64_t feats = args->feats;
    uint64_t n_param = args->n;
    uint64_t k_param = args->k;
    float alpha = args->alpha;// / (float)n_param;
    float beta = args->beta;
    uint64_t width = args->width;

    uint64_t n_shift = n_param / 2;
    __m256 alpha_mm = _mm256_set1_ps(alpha);
    __m256 k_mm = _mm256_set1_ps(k_param);

    const uint64_t BATCH_BLOCKS = BATCH_ACCEPTED_BLOCK / BATCH_SHIFT;

    __m256 accs[BATCH_BLOCKS];
    __m256 compensation[BATCH_BLOCKS];
    for (uint64_t i = 0; i < BATCH_BLOCKS; ++i)
        compensation[i] = _mm256_setzero_ps();
    auto calc = [&](uint64_t i) {
            for (uint64_t j = 0; j < BATCH_BLOCKS; ++j)
            {
                auto temp = _mm256_fmadd_ps(accs[j], alpha_mm, k_mm);
                temp = nn::intrinsics::invpow_075(temp);
                auto src = _mm256_load_ps(input + i * BATCH_ACCEPTED_BLOCK + j * BATCH_SHIFT);
                auto dest = output + i * BATCH_ACCEPTED_BLOCK + j * BATCH_SHIFT;
                _mm256_storeu_ps(dest, _mm256_mul_ps(temp, src));
            }
        };
    auto add_top = [&](uint64_t i) {
            for (uint64_t j = 0; j < BATCH_BLOCKS; ++j)
            {
                auto src = _mm256_load_ps(input + i * BATCH_ACCEPTED_BLOCK + j * BATCH_SHIFT);
                auto y = _mm256_fmsub_ps(src, src, compensation[j]);
                auto t = _mm256_add_ps(accs[j], y);
                compensation[j] = _mm256_sub_ps(_mm256_sub_ps(t, accs[j]), y);
                accs[j] = t;
            }
        };
    auto sub_bottom = [&](uint64_t i) {
            for (uint64_t j = 0; j < BATCH_BLOCKS; ++j)
            {
                auto src = _mm256_load_ps(input + i * BATCH_ACCEPTED_BLOCK + j * BATCH_SHIFT);
                auto y = _mm256_fmadd_ps(src, src, compensation[j]);
                auto t = _mm256_sub_ps(accs[j], y);
                compensation[j] = _mm256_add_ps(_mm256_sub_ps(t, accs[j]), y);
                accs[j] = t;
            }
        };

    for (uint64_t x = 0; x < width; ++x)
    {
        for (uint64_t i = 0; i < BATCH_BLOCKS; ++i)
             accs[i] = _mm256_setzero_ps();
        for (uint64_t i = 0; i < n_shift; ++i) {
            add_top(i);
        }
        for (uint64_t i = n_shift; i < n_param; ++i) {
            add_top(i);
            calc(i - n_shift);
        }
        for (uint64_t i = n_param; i < feats; ++i) {
            add_top(i);
            sub_bottom(i - n_param);
            calc(i - n_shift);
        }
        for (uint64_t i = feats; i < feats + n_shift; ++i) {
            sub_bottom(i - n_param);
            calc(i - n_shift);
        }
        input += feats * BATCH_ACCEPTED_BLOCK;
        output += feats * BATCH_ACCEPTED_BLOCK;
    }
}

//for any beta
void naive(InputData* args)
{
    float* input = args->input;
    float* output = args->output;
    uint64_t feats = args->feats;
    uint64_t n_param = args->n;
    uint64_t k_param = args->k;
    float alpha = args->alpha;// / (float)n_param;
    float beta = args->beta;
    float accs[BATCH_ACCEPTED_BLOCK] = {};

    uint64_t n_shift = n_param / 2;

    auto calc = [&](uint64_t i) {
            for (uint64_t b = 0; b < BATCH_ACCEPTED_BLOCK; ++b)
                output[i * BATCH_ACCEPTED_BLOCK + b] =
                    input[i * BATCH_ACCEPTED_BLOCK + b] * std::pow(accs[b] * alpha + k_param, -beta);
        };
    auto add_top = [&](uint64_t i) {
            for (uint64_t b = 0; b < BATCH_ACCEPTED_BLOCK; ++b)
                accs[b] += input[i * BATCH_ACCEPTED_BLOCK + b] * input[i * BATCH_ACCEPTED_BLOCK + b];
        };
    auto sub_bottom = [&](uint64_t i) {
            for (uint64_t b = 0; b < BATCH_ACCEPTED_BLOCK; ++b)
                accs[b] -= input[i * BATCH_ACCEPTED_BLOCK + b] * input[i * BATCH_ACCEPTED_BLOCK + b];
        };
    for (uint64_t i = 0; i < n_shift; ++i) {
        add_top(i);
    }
    for (uint64_t i = n_shift; i < n_param; ++i) {
        add_top(i);
        calc(i - n_shift);
    }
    for (uint64_t i = n_param; i < feats; ++i) {
        add_top(i);
        sub_bottom(i - n_param);
        calc(i - n_shift);
    }
    for (uint64_t i = feats; i < feats + n_shift; ++i) {
        sub_bottom(i - n_param);
        calc(i - n_shift);
    }
}

} // namespace

normalization_response_across_maps_f32_batch24n::normalization_response_across_maps_f32_batch24n(
    float alpha,
    float beta,
    float k_param,
    uint32_t norm_points,
    OutputDimensions out_dims,
    uint32_t batch_size,
    nn_device_internal* device)
    : alpha(alpha)
    , beta(beta)
    , k_param(k_param)
    , norm_points(norm_points)
    , batch_size(batch_size)
    , out_dims(out_dims)
    , device(device)
{
}

void normalization_response_across_maps_f32_batch24n::forward(
    const nn::workload_data<> *input,
    nn::workload_data<> *output)
{
    assert(input);
    assert(output);
    static const auto naive_wrapper = [](void* data_ptr) {
        auto data = static_cast<InputData*>(data_ptr);
        naive(data);
    };
    static const auto intrinsic_wrapper = [](void* data_ptr) {
        auto data = static_cast<InputData*>(data_ptr);
        intrinsic_version_for_0_75(data);
    };

    assert(input->get_length() == output->get_length());
    assert(input->parent->layout == output->parent->layout);

    uint64_t batch_blocks = output->get_length(NN_DATA_COORD_n);
    uint64_t width = output->get_length(NN_DATA_COORD_x);
    uint64_t height = output->get_length(NN_DATA_COORD_y);
    uint64_t feats = output->get_length(NN_DATA_COORD_z);
    assert(width == out_dims.width);
    assert(height == out_dims.height);
    assert(feats == out_dims.feats);

    std::vector<InputData> jobs_args(batch_blocks * height);
    std::vector<nn_multithreaded_request> jobs(jobs_args.size());


    uint64_t pixel_size = BATCH_ACCEPTED_BLOCK * feats;
    uint64_t input_width = input->parent->lengths.t[NN_DATA_COORD_x];
    uint64_t input_height = input->parent->lengths.t[NN_DATA_COORD_y];
    uint64_t output_width = output->parent->lengths.t[NN_DATA_COORD_x];
    uint64_t output_height = output->parent->lengths.t[NN_DATA_COORD_y];

    auto input_buffer = reinterpret_cast<float*>(input->parent->data_buffer);
    auto output_buffer = reinterpret_cast<float*>(output->parent->data_buffer);

    for (uint64_t b = 0u; b < batch_blocks; ++b)
    {
        for (uint64_t i = 0u; i < height; ++i)
        {
            auto curr_in_buffer = input_buffer + (b * input_height + i) * input_width * pixel_size;
            auto curr_out_buffer = output_buffer + (b * output_height + i) * output_width * pixel_size;
            auto job_index = b * height + i;

            jobs_args[job_index] = InputData(make<InputBuffer>(curr_in_buffer),
                                         make<OutputBuffer>(curr_out_buffer),
                                         make<KParam>(k_param),
                                         make<AlphaParam>(alpha),
                                         make<BetaParam>(beta),
                                         make<FeatsParam>(out_dims.feats),
                                         make<NParam>(norm_points),
                                         make<Width>(width));

            if (beta == 0.75f)
                jobs[job_index] = nn_multithreaded_request{intrinsic_wrapper, (void*)&jobs_args[job_index]};
            else
                jobs[job_index] = nn_multithreaded_request{naive_wrapper, (void*)&jobs_args[job_index]};
        }
    }
    const_cast<nn_device_internal*>(device)->thread_pool.push_job(jobs);
}

void normalization_response_across_maps_f32_batch24n::forward(
    const std::vector<const nn_workload_data_t *> &inputs,
    const std::vector<const nn_workload_data_t *> &parameters,
    const std::vector<nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 1);
    assert(parameters.size() == 0);
    assert(outputs.size() == 1);
    assert(inputs[0]);
    assert(outputs[0]);

    forward(nn::workload_data_cast<>(inputs[0]),
            nn::workload_data_cast<>(outputs[0]));
}

bool normalization_response_across_maps_f32_batch24n::validate_input(size_t, nn_workload_data_t*)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> normalization_response_across_maps_f32_batch24n::create_inputs(bool)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> normalization_response_across_maps_f32_batch24n::create_parameters(bool)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> normalization_response_across_maps_f32_batch24n::create_outputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::create(
                batch_size,
                out_dims.width,
                out_dims.height,
                out_dims.feats,
                BATCH_ACCEPTED_BLOCK,
                allocate_delta)};
}

} // namespace layer

