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
#include "device/common/nn_layer_parameters.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "layer_softmax_avx2_batch24n.h"
#include "device/cpu/api_internal/data_helper.h"
#include "nn_intrinsic_power.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <iostream>

namespace layer
{
typedef std::vector<nn_multithreaded_request> Jobs;

void softmax_f32_batch24n::forward(const nn::workload_data<nn::layout_f32> *input,
                          nn::workload_data<nn::layout_f32> *output)
{
    const uint32_t input_width = input->view_end.t[NN_DATA_COORD_z] - input->view_begin.t[NN_DATA_COORD_z] + 1;
    const uint32_t output_width = output->view_end.t[NN_DATA_COORD_x] - output->view_begin.t[NN_DATA_COORD_x] + 1;
    if (input_width != output_width)
        throw std::runtime_error("input width (" + std::to_string(input_width)
            + ") != output width (" + std::to_string(output_width) + ")");

    const auto input_batch_size = input->parent->lengths.t[NN_DATA_COORD_z] * BATCH_ACCEPTED_BLOCK;
    const auto output_batch_size = output->parent->lengths.t[NN_DATA_COORD_x] * BATCH_ACCEPTED_BLOCK;

    const auto output_view_start = output->view_begin.t[NN_DATA_COORD_x] * BATCH_ACCEPTED_BLOCK;
    const auto input_view_start = input->view_begin.t[NN_DATA_COORD_z] * BATCH_ACCEPTED_BLOCK;
    const auto width = input_width;

    const auto batches = batch_size / BATCH_ACCEPTED_BLOCK;

    __m256 maxes[BATCH_BLOCKS];
    __m256 sums[BATCH_BLOCKS];
    for (auto b = 0; b < batches; ++b)
    {
        auto input_buffer = static_cast<float*>(input->parent->data_buffer)
            + input_view_start + b * input_batch_size;
        auto output_buffer = static_cast<float*>(output->parent->data_buffer)
            + output_view_start + b * output_batch_size;

        {
            auto curr_input = input_buffer;
            for (auto j = 0u; j < BATCH_BLOCKS; ++j)
            {
                maxes[j] = _mm256_load_ps(curr_input);
                curr_input += BATCH_SHIFT;
            }
            for (auto i = 1u; i < width; ++i)
            {
                for (auto j = 0u; j < BATCH_BLOCKS; ++j)
                {
                    maxes[j] = _mm256_max_ps(maxes[j], _mm256_load_ps(curr_input));
                    curr_input += BATCH_SHIFT;
                }
            }
        }
        {
            for (auto j = 0u; j < BATCH_BLOCKS; ++j)
                sums[j] = _mm256_setzero_ps();
            using namespace nn::intrinsics;
            auto curr_input = input_buffer;
            auto curr_output = output_buffer;

            for (auto i = 0; i < width; ++i)
            {
                for (auto j = 0u; j < BATCH_BLOCKS; ++j)
                {
                    auto calculated = _inner_mm256_exp_ps(_mm256_sub_ps(_mm256_load_ps(curr_input), maxes[j]));
                    sums[j] = _mm256_add_ps(sums[j], calculated);
                    _mm256_store_ps(curr_output, calculated);
                    curr_input += BATCH_SHIFT;
                    curr_output += BATCH_SHIFT;
                }
            }
        }
        {
            for (auto j = 0u; j < BATCH_BLOCKS; ++j)
                sums[j] = _mm256_div_ps(_mm256_set1_ps(1.0f), sums[j]);

            auto curr_output = output_buffer;
            for (auto i = 0u; i < width; ++i)
            {
                for (auto j = 0u; j < BATCH_BLOCKS; ++j)
                {
                    _mm256_store_ps(curr_output, _mm256_mul_ps(_mm256_load_ps(curr_output), sums[j]));
                    curr_output += BATCH_SHIFT;
                }
            }
        }
    }
}

softmax_f32_batch24n::softmax_f32_batch24n(uint32_t num_features,
                                           uint32_t batch_size,
                                           nn_device_internal *device)
    : num_features(num_features),
      batch_size(batch_size),
      device(device),
      in_out_layout(nn::layout_t<nn::layout_nxyzpq_f32>::layout)
{}

void softmax_f32_batch24n::forward(const std::vector<const nn_workload_data_t *> &inputs,
                                   const std::vector<const nn_workload_data_t *> &parameters,
                                   const std::vector<nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 1);
    assert(parameters.size() == 0);
    assert(outputs.size() == 1);

    forward(reinterpret_cast<const nn::workload_data<nn::layout_f32> *>(inputs[0]),
        reinterpret_cast<nn::workload_data<nn::layout_f32> *>(outputs[0]));
}

std::vector<nn_workload_data_t *> softmax_f32_batch24n::create_inputs(bool allocate_delta)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> softmax_f32_batch24n::create_outputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::create(
                batch_size,
                num_features,
                1,
                1,
                BATCH_ACCEPTED_BLOCK,
                allocate_delta)};
}

bool softmax_f32_batch24n::validate_input(size_t index, nn_workload_data_t *data)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> softmax_f32_batch24n::create_parameters(bool)
{
    return {};
}

} // namespace layer

