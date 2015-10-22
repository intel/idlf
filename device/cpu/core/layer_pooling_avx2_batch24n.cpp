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
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "layer_pooling_avx2_batch24n.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <map>
#include <tuple>
#include "device/cpu/api_internal/data_helper.h"

namespace layer
{
max_pooling_f32_batch24n::max_pooling_f32_batch24n(PoolingInfo info,
                                                   OutputDimensions out_dims,
                                                   uint32_t batch_size,
                                                   nn_device_internal *device)
    : info(info)
    , out_dims(out_dims)
    , batch_size(batch_size)
    , device(device)
{}

void max_pooling_f32_batch24n::forward(
    const std::vector<const nn_workload_data_t *> &inputs,
    const std::vector<const nn_workload_data_t *> &parameters,
    const std::vector<nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 1);
    assert(parameters.size() == 0);
    assert(outputs.size() == 1);

    forward(reinterpret_cast<const nn::workload_data<> *>(inputs[0]),
            nn::workload_data_cast<>(outputs[0]));
}

std::vector<nn_workload_data_t *> max_pooling_f32_batch24n::create_parameters(bool)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> max_pooling_f32_batch24n::create_inputs(bool)
{
    throw std::logic_error("unimplemented");
}

std::vector<nn_workload_data_t *> max_pooling_f32_batch24n::create_outputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::create(
                batch_size,
                out_dims.width,
                out_dims.height,
                out_dims.feats,
                BATCH_ACCEPTED_BLOCK,
                allocate_delta)};
}

bool max_pooling_f32_batch24n::validate_input(size_t index, nn_workload_data_t*)
{
    throw std::logic_error("unimplemented");
}

namespace
{

void naive(float* input, float* output, InputDimensions input_dims, PoolingInfo info)
{
    const uint64_t BATCH_SHIFT = 8;
    uint64_t batch_accs = BATCH_ACCEPTED_BLOCK / BATCH_SHIFT;
    uint64_t in_pixel_size = input_dims.feats * BATCH_ACCEPTED_BLOCK;

    
    auto feats = input_dims.feats;
    memcpy(output, input, input_dims.feats * BATCH_ACCEPTED_BLOCK * sizeof(float));
    for (uint64_t i = 0; i < info.dims.height; ++i)
    {
        for (uint64_t j = 0; j < info.dims.width; ++j)
        {
            if ((i == 0) and (j == 0)) continue;
            auto curr_output = output;
            auto curr_input = input + (i * input_dims.width + j) * in_pixel_size;
            for (uint64_t feat = 0u; feat < input_dims.feats; ++feat)
            {
                for (uint64_t acc_index = 0; acc_index < batch_accs; ++acc_index)
                {
                    auto result = _mm256_max_ps(_mm256_load_ps(curr_output), _mm256_load_ps(curr_input));
                    _mm256_store_ps(curr_output, result);

                    curr_output += BATCH_SHIFT;
                    curr_input += BATCH_SHIFT;
                }
            }
        }
    }
}

} //namespace

void max_pooling_f32_batch24n::forward(
    const nn::workload_data<> *input,
    nn::workload_data<> *output)
{
    uint64_t width = output->get_length(NN_DATA_COORD_x);
    uint64_t height = output->get_length(NN_DATA_COORD_y);
    uint64_t batch_blocks = output->get_length(NN_DATA_COORD_n);
    assert(width == out_dims.width);
    assert(height == out_dims.height);
    assert(output->get_length(NN_DATA_COORD_z) == out_dims.feats);
    assert(input->get_length(NN_DATA_COORD_z) == out_dims.feats);
    assert(output->get_length(NN_DATA_COORD_p) == BATCH_ACCEPTED_BLOCK);

    auto input_buffer = reinterpret_cast<float*>(input->parent->data_buffer);
    auto output_buffer = reinterpret_cast<float*>(output->parent->data_buffer);

    if (input->get_length() != input->parent->lengths)
        throw std::runtime_error("view on input in max pooling batch 24n");
    if (output->get_length() != output->parent->lengths)
        throw std::runtime_error("view on output in max pooling batch 24n");

    uint64_t input_width = input->parent->lengths.t[NN_DATA_COORD_x];
    uint64_t input_height = input->parent->lengths.t[NN_DATA_COORD_y];
    uint64_t input_pixel_size = BATCH_ACCEPTED_BLOCK * out_dims.feats;
    uint64_t output_width = output->parent->lengths.t[NN_DATA_COORD_x];
    uint64_t output_height = output->parent->lengths.t[NN_DATA_COORD_y];
    uint64_t output_pixel_size = BATCH_ACCEPTED_BLOCK * output->parent->lengths.t[NN_DATA_COORD_z];

    uint64_t pool_b = 0u;
    uint64_t pool_out_row = 0u;
    uint64_t pool_out_col = 0u;

    InputDimensions in_dims = {make<InputHeight>(input_height),
                               make<InputWidth>(input_width),
                               make<InputFeats>(out_dims.feats)};


    std::mutex mtx;
    auto pull_job = [&]{
            uint64_t b = 0u;
            uint64_t out_row = 0u;
            uint64_t out_col = 0u;
            {
                std::unique_lock<std::mutex> guard(mtx);
                out_row = pool_out_row;
                out_col = pool_out_col;
                b = pool_b;
                if (out_row >= output_height) return false;

                ++pool_b;
                if (pool_b >= batch_blocks) {
                    pool_out_col++;
                    pool_b = 0;
                }
                if (pool_out_col >= output_width) {
                    pool_b = 0;
                    pool_out_col = 0;
                    ++pool_out_row;
                }
            }

            auto curr_out_buffer = output_buffer + ((b * output_height + out_row) * output_width + out_col) * output_pixel_size;
            auto in_base_row = out_row * info.stride.rows;
            auto in_base_col = out_col * info.stride.cols;
            auto curr_in_base_buffer = input_buffer + ((b * input_height + in_base_row) * input_width + in_base_col) * input_pixel_size;

            naive(curr_in_base_buffer, curr_out_buffer, in_dims, info);
            return true;
        };
    auto thread_job = [&](void*){ while(pull_job()); };

    std::vector<nn_multithreaded_request> jobs(device->thread_pool.get_num_threads(), {thread_job, nullptr});
    device->thread_pool.push_job(jobs);
}

} //namespace layer

