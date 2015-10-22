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
#include "layer_convert_data_to_batch_block_layout.h"
#include "device/cpu/api_internal/data_helper.h"
#include "device/cpu/core/layer_convolution_avx2_forward.h"
#include "device/common/nn_allocate.h"
#include <iostream>
#include <atomic>

namespace layer
{

namespace
{
const uint32_t aux_block_size = 1024;
} //namespace

convert_from_zxyn_to_batch_block_format_nzxyn::convert_from_zxyn_to_batch_block_format_nzxyn(
    size_t batch_size,
    size_t input_size_x,
    size_t input_size_y,
    size_t input_size_z,
    nn_device_internal *device)
    : batch_size(batch_size)
    , input_size_x(input_size_x)
    , input_size_y(input_size_y)
    , input_size_z(input_size_z)
    , device(device)
    , aux_data(static_cast<float*>(nn_allocate_aligned(
        device->thread_pool.get_num_threads() * aux_block_size * BATCH_ACCEPTED_BLOCK * sizeof(float))))
{}

bool convert_from_zxyn_to_batch_block_format_nzxyn::validate_input(size_t index, nn_workload_data_t *data)
{
    throw std::logic_error("The method or operation is not implemented.");
}
std::vector<nn_workload_data_t *> convert_from_zxyn_to_batch_block_format_nzxyn::create_outputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::create(
        batch_size, input_size_x, input_size_y, input_size_z,  BATCH_ACCEPTED_BLOCK, allocate_delta)};
}
std::vector<nn_workload_data_t *> convert_from_zxyn_to_batch_block_format_nzxyn::create_inputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::create(
        device, input_size_x, input_size_y, input_size_z, batch_size, 0, 0, 0, 0, allocate_delta)};
}

void convert_from_zxyn_to_batch_block_format_nzxyn::forward(
    const std::vector<const nn_workload_data_t*> &inputs,
    const std::vector<const nn_workload_data_t*> &,
    const std::vector<nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == outputs.size());
    for (uint32_t k = 0u; k < inputs.size(); ++k)
    {
        auto input = nn::workload_data_cast<>(inputs[k]);
        auto output = nn::workload_data_cast<>(outputs[k]);
        assert(input->get_length(NN_DATA_COORD_p) == 1);
        assert(input->get_length(NN_DATA_COORD_q) == 1);
        assert(output->get_length(NN_DATA_COORD_q) == 1);
        assert(input->get_length(NN_DATA_COORD_y) == output->get_length(NN_DATA_COORD_y));
        assert(input->get_length(NN_DATA_COORD_x) == output->get_length(NN_DATA_COORD_x));
        assert(input->get_length(NN_DATA_COORD_z) == output->get_length(NN_DATA_COORD_z));
        assert(output->get_length(NN_DATA_COORD_p) == BATCH_ACCEPTED_BLOCK);
        assert(input->get_length(NN_DATA_COORD_n) ==
            output->get_length(NN_DATA_COORD_n) * output->get_length(NN_DATA_COORD_p));
        assert(input->get_length(NN_DATA_COORD_n) == batch_size);

        assert(output->get_length() == output->parent->lengths);

        const uint32_t width = input->get_length(NN_DATA_COORD_x);
        const uint32_t height = input->get_length(NN_DATA_COORD_y);
        const uint32_t feats = input->get_length(NN_DATA_COORD_z);
        assert(width == input_size_x);
        assert(height == input_size_y);
        assert(feats == input_size_z);
        const uint32_t pic_size = width * height * feats;

        auto input_buffer = reinterpret_cast<float*>(input->parent->data_buffer);
        auto output_buffer = reinterpret_cast<float*>(output->parent->data_buffer);
        assert(reinterpret_cast<size_t>(input_buffer) % BATCH_SHIFT == 0);
        assert(reinterpret_cast<size_t>(output_buffer) % BATCH_SHIFT == 0);

        const uint32_t job_pixels = aux_block_size / feats;
        const uint32_t job_buffer_size = job_pixels * feats * BATCH_ACCEPTED_BLOCK;
        const uint32_t num_of_jobs = (width * height + job_pixels - 1) / job_pixels;

        const uint32_t view_begin_x = input->view_begin.t[NN_DATA_COORD_x];
        const uint32_t view_begin_y = input->view_begin.t[NN_DATA_COORD_y];
        const uint32_t view_begin_z = input->view_begin.t[NN_DATA_COORD_z];
        const uint32_t parent_height = input->parent->lengths.t[NN_DATA_COORD_y];
        const uint32_t parent_width = input->parent->lengths.t[NN_DATA_COORD_x];
        const uint32_t parent_feats = input->parent->lengths.t[NN_DATA_COORD_z];
        const uint32_t parent_row_shift = (parent_width - width) * parent_feats;
        const uint32_t parent_pic_size = parent_width * parent_height * parent_feats;

        for (auto b = 0u; b < output->get_length(NN_DATA_COORD_n); ++b)
        {
            std::atomic<uint32_t> job_number(0u);
            auto partial_convert = [&](float* aux_buffer){
                    auto curr_job = std::atomic_fetch_add(&job_number, 1u);
                    if (curr_job >= num_of_jobs) return false;

                    const auto begin_pixel = curr_job * job_pixels;
                    const auto end_pixel = std::min((curr_job + 1) * job_pixels, width * height);
                    const auto curr_block_size = (end_pixel - begin_pixel) * feats;
                    
                    auto begin_x = curr_job * job_pixels % width;
                    auto begin_y = curr_job * job_pixels / width;
                    for (auto i = 0u; i < BATCH_ACCEPTED_BLOCK; ++i)
                    {
                        auto curr_aux_buffer = aux_buffer + i * curr_block_size;
                        auto curr_input = input_buffer + i * parent_pic_size
                                + ((view_begin_y + begin_y) * parent_width + view_begin_x + begin_x) * parent_feats
                                + view_begin_z;

                        auto x = begin_x;
                        for (auto j = begin_pixel; j < end_pixel; ++j)
                        {
                            std::memcpy(curr_aux_buffer, curr_input, feats * sizeof(float));
                            curr_aux_buffer += feats;
                            curr_input += parent_feats;
                            ++x;
                            if (x == width) {
                                curr_input += parent_row_shift;
                                x = 0;
                            }
                        }
                    }

                    const auto offset_reg = _mm256_set_epi32(
                        curr_block_size * 7, curr_block_size * 6, curr_block_size * 5, curr_block_size * 4,
                        curr_block_size * 3, curr_block_size * 2, curr_block_size * 1, curr_block_size * 0);

                    for (auto i = 0u; i < curr_block_size; ++i)
                    {
                        auto curr_output = output_buffer + (begin_pixel * feats + i) * BATCH_ACCEPTED_BLOCK;
                        auto curr_aux = aux_buffer + i;
                        auto data1 = _mm256_i32gather_ps(curr_aux, offset_reg, 4);
                        auto data2 = _mm256_i32gather_ps(curr_aux + BATCH_SHIFT * curr_block_size, offset_reg, 4);
                        auto data3 = _mm256_i32gather_ps(curr_aux + 2 * BATCH_SHIFT * curr_block_size, offset_reg, 4);

                        _mm256_store_ps(curr_output, data1);
                        _mm256_store_ps(curr_output + BATCH_SHIFT, data2);
                        _mm256_store_ps(curr_output + 2 * BATCH_SHIFT, data3);
                    }
                    return true;
                };
            auto job_wrapper = [&](void* ptr){ while (partial_convert(static_cast<float*>(ptr))); };
            std::vector<nn_multithreaded_request> jobs(num_of_jobs);
            for (auto i = 0u; i < num_of_jobs; ++i)
                jobs[i] = {job_wrapper, aux_data + i * job_buffer_size};
            device->thread_pool.push_job(jobs);

            input_buffer += BATCH_ACCEPTED_BLOCK * parent_pic_size;
            output_buffer += BATCH_ACCEPTED_BLOCK * pic_size;
        }
    }
}

} //namespace layer

