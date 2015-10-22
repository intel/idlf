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
#include "layer_convert_data_from_batch_block_layout.h"
#include "device/cpu/api_internal/data_helper.h"
#include "device/cpu/core/layer_convolution_avx2_forward.h"

namespace layer
{

//convert_from_batch_block_format_to_zxyn
bool convert_from_batch_block_format_to_zxyn::validate_input(size_t index, nn_workload_data_t *data)
{
    throw std::logic_error("The method or operation is not implemented.");
}
std::vector<nn_workload_data_t *> convert_from_batch_block_format_to_zxyn::create_outputs(bool allocate_delta)
{
    return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::create(
        device, input_size_x, input_size_y, input_size_z, batch_size, 0, 0, 0, 0, allocate_delta)};
}
std::vector<nn_workload_data_t *> convert_from_batch_block_format_to_zxyn::create_inputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::create(
        batch_size, input_size_x, input_size_y, input_size_z,  BATCH_ACCEPTED_BLOCK, allocate_delta)};
}

void convert_from_batch_block_format_to_zxyn::forward(
    const std::vector<const nn_workload_data_t*> &inputs,
    const std::vector<const nn_workload_data_t*> &,
    const std::vector<nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == outputs.size());
    for (uint32_t k = 0u; k < inputs.size(); ++k)
    {
        auto input = nn::workload_data_cast<>(inputs[k]);
        auto output = nn::workload_data_cast<>(outputs[k]);
        assert(input->get_length(NN_DATA_COORD_y) == output->get_length(NN_DATA_COORD_y));
        assert(input->get_length(NN_DATA_COORD_x) == output->get_length(NN_DATA_COORD_x));
        assert(input->get_length(NN_DATA_COORD_z) == output->get_length(NN_DATA_COORD_z));
        assert(input->get_length(NN_DATA_COORD_n) * input->get_length(NN_DATA_COORD_p)
                == output->get_length(NN_DATA_COORD_n));
        assert(input->get_length(NN_DATA_COORD_q) == 1);
        assert(output->get_length(NN_DATA_COORD_p) == 1);
        assert(output->get_length(NN_DATA_COORD_q) == 1);

        if (input->get_length() != input->parent->lengths)
            throw std::runtime_error("view on input in convert_from_batch_block_format_to_zxyn");
        if (output->get_length() != output->parent->lengths)
            throw std::runtime_error("view on output in convert_from_batch_block_format_to_zxyn");

        const uint64_t width = input->get_length(NN_DATA_COORD_x);
        const uint64_t height = input->get_length(NN_DATA_COORD_y);
        const uint64_t feats = input->get_length(NN_DATA_COORD_z);
        const uint64_t pic_size = width * height * feats;

        const __m256i offset_reg = _mm256_set_epi32(
            BATCH_ACCEPTED_BLOCK * 7, BATCH_ACCEPTED_BLOCK * 6, BATCH_ACCEPTED_BLOCK * 5, BATCH_ACCEPTED_BLOCK * 4,
            BATCH_ACCEPTED_BLOCK * 3, BATCH_ACCEPTED_BLOCK * 2, BATCH_ACCEPTED_BLOCK * 1, BATCH_ACCEPTED_BLOCK * 0);

        auto input_buffer = reinterpret_cast<float*>(input->parent->data_buffer);
        auto output_buffer = reinterpret_cast<float*>(output->parent->data_buffer);
        for (uint32_t i = 0u; i < input->get_length(NN_DATA_COORD_n); ++i)
        {
            for (uint32_t j = 0u; j < pic_size / 8; ++j)
            {
                for (uint32_t b = 0u; b < BATCH_ACCEPTED_BLOCK; ++b)
                {
                    auto acc = _mm256_i32gather_ps(input_buffer + j * 8 * BATCH_ACCEPTED_BLOCK + b, offset_reg, 4);
                    _mm256_store_ps(output_buffer + b * pic_size + j * 8, acc);
                }
            }
            for (uint32_t j = (pic_size / 8) * 8; j < pic_size; ++j)
                for (uint32_t b = 0u; b < BATCH_ACCEPTED_BLOCK; ++b)
                    output_buffer[b * pic_size + j] = input_buffer[j * BATCH_ACCEPTED_BLOCK + b];

            output_buffer += pic_size * BATCH_ACCEPTED_BLOCK;
            input_buffer += pic_size * BATCH_ACCEPTED_BLOCK;
        }
    }
}

} //namespace layer

