/*
Copyright (c) 2015, Intel Corporation

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

#include "helper_z_block_xyz_i16.h"
#include "device/cpu/api_internal/data_helper.h"

namespace int16_fixedpoint {
namespace helper_z_block_xyz_i16 {
    const nn_workload_data_layout_t &primitive_z_block_xyz_i16_base::in_out_layout = nn::workload_data<int16_t>::layout.pxyznq;

    nn::workload_data<int16_t> * primitive_z_block_xyz_i16_base::create_input_impl(const nn::data<int16_t, 4> &input,
        size_t padding_left,
        size_t padding_right,
        size_t padding_top,
        size_t padding_bottom,
        int view_offset_x,
        int view_offset_y,
        size_t view_size_x,
        size_t view_size_y) {
        nn_workload_data_coords_t size = {static_cast<uint32_t>(input.size[3]), // n
                                          static_cast<uint32_t>(input.size[1]), // x
                                          static_cast<uint32_t>(input.size[2]), // y
                                          static_cast<uint32_t>(input.size[0]), // z
                                          1,
                                          1};

        auto buffer = new nn::workload_data<int16_t>(
            NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, size, in_out_layout, (uint32_t)padding_left, (uint32_t)padding_right, (uint32_t)padding_top, (uint32_t)padding_bottom);

        if (padding_left == 0 && padding_right == 0 && padding_top == 0 && padding_bottom == 0)
        {
            memcpy(buffer->parent->data_buffer, input.buffer, buffer->parent->buffer_size);
        }
        else{
            for (size_t n = 0u; n < size.t[0]; ++n)
                for (size_t z = 0u; z < size.t[3]; ++z)
                    for (size_t y = 0u; y < size.t[2]; ++y)
                        for (size_t x = 0u; x < size.t[1]; ++x)
                            //        n, x, y, z, p, q  =          z, x, y, n
                            ((int16_t *)(buffer->parent->data_buffer))[z + size.t[3] * (x + size.t[1] * (y + size.t[2] * n))] = ((int16_t *)(input.buffer))[z + size.t[3] * (x + size.t[1] * (y + size.t[2] * n))];
        }

        buffer->view_begin.t[NN_DATA_COORD_x] += view_offset_x;
        buffer->view_begin.t[NN_DATA_COORD_y] += view_offset_y;
        buffer->view_end.t[NN_DATA_COORD_x] = (uint32_t)(buffer->view_begin.t[NN_DATA_COORD_x] + view_size_x - 1);
        buffer->view_end.t[NN_DATA_COORD_y] = (uint32_t)(buffer->view_begin.t[NN_DATA_COORD_y] + view_size_y - 1);

        return buffer;
    }

    std::vector<nn_workload_data_t *> primitive_z_block_xyz_i16_base::create_inputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, int16_t>::create(
            device, (uint32_t)get_required_input_w(), (uint32_t)get_required_input_h(), (uint32_t)input_size_z, (uint32_t)batch_size, (uint32_t)block_size, 0, 0, 0, 0) };
    }

    std::vector<nn_workload_data_t *> primitive_z_block_xyz_i16_base::create_outputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, int16_t>::create(
            device,
            (uint32_t)output_size_x,
            (uint32_t)output_size_y,
            (uint32_t)output_size_z,
            (uint32_t)batch_size,
            (uint32_t)block_size,
            (uint32_t)output_padding_left,
            (uint32_t)output_padding_right,
            (uint32_t)output_padding_top,
            (uint32_t)output_padding_bottom) };
    }

    nn::workload_data<int16_t> *primitive_z_block_xyz_i16_base::map_input(const nn::data<int16_t, 4> &input)
    {
        if (input.size[3] != batch_size)
            throw std::invalid_argument("input batch size doesn't match");

        if (input.size[0] != input_size_z)
            throw std::invalid_argument("number of input feature maps doesn't match");

        if (input.size[1] != get_required_input_w())
            throw std::invalid_argument("input width doesn't match");

        if (input.size[2] != get_required_input_h())
            throw std::invalid_argument("input height doesn't match");

        nn_workload_data_coords_t size = {static_cast<uint32_t>(input.size[3]), // n
                                          static_cast<uint32_t>(input.size[1]), // x
                                          static_cast<uint32_t>(input.size[2]), // y
                                          static_cast<uint32_t>(input.size[0]), // z
                                          1,
                                          1};

        auto buffer = new nn::workload_data<int16_t>(input.buffer, size, in_out_layout);

        return buffer;
    }

    std::vector<nn::workload_data<int16_t> *> split_workload_data_z(size_t partition_count,
                                                                       nn::workload_data<int16_t> &source)
    {
        const size_t window_size_z = source.get_length(NN_DATA_COORD_z) / partition_count;

        std::vector<nn::workload_data<int16_t> *> result(partition_count);
        for (size_t i = 0; i < partition_count; ++i) {
            nn_workload_data_coords_t view_start = {0, 0, 0, static_cast<uint32_t>(i * window_size_z), 0, 0};
            nn_workload_data_coords_t view_end = {source.get_length(0) - 1,
                                                  source.get_length(1) - 1,
                                                  source.get_length(2) - 1,
                                                  static_cast<uint32_t>((i + 1) * window_size_z) - 1,
                                                  source.get_length(4) - 1,
                                                  source.get_length(5) - 1};

            result[i] = new nn::workload_data<int16_t>(source, view_start, view_end);
        }

        return result;
    }

    std::vector<nn::workload_data<int16_t> *> primitive_z_block_xyz_i16_base::split_input_z(
        size_t partition_count, nn::workload_data<int16_t> &source) {

        assert(input_size_z * partition_count == source.get_length(NN_DATA_COORD_z));
        return split_workload_data_z(partition_count, source);
    }

    bool primitive_z_block_xyz_i16_base::validate_input(size_t index, nn_workload_data_t *data)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    void primitive_z_block_xyz_i16_base::copy_output(nn::data<int16_t, 4> &destination,
                                              const nn::workload_data<int16_t> &source) {
        throw std::logic_error("The method or operation is not implemented.");
    }
}
}
