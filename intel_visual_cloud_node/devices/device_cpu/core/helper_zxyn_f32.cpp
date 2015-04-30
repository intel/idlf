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

#include "helper_zxyn_f32.h"

namespace layer {
namespace helper_zxyn_f32 {

const nn_workload_data_layout_t primitive_zxyn_f32_base::in_out_layout = {
    {0, 0, 0, 0, 0, 0}, // tile in log2(size)
    {0, 0, 0, 0, 0, 0}, // alignment
    {NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q}, // ordering
    NN_DATATYPE_FLOAT};

nn::nn_workload_data_t<float> * primitive_zxyn_f32_base::create_input_impl(const nn::data<float, 4> &input,
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

    auto buffer = new nn::nn_workload_data_t<float>(
        size, in_out_layout, padding_left, padding_right, padding_top, padding_bottom);

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
                        ((float *)(buffer->parent->data_buffer))[z + size.t[3] * (x + size.t[1] * (y + size.t[2] * n))] = ((float *)(input.buffer))[z + size.t[3] * (x + size.t[1] * (y + size.t[2] * n))];
    }

    buffer->view_begin.t[NN_DATA_COORD_x] += view_offset_x;
    buffer->view_begin.t[NN_DATA_COORD_y] += view_offset_y;
    buffer->view_end.t[NN_DATA_COORD_x] = buffer->view_begin.t[NN_DATA_COORD_x] + view_size_x - 1;
    buffer->view_end.t[NN_DATA_COORD_y] = buffer->view_begin.t[NN_DATA_COORD_y] + view_size_y - 1;

    return buffer;
}

nn::nn_workload_data_t<float> *primitive_zxyn_f32_base::create_input(const nn::data<float, 4> &input) {
    if (input.size[3] != batch_size)
        throw std::invalid_argument("input batch size doesn't match");

    if (input.size[0] != input_size_z)
        throw std::invalid_argument("number of input feature maps doesn't match");

    if (input.size[1] != get_required_input_w())
        throw std::invalid_argument("input width doesn't match");

    if (input.size[2] != get_required_input_h())
        throw std::invalid_argument("input height doesn't match");

    return create_input_impl(input, 0, 0, 0, 0, 0, 0, input.size[1], input.size[2]);
}

nn::nn_workload_data_t<float> *primitive_zxyn_f32_base::map_input(const nn::data<float, 4> &input) {
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

    auto buffer = new nn::nn_workload_data_t<float>(input.buffer, size, in_out_layout);

    return buffer;
}

std::vector<nn::nn_workload_data_t<float> *> split_workload_data_z(size_t partition_count,
                                                                   nn::nn_workload_data_t<float> &source) {
    const size_t window_size_z = source.get_length(NN_DATA_COORD_z) / partition_count;

    std::vector<nn::nn_workload_data_t<float> *> result(partition_count);
    for (size_t i = 0; i < partition_count; ++i) {
        nn_workload_data_coords_t view_start = {0, 0, 0, static_cast<uint32_t>(i * window_size_z), 0, 0};
        nn_workload_data_coords_t view_end = {source.get_length(0) - 1,
                                              source.get_length(1) - 1,
                                              source.get_length(2) - 1,
                                              static_cast<uint32_t>((i + 1) * window_size_z) - 1,
                                              source.get_length(4) - 1,
                                              source.get_length(5) - 1};

        result[i] = new nn::nn_workload_data_t<float>(source, view_start, view_end);
    }

    return result;
}

std::vector<nn::nn_workload_data_t<float> *> primitive_zxyn_f32_base::split_input_z(
    size_t partition_count, nn::nn_workload_data_t<float> &source) {

    assert(input_size_z * partition_count == source.get_length(NN_DATA_COORD_z));
    return split_workload_data_z(partition_count, source);
}

bool primitive_zxyn_f32_base::validate_input(const nn::nn_workload_data_t<float> &input) {
    if (0 != memcmp(&input.parent->layout, &in_out_layout, sizeof(nn_workload_data_layout_t)))
        return false;

    const auto view_size = input.get_length();

    if (view_size.t[NN_DATA_COORD_n] != batch_size)
        return false;

    if (view_size.t[NN_DATA_COORD_x] != get_required_input_w())
        return false;

    if (view_size.t[NN_DATA_COORD_y] != get_required_input_h())
        return false;

    if (view_size.t[NN_DATA_COORD_z] != input_size_z)
        return false;

    return true;
}

nn::nn_workload_data_t<float> *primitive_zxyn_f32_base::create_output() { return create_output(0, 0, 0, 0); }

nn::nn_workload_data_t<float> *primitive_zxyn_f32_base::create_output(size_t padding_left,
                                                                      size_t padding_right,
                                                                      size_t padding_top,
                                                                      size_t padding_bottom) {
    nn_workload_data_coords_t size = {static_cast<uint32_t>(batch_size),
                                      static_cast<uint32_t>(output_size_x),
                                      static_cast<uint32_t>(output_size_y),
                                      static_cast<uint32_t>(output_size_z),
                                      1,
                                      1};

    return new nn::nn_workload_data_t<float>(
        size, in_out_layout, padding_left, padding_right, padding_top, padding_bottom);
}

std::vector<nn::nn_workload_data_t<float> *> primitive_zxyn_f32_base::create_output_vector_z(
    size_t partition_count, nn::nn_workload_data_t<float> *&merged_output) {
    nn_workload_data_coords_t size = {static_cast<uint32_t>(batch_size),
                                      static_cast<uint32_t>(output_size_x),
                                      static_cast<uint32_t>(output_size_y),
                                      static_cast<uint32_t>(output_size_z * partition_count),
                                      1,
                                      1};
    merged_output = new nn::nn_workload_data_t<float>(size, in_out_layout);
    return split_workload_data_z(partition_count, *merged_output);
}

void primitive_zxyn_f32_base::copy_output(nn::data<float, 4> &destination,
                                          const nn::nn_workload_data_t<float> &source) {
    assert(destination.size[0] == output_size_z);
    assert(destination.size[1] == output_size_x);
    assert(destination.size[2] == output_size_y);
    assert(destination.size[3] == batch_size);

    auto source_length = source.get_length();
    assert(source_length.t[NN_DATA_COORD_z] == output_size_z);
    assert(source_length.t[NN_DATA_COORD_x] == output_size_x);
    assert(source_length.t[NN_DATA_COORD_y] == output_size_y);
    assert(source_length.t[NN_DATA_COORD_n] == batch_size);

    assert(source_length.t[NN_DATA_COORD_p] == 1);
    assert(source_length.t[NN_DATA_COORD_q] == 1);

    assert(memcmp(&source.parent->layout, &in_out_layout, sizeof(nn_workload_data_layout_t)) == 0);

    if (output_size_z * output_size_x * output_size_y * batch_size ==
        source.parent->buffer_size / source.parent->data_type_size) {
        memcpy(destination.buffer, source.parent->data_buffer, source.parent->buffer_size);
    }else{
        size_t stride_x = source.parent->lengths.t[NN_DATA_COORD_z],
               stride_y = stride_x * source.parent->lengths.t[NN_DATA_COORD_x],
               stride_n = stride_y * source.parent->lengths.t[NN_DATA_COORD_y];

        for (size_t n = 0; n < source_length.t[NN_DATA_COORD_n]; ++n)
            for (size_t y = 0; y < source_length.t[NN_DATA_COORD_y]; ++y)
                for (size_t x = 0; x < source_length.t[NN_DATA_COORD_x]; ++x) {
                    void *source_ptr =
                        reinterpret_cast<char *>(source.parent->data_buffer) +
                        source.parent->data_type_size *
                            (source.view_begin.t[NN_DATA_COORD_z] + stride_x * (source.view_begin.t[NN_DATA_COORD_x] + x) +
                             stride_y * (source.view_begin.t[NN_DATA_COORD_y] + y) +
                             stride_n * (source.view_begin.t[NN_DATA_COORD_n] + n));
                    memcpy(&destination.at(0, x, y, n),
                           source_ptr,
                           source.parent->data_type_size * source_length.t[NN_DATA_COORD_z]);
                }
    }
}

nn_opaque_data_t *NN_API_CALL_CONVENTION
create_input(nn_primitive_handle_t handle, const nn_data_t *input, NN_API_STATUS *status) {
    auto primitive = static_cast<primitive_zxyn_f32_base *>(handle);
    auto result = primitive->create_input(*nn::data_cast<float, 4>(input));
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

nn_opaque_data_t *NN_API_CALL_CONVENTION
map_input(nn_primitive_handle_t handle, const nn_data_t *input, NN_API_STATUS *status) {
    auto primitive = static_cast<primitive_zxyn_f32_base *>(handle);
    auto result = primitive->map_input(*nn::data_cast<float, 4>(input));
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

NN_API_STATUS NN_API_CALL_CONVENTION split_input_z(nn_primitive_handle_t handle,
                                                   const size_t partition_count,
                                                   nn_opaque_data_t *parts[],
                                                   const nn_opaque_data_t *source) {
    auto primitive = static_cast<primitive_zxyn_f32_base *>(handle);

    auto part_vector = primitive->split_input_z(
        partition_count, *reinterpret_cast<nn::nn_workload_data_t<float> *>(const_cast<nn_opaque_data_t *>(source)));
    for (size_t i = 0; i < partition_count; ++i)
        parts[i] = reinterpret_cast<nn_opaque_data_t *>(part_vector[i]);

    return NN_API_STATUS_OK;
}

nn_opaque_data_t *NN_API_CALL_CONVENTION create_output_with_padding(nn_primitive_handle_t handle,
                                                                    size_t left_padding,
                                                                    size_t right_padding,
                                                                    size_t top_padding,
                                                                    size_t bottom_padding,
                                                                    NN_API_STATUS *status) {
    auto primitive = static_cast<primitive_zxyn_f32_base *>(handle);
    auto result = primitive->create_output(left_padding, right_padding, top_padding, bottom_padding);
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

nn_opaque_data_t *NN_API_CALL_CONVENTION create_output(nn_primitive_handle_t handle, NN_API_STATUS *status) {
    auto primitive = static_cast<primitive_zxyn_f32_base *>(handle);
    auto result = primitive->create_output();
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

nn_opaque_data_t *NN_API_CALL_CONVENTION create_output_vector_z(nn_primitive_handle_t handle,
                                                                const size_t partition_count,
                                                                nn_opaque_data_t *parts[],
                                                                NN_API_STATUS *status) {
    auto primitive = static_cast<primitive_zxyn_f32_base *>(handle);

    nn::nn_workload_data_t<float> *result;
    auto part_vector = primitive->create_output_vector_z(partition_count, result);
    for (size_t i = 0; i < partition_count; ++i)
        parts[i] = reinterpret_cast<nn_opaque_data_t *>(part_vector[i]);

    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

int NN_API_CALL_CONVENTION
validate_input(nn_primitive_handle_t handle, /* primitive handle */
               nn_opaque_data_t *opaque_data /* internal data storage handle to validate */) {
    auto primitive = static_cast<primitive_zxyn_f32_base *>(handle);
    return primitive->validate_input(*reinterpret_cast<nn::nn_workload_data_t<float> *>(opaque_data));
}

nn_event_t NN_API_CALL_CONVENTION copy_output_async(nn_primitive_handle_t handle,
                                                    nn_data_t *output,
                                                    nn_opaque_data_t *output_buffer,
                                                    size_t dependencies_count,
                                                    nn_event_t *dependencies,
                                                    NN_API_STATUS *status) {
    auto primitive = static_cast<primitive_zxyn_f32_base *>(handle);
    primitive->copy_output(*nn::data_cast<float, 4>(output),
                           *reinterpret_cast<nn::nn_workload_data_t<float> *>(output_buffer));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}
}
}
