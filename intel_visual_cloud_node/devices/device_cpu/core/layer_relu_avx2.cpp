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

#include "layer_relu_avx2.h"

#include <cassert>

namespace layer {
relu_f32 *relu_f32::create(
    size_t image_size_x, size_t image_size_y, size_t image_size_z, size_t batch_size, nn_device_t *device) {
    return new relu_f32(
        image_size_x, image_size_y, image_size_z, batch_size, reinterpret_cast<nn_device_internal *>(device));
}

void relu_f32::forward(const nn::nn_workload_data_t<float> *input, nn::nn_workload_data_t<float> *output) {
    const size_t z_input_size = input->parent->lengths.t[NN_DATA_COORD_z],
                 z_input_view_size = input->get_length().t[NN_DATA_COORD_z], z_input_stride = 1,
                 z_input_view_start = input->view_begin.t[NN_DATA_COORD_z];
    const size_t x_input_size = input->parent->lengths.t[NN_DATA_COORD_x],
                 x_input_view_size = input->get_length().t[NN_DATA_COORD_x],
                 x_input_stride = z_input_size * z_input_stride,
                 x_input_view_start = input->view_begin.t[NN_DATA_COORD_x];
    const size_t y_input_size = input->parent->lengths.t[NN_DATA_COORD_y],
                 y_input_view_size = input->get_length().t[NN_DATA_COORD_y],
                 y_input_stride = x_input_size * x_input_stride,
                 y_input_view_start = input->view_begin.t[NN_DATA_COORD_y];
    const size_t batch_input_size = input->parent->lengths.t[NN_DATA_COORD_n],
                 batch_input_view_size = input->get_length().t[NN_DATA_COORD_n],
                 batch_input_stride = y_input_size * y_input_stride,
                 batch_input_view_start = input->view_begin.t[NN_DATA_COORD_n];

    const size_t z_output_size = output->parent->lengths.t[NN_DATA_COORD_z],
                 z_output_view_size = output->get_length().t[NN_DATA_COORD_z], z_output_stride = 1,
                 z_output_view_start = output->view_begin.t[NN_DATA_COORD_z];
    const size_t x_output_size = output->parent->lengths.t[NN_DATA_COORD_x],
                 x_output_view_size = output->get_length().t[NN_DATA_COORD_x],
                 x_output_stride = z_output_size * z_output_stride,
                 x_output_view_start = output->view_begin.t[NN_DATA_COORD_x];
    const size_t y_output_size = output->parent->lengths.t[NN_DATA_COORD_y],
                 y_output_view_size = output->get_length().t[NN_DATA_COORD_y],
                 y_output_stride = x_output_size * x_output_stride,
                 y_output_view_start = output->view_begin.t[NN_DATA_COORD_y];
    const size_t batch_output_size = output->parent->lengths.t[NN_DATA_COORD_n],
                 batch_output_view_size = output->get_length().t[NN_DATA_COORD_n],
                 batch_output_stride = y_output_size * y_output_stride,
                 batch_output_view_start = output->view_begin.t[NN_DATA_COORD_n];

    assert(z_output_view_size == z_input_view_size);
    assert(x_output_view_size == x_input_view_size);
    assert(y_output_view_size == y_input_view_size);
    assert(batch_output_view_size == batch_input_view_size);

    const size_t simd_width = sizeof(__m256) / sizeof(float), block_size = simd_width * 4;
    auto input_buf = reinterpret_cast<float *>(input->parent->data_buffer) +
                     batch_input_view_start * batch_input_stride + y_input_view_start * y_input_stride +
                     x_input_view_start * x_input_stride + z_input_view_start * z_input_stride;
    auto output_buf = reinterpret_cast<float *>(output->parent->data_buffer) +
                      batch_output_view_start * batch_output_stride + y_output_view_start * y_output_stride +
                      x_output_view_start * x_output_stride + z_output_view_start * z_output_stride;

    const __m256 zero = _mm256_setzero_ps();

    for (size_t batch = 0; batch < batch_output_view_size; ++batch)
        for (size_t y = 0; y < y_output_view_size; ++y)
            for (size_t x = 0; x < x_output_view_size; ++x) {
                for (size_t z = 0; z <= z_output_view_size - block_size; z += block_size) {
#pragma unroll(block_size / simd_width)
                    for (size_t z_simd_block = 0; z_simd_block < block_size / simd_width; ++z_simd_block)
                        *reinterpret_cast<__m256 *>(output_buf + batch * batch_output_stride + y * y_output_stride +
                                                    x * x_output_stride + z + z_simd_block * simd_width) =
                            _mm256_max_ps(*reinterpret_cast<__m256 *>(input_buf + batch * batch_input_stride +
                                                                      y * y_input_stride + x * x_input_stride + z +
                                                                      z_simd_block * simd_width),
                                          zero);
                }
                for (size_t z = z_output_view_size - block_size; z < z_output_view_size; ++z) {
                    auto value = input_buf[batch * batch_input_stride + y * y_input_stride + x * x_input_stride + z];
                    output_buf[batch * batch_output_stride + y * y_output_stride + x * x_output_stride + z] =
                        value > 0 ? value : 0;
                }
            }
}

relu_f32::relu_f32(
    size_t image_size_x, size_t image_size_y, size_t image_size_z, size_t batch_size, nn_device_internal *device)
    : primitive_zxyn_f32_base(batch_size, image_size_z, image_size_x, image_size_y, image_size_z, device) {}

size_t relu_f32::get_required_input_w() { return output_size_x; }

size_t relu_f32::get_required_input_h() { return output_size_y; }

namespace relu_f32_impl {
nn_event_t NN_API_CALL_CONVENTION forward_async(nn_primitive_handle_t handle,
                                          nn_opaque_data_t *input,
                                          nn_opaque_data_t *output,
                                          size_t dependencies_count,
                                          nn_event_t *dependencies,
                                          NN_API_STATUS *status) {
    auto primitive = static_cast<layer::relu_f32 *>(handle);
    primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(input),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(output));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}

nn_primitive_handle_t NN_API_CALL_CONVENTION create(nn_device_t *device,
                                                    size_t image_size_x,
                                                    size_t image_size_y,
                                                    size_t image_size_z,
                                                    size_t batch_size,
                                                    NN_API_STATUS *status) {
    SET_STATUS(NN_API_STATUS_OK);
    return layer::relu_f32::create(image_size_x, image_size_y, image_size_z, batch_size, device);
}
}
}

nn_primitives_relu_f32_0_t nn_primitives_relu_f32_0{
    layer::relu_f32_impl::create,
    layer::helper_zxyn_f32::create_input,
    layer::helper_zxyn_f32::validate_input,
    layer::helper_zxyn_f32::create_output,
    nullptr, // create_view
    layer::relu_f32_impl::forward_async,
    layer::helper_zxyn_f32::copy_output_async
};
