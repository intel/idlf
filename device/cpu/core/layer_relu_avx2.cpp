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
#include "device/cpu/api_internal/data_helper.h"

#include <cassert>

const uint32_t C_simd_width = sizeof(__m256) / sizeof(float);
const uint32_t C_num_max_acc = 15;

namespace layer {


template<uint32_t T_num_acc>
void forward_inner_macro(
    float *input, 
    float *output)
{
    __m256 acc[T_num_acc];
    __m256 zero = _mm256_setzero_ps();

    #pragma unroll (T_num_acc)
    for(uint32_t acc_id = 0; acc_id < T_num_acc; ++acc_id)
        acc[acc_id] = _mm256_max_ps(zero, _mm256_load_ps(input + acc_id * C_simd_width));

    #pragma unroll (T_num_acc)
    for(uint32_t acc_id = 0; acc_id < T_num_acc; ++acc_id)
        _mm256_store_ps(output + acc_id * C_simd_width, acc[acc_id]);
}

void relu_f32::forward(const nn::workload_data<float> *input, nn::workload_data<float> *output) {
    const auto input_buffer = reinterpret_cast<float*>(input->parent->data_buffer);
    const auto output_buffer = reinterpret_cast<float*>(output->parent->data_buffer);

    const auto input_depth = input->parent->lengths.t[NN_DATA_COORD_z];
    const auto input_width = input->parent->lengths.t[NN_DATA_COORD_x];
    const auto input_height = input->parent->lengths.t[NN_DATA_COORD_y];
    const auto output_depth = output->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_width = output->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_height = output->parent->lengths.t[NN_DATA_COORD_y];

    const auto input_size = input_depth * input_width * input_height;
    const auto output_size = output_depth * output_width * output_height;


    if(output->get_length(NN_DATA_COORD_z) != 1)
    {
        // Three dimensions used.
        assert(output->get_length(NN_DATA_COORD_z) % C_simd_width == 0);

        for (uint32_t n_out = output->view_begin.t[NN_DATA_COORD_n], n_in = input->view_begin.t[NN_DATA_COORD_n]; 
             n_out <= output->view_end.t[NN_DATA_COORD_n]; 
             ++n_out, ++n_in)
            for (uint32_t x_out = output->view_begin.t[NN_DATA_COORD_x], x_in = input->view_begin.t[NN_DATA_COORD_x]; 
                 x_out <= output->view_end.t[NN_DATA_COORD_x]; 
                 ++x_out, ++x_in)
                for (uint32_t y_out = output->view_begin.t[NN_DATA_COORD_y], y_in = input->view_begin.t[NN_DATA_COORD_y]; 
                     y_out <= output->view_end.t[NN_DATA_COORD_y]; 
                     ++y_out, ++y_in)
                {
                    const auto in_ptr_offset =   input->view_begin.t[NN_DATA_COORD_z]
                                               + x_in * input_depth
                                               + y_in * input_depth * input_width
                                               + n_in * input_size;

                    const auto out_ptr_offset =   output->view_begin.t[NN_DATA_COORD_z]
                                                + x_out * output_depth
                                                + y_out * output_depth * output_width
                                                + n_out * output_size;

                    auto input_ptr = input_buffer + in_ptr_offset;
                    auto output_ptr = output_buffer + out_ptr_offset;

                    const auto full_passes = output->get_length(NN_DATA_COORD_z) / (C_num_max_acc * C_simd_width);
                    const auto partial_pass_size = output->get_length(NN_DATA_COORD_z) % (C_num_max_acc * C_simd_width) / C_simd_width;

                    for(uint32_t pass = 0; pass < full_passes; ++pass)
                    {
                        #pragma forceinline recursive
                        forward_inner_macro<C_num_max_acc>(input_ptr, output_ptr);

                        input_ptr += C_num_max_acc * C_simd_width;
                        output_ptr += C_num_max_acc * C_simd_width;
                    }

                    switch(partial_pass_size)
                    {
                        case  1: forward_inner_macro< 1>(input_ptr, output_ptr); break;
                        case  2: forward_inner_macro< 2>(input_ptr, output_ptr); break;
                        case  3: forward_inner_macro< 3>(input_ptr, output_ptr); break;
                        case  4: forward_inner_macro< 4>(input_ptr, output_ptr); break;
                        case  5: forward_inner_macro< 5>(input_ptr, output_ptr); break;
                        case  6: forward_inner_macro< 6>(input_ptr, output_ptr); break;
                        case  7: forward_inner_macro< 7>(input_ptr, output_ptr); break;
                        case  8: forward_inner_macro< 8>(input_ptr, output_ptr); break;
                        case  9: forward_inner_macro< 9>(input_ptr, output_ptr); break;
                        case 10: forward_inner_macro<10>(input_ptr, output_ptr); break;
                        case 11: forward_inner_macro<11>(input_ptr, output_ptr); break;
                        case 12: forward_inner_macro<12>(input_ptr, output_ptr); break;
                        case 13: forward_inner_macro<13>(input_ptr, output_ptr); break;
                        case 14: forward_inner_macro<14>(input_ptr, output_ptr); break;
                    }
                }
    }
    else
    {
        // One dimension used. No view supported.
        auto input_ptr = input_buffer;
        auto output_ptr = output_buffer;

        auto full_passes = output->parent->buffer_size / 4 / (C_num_max_acc * C_simd_width);
        auto partial_pass_size = output->parent->buffer_size / 4 % (C_num_max_acc * C_simd_width) / C_simd_width;

        for(uint32_t pass = 0; pass < full_passes; ++pass)
        {
            #pragma forceinline recursive
            forward_inner_macro<C_num_max_acc>(input_ptr, output_ptr);

            input_ptr += C_num_max_acc * C_simd_width;
            output_ptr += C_num_max_acc * C_simd_width;
        }

        switch(partial_pass_size)
        {
            case  1: forward_inner_macro< 1>(input_ptr, output_ptr); break;
            case  2: forward_inner_macro< 2>(input_ptr, output_ptr); break;
            case  3: forward_inner_macro< 3>(input_ptr, output_ptr); break;
            case  4: forward_inner_macro< 4>(input_ptr, output_ptr); break;
            case  5: forward_inner_macro< 5>(input_ptr, output_ptr); break;
            case  6: forward_inner_macro< 6>(input_ptr, output_ptr); break;
            case  7: forward_inner_macro< 7>(input_ptr, output_ptr); break;
            case  8: forward_inner_macro< 8>(input_ptr, output_ptr); break;
            case  9: forward_inner_macro< 9>(input_ptr, output_ptr); break;
            case 10: forward_inner_macro<10>(input_ptr, output_ptr); break;
            case 11: forward_inner_macro<11>(input_ptr, output_ptr); break;
            case 12: forward_inner_macro<12>(input_ptr, output_ptr); break;
            case 13: forward_inner_macro<13>(input_ptr, output_ptr); break;
            case 14: forward_inner_macro<14>(input_ptr, output_ptr); break;
        }
    }
}

void relu_f32::forward(const std::vector<const nn_workload_data_t *> &inputs,
                       const std::vector<const nn_workload_data_t *> &parameters,
                       const std::vector<nn_workload_data_t *> &outputs) {
    assert(inputs.size() == 1);
    assert(parameters.size() == 0);
    assert(outputs.size() == 1);

    forward(reinterpret_cast<const nn::workload_data<float> *>(inputs[0]),
            reinterpret_cast<nn::workload_data<float> *>(outputs[0]));
}

template<uint32_t T_num_acc>
void backward_inner_macro(
    float *forward_input, 
    float *backward_input,
    float *backward_output)
{
    __m256 acc[T_num_acc];
    __m256 zero = _mm256_setzero_ps();

    #pragma unroll (T_num_acc)
    for(uint32_t acc_id = 0; acc_id < T_num_acc; ++acc_id)
        acc[acc_id] = _mm256_cmp_ps(zero, _mm256_load_ps(forward_input + acc_id * C_simd_width), _CMP_LT_OQ);

    #pragma unroll (T_num_acc)
    for(uint32_t acc_id = 0; acc_id < T_num_acc; ++acc_id)
        acc[acc_id] = _mm256_and_ps(acc[acc_id], _mm256_load_ps(backward_input + acc_id * C_simd_width));

    #pragma unroll (T_num_acc)
    for(uint32_t acc_id = 0; acc_id < T_num_acc; ++acc_id)
        _mm256_store_ps(backward_output + acc_id * C_simd_width, acc[acc_id]);
}

void relu_f32::backward(
    const nn::workload_data<float> *forward_input, 
    const nn::workload_data<float> *forward_output,
    const nn::workload_data<float> *backward_input,
    nn::workload_data<float> *backward_output)
{
    const auto forward_input_buffer = reinterpret_cast<float*>(forward_input->parent->data_buffer);
    const auto backward_output_buffer = reinterpret_cast<float*>(backward_output->parent->data_buffer);
    const auto backward_input_buffer = reinterpret_cast<float*>(backward_input->parent->data_buffer);

    const auto forw_input_depth = forward_input->parent->lengths.t[NN_DATA_COORD_z];
    const auto forw_input_width = forward_input->parent->lengths.t[NN_DATA_COORD_x];
    const auto forw_input_height = forward_input->parent->lengths.t[NN_DATA_COORD_y];
    const auto forw_input_size = forw_input_depth * forw_input_width * forw_input_height;

    const auto back_input_depth = backward_input->parent->lengths.t[NN_DATA_COORD_z];
    const auto back_input_width = backward_input->parent->lengths.t[NN_DATA_COORD_x];
    const auto back_input_height = backward_input->parent->lengths.t[NN_DATA_COORD_y];
    const auto back_input_size = back_input_depth * back_input_width * back_input_height;

    const auto output_depth = backward_output->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_width = backward_output->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_height = backward_output->parent->lengths.t[NN_DATA_COORD_y];
    const auto output_size = output_depth * output_width * output_height;

    if(backward_output->get_length(NN_DATA_COORD_z) != 1)
    {
        assert(backward_output->get_length(NN_DATA_COORD_z) % C_simd_width == 0);

        for (uint32_t n_back_out = backward_output->view_begin.t[NN_DATA_COORD_n], n_back_in = backward_input->view_begin.t[NN_DATA_COORD_n], n_forw_in = forward_input->view_begin.t[NN_DATA_COORD_n]; 
             n_back_out <= backward_output->view_end.t[NN_DATA_COORD_n]; 
             ++n_back_out, ++n_back_in, ++n_forw_in)
            for (uint32_t x_back_out = backward_output->view_begin.t[NN_DATA_COORD_x], x_back_in = backward_input->view_begin.t[NN_DATA_COORD_x], x_forw_in = forward_input->view_begin.t[NN_DATA_COORD_x]; 
                 x_back_out <= backward_output->view_end.t[NN_DATA_COORD_x]; 
                 ++x_back_out, ++x_back_in, ++x_forw_in)
                for (uint32_t y_back_out = backward_output->view_begin.t[NN_DATA_COORD_y], y_back_in = backward_input->view_begin.t[NN_DATA_COORD_y], y_forw_in = forward_input->view_begin.t[NN_DATA_COORD_y];
                     y_back_out <= backward_output->view_end.t[NN_DATA_COORD_y]; 
                     ++y_back_out, ++y_back_in, ++y_forw_in)
                {
                    auto forward_input_ptr =   forward_input_buffer 
                                             + forward_input->view_begin.t[NN_DATA_COORD_z]
                                             + x_forw_in * forw_input_depth
                                             + y_forw_in * forw_input_depth * forw_input_width
                                             + n_forw_in * forw_input_size;

                    auto backward_input_ptr =   backward_input_buffer
                                              + backward_input->view_begin.t[NN_DATA_COORD_z]
                                              + x_back_in * back_input_depth
                                              + y_back_in * back_input_depth * back_input_width
                                              + n_back_in * back_input_size;

                    auto backward_output_ptr =   backward_output_buffer
                                               + backward_output->view_begin.t[NN_DATA_COORD_z]
                                               + x_back_out * output_depth
                                               + y_back_out * output_depth * output_width
                                               + n_back_out * output_size;


                    auto full_passes = backward_output->get_length(NN_DATA_COORD_z) / (C_num_max_acc * C_simd_width);
                    auto partial_pass_size = backward_output->get_length(NN_DATA_COORD_z) % (C_num_max_acc * C_simd_width) / C_simd_width;

                    for(uint32_t pass = 0; pass < full_passes; ++pass)
                    {
                        #pragma forceinline recursive
                        backward_inner_macro<C_num_max_acc>(forward_input_ptr, backward_input_ptr, backward_output_ptr);

                        forward_input_ptr += C_num_max_acc * C_simd_width;
                        backward_input_ptr += C_num_max_acc * C_simd_width;
                        backward_output_ptr += C_num_max_acc * C_simd_width;
                    }

                    switch(partial_pass_size)
                    {
                        case  1: backward_inner_macro< 1>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case  2: backward_inner_macro< 2>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case  3: backward_inner_macro< 3>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case  4: backward_inner_macro< 4>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case  5: backward_inner_macro< 5>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case  6: backward_inner_macro< 6>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case  7: backward_inner_macro< 7>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case  8: backward_inner_macro< 8>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case  9: backward_inner_macro< 9>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case 10: backward_inner_macro<10>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case 11: backward_inner_macro<11>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case 12: backward_inner_macro<12>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case 13: backward_inner_macro<13>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                        case 14: backward_inner_macro<14>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
                    }
                }
    }
    else
    {
        // One dimension used. No views supported.
        auto forward_input_ptr = forward_input_buffer;
        auto backward_input_ptr = backward_input_buffer;
        auto backward_output_ptr = backward_output_buffer;


        auto full_passes = backward_output->parent->buffer_size / 4 / (C_num_max_acc * C_simd_width);
        auto partial_pass_size = backward_output->parent->buffer_size / 4 % (C_num_max_acc * C_simd_width) / C_simd_width;

        for(uint32_t pass = 0; pass < full_passes; ++pass)
        {
            #pragma forceinline recursive
            backward_inner_macro<C_num_max_acc>(forward_input_ptr, backward_input_ptr, backward_output_ptr);

            forward_input_ptr += C_num_max_acc * C_simd_width;
            backward_input_ptr += C_num_max_acc * C_simd_width;
            backward_output_ptr += C_num_max_acc * C_simd_width;
        }

        switch(partial_pass_size)
        {
            case  1: backward_inner_macro< 1>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case  2: backward_inner_macro< 2>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case  3: backward_inner_macro< 3>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case  4: backward_inner_macro< 4>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case  5: backward_inner_macro< 5>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case  6: backward_inner_macro< 6>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case  7: backward_inner_macro< 7>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case  8: backward_inner_macro< 8>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case  9: backward_inner_macro< 9>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case 10: backward_inner_macro<10>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case 11: backward_inner_macro<11>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case 12: backward_inner_macro<12>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case 13: backward_inner_macro<13>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
            case 14: backward_inner_macro<14>(forward_input_ptr, backward_input_ptr, backward_output_ptr); break;
        }
    }
}

relu_f32::relu_f32(size_t image_size_x,
                   size_t image_size_y,
                   size_t image_size_z,
                   size_t batch_size,
                   size_t output_padding_left,
                   size_t output_padding_right,
                   size_t output_padding_top,
                   size_t output_padding_bottom,
                   nn_device_internal *device)
    : primitive_zxyn_f32_base(batch_size,
                              image_size_z,
                              image_size_x,
                              image_size_y,
                              image_size_z,
                              output_padding_left,
                              output_padding_right,
                              output_padding_top,
                              output_padding_bottom,
                              device) {}

size_t relu_f32::get_required_input_w() { return output_size_x; }

size_t relu_f32::get_required_input_h() { return output_size_y; }

void relu_f32::backward(const std::vector<nn_workload_data_t *> &inputs,
                        const std::vector<const nn_workload_data_t *> &parameters,
                        const std::vector<const nn_workload_data_t *> &outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    const nn::workload_data<float> backward_input(outputs[0]->parent->delta_buffer, outputs[0]->parent->lengths, outputs[0]->parent->layout);
    nn::workload_data<float> backward_output(inputs[0]->parent->delta_buffer, inputs[0]->parent->lengths, inputs[0]->parent->layout);

    backward(reinterpret_cast<const nn::workload_data<float> *>(inputs[0]),
             reinterpret_cast<const nn::workload_data<float> *>(outputs[0]),
             &backward_input,
             &backward_output);
}

bool relu_f32::validate_input(size_t index, nn_workload_data_t *data)
{
    switch (index) {
    case 0:
        return nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, float>::validate<true>(
            data, get_required_input_w(), get_required_input_h(), input_size_z, batch_size, 0, 0, 0, 0);
    }

    throw std::invalid_argument("index out of range");
}

void run_relu_backward(nn_workload_item *const work_item)
{
    auto primitive = static_cast<layer::relu_f32 *>(work_item->forward_item->primitive);
    primitive->backward(reinterpret_cast<nn::workload_data<float> *>(work_item->forward_item->input[0].get_data_view()),
                        reinterpret_cast<nn::workload_data<float> *>(work_item->forward_item->output[0]),
                        reinterpret_cast<nn::workload_data<float> *>(work_item->input[0].get_data_view()),
                        reinterpret_cast<nn::workload_data<float> *>(work_item->output[0]));
}
}

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_relu_f32_create_0(nn_device_t *device,
                                                                             size_t image_size_x,
                                                                             size_t image_size_y,
                                                                             size_t image_size_z,
                                                                             size_t batch_size,
                                                                             const nn_primitives_relu_hints_t *hints,
                                                                             NN_API_STATUS *status) {
    SET_STATUS(NN_API_STATUS_OK);
    nn_primitives_relu_hints_t hints_ = {};
    if(hints != nullptr)
        hints_ = *hints;

    return new layer::relu_f32(image_size_x,
                               image_size_y,
                               image_size_z,
                               batch_size,
                               hints_.output_padding.left,
                               hints_.output_padding.right,
                               hints_.output_padding.top,
                               hints_.output_padding.bottom,
                               reinterpret_cast<nn_device_internal *>(device));
}
