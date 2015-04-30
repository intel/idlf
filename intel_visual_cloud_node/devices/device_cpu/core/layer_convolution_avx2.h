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

#include "../api_internal/nn_device_interface_0_internal.h"
#include "../../api/nn_primitives_api_0.h"
#include "helper_zxyn_f32.h"

namespace layer {

class convolution_f32 : public helper_zxyn_f32::primitive_zxyn_f32_base {
  public:
    static convolution_f32 *create(size_t kernel_w,
                                   size_t kernel_h,
                                   size_t num_input,
                                   size_t num_output,
                                   size_t output_w,
                                   size_t output_h,
                                   int32_t center_offset_x,
                                   int32_t center_offset_y,
                                   size_t stride_x,
                                   size_t stride_y,
                                   const nn_argument_activation_t &activation,
                                   size_t batch_size,
                                   nn_device_t *device);
    virtual ~convolution_f32() {}

    virtual void forward(const nn::nn_workload_data_t<float> *input_buffer,
                         const nn::nn_workload_data_t<float> *weights_buffer,
                         const nn::nn_workload_data_t<float> *bias_buffer,
                         nn::nn_workload_data_t<float> *output_buffer);

    // order of weights coordinates is kernel_width, kernel_height, number of input channels, number of filters
    virtual nn::nn_workload_data_t<float> *create_weights(const nn::data<float, 4> &weights);
    virtual nn::nn_workload_data_t<float> *create_bias(const nn::data<float, 1> &bias);

    // input coordinate ordering must be ZXYN
    virtual nn::nn_workload_data_t<float> *create_input(const nn::data<float, 4> &input);
    virtual bool validate_input(const nn::nn_workload_data_t<float>& input);

  protected:
    convolution_f32(const size_t kernel_w,
                    const size_t kernel_h,
                    const size_t num_input,
                    const size_t num_output,
                    const size_t output_w,
                    const size_t output_h,
                    const int32_t center_offset_x,
                    const int32_t center_offset_y,
                    const size_t stride_x,
                    const size_t stride_y,
                    const nn_argument_activation_t &activation,
                    size_t batch_size,
                    nn_device_internal *device);

    virtual size_t get_required_input_w() override;
    virtual size_t get_required_input_h() override;

    const size_t kernel_w;
    const size_t kernel_h;
    const NN_PADDING_MODE padding;
    const int32_t center_offset_x;
    const int32_t center_offset_y;
    const size_t stride_x;
    const size_t stride_y;
    const nn_argument_activation_t activation;
};

void run_multithreaded_convolve_work_item(nn_workload_item *const work_item, nn_device_internal *device);

namespace convolution_f32_impl {
nn_opaque_data_t *NN_API_CALL_CONVENTION
create_weights(nn_primitive_handle_t handle, const nn_data_t *weights, NN_API_STATUS *status);
nn_opaque_data_t *NN_API_CALL_CONVENTION
create_bias(nn_primitive_handle_t handle, const nn_data_t *bias, NN_API_STATUS *status);
nn_event_t NN_API_CALL_CONVENTION forward_async(nn_primitive_handle_t handle,
                                                nn_opaque_data_t *input,
                                                nn_opaque_data_t *weights,
                                                nn_opaque_data_t *bias,
                                                nn_opaque_data_t *output,
                                                size_t dependencies_count,
                                                nn_event_t *dependencies,
                                                NN_API_STATUS *status);
nn_primitive_handle_t NN_API_CALL_CONVENTION nn_convolution_f32_create(nn_device_t *device,
                                                                       const size_t kernel_w,
                                                                       const size_t kernel_h,
                                                                       const size_t num_input,
                                                                       const size_t num_output,
                                                                       const size_t output_w,
                                                                       const size_t output_h,
                                                                       const int32_t center_offset_x,
                                                                       const int32_t center_offset_y,
                                                                       const size_t stride_x,
                                                                       const size_t stride_y,
                                                                       const nn_argument_activation_t *activation,
                                                                       size_t batch_size,
                                                                       NN_API_STATUS *status);
}
}
