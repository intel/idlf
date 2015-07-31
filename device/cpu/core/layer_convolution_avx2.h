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

#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "device/api/nn_primitives_api_0.h"
#include "helper_zxyn_f32.h"

namespace layer {

class convolution_f32 : public helper_zxyn_f32::primitive_zxyn_f32_base {
  public:
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
                    size_t output_padding_left,
                    size_t output_padding_right,
                    size_t output_padding_top,
                    size_t output_padding_bottom,
                    nn_device_internal *device);

    virtual ~convolution_f32() {}

    virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta = false) override;

    virtual std::vector<nn_workload_data_t *> create_parameters(bool allocate_delta = false) override;

    virtual bool validate_input(size_t index, nn_workload_data_t *data) override;

    virtual void forward(const std::vector<const nn_workload_data_t *> &inputs,
                         const std::vector<const nn_workload_data_t *> &parameters,
                         const std::vector<nn_workload_data_t *> &outputs) override;

    virtual void backward(const nn::workload_data<float> *forward_input_buffer,
                          const nn::workload_data<float> *forward_weights_buffer,
                          const nn::workload_data<float> *forward_output_buffer,
                          const nn::workload_data<float> *backward_input_buffer,
                          nn::workload_data<float> *backward_output_buffer,
                          nn::workload_data<float> *backward_weights_delta_buffer,
                          nn::workload_data<float> *backward_bias_delta_buffer);

    virtual void backward_weights_delta(
                          const nn::workload_data<float> *forward_input_view,
                          const nn::workload_data<float> *backward_input_view,
                          nn::workload_data<float> *backward_weights_delta_view);

    virtual void backward_bias_delta(
                               const nn::workload_data<float> *backward_input_view,
                               nn::workload_data<float> *backward_bias_delta_view);

    virtual void backward_input_delta(
                               const nn::workload_data<float> *forward_weights_view,
                               const nn::workload_data<float> *backward_input_view,
                               nn::workload_data<float> *backward_output_view);
    void backward(const std::vector<nn_workload_data_t *> &inputs,
                  const std::vector<const nn_workload_data_t *> &parameters,
                  const std::vector<const nn_workload_data_t *> &outputs) override;

    void backward_parameter(size_t parameter_index,
                            const std::vector<const nn_workload_data_t *> &inputs,
                            const std::vector<nn_workload_data_t *> &parameters,
                            const std::vector<const nn_workload_data_t *> &outputs) override;
    
    // order of weights coordinates is kernel_width, kernel_height, number of input channels, number of filters


  protected:
    virtual void forward(const nn::workload_data<float> *input_buffer,
                         const nn::workload_data<float> *weights_buffer,
                         const nn::workload_data<float> *bias_buffer,
                         nn::workload_data<float> *output_buffer);

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

void run_multithreaded_convolve_work_item_backward(nn_workload_item *const work_item);
}
