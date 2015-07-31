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
#include "layer_convolution_avx2.h"

namespace layer {
class convolution_pooling_f32_2x2stride2 : public convolution_f32 {
  public:
    convolution_pooling_f32_2x2stride2(const size_t kernel_w,
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

    virtual ~convolution_pooling_f32_2x2stride2() {}

  protected:
    virtual void forward(const nn::workload_data<float> *input,
                         const nn::workload_data<float> *weights,
                         const nn::workload_data<float> *bias,
                         nn::workload_data<float> *output);

    virtual size_t get_required_input_w() override;
    virtual size_t get_required_input_h() override;

  private:
    void choose_convolution_maxpool_padding_mode_and_activation(const nn::workload_data<float> *input_view,
                                                                const nn::workload_data<float> *weights_view,
                                                                const nn::workload_data<float> *bias_view,
                                                                nn::workload_data<float> *output_view);
    template <NN_ACTIVATION_FUNCTION T_activation>
    void run_convolution_maxpool(const nn::workload_data<float> *input_view,
                                 const nn::workload_data<float> *weights_view,
                                 const nn::workload_data<float> *bias_view,
                                 nn::workload_data<float> *output_view);

    friend void unpack_convolve_maxpooling2x2_stride2x2_callback_handle(void *void_handle);

  protected:
    static const size_t pooling_size_x = 2;
    static const size_t pooling_size_y = 2;
    static const size_t pooling_stride_x = 2;
    static const size_t pooling_stride_y = 2;
};
}
