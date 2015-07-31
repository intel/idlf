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
class relu_f32 : public helper_zxyn_f32::primitive_zxyn_f32_base {
  public:
    virtual void forward(const std::vector<const nn_workload_data_t *> &inputs,
                         const std::vector<const nn_workload_data_t *> &parameters,
                         const std::vector<nn_workload_data_t *> &outputs) override;

    virtual void backward(const std::vector<nn_workload_data_t *> &inputs,
                          const std::vector<const nn_workload_data_t *> &parameters,
                          const std::vector<const nn_workload_data_t *> &outputs) override;

    virtual void backward(const nn::workload_data<float> *forward_input,
                          const nn::workload_data<float> *forward_output,
                          const nn::workload_data<float> *backward_input,
                          nn::workload_data<float> *backward_output);

    relu_f32(size_t image_size_x,
             size_t image_size_y,
             size_t image_size_z,
             size_t batch_size,
             size_t output_padding_left,
             size_t output_padding_right,
             size_t output_padding_top,
             size_t output_padding_bottom,
             nn_device_internal *device);

  protected:
    virtual size_t get_required_input_w() override;
    virtual size_t get_required_input_h() override;

  private:
    void forward(const nn::workload_data<float> *input, nn::workload_data<float> *output);

    virtual bool validate_input(size_t index, nn_workload_data_t *data) override;
};

void run_relu(nn_workload_item *const work_item);
void run_relu_backward(nn_workload_item *const work_item);
}
