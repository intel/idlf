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
class arithmetic_f32 : public helper_zxyn_f32::primitive_zxyn_f32_base {
  public:
    static arithmetic_f32 *create(size_t image_size_x,
                                  size_t image_size_y,
                                  size_t image_size_z,
                                  NN_ARITHMETIC_FUNCTION arithmetic_function,
                                  size_t batch_size,
                                  nn_device_t *device);

    virtual nn::nn_workload_data_t<float> *create_factor(const nn::data<float, 0> &factor);

    virtual void forward(const nn::nn_workload_data_t<float> *input,
                         const nn::nn_workload_data_t<float> *factor,
                         nn::nn_workload_data_t<float> *output);

    virtual nn::nn_workload_data_t<float> *create_output() override;

  protected:
    arithmetic_f32(size_t image_size_x,
                   size_t image_size_y,
                   size_t image_size_z,
                   NN_ARITHMETIC_FUNCTION arithmetic_function,
                   size_t batch_size,
                   nn_device_internal *device);

    const NN_ARITHMETIC_FUNCTION arithmetic_function;

    virtual size_t get_required_input_w() override;
    virtual size_t get_required_input_h() override;

  private:
    void run_arithmetic_operation_work_item(const nn::nn_workload_data_t<float> *input,
                                            const nn::nn_workload_data_t<float> *factor,
                                            nn::nn_workload_data_t<float> *output);
    template <NN_ARITHMETIC_FUNCTION T_function>
    void process_arithmetic_operation(const nn::nn_workload_data_t<float> *input,
                                      const nn::nn_workload_data_t<float> *factor,
                                      nn::nn_workload_data_t<float> *output);
    friend void unpack_arithmetic_callback_handle(void *void_handle);
};

void wrapper_arithmetic_operation_work_item(nn_workload_item *const work_item, nn_device_internal *device);
} // namespace layer
