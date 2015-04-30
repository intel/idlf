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

struct nn_workload_item;
struct nn_device_internal;

namespace layer {
class pooling_f32 : public helper_zxyn_f32::primitive_zxyn_f32_base {
  public:
    static pooling_f32 *create(NN_POOLING_MODE pooling_mode,
                               size_t pool_size_x,
                               size_t pool_size_y,
                               size_t pool_stride_x,
                               size_t pool_stride_y,
                               size_t num_feature_maps,
                               size_t output_w,
                               size_t output_h,
                               size_t batch_size,
                               nn_device_t *device);

    pooling_f32(NN_POOLING_MODE pooling_mode,
                size_t pool_size_x,
                size_t pool_size_y,
                size_t pool_stride_x,
                size_t pool_stride_y,
                size_t num_feature_maps,
                size_t output_w,
                size_t output_h,
                size_t batch_size,
                nn_device_internal *device);

    virtual ~pooling_f32() {}

    virtual void forward(const nn::nn_workload_data_t<float> *input,
                         nn::nn_workload_data_t<float> *output);


  private:
      void run_pooling(const nn::nn_workload_data_t<float> *input_view, nn::nn_workload_data_t<float> *output_view);

protected:
    virtual size_t get_required_input_w() override;
    virtual size_t get_required_input_h() override;

    const size_t pool_size_x;
    const size_t pool_size_y;
    const size_t pool_stride_x;
    const size_t pool_stride_y;
    const NN_POOLING_MODE pooling_mode;

    friend void unpack_pooling_callback_handle(void *void_handle);
    friend void wrapper_pooling_work_item(nn_workload_item *const work_item, nn_device_internal *device);
};

void wrapper_pooling_work_item(nn_workload_item *const work_item, nn_device_internal *device);
}
