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

#pragma once
#include <assert.h>

#include "../../api/nn_primitives_api_0.h"
#include "../../common/nn_workload_data.h"
#include "../api_internal/nn_device_interface_0_internal.h"

namespace layer {

// tools for layers using common ZXYN data layout
namespace helper_zxyn_f32 {

class primitive_zxyn_f32_base : public nn_primitive_t {

  public:
    virtual ~primitive_zxyn_f32_base() {}

    virtual nn::nn_workload_data_t<float> *create_input(const nn::data<float, 4> &input);
    virtual nn::nn_workload_data_t<float> *map_input(const nn::data<float, 4> &input);

    virtual std::vector<nn::nn_workload_data_t<float> *> split_input_z(size_t partition_count,
                                                                       nn::nn_workload_data_t<float> &source);

    virtual bool validate_input(const nn::nn_workload_data_t<float> &input);
    virtual nn::nn_workload_data_t<float> *create_output();
    virtual nn::nn_workload_data_t<float> *create_output(size_t padding_left,
                                                         size_t padding_right,
                                                         size_t padding_top,
                                                         size_t padding_bottom);
    virtual std::vector<nn::nn_workload_data_t<float> *> create_output_vector_z(
        size_t partition_count, nn::nn_workload_data_t<float> *&merged_output);

    virtual void copy_output(nn::data<float, 4> &destination, const nn::nn_workload_data_t<float> &source);

    static const nn_workload_data_layout_t in_out_layout;

  protected:
    primitive_zxyn_f32_base(size_t batch_size,
                            size_t input_size_z,
                            size_t output_size_x,
                            size_t output_size_y,
                            size_t output_size_z,
                            nn_device_internal *device)
        : batch_size(batch_size),
          input_size_z(input_size_z),
          output_size_x(output_size_x),
          output_size_y(output_size_y),
          output_size_z(output_size_z),
          device(device) {}

    nn::nn_workload_data_t<float> *create_input_impl(const nn::data<float, 4> &input,
                                                size_t padding_left,
                                                size_t padding_right,
                                                size_t padding_top,
                                                size_t padding_bottom,
                                                int view_offset_x,
                                                int view_offset_y,
                                                size_t view_size_x,
                                                size_t view_size_y);

    virtual size_t get_required_input_w() = 0;
    virtual size_t get_required_input_h() = 0;

    const size_t batch_size;

    const size_t input_size_z;

    const size_t output_size_x;
    const size_t output_size_y;
    const size_t output_size_z;

    nn_device_internal *const device;
};

nn_opaque_data_t *NN_API_CALL_CONVENTION
create_input(nn_primitive_handle_t handle, const nn_data_t *input, NN_API_STATUS *status);

nn_opaque_data_t *NN_API_CALL_CONVENTION
map_input(nn_primitive_handle_t handle, const nn_data_t *input, NN_API_STATUS *status);

NN_API_STATUS NN_API_CALL_CONVENTION split_input_z(nn_primitive_handle_t handle,
                                                   const size_t partition_count,
                                                   nn_opaque_data_t *parts[],
                                                   const nn_opaque_data_t *source);

nn_opaque_data_t *NN_API_CALL_CONVENTION create_output_with_padding(nn_primitive_handle_t handle,
                                                                    size_t left_padding,
                                                                    size_t right_padding,
                                                                    size_t top_padding,
                                                                    size_t bottom_padding,
                                                                    NN_API_STATUS *status);

nn_opaque_data_t *NN_API_CALL_CONVENTION create_output(nn_primitive_handle_t handle, NN_API_STATUS *status);

nn_opaque_data_t *NN_API_CALL_CONVENTION create_output_vector_z(nn_primitive_handle_t handle,
                                                                const size_t partition_count,
                                                                nn_opaque_data_t *parts[],
                                                                NN_API_STATUS *status);

int NN_API_CALL_CONVENTION validate_input(nn_primitive_handle_t handle, nn_opaque_data_t *opaque_data);

nn_event_t NN_API_CALL_CONVENTION copy_output_async(nn_primitive_handle_t handle,
                                                    nn_data_t *output,
                                                    nn_opaque_data_t *output_buffer,
                                                    size_t dependencies_count,
                                                    nn_event_t *dependencies,
                                                    NN_API_STATUS *status);
};
};