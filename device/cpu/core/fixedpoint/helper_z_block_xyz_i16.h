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

#include "device/api/nn_primitives_api_0.h"
#include "device/common/nn_workload_data.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"

namespace int16_fixedpoint {

// tools for layers using common ZXYN data layout
namespace helper_z_block_xyz_i16 {

class primitive_z_block_xyz_i16_base : public nn_primitive_t {

 public:
    virtual ~primitive_z_block_xyz_i16_base() {}

    virtual nn::workload_data<int16_t> *map_input(const nn::data<int16_t, 4> &input);

    virtual std::vector<nn::workload_data<int16_t> *> split_input_z(size_t partition_count, nn::workload_data<int16_t> &source);

    virtual bool validate_input(size_t index, nn_workload_data_t *data) override;

    virtual void copy_output(nn::data<int16_t, 4> &destination, const nn::workload_data<int16_t> &source);

    virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta) override;

    virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta) override;

  protected:
      primitive_z_block_xyz_i16_base(
                            size_t batch_size,
                            size_t input_size_z,
                            size_t output_size_x,
                            size_t output_size_y,
                            size_t output_size_z,
                            size_t output_padding_left,
                            size_t output_padding_right,
                            size_t output_padding_top,
                            size_t output_padding_bottom,
                            nn_device_internal *device)
        : batch_size(batch_size),
          input_size_z(input_size_z),
          output_size_x(output_size_x),
          output_size_y(output_size_y),
          output_size_z(output_size_z),
          output_padding_left(output_padding_left),
          output_padding_right(output_padding_right),
          output_padding_top(output_padding_top),
          output_padding_bottom(output_padding_bottom),
          device(device) {}

    nn::workload_data<int16_t> *create_input_impl(const nn::data<int16_t, 4> &input,
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

    const uint32_t block_size = 16;
    const size_t batch_size;
    const size_t input_size_z;
    const size_t output_size_x;
    const size_t output_size_y;
    const size_t output_size_z;
    const size_t output_padding_left;
    const size_t output_padding_right;
    const size_t output_padding_top;
    const size_t output_padding_bottom;
    static const nn_workload_data_layout_t& in_out_layout;

    nn_device_internal *const device;
};
};
};