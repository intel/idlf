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
#include "device/api/nn_primitives_api_0.h"
#include "helper_zxyn_f32.h"
#include "fixedpoint/helper_z_block_xyz_i16.h"
#include <cstdint>
#include <immintrin.h>

struct nn_workload_item;
struct nn_device_internal;

namespace layer {
void run_convert_to_data_layout_work_item(nn_workload_item *const work_item);

class convert_zxyn_nx_f32 : public nn_primitive_t {
  public:
    convert_zxyn_nx_f32(
        size_t input_size_x, size_t input_size_y, size_t input_size_z, size_t batch_size, nn_device_internal *device);

    virtual void forward(const std::vector<const nn_workload_data_t *> &inputs,
                         const std::vector<const nn_workload_data_t *> &parameters,
                         const std::vector<nn_workload_data_t *> &outputs) override;

    void backward(const std::vector<nn_workload_data_t *> &inputs,
                  const std::vector<const nn_workload_data_t *> &parameters,
                  const std::vector<const nn_workload_data_t *> &outputs) override;

    virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta = false) override;

    virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta = false) override;

    virtual bool validate_input(size_t index, nn_workload_data_t *data) override;

  protected:
    const size_t input_size_x, input_size_y, input_size_z, output_size, batch_size;
    nn_device_internal *const device;

  private:
    void forward(const nn::workload_data<float> *input, nn::workload_data<float> *output);
};

class convert_z_block_xyz_z2nz : public nn_primitive_t {
public:
    convert_z_block_xyz_z2nz(size_t input_size_x, size_t input_size_y, size_t input_size_z, size_t batch_size, nn_device_internal *device);

    virtual bool validate_input(size_t index, nn_workload_data_t *data) override;


    void copy_output(nn::data<int16_t, 2> &destination, const nn::workload_data<int16_t> &source);
    void forward(const nn::workload_data<int16_t> *input, nn::workload_data<int16_t> *output);

    virtual void forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs) override;

    static const nn_workload_data_layout_t out_layout;

protected:
    virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta) override;

    virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta) override;

    const uint32_t in_block_size = 16;
    const uint32_t out_block_size = 2;
    const size_t
        input_size_x,
        input_size_y,
        input_size_z,
        output_size,
        batch_size;
    nn_device_internal *const device;
};

class convert_z2nz_n8xn : public nn_primitive_t {
public:
    convert_z2nz_n8xn(size_t input_size_x, size_t batch_size, nn_device_internal *device);

    virtual bool validate_input(size_t index, nn_workload_data_t *data) override;


    void copy_output(nn::data<int32_t, 2> &destination, const nn::workload_data<int32_t> &source);
    void forward(const nn::workload_data<int32_t> *input, nn::workload_data<int32_t> *output);

    virtual void forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs) override;

    static const nn_workload_data_layout_t out_layout;

protected:
    virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta) override;

    virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta) override;

    const uint32_t in_block_size = 2;
    const size_t
        input_size_x,
        output_size,
        batch_size;
    nn_device_internal *const device;
};
} // namespace layer
