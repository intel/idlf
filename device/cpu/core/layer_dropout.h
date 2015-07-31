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

namespace layer
{

    class dropout_f32 : public nn_primitive_t {
    public:
        dropout_f32(
            size_t input_size_x,
            size_t input_size_y,
            size_t input_size_z,
            size_t batch_size,
            float drop_rate,
            nn_device_internal *device);

        virtual void forward(const std::vector<const nn_workload_data_t *> &inputs,
                             const std::vector<const nn_workload_data_t *> &parameters,
                             const std::vector<nn_workload_data_t *> &outputs) override;

        virtual void backward(const std::vector<nn_workload_data_t *> &inputs,
                              const std::vector<const nn_workload_data_t *> &parameters,
                              const std::vector<const nn_workload_data_t *> &outputs) override;

        virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta = false) override;

        virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta = false) override;

        virtual bool validate_input(size_t index, nn_workload_data_t *data) override;

    protected:
        const float drop_rate;
        const size_t input_size_x, input_size_y, input_size_z, batch_size;
        nn_device_internal *const device;
    };
} //namespace layer