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
#include "device/common/nn_layer_parameters.h"
#include "convolution_jit.h"

struct nn_workload_item;
struct nn_device_internal;

namespace layer
{

class fully_connected_f32_batch24n : public nn_primitive_t
{
public:
    fully_connected_f32_batch24n(size_t num_input,
                                 size_t num_output,
                                 const nn_argument_activation_t &activation,
                                 size_t batch_size,
                                 nn_device_internal *device);

    void forward(const std::vector<const nn_workload_data_t *> &inputs,
                 const std::vector<const nn_workload_data_t *> &parameters,
                 const std::vector<nn_workload_data_t *> &outputs) override;
    void prepare_forward(const std::vector<const nn_workload_data_t *> &inputs,
                         const std::vector<const nn_workload_data_t *> &parameters,
                         const std::vector<nn_workload_data_t *> &outputs) override;

    std::vector<nn_workload_data_t *> create_parameters(bool allocate_delta = false) override;
    std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta = false) override;
    std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta = false) override;

    bool validate_input(size_t index, nn_workload_data_t *data) override;

private:
    virtual void forward(const nn::workload_data<> *input,
                         const nn::workload_data<> *weights,
                         const nn::workload_data<> *bias,
                         nn::workload_data<> *output);
    virtual void prepare_forward(const nn::workload_data<> *input,
                                 const nn::workload_data<> *weights,
                                 const nn::workload_data<> *bias,
                                 nn::workload_data<> *output);

    const uint32_t num_input;
    const uint32_t num_output;
    const nn_argument_activation_t activation;
    const uint32_t batch_size;
    nn_device_internal* device;

    std::tuple<float*, float*, float*, float*> prepared_for;
    std::shared_ptr<jit_convolution> compiled;
};

} //namespace layer

