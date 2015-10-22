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
#include "layer_convolution_avx2_forward.h"
#include "helper_zxyn_f32.h"

namespace layer
{
const uint64_t BATCH_ACCEPTED_BLOCK = 24;

template <typename T_Name>
using ValueU64=Value<T_Name, uint64_t>;

using namespace convolution;
using namespace convolution::forward;

class convolution_f32_batch24n : public nn_primitive_t
{
public:
    convolution_f32_batch24n(ValueU64<Batch> batch_size,
                    OutputDimensions out_dims,
                    KernelInfo kernel_info,
                    nn_argument_activation_t activation,
                    nn_device_internal *device);
    ~convolution_f32_batch24n();

    std::vector<nn_workload_data_t *> create_parameters(bool allocate_delta) override;
    std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta) override;
    std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta) override;

    bool validate_input(size_t index, nn_workload_data_t *data) override;

    void forward(const std::vector<const nn_workload_data_t *> &inputs,
                 const std::vector<const nn_workload_data_t *> &parameters,
                 const std::vector<nn_workload_data_t *> &outputs) override;

    void prepare_forward(const std::vector<const nn_workload_data_t*> &inputs,
                         const std::vector<const nn_workload_data_t*> &parameters,
                         const std::vector<nn_workload_data_t *> &outputs) override;

protected:
    void forward(const nn::workload_data<> *input_buffer,
                 const nn::workload_data<> *weights_buffer,
                 const nn::workload_data<> *bias_buffer,
                 nn::workload_data<> *output_buffer);
    void prepare_forward(const nn::workload_data<> *input_buffer,
                         const nn::workload_data<> *weights_buffer,
                         const nn::workload_data<> *bias_buffer,
                         nn::workload_data<> *output_buffer);

    const ValueU64<Batch> batch_size;
    const OutputDimensions out_dims;
    KernelInfo kernel_info;
    const nn_argument_activation_t activation;
    nn_device_internal *const device;

    std::tuple<float*, float*, float*, float*> prepared_for;
    InputDimensions in_full_dims;
    OutputDimensions out_full_dims;
    StrideInfo stride_info;
    convolution::forward::CompiledKernelConvolution* compiled_convolution;
};

} //namespace layer

