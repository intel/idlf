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

#include "layer_dropout.h"
#include <random>
#include <array>
#include "device/cpu/api_internal/data_helper.h"

namespace layer
{
    template <bool backward> void run_dropout(
        float drop_rate,
        const nn_workload_data_t *input_data,
        const nn_workload_data_t *input_seed,
        const nn_workload_data_t *input_if_train,
        nn_workload_data_t *output)
    {
        // Third input - indicates if it's test or training phase.
        auto is_training = nn_workload_data_get<int32_t>(input_if_train, 0, 0, 0, 0, 0, 0) != 0;

        if(!is_training)
        {
            if(backward)
                nn_workload_delta_copy(output, input_data);
            else
                nn_workload_data_copy(output, input_data);
         
            return;
        }

        // Second input - seed.
        auto seed = static_cast<uint32_t>(nn_workload_data_get<int32_t>(input_seed, 0, 0, 0, 0, 0, 0));

        std::mt19937 gen(seed);
        std::bernoulli_distribution dis(drop_rate);
        float scale = 1.0f / (1.0f - drop_rate);

        for (uint32_t n = 0; n < output->parent->lengths.t[NN_DATA_COORD_n]; ++n)
            for (uint32_t x = 0; x < output->parent->lengths.t[NN_DATA_COORD_x]; ++x)
                for (uint32_t y = 0; y < output->parent->lengths.t[NN_DATA_COORD_y]; ++y)
                    for (uint32_t z = 0; z < output->parent->lengths.t[NN_DATA_COORD_z]; ++z)
                        for (uint32_t p = 0; p < output->parent->lengths.t[NN_DATA_COORD_p]; ++p)
                            for (uint32_t q = 0; q < output->parent->lengths.t[NN_DATA_COORD_q]; ++q)
                            {
                                if(backward)
                                    nn_workload_data_get_delta<float>(output, n, x, y, z, p, q) = dis(gen) ? 0.0f : nn_workload_data_get_delta<float>(input_data, n, x, y, z, p, q) * scale;
                                else
                                    nn_workload_data_get<float>(output, n, x, y, z, p, q) = dis(gen) ? 0.0f : nn_workload_data_get<float>(input_data, n, x, y, z, p, q) * scale;
                            }
                                
    }

    dropout_f32::dropout_f32(
        size_t input_size_x, size_t input_size_y, size_t input_size_z, size_t batch_size, float drop_rate, nn_device_internal *device)
            : input_size_x(input_size_x),
              input_size_y(input_size_y),
              input_size_z(input_size_z),
              batch_size(batch_size),
              drop_rate(drop_rate),
              device(device){}

    bool dropout_f32::validate_input(size_t index, nn_workload_data_t *data) {
        switch (index) {
        case 0:
            return nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, float>::create(
                device, input_size_x * input_size_y * input_size_z, batch_size);
        case 1:
        case 2:
        {
            if(data->parent->buffer_size > 0 &&
               data->parent->layout.data_type == nn::type_to_datatype<int32_t>::value &&
               data->parent->data_buffer != nullptr)
                return true;
            else 
                return false;
        }
        }

        throw std::invalid_argument("index out of range");
    }

    void dropout_f32::forward(const std::vector<const nn_workload_data_t *> &inputs,
                              const std::vector<const nn_workload_data_t *> &parameters,
                              const std::vector<nn_workload_data_t *> &outputs) {
        run_dropout<false>(
            drop_rate,
            inputs[0],     // input
            inputs[1],     // seed
            inputs[2],     // if_training
            outputs[0]);   // output
    }

    void dropout_f32::backward(
        const std::vector<nn_workload_data_t *> &inputs,
        const std::vector<const nn_workload_data_t *> &parameters,
        const std::vector<const nn_workload_data_t *> &outputs)
    {
        run_dropout<true>(
            drop_rate,
            outputs[0],    // input for backpropagation
            inputs[1],     // seed
            inputs[2],     // if_training
            inputs[0]);    // output for backpropagation
    }

    std::vector<nn_workload_data_t *> dropout_f32::create_inputs(bool allocate_delta) {
        
        const nn_workload_data_layout_t layout = nn::workload_data<int32_t>::layout.nxyzpq;

        return {
            nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, float>::create(
                device, input_size_x * input_size_y * input_size_z, batch_size, allocate_delta),
            new nn::workload_data<int32_t>({1, 1, 1, 1, 1, 1}, layout),
            new nn::workload_data<int32_t>({1, 1, 1, 1, 1, 1}, layout)};
    }

    std::vector<nn_workload_data_t *> dropout_f32::create_outputs(bool allocate_delta) {
        return {
            nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, float>::create(
                device, input_size_x * input_size_y * input_size_z, batch_size, allocate_delta)};
    }

}  // namespace layer

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_dropout_f32_create_0(
    nn_device_t *device, /* IDLF device handle */
    size_t input_size_x,
    size_t input_size_y,
    size_t input_size_z,
    size_t batch_size,
    float drop_rate,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);
    return new layer::dropout_f32(
        input_size_x, input_size_y, input_size_z, batch_size, drop_rate, reinterpret_cast<nn_device_internal *>(device));
}