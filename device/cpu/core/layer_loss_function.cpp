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

#include "layer_loss_function.h"
#include "device/cpu/api_internal/data_helper.h"

namespace layer
{
    void run_loss_function(nn_workload_item *const item)
    {
        auto primitive = static_cast<layer::loss_function_f32 *>(item->forward_item != nullptr ? item->forward_item->primitive : item->primitive);

        auto output = reinterpret_cast<nn::workload_data<float>*>(item->output[0]);
        auto current_input = reinterpret_cast<nn::workload_data<float>*>(item->input[0].get_data_view());
        auto target_input = reinterpret_cast<nn::workload_data<float>*>(item->input[1].get_data_view());

        // TODO: This is temporary until workflow compilation makes use of buffer_delta
        assert(current_input->parent->delta_buffer == nullptr);
        current_input->parent->delta_buffer = output->parent->data_buffer;

        primitive->backward(
            {current_input, target_input},
            {nullptr},
            {nullptr});

        // Revert the change made above.
        current_input->parent->delta_buffer = nullptr;
    }

    void multinomial_loss(const nn_workload_data_t *label_data,
                          const nn_workload_data_t *input,
                          nn_workload_data_t *output,
                          uint32_t batch_size) 
    {
        const auto C_loss_weight = 1.0f; // Loss weight from caffe.
        const auto C_scale = - C_loss_weight / batch_size;

        const auto input_buffer = reinterpret_cast<float*>(input->parent->data_buffer);
        const auto label_buffer = reinterpret_cast<int32_t*>(label_data->parent->data_buffer);
        const auto output_buffer = reinterpret_cast<float*>(output->parent->data_buffer);

        memset(output->parent->data_buffer, 0, output->parent->buffer_size);

        for (uint32_t n = output->view_begin.t[NN_DATA_COORD_n];
                n <= output->view_end.t[NN_DATA_COORD_n]; 
                ++n)
        {
            // This label is provided purely by user - we need runtime security 
            // check to ensure it's not used to access data not in buffer.
            // This layer is so computationally cheap that we can afford it.
            auto label = *(label_buffer + n);

            if(label < input->parent->lengths.t[NN_DATA_COORD_x])
            {
                auto result = C_scale / *(input_buffer + label * batch_size + n);

                *(output_buffer + label * batch_size + n) = result;
            }
            else
            {
                throw std::invalid_argument("multinomial_loss: label id too big");
            }
        }
    }

    loss_function_f32::loss_function_f32(
        NN_LOSS_FUNCTION function, size_t input_size_x, size_t input_size_y, size_t input_size_z, size_t batch_size, nn_device_internal *device)
            : function(function),
              input_size_x(input_size_x),
              input_size_y(input_size_y),
              input_size_z(input_size_z),
              batch_size(batch_size),
              device(device),
              arithmetic_primitive(
                new arithmetic_f32(
                    input_size_x,
                    input_size_y,
                    input_size_z,
                    NN_ARITHMETIC_FUNCTION_SUBTRACTION,
                    batch_size,
                    1.0f, 1.0f, -1 / float(batch_size),
                    device)){}
    
    bool loss_function_f32::validate_input(size_t index, nn_workload_data_t *data) {
        switch (index) {
        case 0:
        case 1:
            return nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, float>::validate<false>(
                data, input_size_x, input_size_y, input_size_z, batch_size, 0, 0, 0, 0);
        }

        throw std::invalid_argument("index out of range");
    }

    void loss_function_f32::forward(const std::vector<const nn_workload_data_t *> &inputs,
                                    const std::vector<const nn_workload_data_t *> &parameters,
                                    const std::vector<nn_workload_data_t *> &outputs) {
        throw std::runtime_error("loss_function_f32::forward() not implemented");
    }

    void loss_function_f32::backward(
        const std::vector<nn_workload_data_t *> &inputs,
        const std::vector<const nn_workload_data_t *> &parameters,
        const std::vector<const nn_workload_data_t *> &outputs)
    {
        nn::workload_data<float> output(inputs[0]->parent->delta_buffer, inputs[0]->parent->lengths, inputs[0]->parent->layout);
        if(function == NN_LOSS_FUNCTION_SUM_OF_SQUARES)
            arithmetic_primitive->forward({inputs[1]}, {inputs[0]}, {&output});
        else if(function == NN_LOSS_FUNCTION_MULTINOMIAL_LOGISTIC)
            multinomial_loss(inputs[1], inputs[0], &output, batch_size);
        else
            throw std::runtime_error("loss function not implemetned");
    }

    std::vector<nn_workload_data_t *> loss_function_f32::create_inputs(bool allocate_delta) {
        return {
            nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, float>::create(
                device, input_size_x, input_size_y, input_size_z, batch_size, 0, 0, 0, 0, allocate_delta),
            nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, float>::create(
                device, input_size_x, input_size_y, input_size_z, batch_size, 0, 0, 0, 0, false)};
    }

    std::vector<nn_workload_data_t *> loss_function_f32::create_outputs(bool allocate_delta) {
        return {
            nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, float>::create(
                device, input_size_x, input_size_y, input_size_z, batch_size, 0, 0, 0, 0, allocate_delta)};
    }
} // namespace layer

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_loss_f32_create_0(
    nn_device_t *device, /* IDLF device handle */
    NN_LOSS_FUNCTION function,
    size_t input_size_x,
    size_t input_size_y,
    size_t input_size_z,
    size_t batch_size,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);
    return new layer::loss_function_f32(
        function, input_size_x, input_size_y, input_size_z, batch_size, reinterpret_cast<nn_device_internal *>(device));
}