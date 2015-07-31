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

#include "tester/g_ult/unit_tests/cpu/naive_implementations.h"
#include "device/cpu/core/layer_fully_connected_avx2.h"

namespace
{
const auto C_max_acc_batch8 = 13u;
const auto C_max_acc_batch48 = 2u;
///////////////////////////////////////////////////////////////////////////////////////////////////
// Helper classess and functions.
bool compare_work_items(
    nn_workload_item* &work_item,
    nn_workload_item* &work_item_ref)
{
    for (uint32_t batch = 0; batch < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_n]; ++batch)
    {
        for (uint32_t output_element = 0; output_element < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_x]; ++output_element)
        {
            float value = nn_workload_data_get<float>(work_item->output[0], batch, output_element, 0, 0, 0, 0);
            float value_ref = nn_workload_data_get<float>(work_item_ref->output[0], batch, output_element, 0, 0, 0, 0);

            float diff = fabs(value_ref - value);

            if (value_ref == 0.0f || value == 0.0f || diff < FLT_MIN)
            {
                if (diff > FLT_MIN)
                {
                    return false;
                }
            }
            else
            {
                if (fabs(diff / value_ref) > 5.2e-06F)
                {
                    return false;
                }
            }
        }
    }

    return true;
}

bool run_work_item(
    nn_workload_item* &work_item,
    NN_ACTIVATION_FUNCTION activation_function,
    bool is_ref)
{
    //auto &arguments = work_item->arguments.forward_fully_connected;
    if (is_ref)
    {
        // Naive implementation.
        cpu_layer_fullyconnected(
            * &work_item,
            activation_function);
    }
    else
    {
        // Use optimized routine.
        work_item->primitive->forward({work_item->input[0].get_data_view()},
                                      {work_item->parameters.begin(), work_item->parameters.end()},
                                      work_item->output);
    }

    return true;
}

void destroy_work_item(
    nn_workload_item* &work_item)
{
    if (work_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED)
    {
        delete reinterpret_cast<nn::workload_data<float> *>(work_item->parameters[0]);
        delete reinterpret_cast<nn::workload_data<float> *>(work_item->parameters[1]);
    }

    work_item->input.clear();
    delete reinterpret_cast<nn::workload_data<float>*>(work_item->output[0]);

    delete work_item;

    work_item = nullptr;
}

void create_and_initialize_input_item(
    nn_workload_item* &work_item,
    uint32_t input_width,
    uint32_t batch_size)
{
    nn_workload_data_coords_t in_out_coords =
    {
        batch_size,
        input_width,
        1,
        1,
        1,
        1
    };

    work_item = new nn_workload_item();

    work_item->type = NN_WORK_ITEM_TYPE_INPUT;

    work_item->arguments.input.index = 0;

    nn_workload_data_layout_t inp_out_layout = nn::workload_data<float>::layout.nxyzpq;

    work_item->output.push_back(new nn::workload_data<float>(in_out_coords, inp_out_layout));

    for (uint32_t batch = 0; batch < batch_size; ++batch)
    {
        for (uint32_t input_element = 0; input_element < input_width; ++input_element)
        {
            float value = 0.03125f;
            value *= pow(1.01f, input_element);
            value *= pow(1.01f, batch);
            if (input_element % 2) value *= -1.0f;
            nn_workload_data_get<float>(work_item->output[0], batch, input_element, 0, 0, 0, 0) = value;
        }
    }
}

void create_and_initialize_work_item(
    nn_workload_item* &work_item,
    nn_workload_item* input_item,
    uint32_t input_width,
    uint32_t output_width,
    uint32_t batch_size,
    bool check_views,
    bool bias_in_output,
    NN_ACTIVATION_FUNCTION function,
    bool is_ref,
    nn_device_t *device)
{
    nn_workload_data_coords_t input_coords =
    {
        batch_size,
        input_width,
        1,
        1,
        1,
        1
    };

    nn_workload_data_coords_t output_coords =
    {
        batch_size,
        output_width,
        1,
        1,
        1,
        1
    };

    nn_workload_data_coords_t bias_coords =
    {
        1,
        output_width,
        1,
        1,
        1,
        1
    };


    work_item = new nn_workload_item();

    work_item->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;

    work_item->input.push_back({ input_item, 0 });

    work_item->parameters.resize(2);
    auto &weights = work_item->parameters[0];
    auto &biases = work_item->parameters[1];
    nn_argument_activation_t s_activation = {};
    s_activation.function = function;
    work_item->primitive = new layer::fully_connected_f32(
        input_width, output_width, s_activation, batch_size, reinterpret_cast<nn_device_internal *>(device));

    nn_workload_data_layout_t inp_out_bias_layout = nn::workload_data<float>::layout.nxyzpq;

    if (is_ref)
    {
        nn_workload_data_layout_t weights_layout = nn::workload_data<float>::layout.xyzpqn;

        nn_workload_data_coords_t weights_coords =
        {
            1,
            input_width,
            output_width,
            1,
            1,
            1
        };

        work_item->output.push_back(new nn::workload_data<float>(output_coords, inp_out_bias_layout));
        biases = new nn::workload_data<float>(bias_coords, inp_out_bias_layout);
        weights = new nn::workload_data<float>(weights_coords, weights_layout);

        for (uint32_t weight_input = 0; weight_input < input_width; ++weight_input)
        {
            for (uint32_t weight_output = 0; weight_output < output_width; ++weight_output)
            {
                float value = 0.03125f;
                value *= pow(1.01f, weight_input);
                value *= pow(1.01f, weight_output);
                if (weight_output % 2) value *= -1.0f;
                nn_workload_data_get<float>(weights, 0, weight_input, weight_output, 0, 0, 0) = value;
            }
        }

        for (uint32_t bias_element = 0; bias_element < output_width; ++bias_element)
        {
            nn_workload_data_get<float>(biases, 0, bias_element, 0, 0, 0, 0) = static_cast<float>(bias_element);
        }

        for (uint32_t batch = 0; batch < batch_size; ++batch)
        {
            for (uint32_t output_element = 0; output_element < output_width; ++output_element)
            {
                nn_workload_data_get<float>(work_item->output[0], batch, output_element, 0, 0, 0, 0) = 0.0f;
            }
        }
    }
    else
    {
        if (batch_size == 1)
        {
            nn_workload_data_layout_t weights_layout = nn::workload_data<float>::layout.yxzpqn;

            nn_workload_data_coords_t weights_coords =
            {
                1,
                input_width,
                output_width,
                1,
                1,
                1
            };

            weights = new nn::workload_data<float>(weights_coords, weights_layout);

            for (uint32_t weight_input = 0; weight_input < input_width; ++weight_input)
            {
                for (uint32_t weight_output = 0; weight_output < output_width; ++weight_output)
                {
                    float value = 0.03125f;
                    value *= pow(1.01f, weight_input);
                    value *= pow(1.01f, weight_output);
                    if (weight_output % 2) value *= -1.0f;
                    nn_workload_data_get<float>(weights, 0, weight_input, weight_output, 0, 0, 0) = value;
                }
            }
        }
        else if (batch_size == 8)
        {
            nn_workload_data_layout_t weights_layout = nn::workload_data<float>::layout.pxqzyn;

            nn_workload_data_coords_t weights_coords =
            {
                1,
                input_width,
                1,
                1,
                C_max_acc_batch8,
                (output_width + C_max_acc_batch8 - 1) / C_max_acc_batch8
            };

            weights = new nn::workload_data<float>(weights_coords, weights_layout);

            for (uint32_t weight_input = 0; weight_input < input_width; ++weight_input)
            {
                for (uint32_t weight_output = 0; weight_output < output_width; ++weight_output)
                {
                    float value = 0.03125f;
                    value *= pow(1.01f, weight_input);
                    value *= pow(1.01f, weight_output);
                    if (weight_output % 2) value *= -1.0f;
                    nn_workload_data_get<float>(weights, 0, weight_input, 0, 0, weight_output % C_max_acc_batch8, weight_output / C_max_acc_batch8) = value;
                }
            }
        }
        else if (batch_size == 48)
        {
            nn_workload_data_layout_t weights_layout = nn::workload_data<float>::layout.pxqzyn;

            nn_workload_data_coords_t weights_coords =
            {
                1,
                input_width,
                1,
                1,
                C_max_acc_batch48,
                (output_width + C_max_acc_batch48 - 1) / C_max_acc_batch48
            };

            weights = new nn::workload_data<float>(weights_coords, weights_layout);

            for (uint32_t weight_input = 0; weight_input < input_width; ++weight_input)
            {
                for (uint32_t weight_output = 0; weight_output < output_width; ++weight_output)
                {
                    float value = 0.03125f;
                    value *= pow(1.01f, weight_input);
                    value *= pow(1.01f, weight_output);
                    if (weight_output % 2) value *= -1.0f;
                    nn_workload_data_get<float>(weights, 0, weight_input, 0, 0, weight_output % C_max_acc_batch48, weight_output / C_max_acc_batch48) = value;
                }
            }
        }

        work_item->output.push_back(new nn::workload_data<float>(output_coords, inp_out_bias_layout));
        biases = (bias_in_output) ? nullptr : new nn::workload_data<float>(bias_coords, inp_out_bias_layout);

        for (uint32_t batch = 0; batch < batch_size; ++batch)
        {
            for (uint32_t output_element = 0; output_element < output_width && bias_in_output; ++output_element)
            {
                nn_workload_data_get<float>(work_item->output[0], batch, output_element, 0, 0, 0, 0) = static_cast<float>(output_element);
            }
        }

        for (uint32_t bias_element = 0; bias_element < output_width && !bias_in_output; ++bias_element)
        {
            nn_workload_data_get<float>(biases, 0, bias_element, 0, 0, 0, 0) = static_cast<float>(bias_element);
        }
    }
}

bool ult_perform_test(
    uint32_t input_width,
    uint32_t output_width,
    uint32_t batch_size,
    bool bias_in_output,
    NN_ACTIVATION_FUNCTION function)
{
    bool return_value = true;

    // Input item.
    nn_workload_item* input_item = nullptr;
    create_and_initialize_input_item(input_item, input_width, batch_size);

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    nn_device_load(&device_description);
    nn_device_interface_open(0, &device_interface_0);

    // Work item.
    nn_workload_item* work_item = nullptr;
    create_and_initialize_work_item(work_item, input_item, input_width, output_width, batch_size, false, bias_in_output, function, false, device_interface_0.device);

    // Reference workload item.
    nn_workload_item* work_item_ref = nullptr;
    create_and_initialize_work_item(work_item_ref, input_item, input_width, output_width, batch_size, false, bias_in_output, function, true, device_interface_0.device);

    // Execute optimized workload item.
    return_value &= run_work_item(work_item, function, false);

    // Execute reference item.
    return_value &= run_work_item(work_item_ref, function, true);

    // Compare results.
    return_value &= compare_work_items(work_item, work_item_ref);

    // Cleanup.
    destroy_work_item(work_item);
    destroy_work_item(work_item_ref);

    destroy_work_item(input_item);

    nn_device_interface_close(&device_interface_0);
    nn_device_unload();

    return return_value;
}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_fullyconnected_artificial, cpu_fullyconnected)
{
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    uint32_t batches[] = { 1, 8, 48 };
    uint32_t biases_modes[] = { false, true };

    for (auto batch : batches)
        for (auto activation : activations)
            for (auto bias_mode : biases_modes)
                for (uint32_t input_sizes = 1; input_sizes < 20; ++input_sizes)
                    for (uint32_t output_sizes = 1; output_sizes < 20; ++output_sizes)
                        EXPECT_EQ(true, ult_perform_test(
                            input_sizes,   // input width
                            output_sizes,  // output width
                            batch,         // batch size
                            bias_mode,     // bias in output
                            activation));  // activation function
}