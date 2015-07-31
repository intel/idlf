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

namespace
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// Helper classes and functions.
bool run_work_item(nn_workload_item* &work_item, nn_device_t* device)
{
    work_item->primitive->forward({work_item->input[0].get_data_view()}, {}, work_item->output);
    return true;
}

void destroy_work_item(
    nn_workload_item* &work_item)
{
    work_item->input.clear();

    delete reinterpret_cast<nn::workload_data<float>*>(work_item->output[0]);

    delete work_item;

    work_item = nullptr;
}

template<class T_primitive>
nn::data<float, 4> create_input(
    size_t input_width_x, size_t input_width_y, size_t input_width_z, uint32_t batch_size) {
    nn::data<float, 4> input(input_width_z, input_width_x, input_width_y, batch_size);

    for (uint32_t batch = 0; batch < batch_size; ++batch)
    {
        for (uint32_t input_element_y = 0; input_element_y < input_width_y; ++input_element_y)
        {
            for (uint32_t input_element_x = 0; input_element_x < input_width_x; ++input_element_x)
            {
                for (uint32_t input_element_z = 0; input_element_z < input_width_z; ++input_element_z)
                {
                    float value = 0.03125f;
                    value *= pow(1.01f, input_width_x);
                    value *= pow(1.02f, input_width_y);
                    value *= pow(1.03f, input_width_z);
                    value *= pow(1.04f, batch);

                    if (input_element_z % 2)
                        value *= -1.0f;

                    input.at(input_element_z, input_element_x, input_element_y, batch) = value;
                }
            }
        }
    }

    return input;
}

template <class T_primitive>
void create_and_initialize_input_item(nn_workload_item *&work_item,
                                      T_primitive *primitive,
                                      const nn::data<float, 4> &input) {
    work_item = new nn_workload_item();
    work_item->type = NN_WORK_ITEM_TYPE_INPUT;
    work_item->arguments.input.index = 0;
    work_item->output = primitive->create_input(input);
}

template <class T_primitive>
T_primitive *create_and_initialize_work_item(nn_workload_item *&work_item,
                                             nn::data<float, 4> &input,
                                             uint32_t input_width_x,
                                             uint32_t input_width_y,
                                             uint32_t input_width_z,
                                             float coeff_a,
                                             float coeff_b,
                                             uint32_t coeff_n,
                                             uint32_t coeff_k,
                                             uint32_t batch_size,
                                             bool check_views,
                                             nn_device_t *device) {
    auto input_item = new nn_workload_item();
    input_item->type = NN_WORK_ITEM_TYPE_INPUT;
    input_item->arguments.input.index = 0;

    work_item = new nn_workload_item();
    work_item->type = NN_WORK_ITEM_TYPE_NORMALIZATION;
    work_item->input.push_back({ input_item, 0 });

    if (std::is_same<T_primitive, layer::normalization_elementwise_linear_f32>::value) {
        work_item->arguments.forward_normalization.normalization.mode = NN_NORMALIZATION_MODE_LINEAR_SINGLE;
        work_item->primitive =
            new layer::normalization_elementwise_linear_f32(coeff_a,
                                                            coeff_b,
                                                            input_width_x,
                                                            input_width_y,
                                                            input_width_z,
                                                            batch_size,
                                                            reinterpret_cast<nn_device_internal *>(device));
    } else if (std::is_same<T_primitive, layer::normalization_response_across_maps_f32>::value) {
        work_item->arguments.forward_normalization.normalization.mode = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
        work_item->primitive =
            new layer::normalization_response_across_maps_f32(coeff_a,
                                                              coeff_b,
                                                              coeff_k,
                                                              coeff_n,
                                                              input_width_x,
                                                              input_width_y,
                                                              input_width_z,
                                                              batch_size,
                                                              check_views ? 1 : 0,
                                                              check_views ? 2 : 0,
                                                              check_views ? 3 : 0,
                                                              check_views ? 4 : 0,
                                                              reinterpret_cast<nn_device_internal *>(device));
    } else {
        assert(0);
    }

    input_item->output = work_item->primitive->create_inputs();
    copy_data(reinterpret_cast<nn_device_internal *>(device), input_item->output[0], &input);
    work_item->output = work_item->primitive->create_outputs();

    return static_cast<T_primitive*>(work_item->primitive);
}

template<class T_primitive>
bool ult_perform_test(
    uint32_t input_width_x,
    uint32_t input_width_y,
    uint32_t input_width_z,
    float coeff_a,
    float coeff_b,
    uint32_t coeff_n,
    uint32_t coeff_k,
    uint32_t batch_size,
    bool check_views)
{
    bool return_value = true;

    // Use optimized routine.
    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;

    nn_device_load(&device_description);
    nn_device_interface_open(0, &device_interface_0);

    auto input = create_input<T_primitive>(input_width_x, input_width_y, input_width_z, batch_size);

    // Work item.
    nn_workload_item* work_item = nullptr;
    auto primitive = create_and_initialize_work_item<T_primitive>(work_item,
                                                                  input,
                                                                  input_width_x,
                                                                  input_width_y,
                                                                  input_width_z,
                                                                  coeff_a,
                                                                  coeff_b,
                                                                  coeff_n,
                                                                  coeff_k,
                                                                  batch_size,
                                                                  check_views,
                                                                  device_interface_0.device);

    // Execute optimized workload item.
    return_value &= run_work_item(work_item, device_interface_0.device);

    // Execute reference implementation.
    nn::data<float, 4> output_ref(input_width_z, input_width_x, input_width_y, batch_size);
    run_naive_layer_normalization<T_primitive>(coeff_a, coeff_b, coeff_k, coeff_n, input, output_ref);

    // Compare results.
    return_value &= compare_results(work_item, output_ref);

    // Cleanup.
    destroy_work_item(work_item->input.at(0).item);
    destroy_work_item(work_item);

    nn_device_interface_close(&device_interface_0);
    nn_device_unload();

    return return_value;
}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_normalization_artificial_linear, cpu_normalization_base)
{
    uint32_t batches[] = { 1, 8 };
    for (auto batch : batches)
    {
        for (uint32_t input_sizes_y = 1; input_sizes_y <= 8; ++input_sizes_y)
        {
            for (uint32_t input_sizes_x = 1; input_sizes_x <= 8; ++input_sizes_x)
            {
                for (uint32_t input_sizes_z = 1; input_sizes_z <= 8; ++input_sizes_z)
                {
                    EXPECT_EQ(true,
                              ult_perform_test<layer::normalization_elementwise_linear_f32>(
                                  input_sizes_x, // input/output width
                                  input_sizes_y, // input/output height
                                  input_sizes_z, // input/output depth
                                  1.0f,          // A coefficient
                                  1.0f,          // B coefficient
                                  0,             // N coefficient
                                  0,             // K coefficient
                                  batch,         // batch size
                                  false          // check views
                                  ));
                }
            }
        }
    }
}

TEST(cpu_normalization_artificial_localresponse, cpu_normalization_base)
{
    uint32_t batches[] = { 1, 8 };
    for (auto batch : batches)
    {
        for (uint32_t input_sizes_y = 1; input_sizes_y <= 4; ++input_sizes_y)
        {
            for (uint32_t input_sizes_x = 1; input_sizes_x <= 4; ++input_sizes_x)
            {
                for (uint32_t input_sizes_z = 8; input_sizes_z <= 32; input_sizes_z += 8)
                {
                    float b_coeff = 0.75f;
                    {
                        for (uint32_t k_coeff = 0; k_coeff <= 4; ++k_coeff)
                        {
                            for (uint32_t n_coeff = 3; n_coeff <= 9; n_coeff += 2)
                            {
                                EXPECT_EQ(true,
                                          ult_perform_test<layer::normalization_response_across_maps_f32>(
                                              input_sizes_x, // input/output width
                                              input_sizes_y, // input/output height
                                              input_sizes_z, // input/output depth
                                              2.0f,          // A coefficient
                                              b_coeff,       // B coefficient
                                              n_coeff,       // N coefficient
                                              k_coeff,       // K coefficient
                                              batch,         // batch size
                                              false          // check views
                                              ));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(cpu_normalization_artificial_localresponse, cpu_normalization_view)
{
    uint32_t batches[] = { 1, 8 };
    for (auto batch : batches)
    {
        for (uint32_t input_sizes_y = 1; input_sizes_y <= 4; ++input_sizes_y)
        {
            for (uint32_t input_sizes_x = 2; input_sizes_x <= 4; ++input_sizes_x)
            {
                for (uint32_t input_sizes_z = 8; input_sizes_z <= 32; input_sizes_z += 8)
                {
                    float b_coeff = 0.75f;
                    {
                        for (uint32_t k_coeff = 0; k_coeff <= 4; ++k_coeff)
                        {
                            for (uint32_t n_coeff = 3; n_coeff <= 9; n_coeff += 2)
                            {
                                EXPECT_EQ(true,
                                          ult_perform_test<layer::normalization_response_across_maps_f32>(
                                              input_sizes_x, // input/output width
                                              input_sizes_y, // input/output height
                                              input_sizes_z, // input/output depth
                                              2.0f,          // A coefficient
                                              b_coeff,       // B coefficient
                                              n_coeff,       // N coefficient
                                              k_coeff,       // K coefficient
                                              batch,         // batch size
                                              true           // check views
                                              ));
                            }
                        }
                    }
                }
            }
        }
    }
}