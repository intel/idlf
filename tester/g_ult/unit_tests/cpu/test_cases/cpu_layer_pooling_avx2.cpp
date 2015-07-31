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
#include "device/cpu/core/layer_pooling_avx2.h"

namespace
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// Helper classess and functions.
bool compare_output(const nn::data<float, 4> &output, const nn::data<float, 4> &output_ref) {
    assert(output.size[0] == output_ref.size[0]);
    assert(output.size[1] == output_ref.size[1]);
    assert(output.size[2] == output_ref.size[2]);
    assert(output.size[3] == output_ref.size[3]);

    for (uint32_t batch = 0; batch < output.size[3]; ++batch)
    {
        for (uint32_t output_element_y = 0; output_element_y < output.size[2]; ++output_element_y)
        {
            for (uint32_t output_element_x = 0; output_element_x < output.size[1]; ++output_element_x)
            {
                for (uint32_t output_element_z = 0; output_element_z < output.size[0]; ++output_element_z)
                {
                    float value = output.at(output_element_z, output_element_x, output_element_y, batch);
                    float value_ref = output_ref.at(output_element_z, output_element_x, output_element_y, batch);

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
                        if (fabs(diff / value_ref) > 5.2e-04F)
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }

    return true;
}

void run_work_item(nn_workload_item *work_item, nn_device_t *device) {
    // Use optimized routine.
    layer::wrapper_pooling_work_item(work_item);
}

void destroy_work_item(nn_workload_item *&work_item) {
    work_item->input.clear();

    delete reinterpret_cast<nn::workload_data<float> *>(work_item->output[0]);

    delete work_item;

    work_item = nullptr;
}

void create_and_initialize_input_item(
    nn_workload_item* &work_item,
    uint32_t input_size_x,
    uint32_t input_size_y,
    uint32_t input_size_z,
    uint32_t batch_size)
{
    nn_workload_data_coords_t in_out_coords =
    {
        batch_size,
        input_size_x,
        input_size_y,
        input_size_z,
        1,
        1
    };

    work_item = new nn_workload_item();

    work_item->type = NN_WORK_ITEM_TYPE_INPUT;

    work_item->arguments.input.index = 0;

    nn_workload_data_layout_t inp_out_layout = nn::workload_data<float>::layout.zxynpq;

    work_item->output.push_back(new nn::workload_data<float>(in_out_coords, inp_out_layout));

    for (uint32_t batch = 0; batch < batch_size; ++batch)
    {
        for (uint32_t input_element_y = 0; input_element_y < input_size_y; ++input_element_y)
        {
            for (uint32_t input_element_x = 0; input_element_x < input_size_x; ++input_element_x)
            {
                for (uint32_t input_element_z = 0; input_element_z < input_size_z; ++input_element_z)
                {
                    float value = 0.03125f;
                    value *= pow(1.01f, input_size_x);
                    value *= pow(1.02f, input_size_y);
                    value *= pow(1.03f, input_size_z);
                    value *= pow(1.04f, batch);
                    if (input_element_z % 2) value *= -1.0f;
                    nn_workload_data_get<float>(work_item->output[0], batch, input_element_x, input_element_y, input_element_z, 0, 0) = value;
                }
            }
        }

    }
}

void create_and_initialize_work_item(
    nn_workload_item* &work_item,
    nn_workload_item* input_item,
    uint32_t output_size_x,
    uint32_t output_size_y,
    uint32_t size_z,
    uint32_t pool_stride_x,
    uint32_t pool_stride_y,
    uint32_t pool_size_x,
    uint32_t pool_size_y,
    uint32_t batch_size,
    bool check_view,
    nn_device_t *device)
{
    nn_workload_data_coords_t in_out_coords =
    {
        batch_size,
        output_size_x,
        output_size_y,
        size_z,
        1,
        1
    };

    work_item = new nn_workload_item();

    work_item->type = NN_WORK_ITEM_TYPE_POOLING;
    auto primitive = new layer::pooling_f32(NN_POOLING_MODE_MAX,
                                            pool_size_x,
                                            pool_size_y,
                                            pool_stride_x,
                                            pool_stride_y,
                                            size_z,
                                            output_size_x,
                                            output_size_y,
                                            batch_size,
                                            check_view ? 1 : 0,
                                            check_view ? 4 : 0,
                                            check_view ? 1 : 0,
                                            check_view ? 4 : 0,
                                            reinterpret_cast<nn_device_internal *>(device));
    work_item->primitive = primitive;

    work_item->input.push_back({ input_item, 0 });

    nn_workload_data_layout_t inp_out_layout = nn::workload_data<float>::layout.zxynpq;

    work_item->output.push_back(primitive->create_outputs()[0]);

    memset(work_item->output[0]->parent->data_buffer, 0, work_item->output[0]->parent->buffer_size);
}

bool ult_perform_test(
    uint32_t input_size_x,
    uint32_t input_size_y,
    uint32_t input_size_z,
    uint32_t output_size_x,
    uint32_t output_size_y,
    uint32_t pool_stride_x,
    uint32_t pool_stride_y,
    uint32_t pool_size_x,
    uint32_t pool_size_y,
    uint32_t batch_size,
    bool check_views)
{
    bool return_value = true;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;

    nn_device_load(&device_description);
    nn_device_interface_open(0, &device_interface_0);

    // Input item.
    nn_workload_item* input_item = nullptr;
    create_and_initialize_input_item(input_item, input_size_x, input_size_y, input_size_z, batch_size);

    // Work item.
    nn_workload_item* work_item = nullptr;
    create_and_initialize_work_item(work_item, input_item, output_size_x, output_size_y, input_size_z, pool_stride_x, pool_stride_y, pool_size_x, pool_size_y, batch_size, check_views, device_interface_0.device);

    // Execute optimized workload item.
    run_work_item(work_item, device_interface_0.device);

    auto output_ref = nn::data<float, 4>(input_size_z, output_size_x, output_size_y, batch_size);
    auto output = nn::data<float, 4>(input_size_z, output_size_x, output_size_y, batch_size);

    // Execute reference item.
    run_reference(reinterpret_cast<nn::workload_data<float> *>(input_item->output[0]),
                                  output_ref,
                                  pool_size_x,
                                  pool_size_y,
                                  pool_stride_x,
                                  pool_stride_y);

    copy_data(reinterpret_cast<nn_device_internal *>(device_interface_0.device), &output, work_item->output[0]);

    // Compare results.
    return_value &= compare_output(output, output_ref);

    // Cleanup.
    destroy_work_item(work_item);

    destroy_work_item(input_item);

    nn_device_interface_close(&device_interface_0);
    nn_device_unload();

    return return_value;
}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_pooling_artificial_max, cpu_pooling_base)
{
    uint32_t batches[] = { 1, 8, 48 };
    for (auto batch : batches)
    {
        for (uint32_t input_sizes_z = 8; input_sizes_z <= 128; input_sizes_z += 8)
        {
            EXPECT_EQ(true, ult_perform_test(2, 2, input_sizes_z, 1, 1, 2, 2, 2, 2, batch, false));
            EXPECT_EQ(true, ult_perform_test(4, 4, input_sizes_z, 2, 2, 2, 2, 2, 2, batch, false));

            EXPECT_EQ(true, ult_perform_test(2, 1, input_sizes_z, 1, 1, 2, 1, 2, 1, batch, false));
            EXPECT_EQ(true, ult_perform_test(4, 1, input_sizes_z, 2, 1, 2, 1, 2, 1, batch, false));

            EXPECT_EQ(true, ult_perform_test(1, 2, input_sizes_z, 1, 1, 1, 2, 1, 2, batch, false));
            EXPECT_EQ(true, ult_perform_test(1, 4, input_sizes_z, 1, 2, 1, 2, 1, 2, batch, false));

            EXPECT_EQ(true, ult_perform_test(5, 5, input_sizes_z, 2, 2, 2, 2, 3, 3, batch, false));
        }
    }
}

TEST(cpu_pooling_artificial_max, cpu_pooling_base_view)
{
    uint32_t batches[] = { 1, 8, 48 };
    for (auto batch : batches)
    {
        for (uint32_t input_sizes_z = 8; input_sizes_z <= 128; input_sizes_z += 8)
        {
            EXPECT_EQ(true, ult_perform_test(4, 4, input_sizes_z, 2, 2, 2, 2, 2, 2, batch, true));

            EXPECT_EQ(true, ult_perform_test(4, 1, input_sizes_z, 2, 1, 2, 1, 2, 1, batch, true));

            EXPECT_EQ(true, ult_perform_test(5, 5, input_sizes_z, 2, 2, 2, 2, 3, 3, batch, true));
        }
    }
}