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

#include <immintrin.h>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include "gtest/gtest.h"

#include "../../devices/common/nn_workload_data.h"
#include "../../devices/device_cpu/core/layer_softmax_avx2.h"
#include "../../devices/device_cpu/api_internal/nn_device_interface_0_internal.h"

namespace
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// Helper classess and functions.
bool compare_work_items(
    nn_workload_item* &work_item, 
    nn_workload_item* &work_item_ref)
{
    for (uint32_t batch = 0; batch < work_item->output->parent->lengths.t[NN_DATA_COORD_n]; ++batch)
    {
        for (uint32_t output_element = 0; output_element < work_item->output->parent->lengths.t[NN_DATA_COORD_x]; ++output_element)
        {
            float value = nn_workload_data_get<float>(work_item->output, batch, output_element, 0, 0, 0, 0);
            float value_ref = nn_workload_data_get<float>(work_item_ref->output, batch, output_element, 0, 0, 0, 0);

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
                if (diff / value_ref > 5.2e-05F)
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
    bool is_ref,
    nn_device_t *device)
{
    if (is_ref)
    {
        // Naive implementation.
        for (uint32_t batch = 0; 
            batch < work_item->output->parent->lengths.t[NN_DATA_COORD_n];
            ++batch)
        {
            float sum = 0.0f;
            for (uint32_t output_element = 0; 
                output_element < work_item->output->parent->lengths.t[NN_DATA_COORD_x];
                ++output_element)
            {
                float value = exp(nn_workload_data_get<float>(work_item->input[0]->output, batch, output_element, 0, 0, 0, 0));
                sum += value;

                nn_workload_data_get<float>(work_item->output, batch, output_element, 0, 0, 0, 0) = value;
            }

            sum = 1.0f / sum;

            for (uint32_t output_element = 0;
                output_element < work_item->output->parent->lengths.t[NN_DATA_COORD_x];
                ++output_element)
            {
                float value = nn_workload_data_get<float>(work_item->output, batch, output_element, 0, 0, 0, 0);
                value *= sum;

                nn_workload_data_get<float>(work_item->output, batch, output_element, 0, 0, 0, 0) = value;
            }

        }
    }
    else
    {
        // Use optimized routine.
        layer::wrapper_softmax_work_item(work_item, reinterpret_cast<nn_device_internal*>(device));
    }

    return true;
}

void destroy_work_item(
    nn_workload_item* &work_item)
{
    work_item->input.clear();

    delete reinterpret_cast<nn::nn_workload_data_t<float>*>(work_item->output);

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

    nn_workload_data_layout_t inp_out_layout =
    {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_FLOAT
    };

    work_item->output = new nn::nn_workload_data_t<float>(in_out_coords, inp_out_layout);

    for (uint32_t batch = 0; batch < batch_size; ++batch)
    {
        for (uint32_t input_element = 0; input_element < input_width; ++input_element)
        {
            float value = 0.03125f;
            value *= pow(1.01f, input_element);
            value *= pow(1.01f, batch);
            if (input_element % 2) value *= -1.0f;
            nn_workload_data_get<float>(work_item->output, batch, input_element, 0, 0, 0, 0) = value;
        }
    }
}

void create_and_initialize_work_item(
    nn_workload_item* &work_item,
    nn_workload_item* input_item,
    uint32_t input_width,
    uint32_t batch_size,
    bool is_ref,
    nn_device_t *device)
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

    work_item->type = NN_WORK_ITEM_TYPE_SOFTMAX;
    work_item->primitive = layer::softmax_f32::create(input_width, batch_size, device);

    work_item->input.push_back(input_item);

    nn_workload_data_layout_t inp_out_layout =
    {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_FLOAT
    };

    work_item->output = new nn::nn_workload_data_t<float>(in_out_coords, inp_out_layout);

    for (uint32_t batch = 0; batch < batch_size; ++batch)
    {
        for (uint32_t output_element = 0; output_element < input_width; ++output_element)
        {
            nn_workload_data_get<float>(work_item->output, batch, output_element, 0, 0, 0, 0) = 0.0f;
        }
    }
}

bool ult_perform_test(
    uint32_t input_width,
    uint32_t batch_size)
{
    bool return_value = true;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;

    nn_device_load(&device_description);
    nn_device_interface_open(0, &device_interface_0);

    // Input item.
    nn_workload_item* input_item = nullptr;
    create_and_initialize_input_item(input_item, input_width, batch_size);

    // Work item.
    nn_workload_item* work_item = nullptr;
    create_and_initialize_work_item(work_item, input_item, input_width, batch_size, false, device_interface_0.device);

    // Reference workload item.
    nn_workload_item* work_item_ref = nullptr;
    create_and_initialize_work_item(work_item_ref, input_item, input_width, batch_size, true, nullptr);

    // Run items.
    return_value &= run_work_item(work_item, false, device_interface_0.device);
    return_value &= run_work_item(work_item_ref, true, nullptr);

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
TEST(cpu_softmax_artificial, cpu_softmax_base)
{
    uint32_t batches[] = { 1, 8, 48 };
    for (auto batch : batches)
        for (uint32_t input_sizes = 1; input_sizes < 256; ++input_sizes)
            EXPECT_EQ(true, ult_perform_test(
                input_sizes,   // input/output width
                batch          // batch size
                ));
}