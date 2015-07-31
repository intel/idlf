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
#include <algorithm>
#include <random>
#include "gtest/gtest.h"

#include "device/common/nn_workload_data.h"
#include "device/cpu/core/layer_dropout.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"


namespace
{
    nn_device_interface_0_t load_device()
    {
        nn_device_description_t device_description;
        nn_device_interface_0_t di;

        // Check device loading APIs.
        EXPECT_EQ(0, nn_device_load(&device_description));
        EXPECT_EQ(0, device_description.version_first);
        EXPECT_EQ(0, nn_device_interface_open(device_description.version_first, &di));

        return di;
    }

    void unload_device_and_delete_workload(
        nn_device_interface_0_t* di,
        nn_workload_t* workload)
    {
        // Check workload delete API.
        EXPECT_EQ(0, di->workload_delete_function(workload));

        // Check device unloading APIs.
        EXPECT_EQ(0, nn_device_interface_close(di));
        EXPECT_EQ(0, nn_device_unload());
    }

    nn_workload_t* create_workload(
        nn_device_interface_0_t& di,
        uint32_t batch,
        nn_output_format input_format,
        nn_output_format output_format)
    {
        // Create workflow.
        nn_workflow_t *workflow = nullptr;
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&workflow, 3, 0));

        // Create data input.
        nn_workflow_item_t *data_input = nullptr;
        {
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&data_input, 0, nullptr, 1));
            data_input->type = NN_WORK_ITEM_TYPE_INPUT;

            data_input->arguments.input.index = 0;

            data_input->output_format[0] = input_format;
        }

        // Create seed input.
        nn_workflow_item_t *seed_input = nullptr;
        {
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&seed_input, 0, nullptr, 1));
            seed_input->type = NN_WORK_ITEM_TYPE_INPUT;

            seed_input->arguments.input.index = 1;

            seed_input->output_format[0] = input_format;
        }

        // Create if_train input.
        nn_workflow_item_t *if_train_input = nullptr;
        {
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&if_train_input, 0, nullptr, 1));
            if_train_input->type = NN_WORK_ITEM_TYPE_INPUT;

            if_train_input->arguments.input.index = 2;

            if_train_input->output_format[0] = input_format;
        }

        // Create dropout.
        nn_workflow_item_t *dropout = nullptr;
        {
            nn_workflow_use_descriptor_t desc0[] = {{ data_input, 0 }, { seed_input, 0 }, { if_train_input, 0 }};
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&dropout, 3, desc0, 1));
            dropout->type = NN_WORK_ITEM_TYPE_DROPOUT;

            dropout->arguments.dropout.drop_rate = 0.5f; // Should be ignored.

            dropout->output_format[0] = output_format;
        }

        // Pin inputs.
        workflow->input[0] = data_input;
        workflow->input[1] = seed_input;
        workflow->input[2] = if_train_input;

        // Compile workload.
        NN_WORKLOAD_DATA_TYPE io_format[3] = { NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH, NN_WORKLOAD_DATA_TYPE_I32_1D, NN_WORKLOAD_DATA_TYPE_I32_1D };
        nn_workload_t *workload = nullptr;
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_compile_function(&workload, di.device, workflow, io_format, nullptr, batch));

        // Cleanup workflow.
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(data_input));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(seed_input));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(if_train_input));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(dropout));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));

        return workload;
    }

    nn_workload_item_t* get_forward(
        nn_workload_t* workload)
    {
        nn_workload_opaque_t* workload_opaque = reinterpret_cast<nn_workload_opaque_t*>(workload + 1);
        nn_workload_item_t *workload_backprop = workload_opaque->order_of_execution.back();
        assert(workload_backprop->type == NN_WORK_ITEM_TYPE_DROPOUT);

        return workload_backprop;
    }

    void naive(
        nn_workload_data_t *input_data,
        nn_workload_data_t *output)
    {
        // Forward pass of classification is just passthrough...
        nn_workload_data_copy(output, input_data);                       
    }

    bool compare_data_items(
        nn_workload_data_t* data_item,
        nn_workload_data_t* data_item_ref)
    {
        nn::workload_data<float> data(data_item->parent->data_buffer, data_item->parent->lengths, data_item->parent->layout);
        nn::workload_data<float> reference(data_item_ref->parent->data_buffer, data_item_ref->parent->lengths, data_item_ref->parent->layout);
        auto& size = data_item->parent->lengths;
        for (auto n = 0u; n < size.t[0]; ++n)
            for (auto x = 0u; x < size.t[1]; ++x)
                for (auto y = 0u; y < size.t[2]; ++y)
                    for (auto z = 0u; z < size.t[3]; ++z)
                        for (auto p = 0u; p < size.t[4]; ++p)
                            for (auto q = 0u; q < size.t[5]; ++q)
                            {
                                float value = data(n, x, y, z, p, q);
                                float value_ref = reference(n, x, y, z, p, q);

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
                                    if (fabs(diff / value_ref) > 5.2e-05F)
                                    {
                                        return false;
                                    }
                                }
                            }

        return true;
    }

    void run_test(
        uint32_t input_fmaps,
        uint32_t batch,
        uint32_t seed,
        bool negative)
    {
        nn::output_format input_format(input_fmaps);
        nn::output_format output_format = input_format;

        // Load device.
        nn_device_interface_0_t di = load_device();

        // Create workflow and compile it.
        nn_workload_t* workload = create_workload(
            di,
            batch,
            input_format,
            output_format);

        // Get last item (should be forward pass dropout) from internal workload.
        nn_workload_item_t* dropout_forward = get_forward(
            workload);

        // Create input buffers and initialize them.
        void* input_datas[3] =
        {
            new nn::data<float, 2>(input_format.size(0), batch),
            new nn::data<int32_t>(1), // dropout seed
            new nn::data<int32_t>(1), // dropout is_training
        };

        for (uint32_t n = 0; n < reinterpret_cast<nn::data<float>**>(input_datas)[0]->size[1]; ++n)
            for (uint32_t x = 0; x < reinterpret_cast<nn::data<float>**>(input_datas)[0]->size[0]; ++x)
                (*reinterpret_cast<nn::data<float>**>(input_datas)[0])(x, n) = 1.0f;

        (*reinterpret_cast<nn::data<int32_t>**>(input_datas)[1])(0) = seed;
        (*reinterpret_cast<nn::data<int32_t>**>(input_datas)[2])(0) = 0;

        // Get refernces to all buffers used by dropout and its backprop.
        // "Inputs" to forward pass.
        auto forward_data_input = dropout_forward->input[0].get_data_view();

        // Outputs of forward pass.
        auto forward_output = dropout_forward->output[0];

        // Create reference outputs with same layout and sizes.
        nn::workload_data<float> ref_forward_output(forward_output->parent->lengths, forward_output->parent->layout);

        std::memset(ref_forward_output.parent->data_buffer, 0, ref_forward_output.parent->buffer_size);

        // Run optimized code.
        NN_API_STATUS status;
        EXPECT_EQ(NN_API_STATUS_OK, di.workload_execute_function(workload, (void **)input_datas, nullptr, &status));

        // Run naive code.
        forward_data_input->parent->data_buffer = reinterpret_cast<nn::data<int32_t>**>(input_datas)[0]->buffer;

        // Check forward pass of classification.
        naive(
            forward_data_input,
            &ref_forward_output);

        if (negative)
        {
            // Random error for each buffer.
            std::mt19937 gen(1);
            std::uniform_int_distribution<uint32_t> dis_error(0, ref_forward_output.parent->buffer_size / sizeof(float) - 1);

            auto forward_error_index = dis_error(gen);

            static_cast<float*>(forward_output->parent->data_buffer)[forward_error_index] += 1.0f;

            EXPECT_NE(true, compare_data_items(forward_output, &ref_forward_output));
        }
        else
        {
            EXPECT_EQ(true, compare_data_items(forward_output, &ref_forward_output));
        }

        unload_device_and_delete_workload(&di, workload);

        delete reinterpret_cast<nn::data<float>**>(input_datas)[0];
        delete reinterpret_cast<nn::data<int32_t>**>(input_datas)[1];
        delete reinterpret_cast<nn::data<int32_t>**>(input_datas)[2];
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_dropout_classification, standard_square_positive)
{
    for (auto batch : { 1, 8, 48 })
        for (auto in_z = 1; in_z < 35; ++in_z)
            for (auto seed : { 1, 10, 50 } )
                run_test(in_z, batch, seed, false);
}

TEST(cpu_dropout_classification, standard_square_negative)
{
    for (auto batch : { 1, 8, 48 })
        for (auto in_z = 1; in_z < 35; ++in_z)
            for (auto seed : { 1, 10, 50 } )
                run_test(in_z, batch, seed, true);
}

