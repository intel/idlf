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
#include "device/api/nn_primitives_api_0.h"


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
        float drop_rate,
        nn_output_format input_format,
        nn_output_format output_format)
    {
        // Create workflow.
        nn_workflow_t *workflow = nullptr;
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&workflow, 4, 0));

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

            dropout->arguments.dropout.drop_rate = drop_rate;

            dropout->output_format[0] = output_format;
        }

        // Create second input (target values).
        nn_workflow_item_t *input_target = nullptr;
        {
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&input_target, 0, nullptr, 1));
            input_target->type = NN_WORK_ITEM_TYPE_INPUT;

            input_target->arguments.input.index = 3;

            input_target->output_format[0] = output_format;
        }

        // Create loss function.
        nn_workflow_item_t *loss_function = nullptr;
        {
            nn_workflow_use_descriptor_t desc0[2] = { { dropout, 0 }, { input_target, 0 } };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&loss_function, 2, desc0, 1));
            loss_function->type = NN_WORK_ITEM_TYPE_LOSS_FUNCTION;

            loss_function->arguments.loss_function.function = NN_LOSS_FUNCTION_SUM_OF_SQUARES;

            loss_function->output_format[0] = output_format;
        }

        // Create dropout backprop.
        nn_workflow_item_t *dropout_backprop = nullptr;
        {
            nn_workflow_use_descriptor_t desc0[] = {{ loss_function, 0 }, { seed_input, 0 }, { if_train_input, 0 }};
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&dropout_backprop, 3, desc0, 1));
            dropout_backprop->type = NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP;

            dropout_backprop->forward_item = dropout;
        }

        // Pin inputs.
        workflow->input[0] = data_input;
        workflow->input[1] = seed_input;
        workflow->input[2] = if_train_input;
        workflow->input[3] = input_target;

        // Compile workload.
        NN_WORKLOAD_DATA_TYPE io_format[4] = { NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH, NN_WORKLOAD_DATA_TYPE_I32_1D, NN_WORKLOAD_DATA_TYPE_I32_1D, NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH };
        nn_workload_t *workload = nullptr;
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_compile_function(&workload, di.device, workflow, io_format, nullptr, batch));

        // Cleanup workflow.
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(data_input));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(seed_input));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(if_train_input));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input_target));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(dropout));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(loss_function));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(dropout_backprop));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));

        return workload;
    }

    nn_workload_item_t* get_backprop(
        nn_workload_t* workload)
    {
        nn_workload_opaque_t* workload_opaque = static_cast<nn_workload_opaque_t*>(workload);
        nn_workload_item_t *workload_backprop = workload_opaque->order_of_execution.back();
        assert(workload_backprop->type == NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP);

        return workload_backprop;
    }

    float get_multiplier(
        std::mt19937& gen,
        uint32_t threshold,
        float scale)
    {
        uint32_t intermediate = (~static_cast<uint32_t>((static_cast<int64_t>(gen()) - threshold) >> 32)) & *reinterpret_cast<uint32_t*>(&scale);
        return *reinterpret_cast<float*>(&intermediate);
    }

    void naive(float drop_rate,
                        nn_workload_data_t *input_data,
                        nn_workload_data_t *input_seed,
                        nn_workload_data_t *input_if_train,
                        nn_workload_data_t *output)
    {
        // Second input - seed.
        auto seed = static_cast<uint32_t>(nn_workload_data_get<int32_t>(input_seed, 0, 0, 0, 0, 0, 0));

        // Third input - indicates if its test or training phase.
        auto is_training = nn_workload_data_get<int32_t>(input_if_train, 0, 0, 0, 0, 0, 0) != 0;  

        std::mt19937 gen(seed);
        auto threshold = static_cast<uint32_t>(static_cast<double>(drop_rate)*std::numeric_limits<unsigned int>::max());
        float scale = 1.0f / (1.0f - drop_rate);

        for (uint32_t x = 0; x < output->parent->lengths.t[NN_DATA_COORD_x]; ++x)
            for (uint32_t n = 0; n < output->parent->lengths.t[NN_DATA_COORD_n]; ++n)   
            {
                if (is_training)
                    nn_workload_data_get<float>(output, n, x, 0, 0, 0, 0) = nn_workload_data_get<float>(input_data, n, x, 0, 0, 0, 0) * get_multiplier(gen, threshold, scale);
                else
                    nn_workload_data_get<float>(output, n, x, 0, 0, 0, 0) = nn_workload_data_get<float>(input_data, n, x, 0, 0, 0, 0);
            }
                              
    }

    bool compare_data_items(
        nn_workload_data_t* data_item,
        nn_workload_data_t* data_item_ref)
    {
        nn::workload_data<nn::layout_f32> data(data_item->parent->data_buffer, data_item->parent->lengths, data_item->parent->layout);
        nn::workload_data<nn::layout_f32> reference(data_item_ref->parent->data_buffer, data_item_ref->parent->lengths, data_item_ref->parent->layout);
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

    void run_primitiv_api(
        nn::workload_data<>* forward_data_input,
        nn::workload_data<>* forward_seed_input,
        nn::workload_data<>* forward_if_train_input,
        nn::workload_data<>* forward_output,        
        nn::workload_data<>* backward_input,        //< Buffer with input to backprop
        nn::workload_data<>* backward_error_delta,  //< Buffer for output of backprop calculation
        uint32_t input_fmaps,
        uint32_t batch,
        float drop_rate)
    {
        nn_device_primitives_description_t device_primitives_description;
        nn_device_get_primitives_description(&device_primitives_description);
        assert(device_primitives_description.version_first <= 0);
        assert(device_primitives_description.version_last >= 0);

        nn_primitives_0_t primitives;
        nn_device_get_primitives(0, &primitives);
        nn_device_t *device = primitives.create_device_with_thread_count(0, nullptr);

        nn_primitive_handle_t primitive = primitives.create_handle.dropout_f32(
          device,
          1, 1, input_fmaps, batch , drop_rate, nullptr);

        // 1. Make & prepare inputs to dropout. as we have opaque_data given we can
        nn_opaque_data_t* input_internal[3];
        input_internal[0] = reinterpret_cast<nn_opaque_data_t*>(forward_data_input);
        input_internal[1] = reinterpret_cast<nn_opaque_data_t*>(forward_seed_input);
        input_internal[2] = reinterpret_cast<nn_opaque_data_t*>(forward_if_train_input);

        assert(forward_data_input->parent->delta_buffer == nullptr);
        assert(forward_output->parent->delta_buffer == nullptr);

        forward_data_input->parent->delta_buffer = backward_error_delta->parent->data_buffer; // this is backward_output for primitives API
        forward_output->parent->delta_buffer = backward_input->parent->data_buffer; // this is backward_input for primitives API

        // 2. prepare output buffer 
        nn_opaque_data_t* output_internal = reinterpret_cast<nn_opaque_data_t*>(forward_output);

        // 3. Call droput backward
        nn_event_t dropout = primitives.backward_async(
            primitive, 3, input_internal, 0, nullptr, 1, &output_internal, 0, nullptr, nullptr);

        primitives.wait(1, &dropout);

        // Cleanup
        forward_data_input->parent->delta_buffer = nullptr;
        forward_output->parent->delta_buffer = nullptr;

        primitives.delete_event(dropout);
        primitives.delete_primitive(primitive);
        primitives.delete_device(device);
    }


    void run_test(
        uint32_t input_fmaps,
        uint32_t batch,
        float drop_rate,
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
            drop_rate,
            input_format,
            output_format);

        // Get last item (should be backprop) from internal workload.
        nn_workload_item_t* dropout_backprop = get_backprop(
            workload);

        // Create input buffers and initialize them.
        void* input_datas[4] =
        {
            new nn::data<float, 2>(input_format.size(0), batch),
            new nn::data<int32_t>(1), // dropout seed
            new nn::data<int32_t>(1), // dropout is_training
            new nn::data<float, 2>(output_format.size(0), batch)
        };

        for (uint32_t n = 0; n < reinterpret_cast<nn::data<float>**>(input_datas)[0]->size[1]; ++n)
            for (uint32_t x = 0; x < reinterpret_cast<nn::data<float>**>(input_datas)[0]->size[0]; ++x)
                (*reinterpret_cast<nn::data<float>**>(input_datas)[0])(x, n) = 1.0f * static_cast<float>(n + x);

        (*reinterpret_cast<nn::data<int32_t>**>(input_datas)[1])(0) = seed;
        (*reinterpret_cast<nn::data<int32_t>**>(input_datas)[2])(0) = 1;

        for (uint32_t n = 0; n < reinterpret_cast<nn::data<float>**>(input_datas)[3]->size[1]; ++n)
            for (uint32_t x = 0; x < reinterpret_cast<nn::data<float>**>(input_datas)[3]->size[0]; ++x)
                (*reinterpret_cast<nn::data<float>**>(input_datas)[3])(x, n) = 0.5f * static_cast<float>(n + x);

        // Run optimized code.
        NN_API_STATUS status;
        EXPECT_EQ(NN_API_STATUS_OK, di.workload_execute_function(workload, (void **)input_datas, nullptr, &status));

        // Get refernces to all buffers used by dropout and its backprop.
        // "Inputs" to backprop and forward pass.
        auto forward_data_input = dropout_backprop->forward_item->input[0].get_data_view();
        auto forward_seed_input = dropout_backprop->forward_item->input[1].get_data_view();
        auto forward_if_train_input = dropout_backprop->forward_item->input[2].get_data_view();
        auto backward_data_input = dropout_backprop->input[0].get_data_view();

        // Outputs of forward pass.
        auto forward_output = dropout_backprop->forward_item->output[0];

        // Outputs of backprop.
        auto backward_error_delta = dropout_backprop->output[0];

        // Create reference outputs with same layout and sizes.
        nn::workload_data<> ref_backward_error_delta(backward_error_delta->parent->lengths, backward_error_delta->parent->layout);
        nn::workload_data<> ref_forward_output(forward_output->parent->lengths, forward_output->parent->layout);

        std::memset(ref_backward_error_delta.parent->data_buffer, 0, ref_backward_error_delta.parent->buffer_size);
        std::memset(ref_forward_output.parent->data_buffer, 0, ref_forward_output.parent->buffer_size);

        // Check backward pass of training.
        // Also check forward pass of training (do not confuse with test pass).
        naive(
            drop_rate,
            backward_data_input,
            forward_seed_input,
            forward_if_train_input,
            &ref_backward_error_delta);

        naive(
            drop_rate,
            forward_data_input,
            forward_seed_input,
            forward_if_train_input,
            &ref_forward_output);

        // Run dropout training using primitive API 
        nn::workload_data<> prim_backward_error_delta(backward_error_delta->parent->lengths, backward_error_delta->parent->layout);

        run_primitiv_api(
            static_cast<nn::workload_data<>*>(forward_data_input),
            static_cast<nn::workload_data<>*>(forward_seed_input), 
            static_cast<nn::workload_data<>*>(forward_if_train_input), 
            static_cast<nn::workload_data<>*>(forward_output), 
            static_cast<nn::workload_data<>*>(backward_data_input),
            &prim_backward_error_delta,
            input_fmaps,
            batch,
            drop_rate);

        if (negative)
        {
            // Random error for each buffer.
            std::mt19937 gen(1);
            std::uniform_int_distribution<uint32_t> dis_error(0, backward_error_delta->parent->buffer_size / sizeof(float) - 1);

            auto forward_error_index = dis_error(gen);
            auto backward_error_index = dis_error(gen);

            static_cast<float*>(backward_error_delta->parent->data_buffer)[backward_error_index] += 1.0f;
            static_cast<float*>(forward_output->parent->data_buffer)[forward_error_index] += 1.0f;
            static_cast<float*>(prim_backward_error_delta.parent->data_buffer)[backward_error_index] += 1.0f;

            EXPECT_NE(true, compare_data_items(backward_error_delta, &ref_backward_error_delta));
            EXPECT_NE(true, compare_data_items(forward_output, &ref_forward_output));
            EXPECT_NE(true, compare_data_items(&prim_backward_error_delta, &ref_backward_error_delta));
        }
        else
        {
            EXPECT_EQ(true, compare_data_items(backward_error_delta, &ref_backward_error_delta));
            EXPECT_EQ(true, compare_data_items(forward_output, &ref_forward_output));
            EXPECT_EQ(true, compare_data_items(&prim_backward_error_delta, &ref_backward_error_delta));
        }

        unload_device_and_delete_workload(&di, workload);

        delete reinterpret_cast<nn::data<float>**>(input_datas)[0];
        delete reinterpret_cast<nn::data<int32_t>**>(input_datas)[1];
        delete reinterpret_cast<nn::data<int32_t>**>(input_datas)[2];
        delete reinterpret_cast<nn::data<float>**>(input_datas)[3];
    }



}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_dropout_training, standard_square_positive)
{
    for (auto batch : { 1, 8, 48 })
        for (auto in_z = 1; in_z < 35; ++in_z)
            for (auto ratio : { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f })
                for (auto seed : { 1, 10, 50 } )
                    run_test(in_z, batch, ratio, seed, false);
}

TEST(cpu_dropout_training, standard_square_negative)
{
    for (auto batch : { 1, 8, 48 })
        for (auto in_z = 1; in_z < 35; ++in_z)
            for (auto ratio : { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f })
                for (auto seed : { 1, 10, 50 } )
                    run_test(in_z, batch, ratio, seed, true);
}
