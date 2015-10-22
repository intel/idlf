/*
Copyright (c) 2015, Intel Corporation

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
#include "device/cpu/core/layer_loss_function.h"
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
        nn_output_format output_format,
        NN_LOSS_FUNCTION function)
    {
        // Create workflow.
        nn_workflow_t *workflow = nullptr;
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&workflow, 2, 0));

        // Create data input.
        nn_workflow_item_t *data_input = nullptr;
        {
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&data_input, 0, nullptr, 1));
            data_input->type = NN_WORK_ITEM_TYPE_INPUT;

            data_input->arguments.input.index = 0;

            data_input->output_format[0] = input_format;
        }

        // Create second input (target values).
        nn_workflow_item_t *target_input = nullptr;
        {
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&target_input, 0, nullptr, 1));
            target_input->type = NN_WORK_ITEM_TYPE_INPUT;

            target_input->arguments.input.index = 1;

            target_input->output_format[0] = output_format;
        }

        // Create loss function.
        nn_workflow_item_t *loss_function = nullptr;
        {
            nn_workflow_use_descriptor_t desc0[2] = {{data_input, 0}, {target_input, 0}};
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&loss_function, 2, desc0, 1));
            loss_function->type = NN_WORK_ITEM_TYPE_LOSS_FUNCTION;

            loss_function->arguments.loss_function.function = function;

            loss_function->output_format[0] = output_format;
        }

        // Pin inputs.
        workflow->input[0] = data_input;
        workflow->input[1] = target_input;

        // Compile workload.
        NN_WORKLOAD_DATA_TYPE io_format[2] = { NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH, NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH };
        nn_workload_t *workload = nullptr;
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_compile_function(&workload, di.device, workflow, io_format, nullptr, batch));

        // Cleanup workflow.
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(data_input));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(target_input));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(loss_function));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));

        return workload;
    }

    nn_workload_item_t* get_loss_item(
        nn_workload_t* workload)
    {
        nn_workload_opaque_t* workload_opaque = static_cast<nn_workload_opaque_t*>(workload);
        nn_workload_item_t *loss_item = workload_opaque->order_of_execution.back();
        assert(loss_item->type == NN_WORK_ITEM_TYPE_LOSS_FUNCTION);

        return loss_item;
    }

    void naive_loss_function(
        NN_LOSS_FUNCTION function,
        nn_workload_data_t *current_input,
        nn_workload_data_t *target_input,
        nn_workload_data_t *output)
    {
        auto batch = output->parent->lengths.t[NN_DATA_COORD_n];
        // Simulate U[m+1] = -2 * (t-a[m+1]) computation from output, this equation is valid for last layer as we know exact error.
        // For hidden layers it would be U[m+1] = W[m+1]^T * s[m+1] - should (and will) be provided by them.
        if (function == NN_LOSS_FUNCTION_SUM_OF_SQUARES)
            for (uint32_t n = 0; n < output->parent->lengths.t[NN_DATA_COORD_n]; ++n)
                for (uint32_t x = 0; x < output->parent->lengths.t[NN_DATA_COORD_x]; ++x)
                    for (uint32_t y = 0; y < output->parent->lengths.t[NN_DATA_COORD_y]; ++y)
                        for (uint32_t z = 0; z < output->parent->lengths.t[NN_DATA_COORD_z]; ++z)
                            for (uint32_t p = 0; p < output->parent->lengths.t[NN_DATA_COORD_p]; ++p)
                                for (uint32_t q = 0; q < output->parent->lengths.t[NN_DATA_COORD_q]; ++q)
                                    nn_workload_data_get<float>(output, n, x, y, z, p, q) = 
                                        -(nn_workload_data_get<float>(target_input, n, x, y, z, p, q) - nn_workload_data_get<float>(current_input, n, x, y, z, p, q)) / batch; // INFO: caffe ignores 2.0 constant                              
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

    void run_primitives_api(
        NN_LOSS_FUNCTION function,
        nn::workload_data<>* forward_input_current,
        nn::workload_data<>* forward_input_target,
        nn::workload_data<>* backward_error_delta)
    {
        nn_device_primitives_description_t device_primitives_description;
        nn_device_get_primitives_description(&device_primitives_description);
        assert(device_primitives_description.version_first <= 0);
        assert(device_primitives_description.version_last >= 0);

        nn_primitives_0_t primitives;
        nn_device_get_primitives(0, &primitives);
        nn_device_t *device = primitives.create_device_with_thread_count(0, nullptr);

        auto size = forward_input_current->get_length();

        // init primitive handle
        nn_primitive_handle_t primitive =
            primitives.create_handle.loss_f32(device, function, size.t[1], size.t[2], size.t[3], size.t[0], nullptr);

        // TODO: change this when device API starts using parent->delta_buffer
        assert(forward_input_current->parent->delta_buffer == nullptr);
        forward_input_current->parent->delta_buffer = backward_error_delta->parent->data_buffer; // this is backward_output for primitives API

        // execute loss function
        nn_opaque_data_t* inputs[] = {reinterpret_cast<nn_opaque_data_t*> (forward_input_current), reinterpret_cast<nn_opaque_data_t*> (forward_input_target)};

        nn_event_t loss_event = primitives.backward_async(
            primitive, 2, inputs, 0, nullptr, 0, nullptr, 0, nullptr, nullptr);

        primitives.wait(1, &loss_event);

        // revert changes made above
        forward_input_current->parent->delta_buffer = nullptr;

        // cleanup
        primitives.delete_event(loss_event);
        primitives.delete_primitive(primitive);
        primitives.delete_device(device);
    }

    void run_test(
        NN_LOSS_FUNCTION function,
        uint32_t input_fmaps,
        uint32_t batch,
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
            output_format,
            function);

        // Get last item (should be loss function) from internal workload.
        nn_workload_item_t* loss_item = get_loss_item(workload);

        // Create input buffers and initialize them.
        void* input_datas[2] =
        {
            new nn::data<float, 2>(input_format.size(0), batch),
            new nn::data<float, 2>(output_format.size(0), batch)
        };

        for (uint32_t n = 0; n < reinterpret_cast<nn::data<float>**>(input_datas)[0]->size[1]; ++n)
            for (uint32_t x = 0; x < reinterpret_cast<nn::data<float>**>(input_datas)[0]->size[0]; ++x)
                (*reinterpret_cast<nn::data<float>**>(input_datas)[0])(x, n) = 1.0f;

        for (uint32_t n = 0; n < reinterpret_cast<nn::data<float>**>(input_datas)[1]->size[1]; ++n)
            for (uint32_t x = 0; x < reinterpret_cast<nn::data<float>**>(input_datas)[1]->size[0]; ++x)
                (*reinterpret_cast<nn::data<float>**>(input_datas)[1])(x, n) = 0.5f;

        // Get references to all buffers used by loss
        // "Inputs" to loss function.
        auto forward_input_current = loss_item->input[0].get_data_view();
        auto forward_input_target = loss_item->input[1].get_data_view();

        // Outputs of loss function.
        auto backward_error_delta = loss_item->output[0];

        // Create reference outputs with same layout and sizes.
        nn::workload_data<> ref_backward_error_delta(backward_error_delta->parent->lengths, backward_error_delta->parent->layout);

        std::memset(ref_backward_error_delta.parent->data_buffer, 0, ref_backward_error_delta.parent->buffer_size);

        // Run optimized code.
        NN_API_STATUS status;
        EXPECT_EQ(NN_API_STATUS_OK, di.workload_execute_function(workload, (void **)input_datas, nullptr, &status));

        // Create data for naive run.
        nn::workload_data<> naive_forward_input_current(
            reinterpret_cast<nn::data<int32_t>**>(input_datas)[0]->buffer, 
            forward_input_current->parent->lengths, 
            forward_input_current->parent->layout);

        naive_forward_input_current.view_begin = forward_input_current->view_begin;
        naive_forward_input_current.view_end = forward_input_current->view_end;

        nn::workload_data<> naive_forward_input_target(
            reinterpret_cast<nn::data<int32_t>**>(input_datas)[1]->buffer, 
            forward_input_target->parent->lengths, 
            forward_input_target->parent->layout);

        naive_forward_input_target.view_begin = forward_input_target->view_begin;
        naive_forward_input_target.view_end = forward_input_target->view_end;

        // Run naive code.
        naive_loss_function(
            function,
            &naive_forward_input_current,
            &naive_forward_input_target,
            &ref_backward_error_delta);

        // Run optimized code through primitive API
        nn::workload_data<> prim_backward_error_delta(backward_error_delta->parent->lengths, backward_error_delta->parent->layout);
        std::memset(prim_backward_error_delta.parent->data_buffer, 0, prim_backward_error_delta.parent->buffer_size);

        run_primitives_api(
            function,
            &naive_forward_input_current,
            &naive_forward_input_target,
            &prim_backward_error_delta);

        if (negative)
        {
            // Random error for each buffer.
            std::mt19937 gen(1);
            std::uniform_int_distribution<uint32_t> dis_error(0, backward_error_delta->parent->buffer_size / sizeof(float) - 1);

            auto forward_error_index = dis_error(gen);
            auto backward_error_index = dis_error(gen);

            static_cast<float*>(backward_error_delta->parent->data_buffer)[backward_error_index] += 1.0f;
            static_cast<float*>(prim_backward_error_delta.parent->data_buffer)[backward_error_index] += 1.0f;

            EXPECT_NE(true, compare_data_items(backward_error_delta, &ref_backward_error_delta));
            EXPECT_NE(true, compare_data_items(&prim_backward_error_delta, &ref_backward_error_delta));
        }
        else
        {
            EXPECT_EQ(true, compare_data_items(backward_error_delta, &ref_backward_error_delta));
            EXPECT_EQ(true, compare_data_items(&prim_backward_error_delta, &ref_backward_error_delta));
        }

        unload_device_and_delete_workload(&di, workload);

        delete reinterpret_cast<nn::data<float>**>(input_datas)[0];
        delete reinterpret_cast<nn::data<float>**>(input_datas)[1];
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_loss_function, standard_square_positive)
{
    for(auto function : { NN_LOSS_FUNCTION_SUM_OF_SQUARES })
        for (auto batch : { 1, 8, 48 })
            for (auto in_z = 1; in_z < 35; ++in_z)
                run_test(function, in_z, batch, false);
}

TEST(cpu_loss_function, standard_square_negative)
{
    for(auto function : { NN_LOSS_FUNCTION_SUM_OF_SQUARES })
        for (auto batch : { 1, 8, 48 })
            for (auto in_z = 1; in_z < 35; ++in_z)
                run_test(function, in_z, batch, true);
}
