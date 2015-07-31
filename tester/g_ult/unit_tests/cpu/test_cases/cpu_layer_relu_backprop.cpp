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
#include "device/cpu/core/layer_relu_avx2.h"
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
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&workflow, 2, 0));

        // Create input.
        nn_workflow_item_t *input = nullptr;
        {
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&input, 0, nullptr, 1));
            input->type = NN_WORK_ITEM_TYPE_INPUT;

            input->arguments.input.index = 0;

            input->output_format[0] = input_format;
        }

        // Create normalization.
        nn_workflow_item_t *relu = nullptr;
        {
            nn_workflow_use_descriptor_t desc0 = { input, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&relu, 1, &desc0, 1));
            relu->type = NN_WORK_ITEM_TYPE_RELU;

            relu->output_format[0] = output_format;
        }

        // Create second input (target values).
        nn_workflow_item_t *input_target = nullptr;
        {
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&input_target, 0, nullptr, 1));
            input_target->type = NN_WORK_ITEM_TYPE_INPUT;

            input_target->arguments.input.index = 1;

            input_target->output_format[0] = output_format;
        }

        // Create loss function.
        nn_workflow_item_t *loss_function = nullptr;
        {
            nn_workflow_use_descriptor_t desc0[2] = { { relu, 0 }, { input_target, 0 } };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&loss_function, 2, desc0, 1));
            loss_function->type = NN_WORK_ITEM_TYPE_LOSS_FUNCTION;

            loss_function->arguments.loss_function.function = NN_LOSS_FUNCTION_SUM_OF_SQUARES;

            loss_function->output_format[0] = output_format;
        }

        // Create normalization backprop.
        nn_workflow_item_t *relu_backprop = nullptr;
        {
            nn_workflow_use_descriptor_t desc0 = { loss_function, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&relu_backprop, 1, &desc0, 1));
            relu_backprop->type = NN_WORK_ITEM_TYPE_RELU_BACKPROP;

            relu_backprop->forward_item = relu;
        }

        // Pin inputs.
        workflow->input[0] = input;
        workflow->input[1] = input_target;

        // Compile workload.
        NN_WORKLOAD_DATA_TYPE io_format[2] = { NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH, NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH };
        nn_workload_t *workload = nullptr;
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_compile_function(&workload, di.device, workflow, io_format, nullptr, batch));

        // Cleanup workflow.
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input_target));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(relu));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(loss_function));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(relu_backprop));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));

        return workload;
    }

    nn_workload_item_t* get_backprop(
        nn_workload_t* workload)
    {
        nn_workload_opaque_t* workload_opaque = reinterpret_cast<nn_workload_opaque_t*>(workload + 1);
        nn_workload_item_t *workload_backprop = workload_opaque->order_of_execution.back();
        assert(workload_backprop->type == NN_WORK_ITEM_TYPE_RELU_3D_BACKPROP);

        return workload_backprop;
    }

    void backward_naive(const nn::workload_data<float> *forward_input,
                        const nn::workload_data<float> *forward_output,
                        const nn::workload_data<float> *backward_input,
                        nn::workload_data<float> *backward_output)
    {
        for (uint32_t n = 0; n < forward_input->get_length(NN_DATA_COORD_n); ++n)
            for (uint32_t x = 0; x < forward_input->get_length(NN_DATA_COORD_x); ++x)
                for (uint32_t y = 0; y < forward_input->get_length(NN_DATA_COORD_y); ++y)
                    for (uint32_t z = 0; z < forward_input->get_length(NN_DATA_COORD_z); ++z)
                        for (uint32_t p = 0; p < forward_input->get_length(NN_DATA_COORD_p); ++p)
                            for (uint32_t q = 0; q < forward_input->get_length(NN_DATA_COORD_q); ++q)
                                backward_output->at(n, x, y, z, p, q) = ((forward_input->at(n, x, y, z, p, q) <= 0.0f) ? 0.0f : 1.0f)
                                * backward_input->at(n, x, y, z, p, q);
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

    void run_primitives_api(
        nn::workload_data<float>* forward_input,
        nn::workload_data<float>* forward_output,
        nn::workload_data<float>* backward_input,
        nn::workload_data<float>* backward_error_delta)
    {
        nn_device_primitives_description_t device_primitives_description;
        nn_device_get_primitives_description(&device_primitives_description);
        assert(device_primitives_description.version_first <= 0);
        assert(device_primitives_description.version_last >= 0);

        nn_primitives_0_t primitives;
        nn_device_get_primitives(0, &primitives);
        nn_device_t *device = primitives.create_device_with_thread_count(0, nullptr);

        auto size = forward_input->get_length();

        // init primitive handle
        nn_primitive_handle_t primitive =
            primitives.create_handle.relu_f32(device, size.t[1], size.t[2], size.t[3], size.t[0], nullptr, nullptr);

        // TODO: change this when device API starts using parent->delta_buffer
        assert(forward_input->parent->delta_buffer == nullptr);
        forward_input->parent->delta_buffer = backward_error_delta->parent->data_buffer; // this is backward_output for primitives API
        assert(forward_output->parent->delta_buffer == nullptr);
        forward_output->parent->delta_buffer = backward_input->parent->data_buffer; // this is backward_input for primitives API

        // execute relu backward
        nn_opaque_data_t* inputs[] = {reinterpret_cast<nn_opaque_data_t*> (forward_input)};
        nn_opaque_data_t* outputs[] = {reinterpret_cast<nn_opaque_data_t*> (forward_output)};

        nn_event_t relu = primitives.backward_async(
            primitive, 1, inputs, 0, nullptr, 1, outputs, 0, nullptr, nullptr);

        primitives.wait(1, &relu);

        // revert changes made above
        forward_input->parent->delta_buffer = nullptr;
        forward_output->parent->delta_buffer = nullptr;

        // cleanup
        primitives.delete_event(relu);
        primitives.delete_primitive(primitive);
        primitives.delete_device(device);
    }

    void run_test(
        uint32_t fmaps,
        uint32_t input_width,
        uint32_t input_height,
        uint32_t batch,
        bool negative)
    {
        nn::output_format input_format(input_width, input_height, fmaps);
        nn::output_format output_format = input_format;

        // Load device.
        nn_device_interface_0_t di = load_device();

        // Create workflow and compile it.
        nn_workload_t* workload = create_workload(
            di,
            batch,
            input_format,
            output_format);

        // Get last item (should be backprop) from internal workload.
        nn_workload_item_t* pool_backprop = get_backprop(
            workload);

        // Create input buffers and initialize them.
        nn::data<float, 4>* input_datas[2] =
        {
            new nn::data<float, 4>(input_format.size(0), input_format.size(1), input_format.size(2), batch),
            new nn::data<float, 4>(output_format.size(0), output_format.size(1), output_format.size(2), batch)
        };

        std::mt19937 gen(1);
        std::bernoulli_distribution dis(0.5f);

        for (uint32_t n = 0; n < input_datas[0]->size[3]; ++n)
            for (uint32_t z = 0; z < input_datas[0]->size[2]; ++z)
                for (uint32_t y = 0; y < input_datas[0]->size[1]; ++y)
                    for (uint32_t x = 0; x < input_datas[0]->size[0]; ++x)
                    {
                        (*input_datas[0])(x, y, z, n) = (dis(gen) ? 1.0f : -1.0f);
                        (*input_datas[1])(x, y, z, n) = 0.5f;
                    } 


        // Get refernces to all buffers used by convolution and its backprop.
        // "Inputs" to backprop.
        auto forward_input = pool_backprop->forward_item->input[0].get_data_view();
        auto forward_output = pool_backprop->forward_item->output[0];
        auto backward_input = pool_backprop->input[0].get_data_view();

        // Outputs of backprop.
        auto backward_error_delta = pool_backprop->output[0];

        // Create reference outputs with same layout and sizes.
        nn::workload_data<float> ref_backward_error_delta(backward_error_delta->parent->lengths, backward_error_delta->parent->layout);

        std::memset(ref_backward_error_delta.parent->data_buffer, 0, ref_backward_error_delta.parent->buffer_size);

        // Run optimized code.
        NN_API_STATUS status;
        EXPECT_EQ(NN_API_STATUS_OK, di.workload_execute_function(workload, (void **)input_datas, nullptr, &status));

        // Run naive code.
        forward_input->parent->data_buffer = input_datas[0]->buffer;
        backward_naive(
            static_cast<nn::workload_data<float>*>(forward_input),
            static_cast<nn::workload_data<float>*>(forward_output),
            static_cast<nn::workload_data<float>*>(backward_input),
            &ref_backward_error_delta);

        // Run optimized code through primitive API
        nn::workload_data<float> prim_backward_error_delta(backward_error_delta->parent->lengths, backward_error_delta->parent->layout);
        run_primitives_api(
            static_cast<nn::workload_data<float>*>(forward_input),
            static_cast<nn::workload_data<float>*>(forward_output),
            static_cast<nn::workload_data<float>*>(backward_input),
            &prim_backward_error_delta);
            
        if (negative)
        {
            // Random error for each buffer.
            std::mt19937 gen(1);
            std::uniform_int_distribution<uint32_t> dis_error(0, backward_error_delta->parent->buffer_size / sizeof(float) - 1);

            auto error_index = dis_error(gen);

            static_cast<float*>(    backward_error_delta->parent->data_buffer)[error_index] += 1.0f;
            static_cast<float*>(prim_backward_error_delta.parent->data_buffer)[error_index] += 1.0f;

            EXPECT_NE(true, compare_data_items(backward_error_delta,       &ref_backward_error_delta));
            EXPECT_NE(true, compare_data_items(&prim_backward_error_delta, &ref_backward_error_delta));
        }
        else
        {
            EXPECT_EQ(true, compare_data_items(backward_error_delta,       &ref_backward_error_delta));
            EXPECT_EQ(true, compare_data_items(&prim_backward_error_delta, &ref_backward_error_delta));
        }

        unload_device_and_delete_workload(&di, workload);

        delete input_datas[0];
        delete input_datas[1];
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_relu_backprop, standard_square_positive)
{
    for (auto batch : { 1, 8, 48 })
        for (auto in = 1; in < 10; in++)
            for (auto fmaps : { 8, 16 })
                run_test(fmaps, in, in, batch, false);
}

TEST(cpu_relu_backprop, standard_square_negative)
{
    for (auto batch : { 1, 8, 48 })
        for (auto in = 1; in < 10; in++)
            for (auto fmaps : { 8, 16 })
                run_test(fmaps, in, in, batch, true);
}

