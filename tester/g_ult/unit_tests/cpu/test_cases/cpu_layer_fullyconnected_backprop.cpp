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
#include "device/cpu/core/layer_fully_connected_avx2.h"
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
        uint32_t input_fmaps,
        uint32_t output_fmaps,
        uint32_t batch,
        nn_output_format input_format,
        nn_output_format output_format)
    {
        std::mt19937 gen(1);
        std::uniform_real_distribution<float> dis(0.01f, 0.2f);

        nn::data<float, 2> weight_data(input_fmaps, output_fmaps);
        nn::data<float, 1> bias_data(output_fmaps);

        for (uint32_t y = 0; y < weight_data.size[1]; ++y)
        {
            bias_data(y) = dis(gen);

            for (uint32_t x = 0; x < weight_data.size[0]; ++x)
                weight_data(x, y) = dis(gen);
        }

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

        // Create fc.
        nn_workflow_item_t *fc = nullptr;
        {
            nn_workflow_use_descriptor_t desc0 = { input, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&fc, 1, &desc0, 1));
            fc->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;

            auto& args = fc->arguments.forward_fully_connected;
            args.activation.function = NN_ACTIVATION_FUNCTION_NONE;
            args.weights = &weight_data;
            args.biases = &bias_data;

            fc->output_format[0] = output_format;
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
            nn_workflow_use_descriptor_t desc0[2] = { { fc, 0 }, { input_target, 0 } };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&loss_function, 2, desc0, 1));
            loss_function->type = NN_WORK_ITEM_TYPE_LOSS_FUNCTION;

            loss_function->arguments.loss_function.function = NN_LOSS_FUNCTION_SUM_OF_SQUARES;

            loss_function->output_format[0] = output_format;
        }

        // Create fc backprop.
        nn_workflow_item_t *fc_backprop = nullptr;
        {
            nn_workflow_use_descriptor_t desc0 = { loss_function, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&fc_backprop, 1, &desc0, 3));
            fc_backprop->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP;

            fc_backprop->forward_item = fc;
        }

        // Pin inputs.
        workflow->input[0] = input;
        workflow->input[1] = input_target;

        // Compile workload.
        NN_WORKLOAD_DATA_TYPE io_format[2] = { NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH, NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH };
        nn_workload_t *workload = nullptr;
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_compile_function(&workload, di.device, workflow, io_format, nullptr, batch));

        // Cleanup workflow.
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input_target));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(fc));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(loss_function));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(fc_backprop));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));

        return workload;
    }

    nn_workload_item_t* get_backprop(
        nn_workload_t* workload)
    {
        nn_workload_opaque_t* workload_opaque = reinterpret_cast<nn_workload_opaque_t*>(workload + 1);
        nn_workload_item_t *workload_backprop = workload_opaque->order_of_execution.back();
        assert(workload_backprop->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP);

        return workload_backprop;
    }

    void backward_naive(const nn::workload_data<float> *forward_input_view,
                        const nn::workload_data<float> *forward_weights_view,
                        const nn::workload_data<float> *forward_output_view,
                        const nn::workload_data<float> *backward_input_view,
                        nn::workload_data<float> *backward_output_view,
                        nn::workload_data<float> *backward_weights_delta_view,
                        nn::workload_data<float> *backward_bias_delta_view)
    {
        const auto& in_begin = forward_input_view->view_begin;
        const auto& in_end = forward_input_view->view_end;
        const auto& out_begin = forward_output_view->view_begin;
        const auto& out_end = forward_output_view->view_end;

        nn::workload_data<float> forward_input_buffer(forward_input_view->parent->data_buffer, forward_input_view->parent->lengths, forward_input_view->parent->layout);
        nn::workload_data<float> forward_weights_buffer(forward_weights_view->parent->data_buffer, forward_weights_view->parent->lengths, forward_weights_view->parent->layout);
        nn::workload_data<float> backward_input_buffer(backward_input_view->parent->data_buffer, backward_input_view->parent->lengths, backward_input_view->parent->layout);
        nn::workload_data<float> backward_output_buffer(backward_output_view->parent->data_buffer, backward_output_view->parent->lengths, backward_output_view->parent->layout);
        nn::workload_data<float> backward_weights_delta_buffer(backward_weights_delta_view->parent->data_buffer, backward_weights_delta_view->parent->lengths, backward_weights_delta_view->parent->layout);
        nn::workload_data<float> backward_bias_delta_buffer(backward_bias_delta_view->parent->data_buffer, backward_bias_delta_view->parent->lengths, backward_bias_delta_view->parent->layout);

        auto batch_size = forward_input_buffer.parent->lengths.t[NN_DATA_COORD_n];

        // For backpropagation in [m] layer, we require s[m] = F'[m](n[m]) * W[m+1]^T * s[m+1] <=> s[m] = F'[m](n[m]) * U[m+1] => U[m+1] = W[m+1]^T * s[m+1].
        // Backpropagation for next layer already has computed U[m+1], now we need to compute F'[m](n[m]) part.
        // If there is no activation, then F'[m](n[m]) = 1. It means s[m] = U[m+1], so we'll use just s[m].
        // We have access to both s[m] and W[m] and so, we can compute U[m] = W[m]^T * s[m] required for previous layer.
        // Also compute weights gradient as s[m] * a[m-1]^T and compute bias gradients as s[m].
        // We already have s[m], and a[m-1] is just raw input to this layer.
        for (int32_t batch = (int32_t)in_begin.t[NN_DATA_COORD_n];
             batch <= in_end.t[NN_DATA_COORD_n];
             ++batch)
        {
            for (uint32_t output_element = out_begin.t[NN_DATA_COORD_x]; output_element <= out_end.t[NN_DATA_COORD_x]; ++output_element)
            {
                for (uint32_t input_element = in_begin.t[NN_DATA_COORD_x]; input_element <= in_end.t[NN_DATA_COORD_x]; ++input_element)
                {
                    if (batch_size == 1)
                    {
                        backward_output_buffer(batch, input_element, 0, 0, 0, 0)
                            += backward_input_buffer(batch, output_element, 0, 0, 0, 0)
                            * forward_weights_buffer(0, input_element, output_element, 0, 0, 0);


                        backward_weights_delta_buffer(0, input_element, output_element, 0, 0, 0)
                            += backward_input_buffer(batch, output_element, 0, 0, 0, 0)
                            * forward_input_buffer(batch, input_element, 0, 0, 0, 0);
                    }
                    else
                    {
                        const uint32_t C_max_accumulators = (batch_size == 8) ? 13 : 2;

                        backward_output_buffer(batch, input_element, 0, 0, 0, 0)
                            += backward_input_buffer(batch, output_element, 0, 0, 0, 0)
                            * forward_weights_buffer(0, input_element, 0, 0, output_element % C_max_accumulators, output_element / C_max_accumulators);


                        backward_weights_delta_buffer(0, input_element, 0, 0, output_element % C_max_accumulators, output_element / C_max_accumulators)
                            += backward_input_buffer(batch, output_element, 0, 0, 0, 0)
                            * forward_input_buffer(batch, input_element, 0, 0, 0, 0);
                    }
                }

                backward_bias_delta_buffer(0, output_element, 0, 0, 0, 0)
                    += backward_input_buffer(batch, output_element, 0, 0, 0, 0);
            }
        }
    }

    bool compare_data_items(
        nn_workload_data_t* data_item,
        nn_workload_data_t* data_item_ref,
        uint32_t slice_remainder = 0,
        bool is_weight = false)
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
                                // For weights we have to check only these values we really need, because there 
                                // is slice alignment that can be different in optimal and naive versions.
                                if(is_weight && q == size.t[5]-1 && slice_remainder != 0 && p >= slice_remainder)
                                    break;

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
        nn::workload_data<float> *forward_input,
        nn::workload_data<float> *forward_weights,
        nn::workload_data<float> *forward_output,
        nn::workload_data<float> *backward_input,
        nn::workload_data<float> *backward_output,
        nn::workload_data<float> *backward_weights_delta,
        nn::workload_data<float> *backward_bias_delta)
    {
        nn_device_primitives_description_t device_primitives_description;
        nn_device_get_primitives_description(&device_primitives_description);
        assert(device_primitives_description.version_first <= 0);
        assert(device_primitives_description.version_last >= 0);

        nn_primitives_0_t primitives;
        nn_device_get_primitives(0, &primitives);
        nn_device_t *device = primitives.create_device_with_thread_count(0, nullptr);

        auto o_size = forward_output->get_length();
        auto w_size = forward_weights->get_length();

        nn_argument_activation activation;
        activation.function = NN_ACTIVATION_FUNCTION_NONE;

        // init primitive handle
        nn_primitive_handle_t primitive =
            primitives.create_handle.fully_connected_f32(
                device,
                w_size.t[1], // num_input
                o_size.t[1], // num_output
                &activation,
                o_size.t[0], // batch size
                nullptr,
                nullptr);

        // TODO: change this when device API starts using parent->delta_buffer
        assert(forward_input->parent->delta_buffer == nullptr);
        assert(forward_output->parent->delta_buffer == nullptr);
        assert(forward_weights->parent->delta_buffer == nullptr);
        assert(backward_bias_delta->parent->delta_buffer == nullptr);
        forward_input->parent->delta_buffer = backward_output->parent->data_buffer;
        forward_output->parent->delta_buffer = backward_input->parent->data_buffer;
        forward_weights->parent->delta_buffer = backward_weights_delta->parent->data_buffer;
        backward_bias_delta->parent->delta_buffer = backward_bias_delta->parent->data_buffer;

        // prepare buffers
        nn_opaque_data_t* inputs[] = {reinterpret_cast<nn_opaque_data_t*> (forward_input)};
        nn_opaque_data_t* outputs[] = {reinterpret_cast<nn_opaque_data_t*> (forward_output)};

        nn_opaque_data_t* parameters[] = 
            {reinterpret_cast<nn_opaque_data_t*> (forward_weights),
             reinterpret_cast<nn_opaque_data_t*> (backward_bias_delta)};

        // execute convolution backpropagation
        nn_event_t backward = primitives.backward_async(
            primitive, 1, inputs, 2, parameters, 1, outputs, 0, nullptr, nullptr);

        // input buffer needed for calculating weights_delta and bias_delta
        nn_event_t backward_weights = primitives.backward_parameter_async(
            primitive, 0, 1, inputs, 2, parameters, 1, outputs, 0, nullptr, nullptr);

        nn_event_t backward_bias = primitives.backward_parameter_async(
            primitive, 1, 1, inputs, 2, parameters, 1, outputs, 0, nullptr, nullptr);

        primitives.wait(1, &backward);
        primitives.wait(1, &backward_weights);
        primitives.wait(1, &backward_bias);

        // revert changes made above
        forward_input->parent->delta_buffer = nullptr;
        forward_output->parent->delta_buffer = nullptr;
        forward_weights->parent->delta_buffer = nullptr;
        backward_bias_delta->parent->delta_buffer = nullptr;

        // cleanup
        primitives.delete_event(backward);
        primitives.delete_event(backward_weights);
        primitives.delete_event(backward_bias);
        primitives.delete_primitive(primitive);
        primitives.delete_device(device);
    }

    void run_test(
        uint32_t input_fmaps,
        uint32_t output_fmaps,
        uint32_t batch,
        bool negative)
    {
        nn::output_format input_format(input_fmaps);
        nn::output_format output_format(output_fmaps);

        // Load device.
        nn_device_interface_0_t di = load_device();

        // Create workflow and compile it.
        nn_workload_t* workload = create_workload(
            di,
            input_fmaps,
            output_fmaps,
            batch,
            input_format,
            output_format);

        // Get last item (should be backprop) from internal workload.
        nn_workload_item_t* fc_backprop = get_backprop(
            workload);

        // Create input buffers and initialize them.
        nn::data<float, 2>* input_datas[2] =
        {
            new nn::data<float, 2>(input_format.size(0), batch),
            new nn::data<float, 2>(output_format.size(0), batch)
        };

        for (uint32_t n = 0; n < input_datas[0]->size[1]; ++n)
            for (uint32_t x = 0; x < input_datas[0]->size[0]; ++x)
                (*input_datas[0])(x, n) = 1.0f;

        for (uint32_t n = 0; n < input_datas[1]->size[1]; ++n)
            for (uint32_t x = 0; x < input_datas[1]->size[0]; ++x)
                (*input_datas[1])(x, n) = 0.5f;

        // Get refernces to all buffers used by fc and its backprop.
        // "Inputs" to backprop.
        auto forward_input = fc_backprop->forward_item->input[0].get_data_view();
        auto forward_output = fc_backprop->forward_item->output[0];
        auto forward_weight = fc_backprop->forward_item->parameters.at(0);
        auto forward_bias = fc_backprop->forward_item->parameters.at(1);
        auto backward_input = fc_backprop->input[0].get_data_view();

        // Outputs of backprop.
        auto backward_error_delta = fc_backprop->output[0];
        auto backward_weight_delta = fc_backprop->output[1];
        auto backward_bias_delta = fc_backprop->output[2];

        // Create reference outputs with same layout and sizes.
        nn::workload_data<float> ref_backward_error_delta(backward_error_delta->parent->lengths, backward_error_delta->parent->layout);
        nn::workload_data<float> ref_backward_weight_delta(backward_weight_delta->parent->lengths, backward_weight_delta->parent->layout);
        nn::workload_data<float> ref_backward_bias_delta(backward_bias_delta->parent->lengths, backward_bias_delta->parent->layout);

        std::memset(ref_backward_error_delta.parent->data_buffer, 0, ref_backward_error_delta.parent->buffer_size);
        std::memset(ref_backward_weight_delta.parent->data_buffer, 0, ref_backward_weight_delta.parent->buffer_size);
        std::memset(ref_backward_bias_delta.parent->data_buffer, 0, ref_backward_bias_delta.parent->buffer_size);

        // Run optimized code.
        NN_API_STATUS status;
        EXPECT_EQ(NN_API_STATUS_OK, di.workload_execute_function(workload, (void **)input_datas, nullptr, &status));

        // Run naive code.
        forward_input->parent->data_buffer = input_datas[0]->buffer;
        backward_naive(
            static_cast<nn::workload_data<float>*>(forward_input),
            static_cast<nn::workload_data<float>*>(forward_weight),
            static_cast<nn::workload_data<float>*>(forward_output),
            static_cast<nn::workload_data<float>*>(backward_input),
            &ref_backward_error_delta,
            &ref_backward_weight_delta,
            &ref_backward_bias_delta);

        const uint32_t C_max_accumulators = (batch == 8) ? 13 : 2;

        // Run optimized code through primitive API
        nn::workload_data<float> primitive_backward_error_delta(backward_error_delta->parent->lengths, backward_error_delta->parent->layout);
        nn::workload_data<float> primitive_backward_weight_delta(backward_weight_delta->parent->lengths, backward_weight_delta->parent->layout);
        nn::workload_data<float> primitive_backward_bias_delta(backward_bias_delta->parent->lengths, backward_bias_delta->parent->layout);
        std::memset(primitive_backward_error_delta.parent->data_buffer, 0, ref_backward_error_delta.parent->buffer_size);
        std::memset(primitive_backward_weight_delta.parent->data_buffer, 0, ref_backward_weight_delta.parent->buffer_size);
        std::memset(primitive_backward_bias_delta.parent->data_buffer, 0, ref_backward_bias_delta.parent->buffer_size);

        run_primitives_api(
            static_cast<nn::workload_data<float>*>(forward_input),
            static_cast<nn::workload_data<float>*>(forward_weight),
            static_cast<nn::workload_data<float>*>(forward_output),
            static_cast<nn::workload_data<float>*>(backward_input),
            &primitive_backward_error_delta,
            &primitive_backward_weight_delta,
            &primitive_backward_bias_delta);

        if (negative)
        {
            // Random error for each buffer.
            std::mt19937 gen(1);
            std::uniform_int_distribution<uint32_t> dis_error(0, backward_error_delta->parent->buffer_size / sizeof(float) - 1);
            std::uniform_int_distribution<uint32_t> dis_weight(0, backward_weight_delta->parent->buffer_size / sizeof(float) - 1);
            std::uniform_int_distribution<uint32_t> dis_bias(0, backward_bias_delta->parent->buffer_size / sizeof(float) - 1);

            auto error_index = dis_error(gen);
            auto weight_index = dis_weight(gen) / C_max_accumulators * C_max_accumulators;
            auto bias_index = dis_bias(gen);

            static_cast<float*>(backward_error_delta->parent->data_buffer)[error_index] += 1.0f;
            static_cast<float*>(backward_weight_delta->parent->data_buffer)[weight_index] += 1.0f;
            static_cast<float*>(backward_bias_delta->parent->data_buffer)[bias_index] += 1.0f;

            EXPECT_NE(true, compare_data_items(backward_error_delta, &ref_backward_error_delta));
            EXPECT_NE(true, compare_data_items(backward_weight_delta, &ref_backward_weight_delta, output_fmaps % C_max_accumulators, true));
            EXPECT_NE(true, compare_data_items(backward_bias_delta, &ref_backward_bias_delta));

            static_cast<float*>(primitive_backward_error_delta.parent->data_buffer)[error_index] += 1.0f;
            static_cast<float*>(primitive_backward_weight_delta.parent->data_buffer)[weight_index] += 1.0f;
            static_cast<float*>(primitive_backward_bias_delta.parent->data_buffer)[bias_index] += 1.0f;

            EXPECT_NE(true, compare_data_items(&primitive_backward_error_delta, &ref_backward_error_delta));
            EXPECT_NE(true, compare_data_items(&primitive_backward_weight_delta, &ref_backward_weight_delta));
            EXPECT_NE(true, compare_data_items(&primitive_backward_bias_delta, &ref_backward_bias_delta));
        }
        else
        {
            EXPECT_EQ(true, compare_data_items(backward_error_delta, &ref_backward_error_delta));
            EXPECT_EQ(true, compare_data_items(backward_weight_delta, &ref_backward_weight_delta, output_fmaps % C_max_accumulators, true));
            EXPECT_EQ(true, compare_data_items(backward_bias_delta, &ref_backward_bias_delta));

            EXPECT_EQ(true, compare_data_items(&primitive_backward_error_delta, &ref_backward_error_delta));
            EXPECT_EQ(true, compare_data_items(&primitive_backward_weight_delta, &ref_backward_weight_delta));
            EXPECT_EQ(true, compare_data_items(&primitive_backward_bias_delta, &ref_backward_bias_delta));
        }

        unload_device_and_delete_workload(&di, workload);

        delete input_datas[0];
        delete input_datas[1];
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_fc_backprop, standard_square_positive)
{
    for (auto batch : { 1, 8, 48 })
        for (auto in_z = 1; in_z < 17; ++in_z)
            for (auto out_z = 1; out_z < 17; ++out_z)
                run_test(in_z, out_z, batch, false);
}

TEST(cpu_fc_backprop, standard_square_negative)
{
    for (auto batch : { 1, 8, 48 })
        for (auto in_z = 1; in_z < 17; ++in_z)
            for (auto out_z = 1; out_z < 17; ++out_z)
                run_test(in_z, out_z, batch, true);
}

