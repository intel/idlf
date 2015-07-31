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
#include "device/cpu/core/layer_convolution_avx2.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"


namespace
{
    const auto C_simd_width = sizeof(__m256) / sizeof(float);
    const uint32_t C_slice_size = 2 * C_simd_width;

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
        uint32_t stride_x,
        uint32_t stride_y,
        uint32_t weight_width,
        uint32_t weight_height,
        uint32_t padding_x,
        uint32_t padding_y,
        uint32_t input_fmaps,
        uint32_t output_fmaps,
        uint32_t batch,
        nn_output_format_t input_format,
        nn_output_format_t output_format)
    {
        std::mt19937 gen(1);
        std::uniform_real_distribution<float> dis(0.01f, 0.2f);

        nn::data<float, 4> weight_data(weight_width, weight_height, input_fmaps, output_fmaps);
        nn::data<float, 1> bias_data(output_fmaps);

        for (uint32_t p = 0; p < weight_data.size[3]; ++p)
        {
            bias_data(p) = dis(gen);

            for (uint32_t z = 0; z < weight_data.size[2]; ++z)
                for (uint32_t y = 0; y < weight_data.size[1]; ++y)
                    for (uint32_t x = 0; x < weight_data.size[0]; ++x)
                        weight_data(x, y, z, p) = dis(gen);
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

        // Create convolution.
        nn_workflow_item_t *convolution = nullptr;
        {
            nn_workflow_use_descriptor_t desc0 = { input, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&convolution, 1, &desc0, 1));
            convolution->type = NN_WORK_ITEM_TYPE_CONVOLUTION;

            auto& args = convolution->arguments.forward_convolution;
            args.activation.function = NN_ACTIVATION_FUNCTION_NONE;
            args.center_offset[0] = padding_x;
            args.center_offset[1] = padding_y;
            args.padding = NN_PADDING_MODE_DATA_OR_ZERO;
            args.stride[0] = stride_x;
            args.stride[1] = stride_y;
            args.weights = &weight_data;
            args.biases = &bias_data;

            convolution->output_format[0] = output_format;
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
            nn_workflow_use_descriptor_t desc0[2] = { { convolution, 0 }, { input_target, 0 } };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&loss_function, 2, desc0, 1));
            loss_function->type = NN_WORK_ITEM_TYPE_LOSS_FUNCTION;

            loss_function->arguments.loss_function.function = NN_LOSS_FUNCTION_SUM_OF_SQUARES;

            loss_function->output_format[0] = output_format;
        }

        // Create convolution backprop.
        nn_workflow_item_t *conv_backprop = nullptr;
        {
            nn_workflow_use_descriptor_t desc0 = { loss_function, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&conv_backprop, 1, &desc0, 3));
            conv_backprop->type = NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP;

            conv_backprop->forward_item = convolution;
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
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(convolution));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(loss_function));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(conv_backprop));
        EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));

        return workload;
    }

    nn_workload_item_t* get_backprop(
        nn_workload_t* workload)
    {
        nn_workload_opaque_t* workload_opaque = reinterpret_cast<nn_workload_opaque_t*>(workload + 1);
        nn_workload_item_t *workload_backprop = workload_opaque->order_of_execution.back();
        assert(workload_backprop->type == NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP);

        return workload_backprop;
    }

    void backward_naive(const nn::workload_data<float> *forward_input_view,
                        const nn::workload_data<float> *forward_weights_view,
                        const nn::workload_data<float> *forward_output_view,
                        const nn::workload_data<float> *backward_input_view,
                        nn::workload_data<float> *backward_output_view,
                        nn::workload_data<float> *backward_weights_delta_view,
                        nn::workload_data<float> *backward_bias_delta_view,
                        uint32_t stride_x,
                        uint32_t stride_y, 
                        uint32_t center_offset_x,
                        uint32_t center_offset_y)
    {
        const auto& in_begin = forward_input_view->view_begin;
        const auto& in_end = forward_input_view->view_end;
        const auto& out_begin = forward_output_view->view_begin;
        const auto& out_end = forward_output_view->view_end;
        const auto& kernel_begin = forward_weights_view->view_begin;
        const auto& kernel_end = forward_weights_view->view_end;

        const auto output_width = forward_output_view->parent->lengths.t[NN_DATA_COORD_x];
        const auto output_height = forward_output_view->parent->lengths.t[NN_DATA_COORD_y];

        const auto slice_size = C_slice_size;

        nn::workload_data<float> forward_input_buffer(forward_input_view->parent->data_buffer, forward_input_view->parent->lengths, forward_input_view->parent->layout);
        nn::workload_data<float> forward_weights_buffer(forward_weights_view->parent->data_buffer, forward_weights_view->parent->lengths, forward_weights_view->parent->layout);
        nn::workload_data<float> backward_input_buffer(backward_input_view->parent->data_buffer, backward_input_view->parent->lengths, backward_input_view->parent->layout);
        nn::workload_data<float> backward_output_buffer(backward_output_view->parent->data_buffer, backward_output_view->parent->lengths, backward_output_view->parent->layout);
        nn::workload_data<float> backward_weights_delta_buffer(backward_weights_delta_view->parent->data_buffer, backward_weights_delta_view->parent->lengths, backward_weights_delta_view->parent->layout);
        nn::workload_data<float> backward_bias_delta_buffer(backward_bias_delta_view->parent->data_buffer, backward_bias_delta_view->parent->lengths, backward_bias_delta_view->parent->layout);

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
            for (int32_t input_row = (int32_t)in_begin.t[NN_DATA_COORD_y] - (int32_t)center_offset_y, output_row = out_begin.t[NN_DATA_COORD_y];
                 output_row <= out_end.t[NN_DATA_COORD_y];
                 input_row += stride_y, ++output_row)
            {
                for (int32_t input_column = (int32_t)in_begin.t[NN_DATA_COORD_x] - (int32_t)center_offset_x, output_column = out_begin.t[NN_DATA_COORD_x];
                     output_column <= out_end.t[NN_DATA_COORD_x];
                     input_column += stride_x, ++output_column)
                {
                    for (uint32_t input_feature_map = in_begin.t[NN_DATA_COORD_z]; input_feature_map <= in_end.t[NN_DATA_COORD_z]; ++input_feature_map)
                    {
                        for (uint32_t kernel_row = kernel_begin.t[NN_DATA_COORD_y]; kernel_row <= kernel_end.t[NN_DATA_COORD_y]; ++kernel_row)
                        {
                            for (uint32_t kernel_column = kernel_begin.t[NN_DATA_COORD_x]; kernel_column <= kernel_end.t[NN_DATA_COORD_x]; ++kernel_column)
                            {
                                for (uint32_t output_feature_map = out_begin.t[NN_DATA_COORD_z];
                                     output_feature_map <= out_end.t[NN_DATA_COORD_z];
                                     ++output_feature_map)
                                {
                                    // It's data_or_zero padding for now.
                                    // Assume zero-padding for out-of-buffer inputs.
                                    // Otherwise, we'll use the data. 
                                    if ((input_row + kernel_row) >= 0 &&
                                        (input_column + kernel_column) >= 0 &&
                                        (input_row + kernel_row) < (int32_t)forward_input_view->parent->lengths.t[NN_DATA_COORD_y] &&
                                        (input_column + kernel_column) < (int32_t)forward_input_view->parent->lengths.t[NN_DATA_COORD_x])
                                    {
                                        backward_output_buffer(batch, input_column + kernel_column, input_row + kernel_row, input_feature_map, 0, 0)
                                            += backward_input_buffer(batch, output_column, output_row, output_feature_map, 0, 0)
                                            * forward_weights_buffer(0, kernel_column, kernel_row, input_feature_map, output_feature_map % slice_size, output_feature_map / slice_size);

                                        backward_weights_delta_buffer(0, kernel_column, kernel_row, input_feature_map, output_feature_map % slice_size, output_feature_map / slice_size)
                                            += backward_input_buffer(batch, output_column, output_row, output_feature_map, 0, 0)
                                            * forward_input_buffer(batch, input_column + kernel_column, input_row + kernel_row, input_feature_map, 0, 0);
                                        // / (float)(output_width*output_height) // INFO: caffe accumulates error across image, do not average it
                                    }
                                }
                            }
                        }
                    }

                    for (uint32_t output_feature_map = out_begin.t[NN_DATA_COORD_z];
                         output_feature_map <= out_end.t[NN_DATA_COORD_z];
                         ++output_feature_map)
                    {
                        backward_bias_delta_buffer(0, output_feature_map, 0, 0, 0, 0)
                            += backward_input_buffer(batch, output_column, output_row, output_feature_map, 0, 0);
                        // / (float)(output_width*output_height) // INFO: caffe accumulates error across image, do not average it
                    }
                }
            }
        }
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
        nn::workload_data<float> *forward_input,
        nn::workload_data<float> *forward_weights,
        nn::workload_data<float> *forward_output,
        nn::workload_data<float> *backward_input,
        nn::workload_data<float> *backward_output,
        nn::workload_data<float> *backward_weights_delta,
        nn::workload_data<float> *backward_bias_delta,
        uint32_t stride_x,
        uint32_t stride_y,
        uint32_t center_offset_x,
        uint32_t center_offset_y)
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
            primitives.create_handle.convolution_f32(
                device,
                w_size.t[1],     // kernel_w
                w_size.t[2],     // kernel_h
                w_size.t[3],     // num_input
                o_size.t[3],     // num_output
                o_size.t[1],     // output_w
                o_size.t[2],     // output_h
                center_offset_x, // center_offset_x
                center_offset_y, // center_offset_y
                stride_x,        // stride_x
                stride_y,        // stride_y
                &activation,     // activation
                o_size.t[0],
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

        nn_opaque_data_t* primitive_inputs[1];
        // Create input. TODO: this is done because currently forward_input and backward_output
        // created during workflow=>workload compilation may have different views...
        NN_API_STATUS status;
        primitives.create_inputs(primitive, 1, primitive_inputs, 0, &status);
        ASSERT_EQ(status, NN_API_STATUS_OK);
        // this is backward_output for primitives API
        reinterpret_cast<nn::workload_data<float>*>(primitive_inputs[0])->parent->delta_buffer = backward_output->parent->data_buffer;

        // prepare buffers
        nn_opaque_data_t* outputs[] = {reinterpret_cast<nn_opaque_data_t*> (forward_output)};
        nn_opaque_data_t* parameters[] = 
            {reinterpret_cast<nn_opaque_data_t*> (forward_weights),
            reinterpret_cast<nn_opaque_data_t*> (backward_bias_delta)};

        // execute convolution backpropagation
        nn_event_t backward = primitives.backward_async(
            primitive, 1, primitive_inputs, 2, parameters, 1, outputs, 0, nullptr, nullptr);

        // we no longer need this
        reinterpret_cast<nn::workload_data<float>*>(primitive_inputs[0])->parent->delta_buffer = nullptr;
        primitives.delete_opaque_data(primitive_inputs[0]);

        // input buffer needed for calculating weights_delta and bias_delta
        nn_opaque_data_t* inputs[] = {reinterpret_cast<nn_opaque_data_t*> (forward_input)};
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
        uint32_t stride_x,
        uint32_t stride_y,
        uint32_t weight_width,
        uint32_t weight_height,
        uint32_t padding_x,
        uint32_t padding_y,
        uint32_t input_fmaps,
        uint32_t input_width,
        uint32_t input_height,
        uint32_t output_fmaps,
        uint32_t batch,
        bool negative)
    {
        nn::output_format input_format(input_width + padding_x * 2, input_height + padding_y * 2, input_fmaps);
        nn::output_format output_temp_format(input_width - weight_width + 1 + padding_x * 2, input_height - weight_height + 1 + padding_y * 2, output_fmaps);
        nn::output_format output_format(1 + (output_temp_format.size(0) - 1) / stride_x, 1 + (output_temp_format.size(1) - 1) / stride_y, output_temp_format.size(2));

        // Load device.
        nn_device_interface_0_t di = load_device();

        // Create workflow and compile it.
        nn_workload_t* workload = create_workload(
            di,
            stride_x,
            stride_y,
            weight_width,
            weight_height,
            padding_x,
            padding_y,
            input_fmaps,
            output_fmaps,
            batch,
            input_format,
            output_format);

        // Get last item (should be backprop) from internal workload.
        nn_workload_item_t* conv_backprop = get_backprop(
            workload);

        // Create input buffers and initialize them.
        nn::data<float, 4>* input_datas[2] =
        {
            new nn::data<float, 4>(input_format.size(0), input_format.size(1), input_format.size(2), batch),
            new nn::data<float, 4>(output_format.size(0), output_format.size(1), output_format.size(2), batch)
        };

        for (uint32_t n = 0; n < input_datas[0]->size[3]; ++n)
            for (uint32_t z = 0; z < input_datas[0]->size[2]; ++z)
                for (uint32_t y = 0; y < input_datas[0]->size[1]; ++y)
                    for (uint32_t x = 0; x < input_datas[0]->size[0]; ++x)
                        (*input_datas[0])(x, y, z, n) = 1.0f;

        for (uint32_t n = 0; n < input_datas[1]->size[3]; ++n)
            for (uint32_t z = 0; z < input_datas[1]->size[2]; ++z)
                for (uint32_t y = 0; y < input_datas[1]->size[1]; ++y)
                    for (uint32_t x = 0; x < input_datas[1]->size[0]; ++x)
                        (*input_datas[1])(x, y, z, n) = 0.5f;

        // Get refernces to all buffers used by convolution and its backprop.
        // "Inputs" to backprop.
        auto forward_input = conv_backprop->forward_item->input[0].get_data_view();
        auto forward_output = conv_backprop->forward_item->output[0];
        auto forward_weight = conv_backprop->forward_item->parameters.at(0);
        auto forward_bias = conv_backprop->forward_item->parameters.at(1);
        auto backward_input = conv_backprop->input[0].get_data_view();

        // Outputs of backprop.
        auto backward_error_delta = conv_backprop->output[0];
        auto backward_weight_delta = conv_backprop->output[1];
        auto backward_bias_delta = conv_backprop->output[2];

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
            &ref_backward_bias_delta,
            stride_x,
            stride_y,
            padding_x,
            padding_y);

        // Run optimized code through primitive API
        nn::workload_data<float> primitive_backward_error_delta(backward_error_delta->parent->lengths, backward_error_delta->parent->layout);
        nn::workload_data<float> primitive_backward_weight_delta(backward_weight_delta->parent->lengths, backward_weight_delta->parent->layout);
        nn::workload_data<float> primitive_backward_bias_delta(backward_bias_delta->parent->lengths, backward_bias_delta->parent->layout);
        run_primitives_api(
            static_cast<nn::workload_data<float>*>(forward_input),
            static_cast<nn::workload_data<float>*>(forward_weight),
            static_cast<nn::workload_data<float>*>(forward_output),
            static_cast<nn::workload_data<float>*>(backward_input),
            &primitive_backward_error_delta,
            &primitive_backward_weight_delta,
            &primitive_backward_bias_delta,
            stride_x,
            stride_y,
            padding_x,
            padding_y);

        if (negative)
        {
            // Random error for each buffer.
            std::mt19937 gen(1);
            std::uniform_int_distribution<uint32_t> dis_error(0, backward_error_delta->parent->buffer_size / sizeof(float) - 1);
            std::uniform_int_distribution<uint32_t> dis_weight(0, backward_weight_delta->parent->buffer_size / sizeof(float) - 1);
            std::uniform_int_distribution<uint32_t> dis_bias(0, backward_bias_delta->parent->buffer_size / sizeof(float) - 1);

            auto error_index = dis_error(gen);
            auto weight_index = dis_weight(gen);
            auto bias_index = dis_bias(gen);

            static_cast<float*>(backward_error_delta->parent->data_buffer)[error_index] += 1.0f;
            static_cast<float*>(backward_weight_delta->parent->data_buffer)[weight_index] += 1.0f;
            static_cast<float*>(backward_bias_delta->parent->data_buffer)[bias_index] += 1.0f;

            EXPECT_NE(true, compare_data_items(backward_error_delta, &ref_backward_error_delta));
            EXPECT_NE(true, compare_data_items(backward_weight_delta, &ref_backward_weight_delta));
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
            EXPECT_EQ(true, compare_data_items(backward_weight_delta, &ref_backward_weight_delta));
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
TEST(cpu_convolution_backprop, standard_square_positive)
{
    for (auto batch : { 1, 8 })
        for (auto in = 1; in < 5; in++)
            for (auto in_z : { 1, 2, 3, 4 })
                for (auto out_z : { 16, 32 })
                    for (auto wght = in; wght > 0; wght--)
                        for (auto padding = 0; padding < wght-1; ++padding)
                            for (auto stride : { 1, 2 })
                                run_test(stride, stride, wght, wght, padding, padding, in_z, in, in, out_z, batch, false);
}

TEST(cpu_convolution_backprop, standard_square_negative)
{
    for (auto batch : { 1, 8 })
        for (auto in = 1; in < 5; in++)
            for (auto in_z : { 1, 2, 3, 4 })
                for (auto out_z : { 16, 32 })
                    for (auto wght = in; wght > 0; wght--)
                        for (auto padding = 0; padding < wght-1; ++padding)
                            for (auto stride : { 1, 2 })
                                run_test(stride, stride, wght, wght, padding, padding, in_z, in, in, out_z, batch, true);
}

