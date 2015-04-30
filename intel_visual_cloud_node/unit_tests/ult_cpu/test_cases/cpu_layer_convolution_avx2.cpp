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
#include "gtest/gtest.h"

#include "../../devices/common/nn_workload_data.h"
#include "../../devices/device_cpu/core/layer_convolution_avx2.h"
#include "../../devices/device_cpu/api_internal/nn_device_interface_0_internal.h"

const uint32_t C_simd_width = sizeof(__m256)/sizeof(float);
const uint32_t C_slice_size = 2 * C_simd_width;

namespace
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// Helper classess and functions.
void ult_nn_convolution_initialize_work_item(
    nn_workload_item* &work_item,
    nn_workload_item* &input_item,
    float *const input,
    float *const biases,
    float *const output,
    float *const kernel,
    uint32_t batch_size,
    uint32_t num_output_feature_maps,
    uint32_t num_input_feature_maps,
    uint32_t output_feature_map_width,
    uint32_t output_feature_map_height,
    uint32_t input_feature_map_width,
    uint32_t input_feature_map_height,
    uint32_t kernel_width,
    uint32_t kernel_height,
    uint32_t kernel_stride_x,
    uint32_t kernel_stride_y,
    NN_ACTIVATION_FUNCTION activation,
    nn_device *device)
{
    nn_workload_data_layout_t inp_out_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_FLOAT
    };
    nn_workload_data_layout_t weight_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q, NN_DATA_COORD_n }, // ordering
        NN_DATATYPE_FLOAT
    };

    nn_workload_data_coords_t input_coords = 
    { 
        batch_size,
        input_feature_map_width,
        input_feature_map_height,
        num_input_feature_maps, 
        1, 
        1 
    };

    nn_workload_data_coords_t output_coords = 
    { 
        batch_size,
        output_feature_map_width,
        output_feature_map_height,
        num_output_feature_maps,
        1,
        1
    };

    nn_workload_data_coords_t bias_coords =
    {
        1,
        num_output_feature_maps,
        1,
        1,
        1,
        1
    };

    nn_workload_data_coords_t weight_coords = 
    { 
        1, 
        kernel_width,
        kernel_height,
        num_input_feature_maps,
        C_slice_size,
        num_output_feature_maps / C_slice_size
    };

    nn::nn_workload_data_t<float> *output_data = new nn::nn_workload_data_t<float>(output_coords, inp_out_layout);
    nn::nn_workload_data_t<float> *bias_data = new nn::nn_workload_data_t<float>(bias_coords, inp_out_layout);
    nn::nn_workload_data_t<float> *weight_data = new nn::nn_workload_data_t<float>(weight_coords, weight_layout);

    work_item = new nn_workload_item();

    work_item->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
    nn_argument_activation_t s_activation;
    s_activation.function = activation;
    uint32_t center_offset_x = (kernel_width - 1) / 2, center_offset_y = (kernel_width - 1) / 2;
    work_item->primitive = layer::convolution_f32::create(kernel_width,
                                                          kernel_height,
                                                          num_input_feature_maps,
                                                          num_output_feature_maps,
                                                          output_feature_map_width,
                                                          output_feature_map_height,
                                                          center_offset_x,
                                                          center_offset_y,
                                                          kernel_stride_x,
                                                          kernel_stride_y,
                                                          s_activation,
                                                          batch_size,
                                                          device);
    
    auto &arguments = work_item->arguments.forward_convolution;
    work_item->output = output_data;
    arguments.biases = bias_data;
    arguments.weights = weight_data;

    memcpy(work_item->output->parent->data_buffer, output, work_item->output->parent->buffer_size);
    memcpy(arguments.biases->parent->data_buffer, biases, arguments.biases->parent->buffer_size);
    memcpy(arguments.weights->parent->data_buffer, kernel, arguments.weights->parent->buffer_size);

    input_item = new nn_workload_item();
    input_item->type = NN_WORK_ITEM_TYPE_INPUT;

    nn::nn_workload_data_t<float> *input_data = new nn::nn_workload_data_t<float>(input_coords, inp_out_layout);

    nn_workload_data_coords_t input_view_begin =
    {
        0,
        center_offset_x,
        center_offset_y,
        0,
        0,
        0
    };
    nn_workload_data_coords_t input_view_end = {
        input_data->get_length(NN_DATA_COORD_n) - 1,
        input_data->get_length(NN_DATA_COORD_x) - 1,
        input_data->get_length(NN_DATA_COORD_y) - 1,
        input_data->get_length(NN_DATA_COORD_z) - 1,
        input_data->get_length(NN_DATA_COORD_p) - 1,
        input_data->get_length(NN_DATA_COORD_q) - 1
    };

    nn::nn_workload_data_t<float> *view_input_data = new nn::nn_workload_data_t<float>(*input_data, input_view_begin, input_view_end);
    delete input_data;

    memcpy(view_input_data->parent->data_buffer, input, view_input_data->parent->buffer_size);
    input_item->output = view_input_data;

    work_item->input.push_back(input_item);
}

void ult_nn_convolution_initialize_work_items(
    nn_workload_item* work_items[],
    nn_workload_item* &work_item,
    nn_workload_item* input_items[],
    nn_workload_item* &input_item,
    float *const input,
    float *const biases,
    float *const output,
    float *const kernel,
    uint32_t batch_size,
    uint32_t num_output_feature_maps,
    uint32_t num_input_feature_maps,
    uint32_t output_feature_map_width,
    uint32_t output_feature_map_height,
    uint32_t input_feature_map_width,
    uint32_t input_feature_map_height,
    uint32_t kernel_width,
    uint32_t kernel_height,
    uint32_t kernel_stride_x,
    uint32_t kernel_stride_y,
    NN_ACTIVATION_FUNCTION activation,
    nn_device_t *device)
{
    nn_workload_data_layout_t inp_out_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_FLOAT
    };
    nn_workload_data_layout_t weight_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q, NN_DATA_COORD_n }, // ordering
        NN_DATATYPE_FLOAT
    };

    nn_workload_data_coords_t input_coords =
    {
        batch_size,
        input_feature_map_width,
        input_feature_map_height,
        num_input_feature_maps,
        1,
        1
    };

    nn_workload_data_coords_t output_coords =
    {
        batch_size,
        output_feature_map_width,
        output_feature_map_height,
        num_output_feature_maps,
        1,
        1
    };

    nn_workload_data_coords_t bias_coords =
    {
        1,
        num_output_feature_maps,
        1,
        1,
        1,
        1
    };

    nn_workload_data_coords_t weight_coords =
    {
        1,
        kernel_width,
        kernel_height,
        num_input_feature_maps,
        C_slice_size,
        num_output_feature_maps / C_slice_size
    };

    nn_workload_data_coords_t input_end_coords =
    {
        0,
        input_feature_map_width - 1,
        input_feature_map_height - 1,
        num_input_feature_maps - 1,
        0,
        0
    };

    nn::nn_workload_data_t<float>  *input_data = new nn::nn_workload_data_t<float>(input_coords, inp_out_layout);
    nn::nn_workload_data_t<float> *output_data = new nn::nn_workload_data_t<float>(output_coords, inp_out_layout);
    nn::nn_workload_data_t<float>   *bias_data = new nn::nn_workload_data_t<float>(bias_coords, inp_out_layout);
    nn::nn_workload_data_t<float> *weight_data = new nn::nn_workload_data_t<float>(weight_coords, weight_layout);

    work_item = new nn_workload_item();

    work_item->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
    nn_argument_activation_t s_activation;
    s_activation.function = activation;
    uint32_t center_offset_x = (kernel_width - 1) / 2, center_offset_y = (kernel_width - 1) / 2;

    work_item->primitive = layer::convolution_f32::create(kernel_width,
                                                          kernel_height,
                                                          num_input_feature_maps,
                                                          num_output_feature_maps,
                                                          output_feature_map_width,
                                                          output_feature_map_height,
                                                          center_offset_x,
                                                          center_offset_y,
                                                          kernel_stride_x,
                                                          kernel_stride_y,
                                                          s_activation,
                                                          batch_size,
                                                          device);

    auto &arguments = work_item->arguments.forward_convolution;
    work_item->output = output_data;
    arguments.biases = bias_data;
    arguments.weights = weight_data;

    memcpy(output_data->parent->data_buffer, output, output_data->parent->buffer_size);
    memcpy(bias_data->parent->data_buffer, biases, bias_data->parent->buffer_size);
    memcpy(weight_data->parent->data_buffer, kernel, weight_data->parent->buffer_size);

    nn_workload_data_coords_t input_view_begin =
    {
        0,
        center_offset_x,
        center_offset_y,
        0,
        0,
        0
    };
    nn_workload_data_coords_t input_view_end = {
        input_data->get_length(NN_DATA_COORD_n) - 1,
        input_data->get_length(NN_DATA_COORD_x) - 1,
        input_data->get_length(NN_DATA_COORD_y) - 1,
        input_data->get_length(NN_DATA_COORD_z) - 1,
        input_data->get_length(NN_DATA_COORD_p) - 1,
        input_data->get_length(NN_DATA_COORD_q) - 1
    };

    nn::nn_workload_data_t<float> *view_input_data = new nn::nn_workload_data_t<float>(*input_data, input_view_begin, input_view_end);
    nn_workload_data_coords_t sub_input_view_end = {
        view_input_data->get_length(NN_DATA_COORD_n) - 1,
        view_input_data->get_length(NN_DATA_COORD_x) - 1,
        view_input_data->get_length(NN_DATA_COORD_y) - 1,
        view_input_data->get_length(NN_DATA_COORD_z) - 1,
        view_input_data->get_length(NN_DATA_COORD_p) - 1,
        view_input_data->get_length(NN_DATA_COORD_q) - 1
    };
    delete input_data;

    for (unsigned int item = 0; item < 8; item++)
    {
        work_items[item] = new nn_workload_item();
        work_items[item]->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
        nn_argument_activation_t s_activation;
        s_activation.function = activation;
        uint32_t center_offset_x = (kernel_width - 1) / 2, center_offset_y = (kernel_width - 1) / 2;

        work_items[item]->primitive = layer::convolution_f32::create(kernel_width,
                                                                     kernel_height,
                                                                     num_input_feature_maps,
                                                                     num_output_feature_maps,
                                                                     output_feature_map_width,
                                                                     output_feature_map_height,
                                                                     center_offset_x,
                                                                     center_offset_y,
                                                                     kernel_stride_x,
                                                                     kernel_stride_y,
                                                                     s_activation,
                                                                     batch_size,
                                                                     device);

        nn_workload_data_coords_t nn_view_begin = 
        { 
            0, 
            (item & 1) ? 0 : output_feature_map_width/2,
            (item & 2) ? 0 : output_feature_map_height/2,
            (item & 4) ? 0 : num_output_feature_maps/2,
            0, 
            0 
        };

        nn_workload_data_coords_t nn_input_view_begin =
        {
            0,
            (item & 1) ? 0 : (output_feature_map_width / 2) * kernel_stride_x,
            (item & 2) ? 0 : (output_feature_map_height / 2) * kernel_stride_y,
            0,
            0,
            0
        };

        nn_workload_data_coords_t nn_view_end =
        { 
            output_data->get_length(NN_DATA_COORD_n) - 1,
            (item & 1) ? output_feature_map_width/2 - 1: output_feature_map_width - 1,
            (item & 2) ? output_feature_map_height/2 - 1: output_feature_map_height - 1,
            (item & 4) ? num_output_feature_maps/2 - 1: num_output_feature_maps - 1,
            output_data->get_length(NN_DATA_COORD_p) - 1,
            output_data->get_length(NN_DATA_COORD_q) - 1,
        };

        nn_workload_data_coords_t nn_weights_view_begin = 
        { 
            0, 
            0,
            0,
            0,
            0, 
            (item & 4) ? 0 : num_output_feature_maps/2/C_slice_size,
        };

        nn_workload_data_coords_t nn_weights_view_end = 
        { 
            weight_data->get_length(NN_DATA_COORD_n) - 1, 
            weight_data->get_length(NN_DATA_COORD_x) - 1,
            weight_data->get_length(NN_DATA_COORD_y) - 1,
            weight_data->get_length(NN_DATA_COORD_z) - 1,
            weight_data->get_length(NN_DATA_COORD_p) - 1, 
            (item & 4) ? num_output_feature_maps/2/C_slice_size - 1: num_output_feature_maps/C_slice_size - 1
        };

        nn_workload_data_coords_t nn_biases_view_begin = 
        { 
            0, 
            (item & 4) ? 0 : num_output_feature_maps/2,
            0,
            0,
            0,
            0, 
        };

        nn_workload_data_coords_t nn_biases_view_end = 
        { 
            bias_data->get_length(NN_DATA_COORD_n) - 1, 
            (item & 4) ? num_output_feature_maps/2 - 1 : num_output_feature_maps - 1,
            bias_data->get_length(NN_DATA_COORD_y) - 1,
            bias_data->get_length(NN_DATA_COORD_z) - 1,
            bias_data->get_length(NN_DATA_COORD_p) - 1,
            bias_data->get_length(NN_DATA_COORD_q) - 1, 
        };

        auto &sub_arguments = work_items[item]->arguments.forward_convolution;
        sub_arguments.weights = new nn::nn_workload_data_t<float>(*weight_data, nn_weights_view_begin, nn_weights_view_end);
        sub_arguments.biases = new nn::nn_workload_data_t<float>(*bias_data, nn_biases_view_begin, nn_biases_view_end);

        work_items[item]->output = new nn::nn_workload_data_t<float>(*output_data, nn_view_begin, nn_view_end);

        input_items[item] = new nn_workload_item();
        input_items[item]->type = NN_WORK_ITEM_TYPE_INPUT;

        input_items[item]->output = new nn::nn_workload_data_t<float>(*view_input_data, nn_input_view_begin, sub_input_view_end);

        work_items[item]->input.push_back(input_items[item]);
    }

    input_item = new nn_workload_item();
    input_item->type = NN_WORK_ITEM_TYPE_INPUT;

    memcpy(view_input_data->parent->data_buffer, input, view_input_data->parent->buffer_size);
    input_item->output = view_input_data;

    work_item->input.push_back(input_item);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolution_deinitialize_work_item(nn_workload_item* &work_item)
{
    if (work_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION)
    {
        auto &arguments = work_item->arguments.forward_convolution;
        delete reinterpret_cast<nn::nn_workload_data_t<float>*>(arguments.biases);
        delete reinterpret_cast<nn::nn_workload_data_t<float>*>(arguments.weights);
    }

    work_item->input.clear();
    delete reinterpret_cast<nn::nn_workload_data_t<float>*>(work_item->output);

    delete work_item;

    work_item = nullptr;
}

void ult_nn_convolution_deinitialize_work_items(nn_workload_item* work_items[])
{
    for (unsigned int item = 0; item < 8; item++)
    {
        auto &arguments = work_items[item]->arguments.forward_convolution;
        work_items[item]->input.clear();
        delete reinterpret_cast<nn::nn_workload_data_t<float>*>(work_items[item]->output);

        if (work_items[item]->type == NN_WORK_ITEM_TYPE_CONVOLUTION)
        {
            delete reinterpret_cast<nn::nn_workload_data_t<float>*>(arguments.weights);
            delete reinterpret_cast<nn::nn_workload_data_t<float>*>(arguments.biases);
        }

        delete work_items[item];

        work_items[item] = nullptr;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
bool ult_nn_convolution_interface_run(nn_workload_item* work_item, nn_device_internal *device)
{
    bool retvalue = true;
    layer::run_multithreaded_convolve_work_item(work_item, device);
    return retvalue;
}

bool ult_nn_convolution_interface_run(nn_workload_item* work_items[], nn_device_internal *device)
{
    bool retvalue = true;

    for (unsigned int item = 0; item < 8 && retvalue; item++)
    {
        layer::run_multithreaded_convolve_work_item(work_items[item], device);
    }

    return retvalue;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolution_optimized_set_output_value(
    float* output,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    uint_least32_t batch,
    float value,
    uint_least32_t& offset)
{
    offset = batch*output_feature_map_width*output_feature_map_height*num_output_feature_maps + output_row*output_feature_map_width*num_output_feature_maps + output_column*num_output_feature_maps + output_map;
    output[offset] = value;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolution_optimized_set_input_value(
    float* input,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t num_input_feature_maps,
    uint_least32_t input_column,
    uint_least32_t input_row,
    uint_least32_t input_map,
    uint_least32_t batch,
    float value,
    uint_least32_t& offset)
{
    offset = batch*num_input_feature_maps*input_feature_map_width*input_feature_map_height + input_column*num_input_feature_maps + input_row*num_input_feature_maps*input_feature_map_width + input_map;

    input[offset] = value;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolution_optimized_set_kernel_value(
    float* kernel,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t num_input_feature_maps,
    uint_least32_t kernel_column,
    uint_least32_t kernel_row,
    uint_least32_t kernel_input_map,
    uint_least32_t kernel_output_map,
    float value,
    uint_least32_t& offset)
{
    uint_least32_t kernel_output_map_div = kernel_output_map / C_slice_size;
    uint_least32_t kernel_output_map_rem = kernel_output_map % C_slice_size;
    offset = kernel_row*C_slice_size*kernel_width*num_input_feature_maps + kernel_input_map*C_slice_size + kernel_column*C_slice_size*num_input_feature_maps + kernel_output_map_div*kernel_width*kernel_height*num_input_feature_maps*C_slice_size + kernel_output_map_rem;
    kernel[offset] = value;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
float ult_nn_convolution_naive_get_output_value(
    float* output_ref,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    uint_least32_t batch,
    uint_least32_t& offset)
{
    offset = batch*output_feature_map_width*output_feature_map_height*num_output_feature_maps + output_column + output_row*output_feature_map_width + output_map*output_feature_map_width*output_feature_map_height;
    return output_ref[offset];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolution_naive_set_output_value(
    float* output_ref,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    uint_least32_t batch,
    float value,
    uint_least32_t& offset)
{
    offset = batch*output_feature_map_width*output_feature_map_height*num_output_feature_maps + output_column + output_row*output_feature_map_width + output_map*output_feature_map_width*output_feature_map_height;
    output_ref[offset] = value;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolution_naive_set_input_value(
    float* input_ref,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t num_input_feature_maps,
    uint_least32_t input_column,
    uint_least32_t input_row,
    uint_least32_t input_map,
    uint_least32_t batch,
    float value,
    uint_least32_t& offset)
{
    offset = batch*num_input_feature_maps*input_feature_map_width*input_feature_map_height + input_column + input_row*input_feature_map_width + input_map*input_feature_map_width*input_feature_map_height;

    input_ref[offset] = value;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolution_naive_set_kernel_value(
    float* kernel_ref,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t num_input_feature_maps,
    uint_least32_t kernel_column,
    uint_least32_t kernel_row,
    uint_least32_t kernel_input_map,
    uint_least32_t kernel_output_map,
    float value,
    uint_least32_t& offset)
{
    offset = kernel_column + kernel_row*kernel_width + kernel_input_map*kernel_width*kernel_height + kernel_output_map*kernel_width*kernel_height*num_input_feature_maps;
    kernel_ref[offset] = value;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolution_both_initialize_matrices(
    float* input,
    float* output,
    float* biases,
    float* kernel,
    float* input_ref,
    float* output_ref,
    float* biases_ref,
    float* kernel_ref,
    uint_least32_t batch_size,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height )
{
    for (uint_least32_t batch = 0; batch < batch_size; batch++)
    {
        for (uint_least32_t input_map = 0; input_map < num_input_feature_maps; input_map++)
        {
            uint_least32_t element = 0;
            for (uint_least32_t row = 0; row < input_feature_map_height; row++)
            {
                for (uint_least32_t column = 0; column < input_feature_map_width; column++)
                {
                    float value = 0.25f;
                    value *= (float)pow(2.0, input_map);
                    value *= (float)pow(2.0, batch);

                    if (element / input_feature_map_width + element % input_feature_map_width + 1 > (input_feature_map_width + input_feature_map_height) / 2)
                    {
                        value *= 2.0f;
                    }
                    element++;

                    uint_least32_t offset;
                    ult_nn_convolution_optimized_set_input_value(input, input_feature_map_width, input_feature_map_height, num_input_feature_maps, column, row, input_map, batch, value, offset);
                    ult_nn_convolution_naive_set_input_value(input_ref, input_feature_map_width, input_feature_map_height, num_input_feature_maps, column, row, input_map, batch, value, offset);
                }
            }

        }
    }

    for (uint_least32_t outmapa = 0; outmapa < num_output_feature_maps; outmapa++)
    {
        for (uint_least32_t input_map = 0; input_map < num_input_feature_maps; input_map++)
        {
            uint_least32_t element = 0;
            for (uint_least32_t row = 0; row < kernel_height; row++)
            {
                for (uint_least32_t column = 0; column < kernel_width; column++)
                {
                    float value = 0.25f;
                    value *= (float)pow(2.0, input_map);
                    value *= (float)pow(2.0, outmapa);

                    if (element / kernel_width + element % kernel_width + 1 >(kernel_width + kernel_height) / 2)
                    {
                        value *= 2.0f;
                    }
                    element++;
                    uint_least32_t offset;
                    ult_nn_convolution_optimized_set_kernel_value(kernel, kernel_width, kernel_height, num_input_feature_maps, column, row, input_map, outmapa, value, offset);
                    ult_nn_convolution_naive_set_kernel_value(kernel_ref, kernel_width, kernel_height, num_input_feature_maps, column, row, input_map, outmapa, value, offset);
                }
            }
        }
    }

    for (uint_least32_t outmapa = 0; outmapa < num_output_feature_maps; outmapa++)
    {
        uint_least32_t offset;
        biases[outmapa] = 1.0f;
        biases_ref[outmapa] = 1.0f;

        for (uint_least32_t batch = 0; batch < batch_size; batch++)
        {
            for (uint_least32_t row = 0; row < output_feature_map_height; row++)
            {
                for (uint_least32_t column = 0; column < output_feature_map_width; column++)
                {
                    ult_nn_convolution_optimized_set_output_value(output, output_feature_map_width, output_feature_map_height, num_output_feature_maps, column, row, outmapa, batch, 0.0f, offset);
                    ult_nn_convolution_naive_set_output_value(output_ref, output_feature_map_width, output_feature_map_height, num_output_feature_maps, column, row, outmapa, batch, 0.0f, offset);
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool ult_nn_convolution_check_outputs(
    nn_workload_data_t* output,
    float* output_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t batch_size)
{
    bool passed = true;
    for (uint_least32_t batch = 0; batch < batch_size && passed; batch++)
    {
        for (uint_least32_t outmapa = 0; outmapa < num_output_feature_maps && passed; outmapa++)
        {
            for (uint_least32_t row = 0; row < output_feature_map_height && passed; row++)
            {
                for (uint_least32_t column = 0; column < output_feature_map_width && passed; column++)
                {
                    uint_least32_t offset;
                    float ref_value = ult_nn_convolution_naive_get_output_value(output_ref, output_feature_map_width, output_feature_map_height, num_output_feature_maps, column, row, outmapa, batch, offset);
                    float value = nn_workload_data_get<float>(output, batch, column, row, outmapa, 0, 0);

                    float diff = fabs(ref_value - value);

                    if (ref_value == 0.0f || value == 0.0f || diff < FLT_MIN)
                    {
                        if (diff > FLT_MIN)
                        {
                            passed = false;
                        }
                    }
                    else
                    {
                        if (diff / ref_value > 5.2e-06F)
                        {
                            passed = false;
                        }
                    }
                }
            }
        }
    }

    return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolution_both_alloc(
    float* &input,
    float* &output,
    float* &biases,
    float* &kernel,
    float* &input_ref,
    float* &output_ref,
    float* &biases_ref,
    float* &kernel_ref,
    uint_least32_t batch_size,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_width,
    uint_least32_t output_height,
    uint_least32_t input_width,
    uint_least32_t input_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height)
{
    uint_least32_t input_size = batch_size * input_width * input_height * num_input_feature_maps * sizeof(float);
    uint_least32_t output_size = batch_size * output_width * output_height * num_output_feature_maps * sizeof(float);
    uint_least32_t kernel_size = num_input_feature_maps * num_output_feature_maps * kernel_width * kernel_height * sizeof(float);

    input_ref = (float*)_mm_malloc(input_size, 64);
    output_ref = (float*)_mm_malloc(output_size, 64);
    biases_ref = (float*)_mm_malloc(output_size, 64);
    kernel_ref = (float*)_mm_malloc(kernel_size, 64);

    input_size = batch_size * input_width * input_height * num_input_feature_maps * sizeof(float);
    output_size = batch_size * output_width * output_height * num_output_feature_maps * sizeof(float);
    kernel_size = num_input_feature_maps * num_output_feature_maps * kernel_width * kernel_height * sizeof(float);

    input = (float*)_mm_malloc(input_size, 64);
    output = (float*)_mm_malloc(output_size, 64);
    biases = (float*)_mm_malloc(num_output_feature_maps * sizeof(float), 64);
    kernel = (float*)_mm_malloc(kernel_size, 64);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolution_both_dealloc(
    float* &input,
    float* &output,
    float* &biases,
    float* &kernel,
    float* &input_ref,
    float* &output_ref,
    float* &biases_ref,
    float* &kernel_ref)
{
    if (input != 0)
    {
        _mm_free(input);
        input = 0;
    }

    if (output != 0)
    {
        _mm_free(output);
        output = 0;
    }

    if (biases != 0)
    {
        _mm_free(biases);
        biases = 0;
    }

    if (kernel != 0)
    {
        _mm_free(kernel);
        kernel = 0;
    }

    if (input_ref != 0)
    {
        _mm_free(input_ref);
        input_ref = 0;
    }

    if (output_ref != 0)
    {
        _mm_free(output_ref);
        output_ref = 0;
    }

    if (biases_ref != 0)
    {
        _mm_free(biases_ref);
        biases_ref = 0;
    }

    if (kernel_ref != 0)
    {
        _mm_free(kernel_ref);
        kernel_ref = 0;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ult_nn_convolve_naive(
    float* input_ref,
    float* output_ref,
    float* biases_ref,
    float* kernel_ref,
    uint_least32_t batch_size,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    NN_ACTIVATION_FUNCTION activation)
{
    for (uint_least32_t batch = 0; batch < batch_size; batch++)
    {
        for (uint_least32_t output_feature_map = 0; output_feature_map < num_output_feature_maps; output_feature_map += 8)
        {
            for (uint_least32_t input_row = 0, output_row = 0; output_row < output_feature_map_height; input_row += kernel_stride_y, output_row++)
            {
                for (uint_least32_t input_column = 0, output_column = 0; output_column < output_feature_map_width; input_column += kernel_stride_x, output_column++)
                {
                    const uint_least32_t out_ofss = output_feature_map_width*output_feature_map_height;
                    const uint_least32_t out_base = batch*output_feature_map_width*output_feature_map_height*num_output_feature_maps + output_column + output_row*output_feature_map_width + output_feature_map*output_feature_map_width*output_feature_map_height;
                    float accumulator0 = 0.0f;
                    float accumulator1 = 0.0f;
                    float accumulator2 = 0.0f;
                    float accumulator3 = 0.0f;
                    float accumulator4 = 0.0f;
                    float accumulator5 = 0.0f;
                    float accumulator6 = 0.0f;
                    float accumulator7 = 0.0f;

                    for (uint_least32_t input_feature_map = 0; input_feature_map < num_input_feature_maps; input_feature_map++)
                    {
                        for (uint_least32_t kernel_row = 0; kernel_row < kernel_height; kernel_row++)
                        {
                            const uint_least32_t kern_ofss = kernel_width*kernel_height*num_input_feature_maps;
                            uint_least32_t kern_base = kernel_row*kernel_width + input_feature_map*kernel_width*kernel_height + output_feature_map*kernel_width*kernel_height*num_input_feature_maps;

                            for (uint_least32_t kernel_column = 0; kernel_column < kernel_width; kernel_column++)
                            {
                                float weight0 = kernel_ref[kern_base + 0 * kern_ofss];
                                float weight1 = kernel_ref[kern_base + 1 * kern_ofss];
                                float weight2 = kernel_ref[kern_base + 2 * kern_ofss];
                                float weight3 = kernel_ref[kern_base + 3 * kern_ofss];
                                float weight4 = kernel_ref[kern_base + 4 * kern_ofss];
                                float weight5 = kernel_ref[kern_base + 5 * kern_ofss];
                                float weight6 = kernel_ref[kern_base + 6 * kern_ofss];
                                float weight7 = kernel_ref[kern_base + 7 * kern_ofss];

                                ++kern_base;

                                float input = input_ref[batch*num_input_feature_maps*input_feature_map_width*input_feature_map_height + kernel_column + kernel_row*input_feature_map_width + input_column + input_row*input_feature_map_width + input_feature_map*input_feature_map_width*input_feature_map_height];

                                accumulator0 += weight0 * input;
                                accumulator1 += weight1 * input;
                                accumulator2 += weight2 * input;
                                accumulator3 += weight3 * input;
                                accumulator4 += weight4 * input;
                                accumulator5 += weight5 * input;
                                accumulator6 += weight6 * input;
                                accumulator7 += weight7 * input;
                            }
                        }
                    }

                    switch (activation)
                    {
                    case NN_ACTIVATION_FUNCTION_RELU:
                        accumulator0 = std::max(0.0f, accumulator0 + biases_ref[output_feature_map + 0]);
                        accumulator1 = std::max(0.0f, accumulator1 + biases_ref[output_feature_map + 1]);
                        accumulator2 = std::max(0.0f, accumulator2 + biases_ref[output_feature_map + 2]);
                        accumulator3 = std::max(0.0f, accumulator3 + biases_ref[output_feature_map + 3]);
                        accumulator4 = std::max(0.0f, accumulator4 + biases_ref[output_feature_map + 4]);
                        accumulator5 = std::max(0.0f, accumulator5 + biases_ref[output_feature_map + 5]);
                        accumulator6 = std::max(0.0f, accumulator6 + biases_ref[output_feature_map + 6]);
                        accumulator7 = std::max(0.0f, accumulator7 + biases_ref[output_feature_map + 7]);
                        break;

                    case NN_ACTIVATION_FUNCTION_NONE:
                        accumulator0 = accumulator0 + biases_ref[output_feature_map + 0];
                        accumulator1 = accumulator1 + biases_ref[output_feature_map + 1];
                        accumulator2 = accumulator2 + biases_ref[output_feature_map + 2];
                        accumulator3 = accumulator3 + biases_ref[output_feature_map + 3];
                        accumulator4 = accumulator4 + biases_ref[output_feature_map + 4];
                        accumulator5 = accumulator5 + biases_ref[output_feature_map + 5];
                        accumulator6 = accumulator6 + biases_ref[output_feature_map + 6];
                        accumulator7 = accumulator7 + biases_ref[output_feature_map + 7];
                        break;

                    default:
                        break;
                    }

                    output_ref[out_base + 0 * out_ofss] = accumulator0;
                    output_ref[out_base + 1 * out_ofss] = accumulator1;
                    output_ref[out_base + 2 * out_ofss] = accumulator2;
                    output_ref[out_base + 3 * out_ofss] = accumulator3;
                    output_ref[out_base + 4 * out_ofss] = accumulator4;
                    output_ref[out_base + 5 * out_ofss] = accumulator5;
                    output_ref[out_base + 6 * out_ofss] = accumulator6;
                    output_ref[out_base + 7 * out_ofss] = accumulator7;
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool ult_perform_test(
    uint_least32_t batch_size,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    bool check_out_views,
    NN_ACTIVATION_FUNCTION activation )
{
    nn_workload_item* work_item = nullptr;
    nn_workload_item* work_items[8];

    nn_workload_item* input_item = nullptr;
    nn_workload_item* input_items[8];

    std::fill_n(work_items, 8, nullptr);

    bool passed = false;

    float* input = 0;
    float* output = 0;
    float* biases = 0;
    float* kernel = 0;

    float* input_ref = 0;
    float* output_ref = 0;
    float* biases_ref = 0;
    float* kernel_ref = 0;

    uint_least32_t output_feature_map_width = (input_feature_map_width - kernel_width) / kernel_stride_x + 1;
    uint_least32_t output_feature_map_height = (input_feature_map_height - kernel_height) / kernel_stride_y + 1;

    num_output_feature_maps += (C_slice_size - (num_output_feature_maps % C_slice_size)) % C_slice_size;

    // Allocate naive and optimized buffers.
    ult_nn_convolution_both_alloc(
        input,
        output,
        biases,
        kernel,
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref,
        batch_size,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        input_feature_map_width,
        input_feature_map_height,
        kernel_width,
        kernel_height );

    // Initialize both buffers.
    ult_nn_convolution_both_initialize_matrices(
        input,
        output,
        biases,
        kernel,
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref,
        batch_size,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        input_feature_map_width,
        input_feature_map_height,
        kernel_width,
        kernel_height);

    // Naive convolution
    ult_nn_convolve_naive(
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref,
        batch_size,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        input_feature_map_width,
        input_feature_map_height,
        kernel_width,
        kernel_height,
        kernel_stride_x,
        kernel_stride_y,
        activation);

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;

    nn_device_load(&device_description);
    nn_device_interface_open(0, &device_interface_0);

    if (check_out_views)
    {
        // Perform data copy to interface test.
        ult_nn_convolution_initialize_work_items(
            work_items,
            work_item,
            input_items,
            input_item,
            input,
            biases,
            output,
            kernel,
            batch_size,
            num_output_feature_maps,
            num_input_feature_maps,
            output_feature_map_width,
            output_feature_map_height,
            input_feature_map_width,
            input_feature_map_height,
            kernel_width,
            kernel_height,
            kernel_stride_x,
            kernel_stride_y,
            activation,
            device_interface_0.device);

        // Optimized convolution.
        passed = ult_nn_convolution_interface_run(work_items, reinterpret_cast<nn_device_internal *>(device_interface_0.device));
    }
    else
    {
        // Perform data copy to interface test.
        ult_nn_convolution_initialize_work_item(
            work_item,
            input_item,
            input,
            biases,
            output,
            kernel,
            batch_size,
            num_output_feature_maps,
            num_input_feature_maps,
            output_feature_map_width,
            output_feature_map_height,
            input_feature_map_width,
            input_feature_map_height,
            kernel_width,
            kernel_height,
            kernel_stride_x,
            kernel_stride_y,
            activation,
            device_interface_0.device);

        // Optimized convolution.
        passed = ult_nn_convolution_interface_run(work_item, reinterpret_cast<nn_device_internal *>(device_interface_0.device));
    }

    if (passed)
    {
        // Basic check between optimized and naive versions.
        passed = ult_nn_convolution_check_outputs(
            work_item->output,
            output_ref,
            num_output_feature_maps,
            output_feature_map_width,
            output_feature_map_height,
            batch_size);
    }

    // Cleanup.
    ult_nn_convolution_deinitialize_work_item(work_item);
    ult_nn_convolution_deinitialize_work_item(input_item);

    if (check_out_views)
    {
        ult_nn_convolution_deinitialize_work_items(work_items);
        ult_nn_convolution_deinitialize_work_items(input_items);
    }

    ult_nn_convolution_both_dealloc(
        input,
        output,
        biases,
        kernel,
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref);

    nn_device_interface_close(&device_interface_0);
    nn_device_unload();

    return passed;
}

bool ult_perform_padding_test(
    uint_least32_t batch_size,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    NN_ACTIVATION_FUNCTION activation)
{
    uint32_t center_offset_x = (kernel_width - 1) / 2;
    uint32_t center_offset_y = (kernel_height - 1) / 2;

    uint32_t ofm_width = (input_feature_map_width + kernel_stride_x - 1) / kernel_stride_x;
    uint32_t ofm_height = (input_feature_map_height + kernel_stride_y - 1) / kernel_stride_y;

    nn_workload_data_layout_t img_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q },
        NN_DATATYPE_FLOAT
    };

    nn_workload_data_coords_t out_size =
    {
        batch_size,
        ofm_width,
        ofm_height,
        num_output_feature_maps,
        1,
        1
    };

    nn_workload_data_coords_t bias_size =
    {
        1,
        num_output_feature_maps,
        1,
        1,
        1,
        1
    };

    nn_workload_data_coords_t kernel_size =
    {
        1,
        kernel_width,
        kernel_height,
        num_input_feature_maps,
        C_slice_size,
        num_output_feature_maps / C_slice_size
    };

    nn_workload_data_layout_t kernel_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q, NN_DATA_COORD_n },
        NN_DATATYPE_FLOAT
    };

    // Create input items.
    nn_workload_item* reference_input = new nn_workload_item();
    {
        reference_input->type = NN_WORK_ITEM_TYPE_INPUT;

        nn_workload_data_coords_t reference_input_size =
        {
            batch_size,
            input_feature_map_width + (kernel_width - 1),
            input_feature_map_height + (kernel_height - 1),
            num_input_feature_maps,
            1,
            1
        };
        nn::nn_workload_data_t<float>* out_buffer = new nn::nn_workload_data_t<float>(reference_input_size, img_layout);

        nn_workload_data_coords_t reference_input_subview_begin =
        {
            0,
            0 + center_offset_x,
            0 + center_offset_y,
            0,
            0,
            0
        };

        nn_workload_data_coords_t reference_input_subview_end =
        {
            out_buffer->get_length(NN_DATA_COORD_n) - 1,
            out_buffer->get_length(NN_DATA_COORD_x) - 1 - ((kernel_width - 1) - center_offset_x),
            out_buffer->get_length(NN_DATA_COORD_y) - 1 - ((kernel_height - 1) - center_offset_y),
            out_buffer->get_length(NN_DATA_COORD_z) - 1,
            out_buffer->get_length(NN_DATA_COORD_p) - 1,
            out_buffer->get_length(NN_DATA_COORD_q) - 1
        };

        memset(out_buffer->parent->data_buffer, 0, out_buffer->parent->buffer_size);

        reference_input->output = new nn::nn_workload_data_t<float>(*out_buffer, reference_input_subview_begin, reference_input_subview_end);
        delete out_buffer;
    }

    nn_workload_item* tested_input = new nn_workload_item();
    {
        tested_input->type = NN_WORK_ITEM_TYPE_INPUT;

        nn_workload_data_coords_t reference_input_size =
        {
            batch_size,
            input_feature_map_width,
            input_feature_map_height,
            num_input_feature_maps,
            1,
            1
        };
        tested_input->output = new nn::nn_workload_data_t<float>(reference_input_size, img_layout);
    }

    // Initialize input buffers.
    for (uint32_t batch = 0; batch < batch_size; ++batch)
    {
        for (uint32_t row = 0; row < input_feature_map_height; ++row)
        {
            for (uint32_t column = 0; column < input_feature_map_width; ++column)
            {
                for (uint32_t map = 0; map < num_input_feature_maps; ++map)
                {
                    float value = 1.0f;

                    value *= std::pow(-2.0, batch + row + column + map);

                    nn_workload_data_get<float>(reference_input->output, batch, column, row, map, 0, 0) = value;
                    nn_workload_data_get<float>(tested_input->output, batch, column, row, map, 0, 0) = value;
                }
            }
        }
    }

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    nn_device_t *device;

    nn_device_load(&device_description);
    nn_device_interface_open(0, &device_interface_0);
    device = device_interface_0.device;

    nn_workload_item* reference_conv = new nn_workload_item();
    {
        reference_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
        auto& args = reference_conv->arguments.forward_convolution;
        nn_argument_activation_t s_activation;
        s_activation.function = activation;

        reference_conv->primitive = layer::convolution_f32::create(kernel_width,
                                                                   kernel_height,
                                                                   num_input_feature_maps,
                                                                   num_output_feature_maps,
                                                                   ofm_width,
                                                                   ofm_height,
                                                                   center_offset_x,
                                                                   center_offset_y,
                                                                   kernel_stride_x,
                                                                   kernel_stride_y,
                                                                   s_activation,
                                                                   batch_size,
                                                                   device);

        args.biases = new nn::nn_workload_data_t<float>(bias_size, img_layout);
        args.weights = new nn::nn_workload_data_t<float>(kernel_size, kernel_layout);

        reference_conv->output = new nn::nn_workload_data_t<float>(out_size, img_layout);

        reference_conv->input.push_back(reference_input);
        reference_input->use.push_back(reference_conv);
    }

    nn_workload_item* tested_conv = new nn_workload_item();
    {
        tested_conv->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
        auto& args = tested_conv->arguments.forward_convolution;
        nn_argument_activation_t s_activation;
        s_activation.function = activation;

        tested_conv->primitive = layer::convolution_f32::create(kernel_width,
                                                                kernel_height,
                                                                num_input_feature_maps,
                                                                num_output_feature_maps,
                                                                ofm_width,
                                                                ofm_height,
                                                                center_offset_x,
                                                                center_offset_y,
                                                                kernel_stride_x,
                                                                kernel_stride_y,
                                                                s_activation,
                                                                batch_size,
                                                                device);

        args.biases = new nn::nn_workload_data_t<float>(bias_size, img_layout);
        args.weights = new nn::nn_workload_data_t<float>(kernel_size, kernel_layout);

        tested_conv->output = new nn::nn_workload_data_t<float>(out_size, img_layout);

        tested_conv->input.push_back(tested_input);
        tested_input->use.push_back(tested_conv);
    }

    // Initialize kernel buffers.
    for (uint32_t out_map = 0; out_map < num_output_feature_maps; ++out_map)
    {
        for (uint32_t row = 0; row < kernel_height; ++row)
        {
            for (uint32_t column = 0; column < kernel_width; ++column)
            {
                for (uint32_t map = 0; map < num_input_feature_maps; ++map)
                {
                    float value = 1.0f;

                    value *= std::pow(2.0, out_map + row + column + map);

                    nn_workload_data_get<float>(reference_conv->arguments.forward_convolution.weights, 0, column, row, map, out_map % C_slice_size, out_map / C_slice_size) = value;
                    nn_workload_data_get<float>(tested_conv->arguments.forward_convolution.weights, 0, column, row, map, out_map % C_slice_size, out_map / C_slice_size) = value;
                }
            }
        }
    }

    // Initialize bias buffers.
    for (uint32_t out_map = 0; out_map < num_output_feature_maps; ++out_map)
    {
        float value = 1.0f;

        value *= std::pow(2.0, out_map);

        nn_workload_data_get<float>(reference_conv->arguments.forward_convolution.biases, 0, out_map, 0, 0, 0, 0) = value;
        nn_workload_data_get<float>(tested_conv->arguments.forward_convolution.biases, 0, out_map, 0, 0, 0, 0) = value;
    }

    layer::run_multithreaded_convolve_work_item(reference_conv, reinterpret_cast<nn_device_internal *>(device));
    layer::run_multithreaded_convolve_work_item(tested_conv, reinterpret_cast<nn_device_internal *>(device));

    nn_device_interface_close(&device_interface_0);
    nn_device_unload();

    bool passed = true;

    for (uint32_t batch = 0; batch < batch_size && passed; ++batch)
    {
        for (uint32_t out_map = 0; out_map < num_output_feature_maps && passed; ++out_map)
        {
            for (uint32_t row = 0; row < ofm_height && passed; ++row)
            {
                for (uint32_t column = 0; column < ofm_width && passed; ++column)
                {
                    float reference = nn_workload_data_get<float>(reference_conv->output, batch, column, row, out_map, 0, 0);
                    float tested = nn_workload_data_get<float>(tested_conv->output, batch, column, row, out_map, 0, 0);

                    float diff = fabs(reference - tested);

                    if (reference == 0.0f || tested == 0.0f || diff < FLT_MIN)
                    {
                        if (diff > FLT_MIN)
                        {
                            passed = false;
                            std::cout
                                << "Error in B/OFM/R/C Ref/Test: "
                                << batch << "/"
                                << out_map << "/"
                                << row << "/"
                                << column << " "
                                << reference << "/"
                                << tested << std::endl;
                        }
                    }
                    else
                    {
                        if (fabs(diff / reference) > 5.2e-06F)
                        {
                            passed = false;
                            std::cout 
                                << "Error in B/OFM/R/C Ref/Test: " 
                                << batch << "/" 
                                << out_map << "/" 
                                << row << "/" 
                                << column << " "
                                << reference << "/" 
                                << tested << std::endl;
                        }
                    }
                }
            }
        }
    }

    delete reinterpret_cast<nn::nn_workload_data_t<float>*>(tested_conv->arguments.forward_convolution.biases);
    delete reinterpret_cast<nn::nn_workload_data_t<float>*>(tested_conv->arguments.forward_convolution.weights);
    delete reinterpret_cast<nn::nn_workload_data_t<float>*>(tested_conv->output);
    delete tested_conv;

    delete reinterpret_cast<nn::nn_workload_data_t<float>*>(reference_conv->arguments.forward_convolution.biases);
    delete reinterpret_cast<nn::nn_workload_data_t<float>*>(reference_conv->arguments.forward_convolution.weights);
    delete reinterpret_cast<nn::nn_workload_data_t<float>*>(reference_conv->output);
    delete reference_conv;

    delete reinterpret_cast<nn::nn_workload_data_t<float>*>(tested_input->output);
    delete tested_input;

    delete reinterpret_cast<nn::nn_workload_data_t<float>*>(reference_input->output);
    delete reference_input;

    return passed;
}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_convolution_artificial, cpu_convolution_stride1)
{
    EXPECT_EQ(true, ult_perform_test(
        1,                                  // batch size
        1,                                  // output feature maps
        1,                                  // input feature maps
        3,                                  // input width
        3,                                  // input height
        2,                                  // kernel width
        2,                                  // kernel height
        1,                                  // kernel stride x
        1,                                  // kernel stride y
        false,                              // check views
        NN_ACTIVATION_FUNCTION_NONE ));     // activation function

    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 3, 3, 3, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 3, 3, 2, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 3, 3, 1, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 3, 3, 1, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 3, 3, 1, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 3, 3, 2, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 3, 3, 3, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 3, 3, 3, 3, 1, 1, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 3, 3, 2, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 3, 3, 3, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 3, 3, 2, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 3, 3, 1, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 3, 3, 1, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 3, 3, 1, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 3, 3, 2, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 3, 3, 3, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 3, 3, 3, 3, 1, 1, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 3, 3, 2, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 3, 3, 3, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 3, 3, 2, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 3, 3, 1, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 3, 3, 1, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 3, 3, 1, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 3, 3, 2, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 3, 3, 3, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 3, 3, 3, 3, 1, 1, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 3, 3, 2, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 3, 3, 3, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 3, 3, 2, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 3, 3, 1, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 3, 3, 1, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 3, 3, 1, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 3, 3, 2, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 3, 3, 3, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 3, 3, 3, 3, 1, 1, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 3, 3, 2, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 3, 3, 3, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 3, 3, 2, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 3, 3, 1, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 3, 3, 1, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 3, 3, 1, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 3, 3, 2, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 3, 3, 3, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 3, 3, 3, 3, 1, 1, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 3, 3, 2, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 3, 3, 3, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 3, 3, 2, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 3, 3, 1, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 3, 3, 1, 2, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 3, 3, 1, 3, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 3, 3, 2, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 3, 3, 3, 1, 1, 1, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 3, 3, 3, 3, 1, 1, false, activation));
        }
    }
}

TEST(cpu_convolution_artificial, cpu_convolution_stride2)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 8, 8, 2, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 8, 8, 3, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 8, 8, 2, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 8, 8, 1, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 8, 8, 1, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 8, 8, 1, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 8, 8, 2, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 8, 8, 3, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 8, 8, 3, 3, 2, 2, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 8, 8, 2, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 8, 8, 3, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 8, 8, 2, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 8, 8, 1, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 8, 8, 1, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 8, 8, 1, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 8, 8, 2, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 8, 8, 3, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 8, 8, 3, 3, 2, 2, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 8, 8, 2, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 8, 8, 3, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 8, 8, 2, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 8, 8, 1, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 8, 8, 1, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 8, 8, 1, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 8, 8, 2, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 8, 8, 3, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 8, 8, 3, 3, 2, 2, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 8, 8, 2, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 8, 8, 3, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 8, 8, 2, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 8, 8, 1, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 8, 8, 1, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 8, 8, 1, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 8, 8, 2, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 8, 8, 3, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 8, 8, 8, 3, 3, 2, 2, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 8, 8, 2, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 8, 8, 3, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 8, 8, 2, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 8, 8, 1, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 8, 8, 1, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 8, 8, 1, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 8, 8, 2, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 8, 8, 3, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 8, 8, 3, 3, 2, 2, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 8, 8, 2, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 8, 8, 3, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 8, 8, 2, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 8, 8, 1, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 8, 8, 1, 2, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 8, 8, 1, 3, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 8, 8, 2, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 8, 8, 3, 1, 2, 2, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 8, 8, 3, 3, 2, 2, false, activation));
        }
    }
}

TEST(cpu_convolution_artificial, cpu_convolution_stride3)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 16, 16, 2, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 16, 16, 3, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 16, 16, 2, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 16, 16, 1, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 16, 16, 1, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 16, 16, 1, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 16, 16, 2, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 16, 16, 3, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 16, 16, 3, 3, 3, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 16, 16, 2, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 16, 16, 3, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 16, 16, 2, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 16, 16, 1, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 16, 16, 1, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 16, 16, 1, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 16, 16, 2, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 16, 16, 3, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 16, 16, 3, 3, 3, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 16, 16, 2, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 16, 16, 3, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 16, 16, 2, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 16, 16, 1, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 16, 16, 1, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 16, 16, 1, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 16, 16, 2, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 16, 16, 3, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 16, 16, 3, 3, 3, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 4, 16, 16, 8, 2, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 16, 16, 8, 3, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 16, 16, 8, 2, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 16, 16, 8, 1, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 16, 16, 8, 1, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 16, 16, 8, 1, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 16, 16, 8, 2, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 16, 16, 8, 3, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 16, 16, 8, 3, 3, 3, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 16, 16, 2, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 16, 16, 3, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 16, 16, 2, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 16, 16, 1, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 16, 16, 1, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 16, 16, 1, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 16, 16, 2, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 16, 16, 3, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 16, 16, 3, 3, 3, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 16, 16, 2, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 16, 16, 3, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 16, 16, 2, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 16, 16, 1, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 16, 16, 1, 2, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 16, 16, 1, 3, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 16, 16, 2, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 16, 16, 3, 1, 3, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 16, 16, 3, 3, 3, 3, false, activation));
        }
    }
}

TEST(cpu_convolution_artificial, cpu_convolution_stride4)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 2, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 3, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 2, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 1, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 1, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 1, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 2, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 3, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 3, 3, 4, 4, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 2, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 3, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 2, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 1, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 1, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 1, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 2, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 3, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 3, 3, 4, 4, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 2, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 3, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 2, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 1, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 1, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 1, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 2, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 3, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 3, 3, 4, 4, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 2, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 3, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 2, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 1, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 1, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 1, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 2, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 3, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 3, 3, 4, 4, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 2, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 3, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 2, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 1, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 1, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 1, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 2, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 3, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 3, 3, 4, 4, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 2, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 3, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 2, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 1, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 1, 2, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 1, 3, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 2, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 3, 1, 4, 4, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 3, 3, 4, 4, false, activation));
        }
    }
}

TEST(cpu_convolution_artificial, cpu_convolution_stride2x3)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 2, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 3, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 2, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 1, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 1, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 1, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 2, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 3, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 3, 3, 2, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 2, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 3, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 2, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 1, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 1, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 1, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 2, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 3, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 3, 3, 2, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 2, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 3, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 2, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 1, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 1, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 1, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 2, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 3, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 3, 3, 2, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 2, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 3, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 2, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 1, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 1, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 1, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 2, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 3, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 3, 3, 2, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 2, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 3, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 2, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 1, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 1, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 1, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 2, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 3, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 3, 3, 2, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 2, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 3, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 2, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 1, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 1, 2, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 1, 3, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 2, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 3, 1, 2, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 3, 3, 2, 3, false, activation));
        }
    }
}

TEST(cpu_convolution_artificial, cpu_convolution_stride4x3)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 2, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 3, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 2, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 1, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 1, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 1, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 2, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 3, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 1, 32, 32, 3, 3, 4, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 2, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 3, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 2, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 1, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 1, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 1, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 2, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 3, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 2, 1, 32, 32, 3, 3, 4, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 2, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 3, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 2, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 1, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 1, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 1, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 2, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 3, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 1, 2, 32, 32, 3, 3, 4, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 2, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 3, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 2, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 1, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 1, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 1, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 2, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 3, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 4, 32, 32, 8, 3, 3, 4, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 2, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 3, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 2, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 1, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 1, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 1, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 2, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 3, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 8, 4, 32, 32, 3, 3, 4, 3, false, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 2, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 3, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 2, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 1, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 1, 2, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 1, 3, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 2, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 3, 1, 4, 3, false, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 10, 7, 32, 32, 3, 3, 4, 3, false, activation));
        }
    }
}

TEST(cpu_convolution_real, cpu_convolution_LeNet5)
{
    // C1
    EXPECT_EQ(true, ult_perform_test(1, 6, 1, 32, 32, 5, 5, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C3
    EXPECT_EQ(true, ult_perform_test(1, 16, 6, 14, 14, 5, 5, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C5
    EXPECT_EQ(true, ult_perform_test(1, 120, 16, 5, 5, 5, 5, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C1
    EXPECT_EQ(true, ult_perform_test(8, 6, 1, 32, 32, 5, 5, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C3
    EXPECT_EQ(true, ult_perform_test(8, 16, 6, 14, 14, 5, 5, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C5
    EXPECT_EQ(true, ult_perform_test(8, 120, 16, 5, 5, 5, 5, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));
}

TEST(cpu_convolution_real, cpu_convolution_Krizhevsky)
{
    // C1
    EXPECT_EQ(true, ult_perform_test(1, 96, 3, 224, 224, 11, 11, 4, 4, false, NN_ACTIVATION_FUNCTION_NONE));

    // C2
    EXPECT_EQ(true, ult_perform_test(1, 256, 96, 55, 55, 5, 5, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C3
    EXPECT_EQ(true, ult_perform_test(1, 384, 256, 27, 27, 3, 3, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C4
    EXPECT_EQ(true, ult_perform_test(1, 384, 384, 13, 13, 3, 3, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C5
    EXPECT_EQ(true, ult_perform_test(1, 256, 384, 13, 13, 3, 3, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C6
    EXPECT_EQ(true, ult_perform_test(1, 4096, 256, 6, 6, 6, 6, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));
}

TEST(cpu_convolution_real, cpu_convolution_)
{
    // C1
    EXPECT_EQ(true, ult_perform_test(1, 96, 3, 231, 231, 11, 11, 4, 4, false, NN_ACTIVATION_FUNCTION_NONE));

    // C2
    EXPECT_EQ(true, ult_perform_test(1, 256, 96, 28, 28, 5, 5, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C3
    EXPECT_EQ(true, ult_perform_test(1, 512, 256, 14, 14, 3, 3, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C4
    EXPECT_EQ(true, ult_perform_test(1, 1024, 512, 14, 14, 3, 3, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C5
    EXPECT_EQ(true, ult_perform_test(1, 1024, 1024, 14, 14, 3, 3, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));

    // C6
    EXPECT_EQ(true, ult_perform_test(1, 3072, 1024, 6, 6, 6, 6, 1, 1, false, NN_ACTIVATION_FUNCTION_NONE));
}

TEST(cpu_convolution_artificial_view, cpu_convolution_stride1)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 3, 3, 2, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 5, 5, 2, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 9, 9, 2, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 17, 17, 2, 2, 1, 1, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 4, 4, 3, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 6, 6, 3, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 10, 10, 3, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 18, 18, 3, 3, 1, 1, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 3, 4, 2, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 5, 6, 2, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 9, 10, 2, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 17, 18, 2, 3, 1, 1, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 4, 3, 3, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 6, 5, 3, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 10, 9, 3, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 18, 17, 3, 2, 1, 1, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 3, 3, 2, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 5, 5, 2, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 9, 9, 2, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 17, 17, 2, 2, 1, 1, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 4, 4, 3, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 6, 6, 3, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 10, 10, 3, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 18, 18, 3, 3, 1, 1, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 3, 4, 2, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 5, 6, 2, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 9, 10, 2, 3, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 17, 18, 2, 3, 1, 1, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 4, 3, 3, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 6, 5, 3, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 10, 9, 3, 2, 1, 1, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 18, 17, 3, 2, 1, 1, true, activation));
        }
    }
}

TEST(cpu_convolution_artificial_view, cpu_convolution_stride2)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 4, 4, 2, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 8, 8, 2, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 16, 16, 2, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 32, 32, 2, 2, 2, 2, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 5, 5, 3, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 9, 9, 3, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 17, 17, 3, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 33, 33, 3, 3, 2, 2, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 4, 5, 2, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 8, 9, 2, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 16, 17, 2, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 32, 33, 2, 3, 2, 2, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 5, 4, 3, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 9, 8, 3, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 17, 16, 3, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 33, 32, 3, 2, 2, 2, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 4, 4, 2, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 8, 8, 2, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 16, 16, 2, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 32, 32, 2, 2, 2, 2, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 5, 5, 3, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 9, 9, 3, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 17, 17, 3, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 33, 33, 3, 3, 2, 2, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 4, 5, 2, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 8, 9, 2, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 16, 17, 2, 3, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 32, 33, 2, 3, 2, 2, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 5, 4, 3, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 9, 8, 3, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 17, 16, 3, 2, 2, 2, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 33, 32, 3, 2, 2, 2, true, activation));
        }
    }
}

TEST(cpu_convolution_artificial_view, cpu_convolution_stride3)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 5, 5, 2, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 11, 11, 2, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 23, 23, 2, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 47, 47, 2, 2, 3, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 6, 6, 3, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 12, 12, 3, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 24, 24, 3, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 48, 48, 3, 3, 3, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 5, 6, 2, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 11, 12, 2, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 23, 24, 2, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 47, 48, 2, 3, 3, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 6, 5, 3, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 12, 11, 3, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 24, 23, 3, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 48, 47, 3, 2, 3, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 5, 5, 2, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 11, 11, 2, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 23, 23, 2, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 47, 47, 2, 2, 3, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 6, 6, 3, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 12, 12, 3, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 24, 24, 3, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 48, 48, 3, 3, 3, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 5, 6, 2, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 11, 12, 2, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 23, 24, 2, 3, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 47, 48, 2, 3, 3, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 6, 5, 3, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 12, 11, 3, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 24, 23, 3, 2, 3, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 48, 47, 3, 2, 3, 3, true, activation));
        }
    }
}

TEST(cpu_convolution_artificial_view, cpu_convolution_stride4)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 6, 6, 2, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 14, 14, 2, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 30, 30, 2, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 62, 62, 2, 2, 4, 4, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 7, 7, 3, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 15, 15, 3, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 31, 31, 3, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 63, 63, 3, 3, 4, 4, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 6, 7, 2, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 14, 15, 2, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 30, 31, 2, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 62, 63, 2, 3, 4, 4, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 7, 6, 3, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 15, 14, 3, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 31, 30, 3, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 63, 62, 3, 2, 4, 4, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 6, 6, 2, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 14, 14, 2, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 30, 30, 2, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 62, 62, 2, 2, 4, 4, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 7, 7, 3, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 15, 15, 3, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 31, 31, 3, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 63, 63, 3, 3, 4, 4, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 6, 7, 2, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 14, 15, 2, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 30, 31, 2, 3, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 62, 63, 2, 3, 4, 4, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 7, 6, 3, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 15, 14, 3, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 31, 30, 3, 2, 4, 4, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 63, 62, 3, 2, 4, 4, true, activation));
        }
    }
}

TEST(cpu_convolution_artificial_view, cpu_convolution_stride2x3)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 4, 5, 2, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 8, 11, 2, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 16, 23, 2, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 32, 47, 2, 2, 2, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 5, 6, 3, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 9, 12, 3, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 17, 24, 3, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 33, 48, 3, 3, 2, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 4, 6, 2, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 8, 12, 2, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 16, 24, 2, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 32, 48, 2, 3, 2, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 5, 5, 3, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 9, 11, 3, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 17, 23, 3, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 33, 47, 3, 2, 2, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 4, 5, 2, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 8, 11, 2, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 16, 23, 2, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 32, 47, 2, 2, 2, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 5, 6, 3, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 9, 12, 3, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 17, 24, 3, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 33, 48, 3, 3, 2, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 4, 6, 2, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 8, 12, 2, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 16, 24, 2, 3, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 32, 48, 2, 3, 2, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 5, 5, 3, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 9, 11, 3, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 17, 23, 3, 2, 2, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 33, 47, 3, 2, 2, 3, true, activation));
        }
    }
}

TEST(cpu_convolution_artificial_view, cpu_convolution_stride4x3)
{
    uint32_t batches[] = { 1, 8 };
    NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
    for (auto batch : batches)
    {
        for (auto activation : activations)
        {
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 6, 5, 2, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 14, 11, 2, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 30, 23, 2, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 62, 47, 2, 2, 4, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 7, 6, 3, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 15, 12, 3, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 31, 24, 3, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 63, 48, 3, 3, 4, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 6, 6, 2, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 14, 12, 2, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 30, 24, 2, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 62, 48, 2, 3, 4, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 7, 5, 3, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 15, 11, 3, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 31, 23, 3, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 1, 63, 47, 3, 2, 4, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 6, 5, 2, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 14, 11, 2, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 30, 23, 2, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 62, 47, 2, 2, 4, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 7, 6, 3, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 15, 12, 3, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 31, 24, 3, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 63, 48, 3, 3, 4, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 6, 6, 2, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 14, 12, 2, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 30, 24, 2, 3, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 62, 48, 2, 3, 4, 3, true, activation));

            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 7, 5, 3, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 15, 11, 3, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 31, 23, 3, 2, 4, 3, true, activation));
            EXPECT_EQ(true, ult_perform_test(batch, 32, 3, 63, 47, 3, 2, 4, 3, true, activation));
        }
    }
}

TEST(cpu_convolution_real_view, cpu_convolution_Krizhevsky)
{
    // C1
    EXPECT_EQ(true, ult_perform_test(1, 96, 3, 224, 224, 11, 11, 4, 4, true, NN_ACTIVATION_FUNCTION_NONE));
}

TEST(cpu_convolution_real_view, cpu_convolution_)
{
    // C1
    EXPECT_EQ(true, ult_perform_test(1, 96, 3, 231, 231, 11, 11, 4, 4, true, NN_ACTIVATION_FUNCTION_NONE));

    // C2
    EXPECT_EQ(true, ult_perform_test(1, 256, 96, 28, 28, 5, 5, 1, 1, true, NN_ACTIVATION_FUNCTION_NONE));

    // C3
    EXPECT_EQ(true, ult_perform_test(1, 512, 256, 14, 14, 3, 3, 1, 1, true, NN_ACTIVATION_FUNCTION_NONE));

    // C4
    EXPECT_EQ(true, ult_perform_test(1, 1024, 512, 14, 14, 3, 3, 1, 1, true, NN_ACTIVATION_FUNCTION_NONE));

    // C5
    EXPECT_EQ(true, ult_perform_test(1, 1024, 1024, 14, 14, 3, 3, 1, 1, true, NN_ACTIVATION_FUNCTION_NONE));
}

TEST(cpu_convolution_padding, cpu_convolution_padding_stride1)
{
    for (uint32_t num_ofm = 16; num_ofm <= 32; num_ofm += 16)
        for (uint32_t num_ifm = 1; num_ifm < 4; ++num_ifm)
            for (uint32_t fm_size = 1; fm_size < 8; ++fm_size)
                for (uint32_t kernel_width = 1; kernel_width < 3; ++kernel_width)
                    for (uint32_t kernel_height = 1; kernel_height < 3; ++kernel_height)
                    {
                        uint32_t batches[] = { 1, 8 };
                        NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
                        for (auto batch : batches)
                            for (auto activation : activations)
                                EXPECT_EQ(true, ult_perform_padding_test(batch, num_ofm, num_ifm, fm_size, fm_size, kernel_width, kernel_height, 1, 1, activation));
                    }
}

TEST(cpu_convolution_padding, cpu_convolution_padding_stride2)
{
    for (uint32_t num_ofm = 16; num_ofm <= 32; num_ofm += 16)
        for (uint32_t num_ifm = 1; num_ifm < 4; ++num_ifm)
            for (uint32_t fm_size = 1; fm_size < 8; ++fm_size)
                for (uint32_t kernel_width = 1; kernel_width < 3; ++kernel_width)
                    for (uint32_t kernel_height = 1; kernel_height < 3; ++kernel_height)
                    {
                        uint32_t batches[] = { 1, 8 };
                        NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
                        for (auto batch : batches)
                            for (auto activation : activations)
                                EXPECT_EQ(true, ult_perform_padding_test(batch, num_ofm, num_ifm, fm_size, fm_size, kernel_width, kernel_height, 2, 2, activation));
                    }

}

TEST(cpu_convolution_padding, cpu_convolution_padding_stride3)
{
    for (uint32_t num_ofm = 16; num_ofm <= 32; num_ofm += 16)
        for (uint32_t num_ifm = 1; num_ifm < 4; ++num_ifm)
            for (uint32_t fm_size = 1; fm_size < 8; ++fm_size)
                for (uint32_t kernel_width = 1; kernel_width < 3; ++kernel_width)
                    for (uint32_t kernel_height = 1; kernel_height < 3; ++kernel_height)
                    {
                        uint32_t batches[] = { 1, 8 };
                        NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
                        for (auto batch : batches)
                            for (auto activation : activations)
                                EXPECT_EQ(true, ult_perform_padding_test(batch, num_ofm, num_ifm, fm_size, fm_size, kernel_width, kernel_height, 3, 3, activation));
                    }

}

TEST(cpu_convolution_padding, cpu_convolution_padding_stride4x3)
{
    for (uint32_t num_ofm = 16; num_ofm <= 32; num_ofm += 16)
        for (uint32_t num_ifm = 1; num_ifm < 4; ++num_ifm)
            for (uint32_t fm_size = 1; fm_size < 8; ++fm_size)
                for (uint32_t kernel_width = 1; kernel_width < 3; ++kernel_width)
                    for (uint32_t kernel_height = 1; kernel_height < 3; ++kernel_height)
                    {
                        uint32_t batches[] = { 1, 8 };
                        NN_ACTIVATION_FUNCTION activations[] = { NN_ACTIVATION_FUNCTION_NONE, NN_ACTIVATION_FUNCTION_RELU };
                        for (auto batch : batches)
                            for (auto activation : activations)
                                EXPECT_EQ(true, ult_perform_padding_test(batch, num_ofm, num_ifm, fm_size, fm_size, kernel_width, kernel_height, 4, 3, activation));
                    }

}
