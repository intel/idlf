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

#include "device/api/nn_device_interface_0.h"
#include "device/common/nn_workload_data.h"
#include "device/cpu/core/layer_convolution_avx2.h"
#include "device/cpu/core/layer_convolution_pooling_avx2.h"
#include "device/cpu/core/layer_fully_connected_avx2.h"
#include "device/cpu/core/layer_softmax_avx2.h"
#include "device/cpu/core/layer_pooling_avx2.h"
#include "device/cpu/core/layer_normalization_avx2.h"
#include "device/cpu/core/layer_convert_data_layout.h"
#include "device/cpu/core/layers_fixedpoint.h"
#include "device/cpu/core/layer_arithmetic_operation.h"
#include "device/cpu/core/layer_parameter_update.h"
#include "device/cpu/core/layer_average_delta.h"
#include "device/cpu/core/layer_loss_function.h"
#include "device/cpu/core/layer_relu_avx2.h"
#include "device/cpu/core/layer_dropout.h"
#include "nn_device_interface_0_internal.h"

#include <map>
#include <cassert>
#include <stack>
#include <set>
#include <cstring>
#include <memory>
#include <algorithm>
#include <numeric>
#include <immintrin.h>

#define ENABLE_WORKLOAD_MONITORING 0


#if ENABLE_WORKLOAD_MONITORING
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

void nn_workload_item_data_marshaling(nn_workload_item* item, int stage_num) {

    std::ostringstream temp;
    std::string item_name;

    switch(item->type){
    case NN_WORK_ITEM_TYPE_NORMALIZATION: item_name = "norm";break;
    case NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT: item_name = item->name;break;
    case NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT: item_name = "conv_float2int";break;
    case NN_WORK_ITEM_TYPE_CONVOLUTION: item_name =  "cnn_f32";break;
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: item_name = "cnn_pool2x2_f32";break;
    case NN_WORK_ITEM_TYPE_POOLING: item_name = "pooling_f32";break;
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED: item_name = "fc_f32";break;
    case NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT: item_name = "cnn_i16";break;
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT: item_name = "cnn_pool2x2_i16";break;
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN:
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN: item_name =  "fc_i16";break;
    case NN_WORK_ITEM_TYPE_SOFTMAX:
    case NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT: item_name = "softmax";break;
    case NN_WORK_ITEM_TYPE_MERGE: item_name =  "merge";break;
    case NN_WORK_ITEM_TYPE_ARITHMETIC: item_name = "arithmetic";break;
    case NN_WORK_ITEM_TYPE_RELU: item_name = "relu"; break;
    case NN_WORK_ITEM_TYPE_RELU_1D: item_name = "relu_1d"; break;
    case NN_WORK_ITEM_TYPE_RELU_3D: item_name = "relu_3d"; break;
    case NN_WORK_ITEM_TYPE_VIEW: item_name = "view";break;
    case NN_WORK_ITEM_TYPE_INPUT: item_name = "input";break;
    case NN_WORK_ITEM_TYPE_OUTPUT: item_name = "output";break;
    case NN_WORK_ITEM_TYPE_AVERAGE_DELTAS: item_name = "avg_deltas"; break;
    case NN_WORK_ITEM_TYPE_DROPOUT: item_name = "dropout"; break;
    case NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP: item_name = "dropout_back"; break;
    case NN_WORK_ITEM_TYPE_LOSS_FUNCTION: item_name = "loss_func"; break;
    case NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS: item_name = "update_args"; break;
    case NN_WORK_ITEM_TYPE_RELU_BACKPROP: item_name = "relu_back"; break;
    case NN_WORK_ITEM_TYPE_RELU_1D_BACKPROP: item_name = "relu_1d_back"; break;
    case NN_WORK_ITEM_TYPE_RELU_3D_BACKPROP: item_name = "relu_3d_back"; break;
    case NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP: item_name = "cnn_back"; break;
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP: item_name = "fc_back"; break;
    case NN_WORK_ITEM_TYPE_POOLING_BACKPROP: item_name = "pool_back"; break;
    case NN_WORK_ITEM_TYPE_NORMALIZATION_BACKPROP: item_name = "norm_back"; break;
    case NN_WORK_ITEM_TYPE_SOFTMAX_BACKPROP: item_name = "softmax_back"; break;
    default: item_name = "no_name";
    }

    temp << "[IDLF]_node_"
        << std::setfill('0')
        << std::setw(2)
        << std::to_string(stage_num)
        << "_[" << item_name <<"].csv";

    std::fstream file;

    file.open(temp.str(), std::ios::out | std::ios::trunc);
    if(file.is_open()) {
        const auto output = static_cast<nn::workload_data<float>*>(item->output[0]);
        if(output) {
            uint32_t
                x, y, z,
                cx = output->get_length(NN_DATA_COORD_x),
                cy = output->get_length(NN_DATA_COORD_y),
                cz = output->get_length(NN_DATA_COORD_z);
            if(stage_num<3) {
                for(uint32_t z=0;z<cz;++z) {
                    file << "z=" << z <<std::endl;
                    for(uint32_t y=0;y<cy;++y){
                        for(uint32_t x=0;x<cx;++x)
                            file << std::setprecision(7) << (*output)(0, x, y, z, 0, 0)() << ";";
                        file << std::endl;
                    }
                }
            }
            else {
                file << "z;x;y;val" <<std::endl;
                for(uint32_t y=0;y<cy;++y)
                    for(uint32_t x=0;x<cx;++x)
                        for(uint32_t z=0;z<cz;++z) {
                            file << z << ";" << x << ";" << y << ";"
                                << std::setprecision(7) << (*output)(0, x, y, z, 0, 0)()
                                << std::endl;
                        }
            }
        }
        file.close();
    }
}
#endif // ENABLE_WORKLOAD_MONITORING


// Function for copying arguments between workflow and workload items [batch, &flow_to_work]
namespace {

template<NN_WORK_ITEM_TYPE T_work_item_type>
struct flow_item_helper;

template <> struct flow_item_helper<NN_WORK_ITEM_TYPE_CONVOLUTION> {
    static const nn_arguments_forward_convolution_t &get_arguments(const nn_workflow_item *flow_item) {
        return flow_item->arguments.forward_convolution;
    }

    static void calculate_padding(const nn_workflow_item *flow_item, size_t input_w, size_t input_h, size_t &left_padding, size_t &right_padding, size_t &top_padding, size_t &bottom_padding){
        auto& arguments = get_arguments(flow_item);

        size_t padding_w = (flow_item->output_format[0].format_3d.size[0] - 1) * arguments.stride[0] - input_w + arguments.weights->size[0];
        size_t padding_h = (flow_item->output_format[0].format_3d.size[1] - 1) * arguments.stride[1] - input_h + arguments.weights->size[1];

        left_padding = arguments.center_offset[0];
        right_padding = padding_w - arguments.center_offset[0];

        top_padding = arguments.center_offset[1];
        bottom_padding = padding_h - arguments.center_offset[1];
    }
};

template <> struct flow_item_helper<NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2> {
    static const nn_arguments_forward_convolution_pooling_max_2x2_stride_2x2 &get_arguments(const nn_workflow_item *flow_item) {
        return flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2;
    }

    static void calculate_padding(const nn_workflow_item *flow_item, size_t input_w, size_t input_h, size_t &left_padding, size_t &right_padding, size_t &top_padding, size_t &bottom_padding){
        auto& arguments = get_arguments(flow_item);
        const size_t pooling_stride = 2, pooling_size = 2;

        size_t padding_w = ((flow_item->output_format[0].format_3d.size[0] - 1) * pooling_stride + pooling_size - 1) *
                               arguments.stride[0] -
                           input_w + arguments.weights->size[0];
        size_t padding_h = ((flow_item->output_format[0].format_3d.size[1] - 1) * pooling_stride + pooling_size - 1) *
                               arguments.stride[1] -
                           input_h + arguments.weights->size[1];

        left_padding = arguments.center_offset[0];
        right_padding = padding_w - arguments.center_offset[0];

        top_padding = arguments.center_offset[1];
        bottom_padding = padding_h - arguments.center_offset[1];
    }
};

template <> struct flow_item_helper<NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT> {
    static const nn_arguments_forward_convolution_fixedpoint_t &get_arguments(const nn_workflow_item *flow_item) {
        return flow_item->arguments.forward_convolution_int16_fixedpoint;
    }

    static void calculate_padding(const nn_workflow_item *flow_item, size_t input_w, size_t input_h, size_t &left_padding, size_t &right_padding, size_t &top_padding, size_t &bottom_padding){
        auto& arguments = get_arguments(flow_item);

        size_t padding_w = (flow_item->output_format[0].format_3d.size[0] - 1) * arguments.stride[0] - input_w + arguments.weights->size[0];
        size_t padding_h = (flow_item->output_format[0].format_3d.size[1] - 1) * arguments.stride[1] - input_h + arguments.weights->size[1];

        left_padding = arguments.center_offset[0];
        right_padding = padding_w - arguments.center_offset[0];

        top_padding = arguments.center_offset[1];
        bottom_padding = padding_h - arguments.center_offset[1];
    }
};

template <> struct flow_item_helper<NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT> {
    static const nn_arguments_forward_merged_convolution_pooling_max_2x2_stride_2x2_fixedpoint_t &get_arguments(const nn_workflow_item *flow_item) {
        return flow_item->arguments.forward_convolution_pooling_fixedpoint;
    }

    static void calculate_padding(const nn_workflow_item *flow_item, size_t input_w, size_t input_h, size_t &left_padding, size_t &right_padding, size_t &top_padding, size_t &bottom_padding){
        auto& arguments = get_arguments(flow_item);
        const size_t pooling_stride = 2, pooling_size = 2;

        size_t padding_w = ((flow_item->output_format[0].format_3d.size[0] - 1) * pooling_stride + pooling_size - 1) *
                               arguments.stride[0] -
                           input_w + arguments.weights->size[0];
        size_t padding_h = ((flow_item->output_format[0].format_3d.size[1] - 1) * pooling_stride + pooling_size - 1) *
                               arguments.stride[1] -
                           input_h + arguments.weights->size[1];

        left_padding = arguments.center_offset[0];
        right_padding = padding_w - arguments.center_offset[0];

        top_padding = arguments.center_offset[1];
        bottom_padding = padding_h - arguments.center_offset[1];
    }
};

template <NN_WORK_ITEM_TYPE T_use_item_type>
static void nn_workflow_compile_0_function_update_output_padding_for_use(const nn_workflow_item *use_flow_item,
                                                                         size_t output_w,
                                                                         size_t output_h,
                                                                         size_t &left_padding,
                                                                         size_t &right_padding,
                                                                         size_t &top_padding,
                                                                         size_t &bottom_padding) {
    using helper_zxyn_f32 = flow_item_helper<T_use_item_type>;

    if (use_flow_item->type == T_use_item_type) {
        auto &arguments = helper_zxyn_f32::get_arguments(use_flow_item);
        if (arguments.padding == NN_PADDING_MODE_ZERO || arguments.padding == NN_PADDING_MODE_DATA_OR_ZERO) {
            size_t temp_left_padding, temp_right_padding, temp_top_padding, temp_bottom_padding;
            helper_zxyn_f32::calculate_padding(use_flow_item, output_w, output_h, temp_left_padding, temp_right_padding, temp_top_padding, temp_bottom_padding);

            if(temp_left_padding > left_padding) left_padding = temp_left_padding;
            if(temp_right_padding > right_padding) right_padding = temp_right_padding;
            if(temp_top_padding> top_padding) top_padding = temp_top_padding;
            if(temp_bottom_padding > bottom_padding) bottom_padding = temp_bottom_padding;
        }
    }
}

static void nn_workflow_compile_0_function_calculate_output_padding(const nn_workflow_item *flow_item,
                                                                    size_t &left_padding,
                                                                    size_t &right_padding,
                                                                    size_t &top_padding,
                                                                    size_t &bottom_padding) {
    const size_t output_w = flow_item->output_format[0].format_3d.size[0];
    const size_t output_h = flow_item->output_format[0].format_3d.size[1];

    for(size_t it_use = 0; it_use < flow_item->use_count; ++it_use){
        auto& use_item = flow_item->use[it_use].item;
        if (use_item->type == NN_WORK_ITEM_TYPE_VIEW)
        {
            nn_workflow_compile_0_function_calculate_output_padding(use_item, left_padding, right_padding, top_padding, bottom_padding);
        }
        else
        {
            nn_workflow_compile_0_function_update_output_padding_for_use<NN_WORK_ITEM_TYPE_CONVOLUTION>(use_item, output_w, output_h, left_padding, right_padding, top_padding, bottom_padding);
            nn_workflow_compile_0_function_update_output_padding_for_use<NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2>(use_item, output_w, output_h, left_padding, right_padding, top_padding, bottom_padding);
            nn_workflow_compile_0_function_update_output_padding_for_use<NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT>(use_item, output_w, output_h, left_padding, right_padding, top_padding, bottom_padding);
            nn_workflow_compile_0_function_update_output_padding_for_use<NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT>(use_item, output_w, output_h, left_padding, right_padding, top_padding, bottom_padding);
        }
    }
}

void nn_workflow_compile_0_function_create_primitive(nn_workload_item_t *load_item,
                                                     nn_workflow_item_t *flow_item,
                                                     uint32_t batch,
                                                     nn_device_internal *device,
                                                     size_t output_left_padding,
                                                     size_t output_right_padding,
                                                     size_t output_top_padding,
                                                     size_t output_bottom_padding) {
    switch (load_item->type) {
    case NN_WORK_ITEM_TYPE_CONVOLUTION: {
        auto &args = flow_item->arguments.forward_convolution;
        assert((flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2]
                                                                     : 1) == args.weights->size[3]);
        assert(args.padding == NN_PADDING_MODE_DATA_OR_ZERO);

        load_item->primitive = new layer::convolution_f32(
            args.weights->size[0],
            args.weights->size[1],
            args.weights->size[2],
            args.weights->size[3],
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            args.center_offset[0],
            args.center_offset[1],
            args.stride[0],
            args.stride[1],
            args.activation,
            batch,
            output_left_padding,
            output_right_padding,
            output_top_padding,
            output_bottom_padding,
            device);
        break;
    }
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: {
        auto &args = flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2;
        assert((flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2]
                                                                     : 1) == args.weights->size[3]);
        assert(args.padding == NN_PADDING_MODE_DATA_OR_ZERO);

        load_item->primitive = new layer::convolution_pooling_f32_2x2stride2(
            args.weights->size[0],
            args.weights->size[1],
            args.weights->size[2],
            args.weights->size[3],
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            args.center_offset[0],
            args.center_offset[1],
            args.stride[0],
            args.stride[1],
            args.activation,
            batch,
            output_left_padding,
            output_right_padding,
            output_top_padding,
            output_bottom_padding,
            device);
        break;
    }
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED: {
        auto &args = flow_item->arguments.forward_fully_connected;
        auto& input = flow_item->input[0];

        assert(input.item->output_format[input.index].format == NN_DATA_FORMAT_1D ||
               input.item->output_format[input.index].format == NN_DATA_FORMAT_3D);

        bool use_3d_input = input.item->output_format[input.index].format == NN_DATA_FORMAT_3D;

        if(use_3d_input)
            load_item->primitive = new layer::fully_connected_f32(
                args.weights->size[0], args.weights->size[1], args.weights->size[2],
                args.weights->size[3],
                args.activation,
                batch,
                device);
        else
            load_item->primitive = new layer::fully_connected_f32(
                args.weights->size[0],
                args.weights->size[1],
                args.activation,
                batch,
                device);

        break;
    }
    case NN_WORK_ITEM_TYPE_POOLING: {
        auto &args = flow_item->arguments.forward_pooling;
        auto& input = flow_item->input[0];

        assert(input.item->output_format[input.index].format >= NN_DATA_FORMAT_2D);
        assert(flow_item->output_format[0].format >= NN_DATA_FORMAT_2D);
        assert(get_format_size<2>(input.item->output_format[input.index]) ==
               get_format_size<2>(flow_item->output_format[0])); // input and output have same depth

        load_item->primitive = new layer::pooling_f32(args.mode,
                                                      args.size[0],
                                                      args.size[1],
                                                      args.stride[0],
                                                      args.stride[1],
                                                      get_format_size<2>(flow_item->output_format[0]),
                                                      get_format_size<0>(flow_item->output_format[0]),
                                                      get_format_size<1>(flow_item->output_format[0]),
                                                      batch,
                                                      output_left_padding,
                                                      output_right_padding,
                                                      output_top_padding,
                                                      output_bottom_padding,
                                                      device);
        break;
    }
    case NN_WORK_ITEM_TYPE_ARITHMETIC: {
        auto &args = flow_item->arguments.forward_arithmetic;
        assert(output_left_padding == 0 && output_right_padding == 0 && output_top_padding == 0 &&
               output_bottom_padding == 0);
        load_item->primitive = new layer::arithmetic_f32(get_format_size<0>(flow_item->output_format[0]),
                                                         get_format_size<1>(flow_item->output_format[0]),
                                                         get_format_size<2>(flow_item->output_format[0]),
                                                         args.arithmetic_function,
                                                         batch,
                                                         device);
        break;
    }
    case NN_WORK_ITEM_TYPE_RELU_1D: // TODO: different primitive implementation for 1D.
    case NN_WORK_ITEM_TYPE_RELU_3D:
    {
        load_item->primitive = new layer::relu_f32(get_format_size<0>(flow_item->output_format[0]),
                                                   get_format_size<1>(flow_item->output_format[0]),
                                                   get_format_size<2>(flow_item->output_format[0]),
                                                   batch,
                                                   output_left_padding,
                                                   output_right_padding,
                                                   output_top_padding,
                                                   output_bottom_padding,
                                                   device);
        break;
    }
    case NN_WORK_ITEM_TYPE_NORMALIZATION: {
        auto &args = flow_item->arguments.forward_normalization;
        switch (args.normalization.mode) {
        case NN_NORMALIZATION_MODE_LINEAR_SINGLE: {
            assert(output_left_padding == 0 && output_right_padding == 0 && output_top_padding == 0 &&
                   output_bottom_padding == 0);

            load_item->primitive =
                new layer::normalization_elementwise_linear_f32(args.normalization.alpha,
                                                                args.normalization.beta,
                                                                get_format_size<0>(flow_item->output_format[0]),
                                                                get_format_size<1>(flow_item->output_format[0]),
                                                                get_format_size<2>(flow_item->output_format[0]),
                                                                batch,
                                                                device);
            break;
        }
        case NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS: {
            load_item->primitive =
                new layer::normalization_response_across_maps_f32(args.normalization.alpha,
                                                                  args.normalization.beta,
                                                                  args.normalization.k,
                                                                  args.normalization.n,
                                                                  get_format_size<0>(flow_item->output_format[0]),
                                                                  get_format_size<1>(flow_item->output_format[0]),
                                                                  get_format_size<2>(flow_item->output_format[0]),
                                                                  batch,
                                                                  output_left_padding,
                                                                  output_right_padding,
                                                                  output_top_padding,
                                                                  output_bottom_padding,
                                                                  device);
            break;
        }
        default:
            // unsupported mode
            assert(0);
        }

        break;
    }
    case NN_WORK_ITEM_TYPE_SOFTMAX: {
        load_item->primitive = new layer::softmax_f32(get_format_size<0>(flow_item->output_format[0]), batch, device);
        break;
    }
    case NN_WORK_ITEM_TYPE_LOSS_FUNCTION: {
        auto &args = flow_item->arguments.loss_function;
        load_item->primitive =
            new layer::loss_function_f32(
                args.function,
                get_format_size<0>(flow_item->output_format[0]),
                get_format_size<1>(flow_item->output_format[0]),
                get_format_size<2>(flow_item->output_format[0]),
                batch,
                device);
        break;
    }
    case NN_WORK_ITEM_TYPE_DROPOUT: {
        auto &args = flow_item->arguments.dropout;
        load_item->primitive =
            new layer::dropout_f32(
                get_format_size<0>(flow_item->output_format[0]),
                get_format_size<1>(flow_item->output_format[0]),
                get_format_size<2>(flow_item->output_format[0]),
                batch,
                args.drop_rate,
                device);
        break;
    }
    case NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT: {
        auto &args = flow_item->arguments.forward_softmax_fixedpoint;
        load_item->primitive = new int16_fixedpoint::softmax_i32(get_format_size<0>(flow_item->output_format[0]), batch, args.input_fraction, device);
        break;
    }
    case NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT: {
        auto &args = flow_item->arguments.forward_pooling_fixedpoint;

        load_item->primitive = new int16_fixedpoint::pooling_i16(
            flow_item->output_format[0].format_3d.size[2],
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            args.pool_size[0],
            args.pool_size[1],
            args.pool_stride[0],
            args.pool_stride[1],
            batch,
            output_left_padding,
            output_right_padding,
            output_top_padding,
            output_bottom_padding,
            device);
        break;
    }
    case NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN: {
        auto &args = flow_item->arguments.normalization_response_across_maps_forward_i16qn;

        load_item->primitive = new int16_fixedpoint::normalization_response_across_maps_i16(
            args.k,
            args.n,
            args.alpha,
            args.beta,
            static_cast<float>(1 << args.fractions.input),
            static_cast<float>(1 << args.fractions.output),
            get_format_size<0>(flow_item->output_format[0]),
            get_format_size<1>(flow_item->output_format[0]),
            get_format_size<2>(flow_item->output_format[0]),
            batch,
            output_left_padding,
            output_right_padding,
            output_top_padding,
            output_bottom_padding,
            device);

        break;
    }
    case NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT: {
        auto &args = flow_item->arguments.forward_convolution_int16_fixedpoint;
        assert((flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2]
                                                                     : 1) == args.weights->size[3]);
        assert(args.padding == NN_PADDING_MODE_DATA_OR_ZERO);

        load_item->primitive = new int16_fixedpoint::convolution_i16(
            args.weights->size[0],
            args.weights->size[1],
            args.weights->size[2],
            args.weights->size[3],
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            args.center_offset[0],
            args.center_offset[1],
            args.stride[0],
            args.stride[1],
            args.activation,
            batch,
            output_left_padding,
            output_right_padding,
            output_top_padding,
            output_bottom_padding,
            device);
        break;
    }
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT: {
        auto &args = flow_item->arguments.forward_convolution_pooling_fixedpoint;
        assert((flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2]
                                                                     : 1) == args.weights->size[3]);
        assert(args.padding == NN_PADDING_MODE_DATA_OR_ZERO);

        load_item->primitive = new int16_fixedpoint::convolution_pooling_i16(
            args.weights->size[0],
            args.weights->size[1],
            args.weights->size[2],
            args.weights->size[3],
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            args.center_offset[0],
            args.center_offset[1],
            args.stride[0],
            args.stride[1],
            args.activation,
            batch,
            output_left_padding,
            output_right_padding,
            output_top_padding,
            output_bottom_padding,
            device);
        break;
    }
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN: {
        auto &args = flow_item->arguments.fully_connected_forward_i16qn_i16qn;
        auto& input = flow_item->input[0];

        assert(input.item->output_format[input.index].format == NN_DATA_FORMAT_1D ||
                input.item->output_format[input.index].format == NN_DATA_FORMAT_3D);

        bool use_3d_input = input.item->output_format[input.index].format == NN_DATA_FORMAT_3D;

        load_item->primitive = new int16_fixedpoint::fully_connected_i16<int16_t>(
            use_3d_input ? args.weights->size[0] * args.weights->size[1] * args.weights->size[2]
                            : args.weights->size[0],
            use_3d_input ? args.weights->size[3] : args.weights->size[1],
            args.activation,
            batch,
            device);
        break;
    }
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN: {
        auto &args = flow_item->arguments.fully_connected_forward_i16qn_i32qn;
        auto& input = flow_item->input[0];

        assert(input.item->output_format[input.index].format == NN_DATA_FORMAT_1D ||
                input.item->output_format[input.index].format == NN_DATA_FORMAT_3D);

        bool use_3d_input = input.item->output_format[input.index].format == NN_DATA_FORMAT_3D;

        load_item->primitive = new int16_fixedpoint::fully_connected_i16<int32_t>(
            use_3d_input ? args.weights->size[0] * args.weights->size[1] * args.weights->size[2]
                            : args.weights->size[0],
            use_3d_input ? args.weights->size[3] : args.weights->size[1],
            args.activation,
            batch,
            device);
        break;
    }
    case NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT: {
        auto &args = flow_item->arguments.forward_convert_float_to_int16_fixedpoint;

        load_item->primitive = new int16_fixedpoint::convert_float_to_int16(
            flow_item->output_format[0].format_3d.size[2],
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            batch,
            output_left_padding,
            output_right_padding,
            output_top_padding,
            output_bottom_padding,
            args.output_fraction,
            device);
        break;
    }
    }
}

void nn_workflow_compile_0_function_copy_item(
             nn_workload_item_t *load_item,
             nn_workflow_item_t *flow_item,
             nn_workflow_t      *workflow,
             nn_workload_t      *workload,
             uint32_t batch,
             std::map<nn_workflow_item_t *, nn_workload_item_t *> &flow_to_work,
             nn_device_internal *device
             ){

            // copy name
            load_item->name = flow_item->name;

            // copy type & create nn_workload_data_t
            load_item->type = flow_item->type;
            
            if(flow_item->type == NN_WORK_ITEM_TYPE_RELU_BACKPROP)
            {
                if(flow_item->forward_item->output_format[0].format == NN_DATA_FORMAT_1D)
                    load_item->type = NN_WORK_ITEM_TYPE_RELU_1D_BACKPROP;
                else if(flow_item->forward_item->output_format[0].format == NN_DATA_FORMAT_3D)
                    load_item->type = NN_WORK_ITEM_TYPE_RELU_3D_BACKPROP;
                else
                    assert(0);
            }
            else if(flow_item->type == NN_WORK_ITEM_TYPE_RELU)
            {
                if(flow_item->output_format[0].format == NN_DATA_FORMAT_1D)
                    load_item->type = NN_WORK_ITEM_TYPE_RELU_1D;
                else if(flow_item->output_format[0].format == NN_DATA_FORMAT_3D)
                    load_item->type = NN_WORK_ITEM_TYPE_RELU_3D;
                else
                    assert(0);
            }

            // calculate needed output buffer paddings (only forward passes)
            size_t padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0;
            if (flow_item->forward_item == nullptr)
                nn_workflow_compile_0_function_calculate_output_padding(flow_item, padding_left, padding_right, padding_top, padding_bottom);

            // setup primitive handle (only forward passes)
            if (flow_item->forward_item == nullptr)
            {
                load_item->primitive = nullptr;
                nn_workflow_compile_0_function_create_primitive(
                    load_item, flow_item, batch, device, padding_left, padding_right, padding_top, padding_bottom);
                load_item->forward_item = nullptr;
            }
            else
                load_item->forward_item = flow_to_work[flow_item->forward_item];

            // creating outputs
            for (uint32_t out = 0; out < flow_item->output_count; ++out)
            {
                load_item->output.push_back(nullptr);
            }

            // copying inputs
            load_item->input.resize(flow_item->input_count);
            for (auto index = 0u; index<flow_item->input_count; ++index)
            {
                assert(flow_to_work.find(flow_item->input[index].item) != flow_to_work.end());
                load_item->input[index].item = flow_to_work[flow_item->input[index].item];
                load_item->input[index].index = flow_item->input[index].index;
            }

            // copying uses
            load_item->use.resize(flow_item->use_count);
            for (auto index = 0u; index<flow_item->use_count; ++index)
            {
                assert(flow_to_work.find(flow_item->use[index].item) != flow_to_work.end());
                load_item->use[index].item = flow_to_work[flow_item->use[index].item];
                load_item->use[index].index = flow_item->use[index].index;
            }

            // layout for inputs & outputs
            auto get_workload_layout = [](NN_WORKLOAD_DATA_TYPE type) -> nn_workload_data_layout_t
            {
                switch (type)
                {
                case NN_WORKLOAD_DATA_TYPE_F32_1D:
                case NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH:
                case NN_WORKLOAD_DATA_TYPE_F32_2D:
                case NN_WORKLOAD_DATA_TYPE_F32_2D_BATCH:
                case NN_WORKLOAD_DATA_TYPE_F32_3D:
                case NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH:
                    return nn::workload_data<float>::layout.xyzpqn;

                case NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH:
                case NN_WORKLOAD_DATA_TYPE_F32_ZXY:
                    return nn::workload_data<float>::layout.zxynpq;

                case NN_WORKLOAD_DATA_TYPE_I16_1D:
                case NN_WORKLOAD_DATA_TYPE_I16_1D_BATCH:
                case NN_WORKLOAD_DATA_TYPE_I16_3D:
                case NN_WORKLOAD_DATA_TYPE_I16_3D_BATCH:
                    return nn::workload_data<int16_t>::layout.xyzpqn;

                case NN_WORKLOAD_DATA_TYPE_I16_ZXY:
                case NN_WORKLOAD_DATA_TYPE_I16_ZXY_BATCH:
                    return nn::workload_data<int16_t>::layout.zxynpq;

                case NN_WORKLOAD_DATA_TYPE_I32_1D:
                case NN_WORKLOAD_DATA_TYPE_I32_1D_BATCH:
                    return nn::workload_data<int32_t>::layout.xyzpqn;

                default:
                    throw std::out_of_range("unsupported data type");
                }
            };

            // calculates 6D size from nn::data, returns it as nn_workload_data_coords_t
            auto calculate_size = [](uint32_t batch, NN_WORKLOAD_DATA_TYPE type, nn_output_format *data) -> nn_workload_data_coords_t
            {
                uint32_t size_n = batch, size_x = 1, size_y = 1, size_z = 1, size_p = 1, size_q = 1;
                switch (type)
                {
                case NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH:
                case NN_WORKLOAD_DATA_TYPE_F32_ZXY:
                case NN_WORKLOAD_DATA_TYPE_I16_ZXY_BATCH:
                case NN_WORKLOAD_DATA_TYPE_I16_ZXY:
                case NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH:
                case NN_WORKLOAD_DATA_TYPE_F32_3D:
                case NN_WORKLOAD_DATA_TYPE_I16_3D_BATCH:
                case NN_WORKLOAD_DATA_TYPE_I16_3D:
                    size_z = data->format_3d.size[2];
                    // fall through
                case NN_WORKLOAD_DATA_TYPE_F32_2D_BATCH:
                case NN_WORKLOAD_DATA_TYPE_F32_2D:
                    size_y = data->format_2d.size[1];
                    // fall through
                case NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH:
                case NN_WORKLOAD_DATA_TYPE_F32_1D:
                case NN_WORKLOAD_DATA_TYPE_I16_1D_BATCH:
                case NN_WORKLOAD_DATA_TYPE_I16_1D:
                case NN_WORKLOAD_DATA_TYPE_I32_1D_BATCH:
                case NN_WORKLOAD_DATA_TYPE_I32_1D:
                    size_x = data->format_1d.size[0];
                    break;
                default:
                    assert(0);
                }
                return nn_workload_data_coords_t( size_n, size_x, size_y, size_z, size_p, size_q );
            };

            // setup output buffer
            switch(load_item->type) {
            case NN_WORK_ITEM_TYPE_INPUT: {
                auto input_index = flow_item->arguments.input.index;
                auto input_item_format = workload->input_format[input_index];
                auto input_item_size = calculate_size(workload->batch, input_item_format, flow_item->output_format);
                auto input_item_layout = get_workload_layout(input_item_format);

                load_item->output[0] =
                    new nn::workload_data<float /* NOTE: this type is disregarded in this case */>(
                        NN_WORKLOAD_DATA_TAG_UNKNOWN,
                        input_item_size,
                        input_item_layout,
                        padding_left,
                        padding_right,
                        padding_top,
                        padding_bottom,
                        true);
                break;
            }
            case NN_WORK_ITEM_TYPE_OUTPUT: {
                auto output_index = flow_item->arguments.output.index;
                auto output_item_format = workload->output_format[output_index];
                auto output_item_size = calculate_size(workload->batch, output_item_format, flow_item->output_format);
                auto output_item_layout = get_workload_layout(output_item_format);

                load_item->output[0] =
                    new nn::workload_data<float /* NOTE: this type is disregarded in this case */>(
                        NN_WORKLOAD_DATA_TAG_UNKNOWN, output_item_size, output_item_layout, true);
                break;
            }
            case NN_WORK_ITEM_TYPE_VIEW: {
                auto& origin = flow_item->arguments.view.origin;
                auto& input = flow_item->input[0];
                nn::workload_data<float> *input_data = reinterpret_cast<nn::workload_data<float> *>(flow_to_work[input.item]->output[input.index]);
                nn_workload_data_coords_t start(0, origin[0], origin[1], origin[2] / input_data->parent->lengths.t[NN_DATA_COORD_p], 0, 0);
                nn_workload_data_coords_t end(
                    batch - 1,
                    origin[0] +  flow_item->output_format[0].format_1d.size[0] - 1,
                    origin[1] + (flow_item->output_format[0].format>=NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1) - 1,
                    (origin[2] + (flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1)) / input_data->parent->lengths.t[NN_DATA_COORD_p] - 1,
                    input_data->view_end.t[NN_DATA_COORD_p] - input_data->view_begin.t[NN_DATA_COORD_p],
                    0
                );

                load_item->output[0] = new nn::workload_data<float>(*input_data, start, end);
                break;
            }
            case NN_WORK_ITEM_TYPE_MERGE: {
                uint16_t axis = flow_item->arguments.forward_merge.axis; // x = 0, y = 1, z = 2
                auto& input = flow_item->input[0];
                auto input_data = flow_to_work[input.item]->output[input.index];
                nn_workload_data_layout_t previous_layout = input_data->parent->layout;

                uint32_t x_size = input_data->parent->lengths.t[1];
                uint32_t y_size = input_data->parent->lengths.t[2];
                uint32_t z_size = input_data->parent->lengths.t[3];
                for (int index = 1; index < flow_item->input_count; index++)
                {
                    auto& input_indexed = flow_item->input[index];
                    auto input_data_local = flow_to_work[input_indexed.item]->output[input_indexed.index];
                    if (input_data->parent->layout!=previous_layout)
                        assert(0);

                    if (axis == 0)
                        x_size += input_data_local->parent->lengths.t[1];
                    else if (axis == 1)
                        y_size += input_data_local->parent->lengths.t[2];
                    else if (axis == 2)
                        z_size += input_data_local->parent->lengths.t[3];
                }

                nn_workload_data_coords_t size(
                    input_data->parent->lengths.t[0],
                    x_size + padding_left + padding_right,
                    y_size + padding_top + padding_bottom,
                    z_size,
                    input_data->parent->lengths.t[4],
                    input_data->parent->lengths.t[5]
                );

                // allocate
                load_item->output[0] = new nn_workload_data_t;
                nn_workload_data_placement_create(load_item->output[0], nullptr, &size, &previous_layout);

                nn_workload_data_coords_t start_coords(
                    0, 
                    input_data->view_begin.t[NN_DATA_COORD_x] + padding_left, 
                    input_data->view_begin.t[NN_DATA_COORD_y] + padding_top, 
                    0, 
                    0, 
                    0);

                nn_workload_data_coords_t end_coords(
                    input_data->parent->lengths.t[0] - 1,
                    input_data->view_end.t[NN_DATA_COORD_x] - padding_right,
                    input_data->view_end.t[NN_DATA_COORD_y] - padding_bottom,
                    z_size - 1,
                    input_data->parent->lengths.t[4] - 1,
                    input_data->parent->lengths.t[5] - 1
                );

                uint32_t x_position = 0, y_position = 0, z_position = 0;

                // pin to the input buffers
                for (int index = 0; index < flow_item->input_count; index++)
                {
                    auto& input_indexed = flow_item->input[index];

                    input_data = flow_to_work[input_indexed.item]->output[input_indexed.index];

                    if (axis == 0)
                    {
                        start_coords.t[1] = x_position;
                        x_position += input_data->parent->lengths.t[1];
                        end_coords.t[1] = x_position - 1;
                    }
                    else if (axis == 1)
                    {
                        start_coords.t[2] = y_position;
                        y_position += input_data->parent->lengths.t[2];
                        end_coords.t[2] = y_position - 1;
                    }
                    else if (axis == 2)
                    {
                        start_coords.t[3] = z_position;
                        z_position += input_data->parent->lengths.t[3];
                        end_coords.t[3] = z_position - 1;
                    }

                    delete flow_to_work[input_indexed.item]->output[input_indexed.index];
                    nn_workload_data_t *merge_output = load_item->output[0];
                    flow_to_work[input_indexed.item]->output[input_indexed.index] = nn_workload_data_create_view(merge_output, &start_coords, &end_coords);
                }

                break;
            }
            case NN_WORK_ITEM_TYPE_ARITHMETIC:
            case NN_WORK_ITEM_TYPE_CONVOLUTION:
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2:
            case NN_WORK_ITEM_TYPE_POOLING:
            case NN_WORK_ITEM_TYPE_NORMALIZATION:
            case NN_WORK_ITEM_TYPE_RELU_3D:
            {
                // views broken in arithmetic and element wise normalization
                if (load_item->type == NN_WORK_ITEM_TYPE_ARITHMETIC ||
                    (load_item->type == NN_WORK_ITEM_TYPE_NORMALIZATION &&
                     flow_item->arguments.forward_normalization.normalization.mode ==
                         NN_NORMALIZATION_MODE_LINEAR_SINGLE)) {
                    if (flow_item->use[0].item->type == NN_WORK_ITEM_TYPE_MERGE ||
                        flow_item->input[0].item->type == NN_WORK_ITEM_TYPE_VIEW)
                        throw NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;
                    if (padding_left != 0 || padding_right != 0 || padding_top != 0 || padding_bottom != 0)
                        throw NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;
                }
            }
            case NN_WORK_ITEM_TYPE_RELU_1D:
            case NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT:
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED:
            case NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN:
            case NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT:
            case NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT:
            case NN_WORK_ITEM_TYPE_SOFTMAX:
            case NN_WORK_ITEM_TYPE_DROPOUT:
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN:
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN: 
            case NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT:
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT:{

                load_item->output[0] = load_item->primitive->create_outputs()[0];
                break;
            }
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP:
            case NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP: {
                auto parameters = load_item->forward_item->primitive->create_parameters();
                load_item->output[1] = parameters[0];
                load_item->output[2] = parameters[1];
            }
            case NN_WORK_ITEM_TYPE_RELU_1D_BACKPROP:
            case NN_WORK_ITEM_TYPE_RELU_3D_BACKPROP:
            case NN_WORK_ITEM_TYPE_POOLING_BACKPROP:
            case NN_WORK_ITEM_TYPE_SOFTMAX_BACKPROP:
            case NN_WORK_ITEM_TYPE_NORMALIZATION_BACKPROP:
            case NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP:{
                auto inputs = load_item->forward_item->primitive->create_inputs();
                load_item->output[0] = inputs[0];
                break;
            }
            case NN_WORK_ITEM_TYPE_LOSS_FUNCTION:
            {
                load_item->output[0] =
                    new nn::workload_data<float>(NN_WORKLOAD_DATA_TAG_UNKNOWN,
                                                      load_item->input[0].get_data_view()->parent->lengths,
                                                      load_item->input[0].get_data_view()->parent->layout);
                break;
            }

            case NN_WORK_ITEM_TYPE_AVERAGE_DELTAS:
            {
                assert(flow_item->input_count == flow_item->output_count);

                for (uint32_t parameter = 0; parameter < flow_item->input_count; ++parameter)
                {
                    // TODO: add and use primitive function for this.
                    auto input = load_item->input[parameter].get_data_view();

                    nn_workload_data_coords_t size (
                        1,
                        input->parent->lengths.t[NN_DATA_COORD_x],
                        input->parent->lengths.t[NN_DATA_COORD_y],
                        input->parent->lengths.t[NN_DATA_COORD_z],
                        input->parent->lengths.t[NN_DATA_COORD_p],
                        input->parent->lengths.t[NN_DATA_COORD_q]
                    );

                    load_item->output[parameter] =
                        new nn::workload_data<float>(NN_WORKLOAD_DATA_TAG_UNKNOWN, size, input->parent->layout);
                }

                break;
            }

            case NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS:
            {
                // Do nothing..
                break;
            }
            default:
                // If this assert fires it meant that new workflow item type was added, but its support
                // was not added to compile function.
                // This switch must contain *all* workflow items on the API.
                assert(0);
            } // switch

            // copy arguments, fill buffers
            // TODO: flow_item and load_item arguments union use different structures - possible corruption.
            assert(sizeof(load_item->arguments) >= sizeof(flow_item->arguments));
            std::memcpy(&load_item->arguments, &flow_item->arguments, sizeof(load_item->arguments));

            switch(load_item->type) {
            case NN_WORK_ITEM_TYPE_CONVOLUTION: {
                auto parameters = load_item->primitive->create_parameters();

                copy_data(device, parameters[0], flow_item->arguments.forward_convolution.weights);
                copy_data(device, parameters[1], flow_item->arguments.forward_convolution.biases);

                load_item->parameters.resize(parameters.size());
                for (size_t i = 0; i < parameters.size(); ++i)
                    load_item->parameters[i] = parameters[i];

                break;
            }
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: {
                auto parameters = load_item->primitive->create_parameters();

                copy_data(device, parameters[0], flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights);
                copy_data(device, parameters[1], flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.biases);

                load_item->parameters.resize(parameters.size());
                for (size_t i = 0; i < parameters.size(); ++i)
                    load_item->parameters[i] = parameters[i];

                break;
            }
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED: {
                auto parameters = load_item->primitive->create_parameters();

                copy_data(device, parameters[0], flow_item->arguments.forward_fully_connected.weights);
                copy_data(device, parameters[1], flow_item->arguments.forward_fully_connected.biases);

                load_item->parameters.resize(parameters.size());
                for (size_t i = 0; i < parameters.size(); ++i)
                    load_item->parameters[i] = parameters[i];

                break;
            }
            case NN_WORK_ITEM_TYPE_ARITHMETIC: {
                auto parameters = load_item->primitive->create_parameters();

                copy_data(device, parameters[0], flow_item->arguments.forward_arithmetic.factor);

                load_item->parameters.resize(parameters.size());
                for (size_t i = 0; i < parameters.size(); ++i)
                    load_item->parameters[i] = parameters[i];

                break;
            }
            case NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT: {
                auto parameters = load_item->primitive->create_parameters();

                copy_data(device, parameters[0], flow_item->arguments.forward_convolution_int16_fixedpoint.weights);
                copy_data(device, parameters[1], flow_item->arguments.forward_convolution_int16_fixedpoint.biases);

                load_item->parameters.resize(parameters.size());
                for (size_t i = 0; i < parameters.size(); ++i)
                    load_item->parameters[i] = parameters[i];

                break;
            }
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT: {
                auto parameters = load_item->primitive->create_parameters();

                copy_data(device, parameters[0], flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights);
                copy_data(device, parameters[1], flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.biases);

                load_item->parameters.resize(parameters.size());
                for (size_t i = 0; i < parameters.size(); ++i)
                    load_item->parameters[i] = parameters[i];

                break;
            }
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN:
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN: {
                auto parameters = load_item->primitive->create_parameters();

                copy_data(device, parameters[0], flow_item->arguments.fully_connected_forward_i16qn_i16qn.weights);
                copy_data(device, parameters[1], flow_item->arguments.fully_connected_forward_i16qn_i16qn.biases);

                load_item->parameters.resize(parameters.size());
                for (size_t i = 0; i < parameters.size(); ++i)
                    load_item->parameters[i] = parameters[i];

                break;
            }
            default:
                // This is the case when all workflow item arguments are empty or do not contain buffers.
                ;
            }

} // end of function nn_workflow_compile_0_function_copy_item
} // end of namespace

void nn_workflow_compile_0_function_add_conversions(
    nn_workload_item_t *load_item,
    nn_workflow_item_t *flow_item,
    uint32_t            batch,
    NN_WORKLOAD_DATA_TYPE  *input_format,   /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE  *output_format  /* array containing formats of outputs */
    ){
    if (batch > 1){
        // add type2 conversions
        auto init_type2_conversion = [&batch, &flow_item, &load_item] {
            auto conversion = new nn_workload_item_t;
            conversion->primitive = nullptr;
            conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            conversion->arguments.convert_data_layout.type = 2;

            nn_workload_data_layout_t layout = nn::workload_data<int32_t>::layout.nxyzpq;

            const uint32_t OutBlock = 2;
            nn_workload_data_coords_t size(
                batch,
                1,
                1,
                flow_item->output_format[0].format_1d.size[0] / OutBlock,
                OutBlock,
                1);

            conversion->output.push_back(new nn::workload_data<std::int32_t>(NN_WORKLOAD_DATA_TAG_UNKNOWN, size, layout));
            conversion->input.push_back({ load_item, 0 });
            return conversion;
        };

        if (load_item->output.size() > 0 && load_item->output[0] != nullptr && load_item->output[0]->parent->layout.ordering.t[0] != NN_DATA_COORD_n) {
            for (auto &next_load_item : load_item->use) {
                if (next_load_item.item->type == NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT) {
                    auto type2_conversion = init_type2_conversion();
                    type2_conversion->name = std::string("convert_layout2_before_") + next_load_item.item->name;

                    for (auto &useinput : next_load_item.item->input)
                    if (useinput.item == load_item) {
                        useinput.item = type2_conversion;
                        type2_conversion->use.push_back(next_load_item);
                    }

                    next_load_item.item = type2_conversion;
                }
            }
        }

        // add type3 conversions
        auto init_type3_conversion = [&batch, &flow_item, &load_item] {
            auto conversion = new nn_workload_item_t;
            conversion->primitive = nullptr;
            conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            conversion->arguments.convert_data_layout.type = 3;

            nn_workload_data_layout_t layout = nn::workload_data<int16_t>::layout.pnzxyq;

            const uint32_t OutBlock = 2;
            nn_workload_data_coords_t size(
                batch,
                1,
                1,
                flow_item->output_format[0].format_3d.size[0] * flow_item->output_format[0].format_3d.size[1] * flow_item->output_format[0].format_3d.size[2] / OutBlock,
                OutBlock,
                1);

            conversion->output.push_back(new nn::workload_data<std::int16_t>(NN_WORKLOAD_DATA_TAG_UNKNOWN, size, layout));
            conversion->input.push_back({ load_item, 0 });
            return conversion;
        };

        if (load_item->output.size() > 0 && load_item->output[0] != nullptr && load_item->output[0]->parent->layout.ordering.t[1] != NN_DATA_COORD_n) {
            for (auto &next_load_item : load_item->use) {
                if (next_load_item.item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN ||
                    next_load_item.item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN) {
                    auto type3_conversion = init_type3_conversion();
                    type3_conversion->name = std::string("convert_layout3_before_") + next_load_item.item->name;

                    for (auto &useinput : next_load_item.item->input)
                    if (useinput.item == load_item) {
                        useinput.item = type3_conversion;
                        type3_conversion->use.push_back(next_load_item);
                    }

                    next_load_item.item = type3_conversion;
                    next_load_item.item->primitive = new layer::convert_z_block_xyz_z2nz(
                        get_format_size<0>(flow_item->output_format[0]),
                        get_format_size<1>(flow_item->output_format[0]),
                        get_format_size<2>(flow_item->output_format[0]),
                        batch,
                        nullptr);
                }
            }
        }
    }
    else { // batch == 1
        // add type0 conversions
        auto init_type0_conversion = [&batch, &flow_item, &load_item] {
            auto conversion = new nn_workload_item_t;
            conversion->primitive = nullptr;
            conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            conversion->arguments.convert_data_layout.type = 0;

            nn_workload_data_layout_t layout = nn::workload_data<int16_t>::layout.pnzxyq;

            const uint32_t OutBlock = 2;
            nn_workload_data_coords_t size(
                batch,
                1,
                1,
                flow_item->output_format[0].format_3d.size[0] * flow_item->output_format[0].format_3d.size[1] * flow_item->output_format[0].format_3d.size[2] / OutBlock,
                OutBlock,
                1);

            conversion->output.push_back(
                new nn::workload_data<std::int16_t>(NN_WORKLOAD_DATA_TAG_UNKNOWN, size, layout));
            conversion->input.push_back({ load_item, 0 });
            return conversion;
        };

        if (load_item->output.size() > 0 && load_item->output[0] != nullptr && flow_item->output_format[0].format == NN_DATA_FORMAT_3D) {
            for (auto &next_load_item : load_item->use) {
                if (next_load_item.item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN ||
                    next_load_item.item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN) {
                    auto type0_conversion = init_type0_conversion();
                    type0_conversion->name = std::string("convert_layout3_before_") + next_load_item.item->name;

                    for (auto &useinput : next_load_item.item->input)
                    if (useinput.item == load_item) {
                        useinput.item = type0_conversion;
                        type0_conversion->use.push_back(next_load_item);
                    }

                    next_load_item.item = type0_conversion;
                    next_load_item.item->primitive = new layer::convert_z_block_xyz_z2nz(
                        get_format_size<0>(flow_item->output_format[0]),
                        get_format_size<1>(flow_item->output_format[0]),
                        get_format_size<2>(flow_item->output_format[0]),
                        batch,
                        nullptr);
                }
            }
        }
    }

    // Add type4 conversions - float batch conversion (...n to n... OR n... to ...n)
    auto init_type4_conversion = [&batch, &flow_item, &load_item] (nn_workload_data_layout_t& next_load_item_layout)
    {
        auto conversion = new nn_workload_item_t;
        conversion->primitive = nullptr;
        conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
        conversion->arguments.convert_data_layout.type = 4;

        if(batch != 1)
        {
            // Actual conversion takes place.
            conversion->output.push_back(new nn::workload_data<float>(
                NN_WORKLOAD_DATA_TAG_UNKNOWN,
                load_item->output[0]->parent->lengths,
                next_load_item_layout));
        }
        else
        {
            // Just add view.
            conversion->output.push_back(new nn::workload_data<float>(
                *static_cast<nn::workload_data<float>*>(load_item->output[0]),
                load_item->output[0]->view_begin, 
                load_item->output[0]->view_end));
        }

        conversion->input.push_back({ load_item, 0 });
        return conversion;
    };

    if (load_item->output.size() > 0 && 
        load_item->output[0] != nullptr)
    {
        auto load_item_layout = (load_item->type == NN_WORK_ITEM_TYPE_RELU_1D || load_item->type == NN_WORK_ITEM_TYPE_RELU_1D_BACKPROP ) 
                                    ? nn::layout_t<float>::nxyzpq 
                                    : load_item->output[0]->parent->layout;

        for (auto &next_load_item : load_item->use)
        {
            if (next_load_item.item->output.size() > 0 &&
                next_load_item.item->type != NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT)
            {
                auto next_item_layout = (next_load_item.item->type == NN_WORK_ITEM_TYPE_RELU_1D || next_load_item.item->type == NN_WORK_ITEM_TYPE_RELU_1D_BACKPROP ) 
                                            ? nn::layout_t<float>::nxyzpq 
                                            : next_load_item.item->output[0]->parent->layout;

                // Check if its one of allowed conversions.
                if((load_item_layout == nn::layout_t<float>::zxynpq && next_item_layout == nn::layout_t<float>::nzxypq) || // ZXY-N -> N-ZXY
                   (load_item_layout == nn::layout_t<float>::zxynpq && next_item_layout == nn::layout_t<float>::nxyzpq) || // ZXY-N -> N-X
                   (load_item_layout == nn::layout_t<float>::nzxypq && next_item_layout == nn::layout_t<float>::zxynpq) || // N-ZXY -> ZXY-N
                   (load_item_layout == nn::layout_t<float>::nxyzpq && next_item_layout == nn::layout_t<float>::zxynpq) || // N-X   -> ZXY-N
                   (load_item_layout == nn::layout_t<float>::xyzpqn && next_item_layout == nn::layout_t<float>::nxyzpq) || // XYZ-N -> N-XYZ
                   (load_item_layout == nn::layout_t<float>::nxyzpq && next_item_layout == nn::layout_t<float>::xyzpqn))   // N-XYZ -> XYZ-N
                {
                    auto type4_conversion = init_type4_conversion(next_item_layout);
                    type4_conversion->name = std::string("convert_layout4_before_") + next_load_item.item->name;

                    for (auto &useinput : next_load_item.item->input)
                    if (useinput.item == load_item)
                    {
                        useinput.item = type4_conversion;
                        type4_conversion->use.push_back(next_load_item);
                    }

                    next_load_item.item = type4_conversion;
                }
            }
        }
    }

    auto init_type5_conversion = [&batch, &flow_item, &load_item]
    {
        auto conversion = new nn_workload_item_t;
        conversion->primitive = nullptr;
        conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
        conversion->arguments.convert_data_layout.type = 5;

        nn_workload_data_layout_t layout = nn::workload_data<int16_t>::layout.pxyznq;

        uint32_t z_size = flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1;
        uint32_t z_block = z_size > 4 ? 16 : 4;

        nn_workload_data_coords_t size(
            batch,
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            (z_size - 1) / z_block + 1,
            z_block,
            1 );

        conversion->output.push_back(new nn::workload_data<float>(NN_WORKLOAD_DATA_TAG_UNKNOWN, size, layout));
        conversion->input.push_back({ load_item, 0 });
        return conversion;
    };

    if (load_item->type == NN_WORK_ITEM_TYPE_INPUT) {
        for (auto &next_load_item : load_item->use) {
            if (next_load_item.item->type == NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN ||
                next_load_item.item->type == NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT ||
                next_load_item.item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT ||
                next_load_item.item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT) {
                assert(input_format[load_item->arguments.input.index] == NN_WORKLOAD_DATA_TYPE_I16_ZXY); // TODO support other formats
                auto type5_conversion = init_type5_conversion();
                type5_conversion->name = std::string("convert_layout5_before_") + next_load_item.item->name;

                for (auto &useinput : next_load_item.item->input)
                if (useinput.item == load_item) {
                    useinput.item = type5_conversion;
                    type5_conversion->use.push_back(next_load_item);
                }

                next_load_item.item = type5_conversion;
            }
        }
    }

    auto init_type6_conversion = [&batch, &flow_item, &load_item]
    {
        auto conversion = new nn_workload_item_t;
        conversion->primitive = nullptr;
        conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
        conversion->arguments.convert_data_layout.type = 6;

        nn_workload_data_layout_t layout = nn::workload_data<int16_t>::layout.zxynpq;

        nn_workload_data_coords_t size(
            batch,
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1,
            1,
            1 );

        conversion->output.push_back(new nn::workload_data<float>(NN_WORKLOAD_DATA_TAG_UNKNOWN, size, layout));
        conversion->input.push_back({ load_item, 0 });
        return conversion;
    };

    if ((load_item->type == NN_WORK_ITEM_TYPE_MERGE && load_item->input[0].get_data_view()->parent->layout.data_type == NN_DATATYPE_INT16) ||
        load_item->type == NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN ||
        load_item->type == NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT ||
        load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT ||
        load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT) {
        for (auto &next_load_item : load_item->use) {
            if (next_load_item.item->type == NN_WORK_ITEM_TYPE_OUTPUT) {
                auto type6_conversion = init_type6_conversion();
                type6_conversion->name = std::string("convert_layout6_before_") + next_load_item.item->name;

                for (auto &useinput : next_load_item.item->input)
                if (useinput.item == load_item) {
                    useinput.item = type6_conversion;
                    type6_conversion->use.push_back(next_load_item);
                }

                next_load_item.item = type6_conversion;
            }
        }
    }
}

/* compile workflow into workload */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_compile_0_function(
    nn_workload_t         **workload,       /* resulting workload */
    nn_device_t            *device,         /* device context */
    nn_workflow_t          *workflow,       /* workflow to be compiled */
    NN_WORKLOAD_DATA_TYPE  *input_format,   /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE  *output_format,  /* array containing formats of outputs */
    uint32_t                batch           /* batch size for compilation */
    ) {
    if(!workload || !device || !workflow)       return NN_API_STATUS_ERROR_INVALID_POINTER;
    for(auto index=0u; index<workflow->input_count; ++index)
        if(workflow->input[index] ->type!=NN_WORK_ITEM_TYPE_INPUT)
            return NN_API_STATUS_ERROR_INVALID_WORKFLOW; // TODO: more granular error code here
    for(auto index=0u; index<workflow->output_count; ++index)
        if(workflow->output[index]->type!=NN_WORK_ITEM_TYPE_OUTPUT)
            return NN_API_STATUS_ERROR_INVALID_WORKFLOW; // TODO: more granular error code here
    try {
        // allocate memory for workload (public & opaque parts & data buffers);
        const size_t  input_size = sizeof(NN_WORKLOAD_DATA_TYPE)*workflow-> input_count;
        const size_t output_size = sizeof(NN_WORKLOAD_DATA_TYPE)*workflow->output_count;
        const size_t buffer_size = sizeof(nn_workload_t)+sizeof(nn_workload_opaque_t)+input_size+output_size;
        uint8_t *buffer = new uint8_t[buffer_size];
        // fill data structure
        nn_workload_t          *workload_public = reinterpret_cast<nn_workload_t *>(buffer);
        buffer += sizeof(nn_workload_t);
        nn_workload_opaque_t   *workload_opaque = new(buffer) nn_workload_opaque_t; // placement new
        buffer += sizeof(nn_workload_opaque_t);
        *const_cast<nn_device_t **>(&workload_public->device) = device;
        *const_cast<uint32_t *>(&workload_public->input_count)  = workflow->input_count;
        *const_cast<uint32_t *>(&workload_public->output_count) = workflow->output_count;
        *const_cast<NN_WORKLOAD_DATA_TYPE **>(&workload_public->input_format) = reinterpret_cast<NN_WORKLOAD_DATA_TYPE *>(buffer);
        std::memcpy(workload_public->input_format, input_format, input_size);
        buffer += sizeof(NN_WORKLOAD_DATA_TYPE)*workflow->input_count;
        *const_cast<NN_WORKLOAD_DATA_TYPE **>(&workload_public->output_format) = (workflow->output_count) ? reinterpret_cast<NN_WORKLOAD_DATA_TYPE *>(buffer) : nullptr;
        *const_cast<uint32_t *>(&workload_public->batch) = batch;
        std::memcpy(workload_public->output_format, output_format, output_size);

        // lookup for matching workflow items to workload items
        std::map<nn_workflow_item_t *, nn_workload_item_t *> flow_to_work;

        {   // traverse workflow items and create workload items
            std::queue<nn_workflow_item_t *> todo;
            for(auto index = 0u; index<workflow->input_count; ++index)
                todo.push(workflow->input[index]);
            while(!todo.empty()) {
                nn_workflow_item_t *flow_item = todo.front();
                todo.pop();
                if(flow_to_work.find(flow_item)==flow_to_work.end()) {
                    flow_to_work[flow_item] = new nn_workload_item_t;
                    for(auto index=0u; index<flow_item->use_count; ++index)
                        todo.push(flow_item->use[index].item);
                }
            }
        }

        { // now for every workflow item there's a workload item
            std::set<nn_workflow_item_t *> done;
            std::deque<nn_workflow_item_t *> todo;
            for (auto index = 0u; index<workflow->input_count; ++index)
                todo.push_front(workflow->input[index]);
            while(!todo.empty()) {
                nn_workflow_item_t *flow_item = todo.front();
                if(done.find(flow_item)==done.end()) {   // Not yet marked.
                    bool all_inputs_done = true;
                    // Go through all inputs.
                    for (auto index = 0u; index < flow_item->input_count; ++index) {
                        if (done.find(flow_item->input[index].item) == done.end())
                        {
                            // Input not marked, add to todo.
                            todo.push_front(flow_item->input[index].item);
                            all_inputs_done = false;
                        }
                    }
                    if (all_inputs_done) {
                        // All inputs already marked - mark this item as well and remove from todo.
                        // Also add all its uses to todo (if they arent there yet.)
                        nn_workload_item_t *load_item = flow_to_work[flow_item];
                        nn_workflow_compile_0_function_copy_item(load_item, flow_item, workflow, workload_public, batch, flow_to_work, reinterpret_cast<nn_device_internal*>(device));
                        done.insert(flow_item);
                        todo.pop_front();
                        for (auto index = 0u; index < flow_item->use_count; ++index)
                            if (std::find(std::begin(todo), std::end(todo), flow_item->use[index].item) == std::end(todo))
                                todo.push_front(flow_item->use[index].item);
                    }
                }
                else {
                    // Marked, remove from todo list.
                    todo.pop_front();
                }
            }
        }

        { // add data layout conversions if required
            std::queue<nn_workflow_item_t *> todo;
            for (auto index = 0u; index < workflow->input_count; ++index)
                todo.push(workflow->input[index]);
            while (!todo.empty()) {
                auto flow_item = todo.front();
                auto load_item = flow_to_work[flow_item];
                todo.pop();
                nn_workflow_compile_0_function_add_conversions(load_item, flow_item, batch, input_format, output_format);
                for (auto index = 0u; index < flow_item->use_count; ++index)
                    todo.push(flow_item->use[index].item);
            }
        }

        // copying inputs & outputs
        workload_opaque->input.resize(workflow->input_count);
        for(auto index=0u; index<workflow->input_count; ++index)
            workload_opaque->input[index] = flow_to_work[workflow->input[index]];

        workload_opaque->output.resize(workflow->output_count);
        for(auto index=0u; index<workflow->output_count; ++index)
            workload_opaque->output[index] = flow_to_work[workflow->output[index]];

        { // creating workload items in order of execution
            std::set<nn_workload_item_t *> done;
            std::deque<nn_workload_item_t *> todo;
            for (auto wrkl_input_item : workload_opaque->input)
                todo.push_front(wrkl_input_item);
            while(!todo.empty()) {
                nn_workload_item_t *load_item = todo.front();
                if(done.find(load_item)==done.end()) {   // Not yet marked.
                    bool all_inputs_done = true;
                    // Go through all inputs.
                    for (auto input_item : load_item->input) {
                        if (done.find(input_item.item) == done.end()) {
                            // Input not marked, add to todo.
                            todo.push_front(input_item.item);
                            all_inputs_done = false;
                        }
                    }
                    if (all_inputs_done) {
                        // All inputs already marked - mark this item as well and remove from todo.
                        // Also add all its uses to todo (if they arent there yet.)
                        workload_opaque->order_of_execution.push_back(load_item);
                        done.insert(load_item);
                        todo.pop_front();
                        for (auto use_item : load_item->use)
                            if (std::find(std::begin(todo), std::end(todo), use_item.item) == std::end(todo))
                                todo.push_front(use_item.item);
                    }
                }
                else {
                    // Marked, remove from todo list.
                    todo.pop_front();
                }
            }
        }

        // set result
        *workload = workload_public;
    }
    catch (...) {
        return NN_API_STATUS_ERROR_OUT_OF_MEMORY;
    }

    return NN_API_STATUS_OK;
}

/* executes workload with given inputs & outputs */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workload_execute_0_function(
    nn_workload_t      *workload_public,/* workload to be started */
    void *             *input,          /* array of pointers with input data;  format is in workload->input_format */
    void *             *output,         /* array of pointers with output data; format is in workload->output_format */
    NN_API_STATUS      *status          /* asynchronous status */
    ) {
    if (!workload_public || !input || (!output && workload_public->output_count) ) return NN_API_STATUS_ERROR_INVALID_POINTER;
    else {
        try {
            *status = NN_API_WORK_IN_PROGRESS;
            nn_workload_opaque_t *workload_opaque = reinterpret_cast<nn_workload_opaque_t *>(workload_public + 1);
#if  ENABLE_WORKLOAD_MONITORING
            uint16_t  item_count=0;
#endif // ENABLE_WORKLOAD_MONITORING
            for(auto item : workload_opaque->order_of_execution) {
#if ENABLE_WORKLOAD_PROFILING

                auto t0 = __rdtsc();
#endif

                switch(item->type) {
                case NN_WORK_ITEM_TYPE_INPUT: {
                    // Copy input.
                    auto item_input = reinterpret_cast<nn_data_t*>(input[item->arguments.input.index]);

                    // TODO validate if input have same lengths and layout as one given during compilation.
                    item->output[0]->parent->data_buffer = item_input->buffer;
                    break;
                }
                case NN_WORK_ITEM_TYPE_OUTPUT: {
                    // Copy result to workload output.
                    auto item_output = reinterpret_cast<nn_data_t*>(output[item->arguments.output.index]);

                    // TODO validate if output have same lengths and layout as one given during compilation.
                    item->output[0]->parent->data_buffer = item_output->buffer;
                    nn_workload_data_copy(item->output[0], item->input[0].get_data_view());
                    break;
                }
                case NN_WORK_ITEM_TYPE_MERGE:
                case NN_WORK_ITEM_TYPE_VIEW: {
                    // Nothing to do here - it's just limiting access to input by attaching view to it.
                    break;
                }
                case NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP: {
                    layer::run_multithreaded_convolve_work_item_backward(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP: {
                    layer::run_multithreaded_FC_work_item_backward(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_POOLING_BACKPROP: {
                    layer::wrapper_pooling_work_item_backward(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_NORMALIZATION_BACKPROP: {
                    layer::wrapper_normalization_work_item_backward(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_SOFTMAX_BACKPROP: {
                    layer::wrapper_softmax_work_item_backward(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_RELU_1D_BACKPROP:
                case NN_WORK_ITEM_TYPE_RELU_3D_BACKPROP:
                {
                    layer::run_relu_backward(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_LOSS_FUNCTION: {
                    layer::run_loss_function(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS: {
                    layer::run_parameter_update(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_DROPOUT: {
                    item->primitive->forward(
                        {item->input[0].get_data_view(), item->input[1].get_data_view(), item->input[2].get_data_view()},
                        {nullptr},
                        item->output);
                    break;
                }
                case NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP: {
                    auto primitive = (item->forward_item != nullptr) ? item->forward_item->primitive : item->primitive;
                    auto input = item->input[0].get_data_view();
                    auto output = item->output[0];

                    // TODO: change this when workflow compilation starts using parent->delta_buffer
                    assert(input->parent->delta_buffer == nullptr);
                    assert(output->parent->delta_buffer == nullptr);
                    input->parent->delta_buffer = input->parent->data_buffer;
                    output->parent->delta_buffer = output->parent->data_buffer;

                    primitive->backward(
                        {output, item->input[1].get_data_view(), item->input[2].get_data_view()},
                        {nullptr},
                        {input});

                    // TODO: change this when workflow compilation starts using parent->delta_buffer
                    // revert changes made above
                    input->parent->delta_buffer = nullptr;
                    output->parent->delta_buffer = nullptr;

                    break;
                }
                case NN_WORK_ITEM_TYPE_AVERAGE_DELTAS: {
                    layer::run_average_delta(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT: {
                    if (item->primitive == nullptr) {
                        layer::run_convert_to_data_layout_work_item(item);
                        break;
                    }
                }
                default: {
                    assert(item->primitive != nullptr);

                    std::vector<const nn_workload_data_t *> inputs;
                    for(auto& input_descriptor : item->input)
                        inputs.push_back(input_descriptor.get_data_view());

                    item->primitive->forward(inputs, {item->parameters.begin(), item->parameters.end()}, item->output);
                }
                } // switch

#if ENABLE_WORKLOAD_PROFILING
                auto t1 = __rdtsc();
                workload_opaque->profiling_data.work_item_cycles[item].push_back(t1 - t0);
#endif

#if ENABLE_WORKLOAD_MONITORING
                nn_workload_item_data_marshaling(item, ++item_count);
#endif // ENABLE_WORKLOAD_MONITORING
            }

            // unset input and output buffer
            for (auto item : workload_opaque->order_of_execution)
            {
                if (item->type == NN_WORK_ITEM_TYPE_INPUT || item->type == NN_WORK_ITEM_TYPE_OUTPUT)
                    item->output[0]->parent->data_buffer = nullptr;
            }
        }
        catch(NN_API_STATUS status) {
            return status;
        }
        catch(...) {
            return NN_API_STATUS_ERROR_OUT_OF_MEMORY;
        }
    }
    return NN_API_STATUS_OK;
}

#if ENABLE_WORKLOAD_PROFILING
void nn_workload_print_profiling_data(nn_workload_opaque_t* workload_opaque){
    printf("\n------------profiling--------------\n");
    printf("%20s %20s %10s %10s\n", "layer_type", "layer_name", "cycles", "");
    printf("%20s %20s %10s %10s\n", "", "", "minimum", "average");

    assert(workload_opaque->order_of_execution.size() != 0); // Empty workload (?).

    for (const auto &item : workload_opaque->order_of_execution) {
        const auto &times = workload_opaque->profiling_data.work_item_cycles[item];

        if(times.size() == 0)
            continue; // Workload was never executed so there are no results.

        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        uint64_t minimum = *std::min_element(times.begin(), times.end());

        auto item_name = [](nn_workload_item* item){
            switch (item->type){
            case NN_WORK_ITEM_TYPE_NORMALIZATION: return "norm";
            case NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT: return "conv_layout";
            case NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT: return "conv_float2int";
            case NN_WORK_ITEM_TYPE_CONVOLUTION: return  "cnn_f32";
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: return "cnn_pool2x2_f32";
            case NN_WORK_ITEM_TYPE_POOLING: return "pooling_f32";
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED: return "fc_f32";
            case NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT: return "cnn_i16";
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT: return "cnn_pool2x2_i16";
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN:
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN: return  "fc_i16";
            case NN_WORK_ITEM_TYPE_SOFTMAX:
            case NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT: return "softmax";
            case NN_WORK_ITEM_TYPE_MERGE: return  "merge";
            case NN_WORK_ITEM_TYPE_ARITHMETIC: return "arithmetic";
            case NN_WORK_ITEM_TYPE_RELU: return "relu";
            case NN_WORK_ITEM_TYPE_RELU_1D: return "relu_1d";
            case NN_WORK_ITEM_TYPE_RELU_3D: return "relu_3d";
            case NN_WORK_ITEM_TYPE_VIEW: return "view";
            case NN_WORK_ITEM_TYPE_INPUT: return "input";
            case NN_WORK_ITEM_TYPE_OUTPUT: return "output";
            case NN_WORK_ITEM_TYPE_AVERAGE_DELTAS: return "avg_deltas";
            case NN_WORK_ITEM_TYPE_DROPOUT: return "dropout";
            case NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP: return "dropout_back";
            case NN_WORK_ITEM_TYPE_LOSS_FUNCTION: return "loss_func";
            case NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS: return "update_args";
            case NN_WORK_ITEM_TYPE_RELU_BACKPROP: return "relu_back";
            case NN_WORK_ITEM_TYPE_RELU_1D_BACKPROP: return "relu_1d_back";
            case NN_WORK_ITEM_TYPE_RELU_3D_BACKPROP: return "relu_3d_back";
            case NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP: return "cnn_back";
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP: return "fc_back";
            case NN_WORK_ITEM_TYPE_POOLING_BACKPROP: return "pool_back";
            case NN_WORK_ITEM_TYPE_NORMALIZATION_BACKPROP: return "norm_back";
            case NN_WORK_ITEM_TYPE_SOFTMAX_BACKPROP: return "softmax_back";
            default: return "no_name";
            }
        };

        printf("%20s %20s %10llu %10.0f\n", item_name(item), item->name.c_str(), minimum, avg);
    }
    printf("\n-----------------------------------\n");
}
#endif



/* delete workload */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workload_delete_0_function(
    nn_workload_t       *workload_public       /* workload to be deleted */
    ) {
    if(!workload_public) return NN_API_STATUS_ERROR_INVALID_POINTER;
    else {
        try {
            uint8_t *buffer = reinterpret_cast<uint8_t *>(workload_public);
            nn_workload_opaque_t *workload_opaque = reinterpret_cast<nn_workload_opaque_t *>(buffer + sizeof(nn_workload_t));

#if ENABLE_WORKLOAD_PROFILING
            nn_workload_print_profiling_data(workload_opaque);
#endif

            std::stack<nn_workload_item_t *> todo;
            std::set<nn_workload_item_t *> done;
            for(auto element : workload_opaque->input) todo.push(element);
            while(!todo.empty()) {
                nn_workload_item_t *load_item = todo.top();
                todo.pop();
                if(done.find(load_item)==done.end()) {
                    done.insert(load_item);
                    for(auto element : load_item->use) todo.push(element.item);
                    for(auto parameter : load_item->parameters)
                        delete parameter;
                    load_item->parameters.clear();
                    delete load_item;
                }
            }
            workload_opaque->~nn_workload_opaque_t(); // placement delete
            delete[] buffer;
        }
        catch(...) {
            return NN_API_STATUS_ERROR_OUT_OF_MEMORY;
        }
    }
    return NN_API_STATUS_OK;
}

/* query workflow for metrics of compilation variants */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_metrics_query_0_function(
    nn_workflow_metrics_array_t **array,/* resulting array of variants */
    nn_device_t        *device,         /* device context */
    nn_workflow_t      *workflow        /* workflow to be querried */
    ) {
    return NN_API_STATUS_ERROR_OTHER;
}

/* delete array of workload metrics */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_metrics_delete_0_function(
    nn_workflow_metrics_array_t *array  /* array to delete */
    ) {
    return NN_API_STATUS_ERROR_OTHER;
}

/* validate parameters of work_item for this particular device */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_item_validate_0_function(
    nn_device_t        *device,         /* target device */
    nn_workflow_item_t *work_item       /* work item to be validated */
    ) {
    return NN_API_STATUS_ERROR_OTHER;
}

NN_API_STATUS NN_API_CALL_CONVENTION nn_device_parameter_get_0_function(
    nn_device_t        *device,         /* target context */
    NN_PARAMETER        parameter,      /* parameter to get */
    void               *buffer,         /* buffer to store result to */
    uint32_t            size            /* size of buffer */
    ) {
    return NN_API_STATUS_ERROR_OTHER;
}

NN_API_STATUS NN_API_CALL_CONVENTION nn_device_parameter_set_0_function(
    nn_device_t        *device,         /* target context */
    NN_PARAMETER        parameter,      /* parameter to set */
    void               *buffer,         /* buffer with argument */
    uint32_t            size            /* size of buffer */
    ) {
    return NN_API_STATUS_ERROR_OTHER;
}
