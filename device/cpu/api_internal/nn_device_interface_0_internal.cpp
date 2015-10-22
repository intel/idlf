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
#include "device/cpu/core/layer_convolution_avx2_batch24n.h"
#include "device/cpu/core/layer_convolution_pooling_avx2.h"
#include "device/cpu/core/layer_fully_connected_avx2.h"
#include "device/cpu/core/layer_softmax_avx2.h"
#include "device/cpu/core/layer_softmax_avx2_batch24n.h"
#include "device/cpu/core/layer_softmax_loss_avx2.h"
#include "device/cpu/core/layer_pooling_avx2.h"
#include "device/cpu/core/layer_pooling_avx2_batch24n.h"
#include "device/cpu/core/layer_normalization_avx2.h"
#include "device/cpu/core/layer_normalization_avx2_batch24n.h"
#include "device/cpu/core/layer_convert_data_layout.h"
#include "device/cpu/core/layer_convert_data_from_batch_block_layout.h"
#include "device/cpu/core/layer_convert_data_to_batch_block_layout.h"
#include "device/cpu/core/layers_fixedpoint.h"
#include "device/cpu/core/layer_arithmetic_operation.h"
#include "device/cpu/core/layer_parameter_update.h"
#include "device/cpu/core/layer_average_delta.h"
#include "device/cpu/core/layer_loss_function.h"
#include "device/cpu/core/layer_relu_avx2.h"
#include "device/cpu/core/layer_dropout.h"
#include "nn_device_interface_0_internal.h"
#include "device/cpu/core/layer_fully_connected_avx2_batch24n.h"
#include "data_helper.h"

#include <map>
#include <cassert>
#include <stack>
#include <set>
#include <cstring>
#include <memory>
#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include <iostream>
#include <cstdlib>
#include <unordered_set>
#include <unordered_map>
#include <iterator>

#define ENABLE_WORKLOAD_MONITORING 0

namespace nn
{
bool use_asmjit_primitives = false;
} //namespace nn

namespace
{
#if ENABLE_WORKLOAD_MONITORING
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

void nn_workload_item_data_marshaling(nn_workload_item* item, int stage_num)
{
    std::ostringstream temp;
    float *fp;
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
    case NN_WORK_ITEM_TYPE_SOFTMAX_LOSS: item_name = "softmax_loss"; break;
    case NN_WORK_ITEM_TYPE_SOFTMAX_LOSS_BACKPROP: item_name = "softmax_loss_backprop"; break;
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
        const auto output = nn::workload_data_cast<>(item->output[0]);
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

template <> struct flow_item_helper<NN_WORK_ITEM_TYPE_POOLING> {
    static const nn_arguments_forward_pooling &get_arguments(const nn_workflow_item *flow_item) {
        return flow_item->arguments.forward_pooling;
    }

    static void calculate_padding(const nn_workflow_item *flow_item, size_t input_w, size_t input_h, size_t &left_padding, size_t &right_padding, size_t &top_padding, size_t &bottom_padding){
        auto& arguments = get_arguments(flow_item);

        size_t padding_w = ((flow_item->output_format[0].format_3d.size[0] - 1) * arguments.stride[0] + arguments.size[0]) - input_w;
        size_t padding_h = ((flow_item->output_format[0].format_3d.size[1] - 1) * arguments.stride[1] + arguments.size[1]) - input_h;

        assert(padding_w >= arguments.center_offset[0]);
        assert(padding_h >= arguments.center_offset[1]);

        left_padding = arguments.center_offset[0];
        right_padding = padding_w - arguments.center_offset[0];

        top_padding = arguments.center_offset[1];
        bottom_padding = padding_h - arguments.center_offset[1];
    }
};

template <> struct flow_item_helper<NN_WORK_ITEM_TYPE_NORMALIZATION> {
    static const nn_arguments_forward_normalization &get_arguments(const nn_workflow_item *flow_item) {
        return flow_item->arguments.forward_normalization;
    }

    static void calculate_padding(const nn_workflow_item *flow_item, size_t input_w, size_t input_h, size_t &left_padding, size_t &right_padding, size_t &top_padding, size_t &bottom_padding){
        auto& arguments = get_arguments(flow_item);

        size_t padding_w = flow_item->output_format[0].format_3d.size[0] - input_w;
        size_t padding_h = flow_item->output_format[0].format_3d.size[1] - input_h;

        left_padding = 0;
        right_padding = padding_w;

        top_padding = 0;
        bottom_padding = padding_h;
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
            nn_workflow_compile_0_function_update_output_padding_for_use<NN_WORK_ITEM_TYPE_POOLING>(use_item, output_w, output_h, left_padding, right_padding, top_padding, bottom_padding);
            nn_workflow_compile_0_function_update_output_padding_for_use<NN_WORK_ITEM_TYPE_NORMALIZATION>(use_item, output_w, output_h, left_padding, right_padding, top_padding, bottom_padding);
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
                                                     size_t output_bottom_padding)
{
    switch (load_item->type)
    {
    case NN_WORK_ITEM_TYPE_CONVOLUTION:
    {
        auto &args = flow_item->arguments.forward_convolution;
        assert((flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2]
                                                                     : 1) == args.weights->size[3]);
        assert(args.padding == NN_PADDING_MODE_DATA_OR_ZERO);

        using namespace convolution;
        if (nn::use_asmjit_primitives && (batch % 24 == 0) && (args.weights->size[0] < 7))
        {
            load_item->primitive = new layer::convolution_f32_batch24n(
                make<Batch>(batch),
                OutputDimensions{make<OutputHeight>(flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1),
                                 make<OutputWidth>(flow_item->output_format[0].format_1d.size[0]),
                                 make<OutputFeats>(args.weights->size[3])},
                KernelInfo{KernelDimensions{make<KernelHeight>(args.weights->size[1]),
                                            make<KernelWidth>(args.weights->size[0]),
                                            make<KernelFeats>(args.weights->size[2])},
                           make<OutputFeats>(args.weights->size[3]),
                           KernelCenter{make<Rows>(args.center_offset[1]), make<Cols>(args.center_offset[0])},
                           Stride{make<Rows>(args.stride[1]), make<Cols>(args.stride[0])}},
                args.activation,
                device);
        }
        else
        {
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
        }
        break;
    }
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2:
    {
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
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED:
    {
        auto &args = flow_item->arguments.forward_fully_connected;
        auto& input = flow_item->input[0];

        assert(input.item->output_format[input.index].format == NN_DATA_FORMAT_1D ||
               input.item->output_format[input.index].format == NN_DATA_FORMAT_3D);

        bool use_3d_input = input.item->output_format[input.index].format == NN_DATA_FORMAT_3D;

        if (nn::use_asmjit_primitives and (batch % 24 == 0))
            load_item->primitive = new layer::fully_connected_f32_batch24n(
                (use_3d_input ? (std::max(args.weights->size[0], size_t(1))
                                * std::max(args.weights->size[1], size_t(1))
                                * std::max(args.weights->size[2], size_t(1)))
                    : args.weights->size[0]),
                (use_3d_input ? args.weights->size[3] : args.weights->size[1]),
                args.activation,
                batch,
                device);
        else if(use_3d_input)
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
    case NN_WORK_ITEM_TYPE_POOLING:
    {
        auto &args = flow_item->arguments.forward_pooling;
        auto& input = flow_item->input[0];

        assert(input.item->output_format[input.index].format >= NN_DATA_FORMAT_2D);
        assert(flow_item->output_format[0].format >= NN_DATA_FORMAT_2D);
        assert(get_format_size<2>(input.item->output_format[input.index]) ==
               get_format_size<2>(flow_item->output_format[0])); // input and output have same depth

        auto load_input = load_item->input[0];
        auto prev_output = load_input.item->output[load_input.index];
        if (nn::use_asmjit_primitives
            && (args.mode == NN_POOLING_MODE_MAX)
            && (prev_output->parent->layout == nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::layout))
            load_item->primitive = new layer::max_pooling_f32_batch24n(
                PoolingInfo{PoolingDimensions(make<PoolingHeight>(args.size[1]),
                                              make<PoolingWidth>(args.size[0])),
                            Stride{make<Rows>(args.stride[1]), make<Cols>(args.stride[0])}},
                OutputDimensions(make<OutputHeight>(get_format_size<1>(flow_item->output_format[0])),
                                 make<OutputWidth>(get_format_size<0>(flow_item->output_format[0])),
                                 make<OutputFeats>(get_format_size<2>(flow_item->output_format[0]))),
                batch,
                device);
        else
            load_item->primitive = new layer::pooling_f32(args.mode,
                                                          args.size[0],
                                                          args.size[1],
                                                          args.stride[0],
                                                          args.stride[1],
                                                          get_format_size<2>(flow_item->output_format[0]),
                                                          get_format_size<0>(flow_item->output_format[0]),
                                                          get_format_size<1>(flow_item->output_format[0]),
                                                          args.center_offset[0],
                                                          args.center_offset[1],
                                                          batch,
                                                          output_left_padding,
                                                          output_right_padding,
                                                          output_top_padding,
                                                          output_bottom_padding,
                                                          device);
        break;
    }
    case NN_WORK_ITEM_TYPE_ARITHMETIC:
    {
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
    case NN_WORK_ITEM_TYPE_RELU_1D:
    {
        load_item->primitive = new layer::relu_1d_f32(
            get_format_size<0>(flow_item->output_format[0]),
            batch,
            device);
        break;
    }
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
    case NN_WORK_ITEM_TYPE_NORMALIZATION:
    {
        auto &args = flow_item->arguments.forward_normalization;
        switch (args.normalization.mode) {
        case NN_NORMALIZATION_MODE_LINEAR_SINGLE:
        {
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
        case NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS:
        {
            auto load_input = load_item->input[0];
            auto prev_output = load_input.item->output[load_input.index];
            if (nn::use_asmjit_primitives and (batch % 24 == 0))
                load_item->primitive =
                    new layer::normalization_response_across_maps_f32_batch24n(
                        args.normalization.alpha,
                        args.normalization.beta,
                        args.normalization.k,
                        args.normalization.n,
                        OutputDimensions{make<OutputHeight>(get_format_size<1>(flow_item->output_format[0])),
                                         make<OutputWidth>(get_format_size<0>(flow_item->output_format[0])),
                                         make<OutputFeats>(get_format_size<2>(flow_item->output_format[0]))},
                        batch,
                        device);
            else
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
        if (nn::use_asmjit_primitives and (batch % 24 == 0))
            load_item->primitive = new layer::softmax_f32_batch24n(get_format_size<0>(flow_item->output_format[0]), batch, device);
        else
            load_item->primitive = new layer::softmax_f32(get_format_size<0>(flow_item->output_format[0]), batch, device);
        break;
    }
    case NN_WORK_ITEM_TYPE_SOFTMAX_LOSS:
    {
        load_item->primitive = new layer::softmax_loss_f32(get_format_size<0>(flow_item->output_format[0]), batch, device);
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
            args.mode,
            flow_item->output_format[0].format_3d.size[2],
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            args.pool_size[0],
            args.pool_size[1],
            args.pool_stride[0],
            args.pool_stride[1],
            args.center_offset[0],
            args.center_offset[1],
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

nn_workload_data_layout_t get_workload_layout(NN_WORKLOAD_DATA_TYPE type)
{
    // layout for inputs & outputs
    switch (type)
    {
    case NN_WORKLOAD_DATA_TYPE_F32_1D:
    case NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH:
    case NN_WORKLOAD_DATA_TYPE_F32_2D:
    case NN_WORKLOAD_DATA_TYPE_F32_2D_BATCH:
    case NN_WORKLOAD_DATA_TYPE_F32_3D:
    case NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH:
        return nn::layout_t<nn::layout_xyzpqn_f32>::layout;

    case NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH:
    case NN_WORKLOAD_DATA_TYPE_F32_ZXY:
        return nn::layout_t<nn::layout_zxynpq_f32>::layout;

    case NN_WORKLOAD_DATA_TYPE_I16_1D:
    case NN_WORKLOAD_DATA_TYPE_I16_1D_BATCH:
    case NN_WORKLOAD_DATA_TYPE_I16_3D:
    case NN_WORKLOAD_DATA_TYPE_I16_3D_BATCH:
        return nn::layout_t<nn::layout_xyzpqn_i16>::layout;

    case NN_WORKLOAD_DATA_TYPE_I16_ZXY:
    case NN_WORKLOAD_DATA_TYPE_I16_ZXY_BATCH:
        return nn::layout_t<nn::layout_zxynpq_i16>::layout;

    case NN_WORKLOAD_DATA_TYPE_I32_1D:
    case NN_WORKLOAD_DATA_TYPE_I32_1D_BATCH:
        return nn::layout_t<nn::layout_xyzpqn_i32>::layout;

    default:
        throw std::out_of_range("unsupported data type");
    }
};

nn_workload_data_coords_t calculate_size(uint32_t batch, NN_WORKLOAD_DATA_TYPE type, nn_output_format *data)
{
    // calculates 6D size from nn::data, returns it as nn_workload_data_coords_t
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

    switch (type)
    {
    case NN_WORKLOAD_DATA_TYPE_F32_ZXY:
    case NN_WORKLOAD_DATA_TYPE_I16_ZXY:
    case NN_WORKLOAD_DATA_TYPE_F32_3D:
    case NN_WORKLOAD_DATA_TYPE_I16_3D:
    case NN_WORKLOAD_DATA_TYPE_F32_2D:
    case NN_WORKLOAD_DATA_TYPE_F32_1D:
    case NN_WORKLOAD_DATA_TYPE_I16_1D:
    case NN_WORKLOAD_DATA_TYPE_I32_1D:
        size_n = 1;
        break;
    }

    return nn_workload_data_coords_t( size_n, size_x, size_y, size_z, size_p, size_q );
}

template <typename T_FindLoadItem>
void nn_workflow_compile_0_function_copy_item(
             nn_workload_item_t *load_item,
             nn_workflow_item_t *flow_item,
             nn_workflow_t      *workflow,
             nn_workload_t      *workload,
             uint32_t batch,
             T_FindLoadItem find_load_item,
             nn_device_internal *device
             ) {

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

    // creating outputs
    for (uint32_t out = 0; out < flow_item->output_count; ++out)
    {
        load_item->output.push_back(nullptr);
    }

    // copying inputs
    load_item->input.resize(flow_item->input_count);
    for (auto index = 0u; index < flow_item->input_count; ++index)
    {
        auto input_load_item = find_load_item(flow_item->input[index].item);
        assert(input_load_item);
        load_item->input[index].item = input_load_item;
        load_item->input[index].index = flow_item->input[index].index;
    }

    load_item->primitive = nullptr;

    // setup primitive handle (only forward passes)
    if (flow_item->forward_item == nullptr)
    {
        nn_workflow_compile_0_function_create_primitive(
            load_item,
            flow_item,
            batch,
            device,
            padding_left,
            padding_right,
            padding_top,
            padding_bottom);
        load_item->forward_item = nullptr;
    }
    else
    {

        load_item->forward_item = find_load_item(flow_item->forward_item);
        if(load_item->forward_item==nullptr) throw std::runtime_error("Couldn't find forward item");

    }

    // setup output buffer
    switch(load_item->type) {
    case NN_WORK_ITEM_TYPE_INPUT: {
        auto input_index = flow_item->arguments.input.index;
        auto input_item_format = workload->input_format[input_index];
        auto input_item_size = calculate_size(workload->batch, input_item_format, flow_item->output_format);
        auto input_item_layout = get_workload_layout(input_item_format);

        load_item->output[0] =
            new nn::workload_data<>(
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
            new nn::workload_data<>(
                NN_WORKLOAD_DATA_TAG_UNKNOWN, output_item_size, output_item_layout, true);
        break;
    }
    case NN_WORK_ITEM_TYPE_VIEW:
    {
        auto& origin = flow_item->arguments.view.origin;
        assert(flow_item->input_count == 1);
        auto& input = load_item->input[0];
        auto input_data = reinterpret_cast<nn::workload_data<> *>(input.item->output[input.index]);
        auto div = (input_data->parent->tag == NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN) ? input_data->parent->lengths.t[NN_DATA_COORD_p] : 1;
        nn_workload_data_coords_t start(
            0,
            origin[0],
            origin[1],
            origin[2] / div,
            0,
            0);
        auto x_size = flow_item->output_format[0].format_1d.size[0];
        auto y_size = (flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1);
        auto z_size = (flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1);
        nn_workload_data_coords_t end(
            input_data->get_length(NN_DATA_COORD_n) - 1,
            origin[0] + x_size - 1,
            origin[1] + y_size - 1,
            (origin[2] + z_size - 1) / div,
            input_data->get_length(NN_DATA_COORD_p) - 1,
            input_data->get_length(NN_DATA_COORD_q) - 1
        );

        load_item->output[0] = new nn::workload_data<>(*input_data, start, end);
        break;
    }
    case NN_WORK_ITEM_TYPE_MERGE:
    {
        uint16_t axis = flow_item->arguments.forward_merge.axis; // x = 0, y = 1, z = 2
        auto& input = load_item->input[0];
        auto input_data = input.item->output[input.index];// flow_to_work[input.item]->output[input.index];
        nn_workload_data_layout_t previous_layout = input_data->parent->layout;

        uint32_t x_size = input_data->parent->lengths.t[1];
        uint32_t y_size = input_data->parent->lengths.t[2];
        uint32_t z_size = input_data->parent->lengths.t[3];
        for (int index = 1; index < flow_item->input_count; index++)
        {
            auto& input_indexed = load_item->input[index];
            auto input_data_local = input_indexed.item->output[input_indexed.index]; //flow_to_work[input_indexed.item]->output[input_indexed.index];
            if (input_data->parent->layout != previous_layout)
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

        memset(load_item->output[0]->parent->data_buffer, 0, load_item->output[0]->parent->buffer_size);

        nn_workload_data_coords_t start_coords(
            0,
            input_data->view_begin.t[NN_DATA_COORD_x] + padding_left,
            input_data->view_begin.t[NN_DATA_COORD_y] + padding_top,
            0,
            0,
            0);

        nn_workload_data_coords_t end_coords(
            input_data->parent->lengths.t[0] - 1,
            input_data->view_end.t[NN_DATA_COORD_x] + padding_left,
            input_data->view_end.t[NN_DATA_COORD_y] + padding_top,
            z_size - 1,
            input_data->parent->lengths.t[4] - 1,
            input_data->parent->lengths.t[5] - 1
            );

        load_item->output[0]->view_begin = start_coords;
        load_item->output[0]->view_end = end_coords;

        uint32_t x_position = 0, y_position = 0, z_position = 0;

        // pin to the input buffers
        for (int index = 0; index < flow_item->input_count; index++)
        {
            auto& input_indexed = load_item->input[index];

            input_data = input_indexed.item->output[input_indexed.index];

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
                start_coords.t[1] = 0;
                start_coords.t[2] = 0;
                end_coords.t[1] = x_size - 1;
                end_coords.t[2] = y_size - 1;
                start_coords.t[3] = z_position;
                z_position += input_data->parent->lengths.t[3];
                end_coords.t[3] = z_position - 1;
            }

            delete input_indexed.item->output[input_indexed.index];
            nn_workload_data_t *merge_output = load_item->output[0];
            auto &input_items_output = input_indexed.item->output[input_indexed.index];
            input_items_output = nn_workload_data_create_view(merge_output, &start_coords, &end_coords);
            assert(input_items_output != nullptr);
        }

        break;
    }
    case NN_WORK_ITEM_TYPE_NORMALIZATION:
    case NN_WORK_ITEM_TYPE_POOLING:
        if(flow_item->output_count == 2)
        {   // These layers needs intermediate output during learning.
            load_item->output[1] = load_item->primitive->create_outputs()[0];
        }
    case NN_WORK_ITEM_TYPE_ARITHMETIC:
    case NN_WORK_ITEM_TYPE_CONVOLUTION:
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2:
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
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT:
    {
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
    case NN_WORK_ITEM_TYPE_SOFTMAX_LOSS_BACKPROP:
    case NN_WORK_ITEM_TYPE_NORMALIZATION_BACKPROP:
    case NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP:{
        auto inputs = load_item->forward_item->primitive->create_inputs();
        load_item->output[0] = inputs[0];

        // Cleanup all unnecessary buffers that were created.
        for(uint32_t input_index = 1; input_index < inputs.size(); ++input_index)
            delete inputs[input_index];
        break;
    }
    case NN_WORK_ITEM_TYPE_SOFTMAX_LOSS:
    {
        auto outputs = load_item->primitive->create_outputs();
        load_item->output[0] = outputs[0];
        load_item->output[1] = outputs[1];
        break;
    }
    case NN_WORK_ITEM_TYPE_LOSS_FUNCTION:
    {
        load_item->output[0] =
            new nn::workload_data<>(NN_WORKLOAD_DATA_TAG_UNKNOWN,
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
                new nn::workload_data<>(NN_WORKLOAD_DATA_TAG_UNKNOWN, size, input->parent->layout);
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

    if( NN_WORK_ITEM_TYPE_INPUT == flow_item->type ) {
        // If input is connected directly to merge layer switch copy flag
        if( NN_WORK_ITEM_TYPE_MERGE == flow_item->use[0].item->type )
            load_item->arguments.input.copy_on_merge = true;
        else
            load_item->arguments.input.copy_on_merge = false;
    }

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

}

template <typename T_FlowUpdateFunc>
void add_conversions_to_batch_block(
    nn_workload_item_t *load_item,
    T_FlowUpdateFunc update_flow,
    uint32_t batch_size,
    nn_device_internal* device)
{
    auto create_to_batch_block_conversion = [&](nn_workload_use_descriptor input,
                                                uint32_t x_size,
                                                uint32_t y_size,
                                                uint32_t z_size)
        {
            auto ret = new nn_workload_item_t();
            ret->primitive = new layer::convert_from_zxyn_to_batch_block_format_nzxyn(
                        batch_size, x_size, y_size, z_size, device);
            ret->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            ret->name = "convert_to_batch_block_before_convolution";

            ret->output.push_back(
                nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::create(
                    batch_size,
                    x_size,
                    y_size,
                    z_size,
                    BATCH_ACCEPTED_BLOCK));
            ret->input = decltype(ret->input)({input});
            return ret;
        };

    //input conversions
    for (auto& input_descr : load_item->input)
    {
        auto input_data = input_descr.get_data_view();
        if (input_data->parent->layout == (nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::layout))
            continue;
        if (input_data->parent->layout != (nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::layout))
            throw std::runtime_error("implementation error: invalid input data layout for conversion to batch block layout");
        auto conversion = create_to_batch_block_conversion(
            input_descr,
            get_length(*input_data, NN_DATA_COORD_x),
            get_length(*input_data, NN_DATA_COORD_y),
            get_length(*input_data, NN_DATA_COORD_z));

        update_flow(conversion, load_item);
        input_descr.item = conversion;
        input_descr.index = 0;
    }
}

template <typename T_FlowUpdateFunc, typename T_UsedByCont>
void add_conversions_from_batch_block(
    nn_workload_item_t *load_item,
    T_FlowUpdateFunc update_flow,
    const T_UsedByCont& used_by,
    uint32_t batch_size,
    nn_device_internal* device)
{
    auto create_from_batch_block_conversion = [&](nn_workload_data_layout_t output_layout,
                                                  uint32_t x_size,
                                                  uint32_t y_size,
                                                  uint32_t z_size)
        {
            auto ret = new nn_workload_item_t();
            ret->primitive = new layer::convert_from_batch_block_format_to_zxyn(
                        batch_size, x_size, y_size, z_size, device);
            ret->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            ret->name = "convert_from_batch_block_after_convolution";

            if (output_layout == nn::layout_t<nn::layout_zxyn_f32>::layout)
            {
                ret->output.push_back(
                    nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::create(
                        nullptr,
                        x_size,
                        y_size,
                        z_size,
                        batch_size,
                        0, 0, 0, 0, false));
            }
            else if (output_layout == nn::layout_t<nn::layout_nx_f32>::layout)
            {
                ret->output.push_back(
                    nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::create(
                        nullptr,
                        x_size * y_size * z_size,
                        batch_size));
            }
            else if (output_layout == nn::layout_t<nn::layout_xyzpqn_f32>::layout)
            {
                auto output_data_buffer = new nn::workload_data<>(
                    NN_WORKLOAD_DATA_TAG_UNKNOWN,
                    nn_workload_data_coords_t(batch_size, x_size * y_size * z_size, 1, 1, 1, 1),
                    nn::layout_t<nn::layout_xyzpqn_f32>::layout);
                ret->output.push_back(output_data_buffer);
            }
            ret->input = decltype(ret->input)({{load_item, 0}});
            return ret;
        };
    //output conversions
    for (auto next : used_by)
    {
        if (next->type == NN_WORK_ITEM_TYPE_CONVOLUTION) continue;
        if (next->type == NN_WORK_ITEM_TYPE_MERGE) continue;
        auto it = std::find_if(next->input.begin(), next->input.end(),
            [&](nn_workload_use_descriptor descr){ return descr.item == load_item; });
        assert(it != next->input.end());
        auto output_data = it->get_data_view();
        if (next->output[0]->parent->layout == (nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::layout))
            continue;

        auto conversion = create_from_batch_block_conversion(
            next->output[0]->parent->layout,
            get_length(*output_data, NN_DATA_COORD_x),
            get_length(*output_data, NN_DATA_COORD_y),
            get_length(*output_data, NN_DATA_COORD_z));

        update_flow(conversion, next);
        it->item = conversion;
        it->index = 0;
    }
}

template <typename T_Flow>
void nn_workflow_compile_add_batch_block_conversions(
    nn_workload_item_t* load_item,
    uint32_t            batch,
    T_Flow&             all_items,
    nn_device_internal* device)
{
    auto update_flow = [&](nn_workload_item_t* new_item, nn_workload_item_t* used_by) {
            typedef typename T_Flow::value_type FlowElemType;
            auto it = std::find_if(all_items.begin(), all_items.end(),
                                   [&](FlowElemType elem){ return elem.first == used_by; });
            all_items.insert(it, std::make_pair(new_item, nullptr));
        };

    std::vector<nn_workload_item_t*> used_by;
    for (auto elem : all_items)
        for (auto input_descr : elem.first->input)
            if (input_descr.item == load_item)
                used_by.push_back(elem.first);
    if (batch % BATCH_ACCEPTED_BLOCK == 0)
    {
        if (load_item->output.empty()) return;
        if (load_item->output.front()->parent->layout == nn::data_helper<NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, nn::layout_nblockzxyn_f32>::layout)
        {
            if (load_item->type != NN_WORK_ITEM_TYPE_MERGE)
                add_conversions_to_batch_block(load_item, update_flow, batch, device);
            add_conversions_from_batch_block(load_item, update_flow, used_by, batch, device);
        }
    }
}

template <typename T_Flow>
void nn_workflow_compile_0_function_add_conversions(
    nn_workload_item_t *load_item,
    nn_workflow_item_t *flow_item,
    uint32_t            batch,
    NN_WORKLOAD_DATA_TYPE  *input_format,   /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE  *output_format,  /* array containing formats of outputs */
    T_Flow& all_items
    ) {

    auto update_flow = [&](nn_workload_item_t* new_item, nn_workload_item_t* used_by) {
            typedef typename T_Flow::value_type FlowElemType;
            auto it = std::find_if(all_items.begin(), all_items.end(),
                                   [&](FlowElemType elem){ return elem.first == used_by; });
            all_items.insert(it, std::make_pair(new_item, nullptr));
        };
    auto update_load = [&](nn_workload_item_t* conversion, nn_workload_item_t* next_load_item) {
            update_flow(conversion, next_load_item);
            for (auto &useinput : next_load_item->input)
                if (useinput.item == load_item)
                    useinput.item = conversion;
        };

    std::vector<nn_workload_item_t*> used_by;
    for (auto elem : all_items)
        for (auto input_descr : elem.first->input)
            if (input_descr.item == load_item)
                used_by.push_back(elem.first);

    if (batch > 1)
    {
        // add type2 conversions
        auto init_type2_conversion = [&batch, &flow_item, &load_item](std::string before) {
            auto conversion = new nn_workload_item_t;
            conversion->primitive = nullptr;
            conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            conversion->arguments.convert_data_layout.type = 2;
            conversion->name = "convert_layout2_before_"+ before;

            auto layout = nn::layout_t<nn::layout_nxyzpq_i32>::layout;

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
            for (auto &next_load_item : used_by) {
                if (next_load_item->type == NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT)
                    update_load(init_type2_conversion(next_load_item->name), next_load_item);
            }
        }

        // add type3 conversions
        auto init_type3_conversion = [&batch, &flow_item, &load_item](std::string before) {
            auto conversion = new nn_workload_item_t;
            conversion->primitive = new layer::convert_z_block_xyz_z2nz(
                        get_format_size<0>(flow_item->output_format[0]),
                        get_format_size<1>(flow_item->output_format[0]),
                        get_format_size<2>(flow_item->output_format[0]),
                        batch,
                        nullptr);
            conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            conversion->arguments.convert_data_layout.type = 3;
            conversion->name = "convert_layout3_before_"+ before;

            nn_workload_data_layout_t layout = nn::layout_t<nn::layout_pnzxyq_i16>::layout;

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

        if (load_item->output.size() > 0 && load_item->output[0] != nullptr && load_item->output[0]->parent->layout.ordering.t[1] != NN_DATA_COORD_n)
        {
            for (auto &next_load_item : used_by)
            {
                if (next_load_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN ||
                    next_load_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN)
                {
                    update_load(init_type3_conversion(next_load_item->name), next_load_item);
                }
            }
        }
    }
    else { // batch == 1
        // add type0 conversions
        auto init_type0_conversion = [&batch, &flow_item, &load_item](std::string before) {
            auto conversion = new nn_workload_item_t;
            conversion->primitive = new layer::convert_z_block_xyz_z2nz(
                        get_format_size<0>(flow_item->output_format[0]),
                        get_format_size<1>(flow_item->output_format[0]),
                        get_format_size<2>(flow_item->output_format[0]),
                        batch,
                        nullptr);
            conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            conversion->arguments.convert_data_layout.type = 0;
            conversion->name = "convert_layout0_before_"+ before;

            nn_workload_data_layout_t layout = nn::layout_t<nn::layout_pnzxyq_i16>::layout;

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
            for (auto &next_load_item : used_by) {
                if (next_load_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN ||
                    next_load_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN)
                {
                    update_load(init_type0_conversion(next_load_item->name), next_load_item);
                }
            }
        }
    }

    // Add type4 conversions - float batch conversion (...n to n... OR n... to ...n)
    auto init_type4_conversion = [&flow_item, &load_item] (nn_workload_item_t* next_load_item, uint32_t type, uint32_t batch)
    {
        auto conversion = new nn_workload_item_t;
        conversion->primitive = nullptr;
        conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
        conversion->arguments.convert_data_layout.type = 4;
        conversion->name = std::string("convert_layout4_before_") + next_load_item->name;

        auto next_load_item_layout = next_load_item->output[0]->parent->layout;

        nn_workload_data_coords_t new_lengths;

        if(type == 0)
        {
            // Standard conversion.
            new_lengths = load_item->output[0]->parent->lengths;
        }
        else if(type == 1)
        {
            // Squashing conversion.
            new_lengths =
            {
                load_item->output[0]->parent->lengths.t[NN_DATA_COORD_n],
                load_item->output[0]->parent->buffer_size / 4 / load_item->output[0]->parent->lengths.t[NN_DATA_COORD_n],
                1,
                1,
                1,
                1
            };
        }
        else if(type == 2)
        {
            // Un-squashing conversion. Special case, can occur only in backward passes.
            conversion->arguments.convert_data_layout.type = 7;
            if(next_load_item->forward_item == nullptr)
                throw std::runtime_error("add_conversion: unsquashing on forward passes occured");

            new_lengths =
            {
                next_load_item->forward_item->output[0]->parent->lengths.t[NN_DATA_COORD_n],
                next_load_item->forward_item->output[0]->parent->lengths.t[NN_DATA_COORD_x],
                next_load_item->forward_item->output[0]->parent->lengths.t[NN_DATA_COORD_y],
                next_load_item->forward_item->output[0]->parent->lengths.t[NN_DATA_COORD_z],
                next_load_item->forward_item->output[0]->parent->lengths.t[NN_DATA_COORD_p],
                next_load_item->forward_item->output[0]->parent->lengths.t[NN_DATA_COORD_q]
            };
        }

        if(batch != 1)
        {
            // Actual conversion takes place - new buffer.
            conversion->output.push_back(new nn::workload_data<>(
                NN_WORKLOAD_DATA_TAG_UNKNOWN,
                new_lengths,
                next_load_item_layout));
        }
        else
        {
            // Just add view for old buffer.
            conversion->output.push_back(new nn::workload_data<>(
                load_item->output[0]->parent->data_buffer,
                new_lengths,
                next_load_item_layout));
        }

        conversion->input.push_back({load_item, 0});
        return conversion;
    };

    if (load_item->output.size() > 0 &&
        load_item->output[0] != nullptr)
    {
        auto load_item_layout = load_item->output[0]->parent->layout;
        auto data_batch = load_item->output[0]->parent->lengths.t[NN_DATA_COORD_n];

        for (auto &next_load_item : used_by)
        {
            if (next_load_item->output.size() > 0 &&
                next_load_item->type != NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT &&
                data_batch == next_load_item->output[0]->parent->lengths.t[NN_DATA_COORD_n])
            {
                auto next_item_layout = next_load_item->output[0]->parent->layout;

                nn_workload_item_t* type4_conversion = nullptr;
                //nn::layout_t<nn::layout_nxyzpq_f32>::layout
                // Check if its one of allowed conversions.
                if((load_item_layout == nn::layout_t<nn::layout_zxynpq_f32>::layout && next_item_layout == nn::layout_t<nn::layout_nzxypq_f32>::layout) || // ZXY-N -> N-ZXY
                   (load_item_layout == nn::layout_t<nn::layout_xyzpqn_f32>::layout && next_item_layout == nn::layout_t<nn::layout_nxyzpq_f32>::layout) || // XYZ-N -> N-XYZ
                   (load_item_layout == nn::layout_t<nn::layout_nzxypq_f32>::layout && next_item_layout == nn::layout_t<nn::layout_zxynpq_f32>::layout) || // N-ZXY -> ZXY-N
                   (load_item_layout == nn::layout_t<nn::layout_nxyzpq_f32>::layout && next_item_layout == nn::layout_t<nn::layout_xyzpqn_f32>::layout))   // N-XYZ -> XYZ-N
                {
                    // Standard conversion.
                    type4_conversion = init_type4_conversion(next_load_item, 0, data_batch);
                }
                else if(load_item_layout == nn::layout_t<nn::layout_zxynpq_f32>::layout && next_item_layout == nn::layout_t<nn::layout_nxyzpq_f32>::layout) // ZXY-N -> N-X
                {
                    // Squashing conversion.
                    type4_conversion = init_type4_conversion(next_load_item, 1, data_batch);
                }
                else if(load_item_layout == nn::layout_t<nn::layout_nxyzpq_f32>::layout && next_item_layout == nn::layout_t<nn::layout_zxynpq_f32>::layout) // N-X -> ZXY-N
                {
                    // Un-squashing conversion.
                    type4_conversion = init_type4_conversion(next_load_item, 2, data_batch);
                }

                if(type4_conversion != nullptr)
                    update_load(type4_conversion, next_load_item);
            }
        }
    }

    auto init_type5_conversion = [&batch, &flow_item, &load_item]
    {
        auto conversion = new nn_workload_item_t;
        conversion->primitive = nullptr;
        conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
        conversion->arguments.convert_data_layout.type = 5;

        nn_workload_data_layout_t layout = nn::layout_t<nn::layout_pxyznq_i16>::layout;

        uint32_t z_size = flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1;
        uint32_t z_block = z_size > 4 ? 16 : 4;

        nn_workload_data_coords_t size(
            batch,
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            (z_size - 1) / z_block + 1,
            z_block,
            1 );

        conversion->output.push_back(new nn::workload_data<>(NN_WORKLOAD_DATA_TAG_UNKNOWN, size, layout));
        conversion->input.push_back({ load_item, 0 });
        return conversion;
    };

    if (load_item->type == NN_WORK_ITEM_TYPE_INPUT) {
        for (auto &next_load_item : used_by) {
            if (next_load_item->type == NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN ||
                next_load_item->type == NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT ||
                next_load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT ||
                next_load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT) {
                assert(input_format[load_item->arguments.input.index] == NN_WORKLOAD_DATA_TYPE_I16_ZXY); // TODO support other formats
                auto type5_conversion = init_type5_conversion();
                type5_conversion->name = std::string("convert_layout5_before_") + next_load_item->name;

                update_load(type5_conversion, next_load_item);
            }
        }
    }

    auto init_type6_conversion = [&batch, &flow_item, &load_item]
    {
        auto conversion = new nn_workload_item_t;
        conversion->primitive = nullptr;
        conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
        conversion->arguments.convert_data_layout.type = 6;

        nn_workload_data_layout_t layout = nn::layout_t<nn::layout_zxynpq_i16>::layout;

        nn_workload_data_coords_t size(
            batch,
            flow_item->output_format[0].format_1d.size[0],
            flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
            flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1,
            1,
            1 );

        conversion->output.push_back(new nn::workload_data<>(NN_WORKLOAD_DATA_TAG_UNKNOWN, size, layout));
        conversion->input.push_back({ load_item, 0 });
        return conversion;
    };

    if ((load_item->type == NN_WORK_ITEM_TYPE_MERGE && load_item->input[0].get_data_view()->parent->layout.data_type == NN_DATATYPE_INT16) ||
        load_item->type == NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN ||
        load_item->type == NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT ||
        load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT ||
        load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT) {
        for (auto &next_load_item : used_by) {
            if (next_load_item->type == NN_WORK_ITEM_TYPE_OUTPUT) {
                auto type6_conversion = init_type6_conversion();
                type6_conversion->name = std::string("convert_layout6_before_") + next_load_item->name;

                update_load(type6_conversion, next_load_item);
            }
        }
    }
}


std::vector<nn_workflow_item_t*> all_items(nn_workflow_t* workflow)
{
    typedef std::vector<nn_workflow_item_t*> WorkflowVec;
    typedef std::unordered_set<nn_workflow_item_t*> WorkflowSet;

    WorkflowVec all(workflow->input, workflow->input + workflow->input_count);
    WorkflowSet visited(all.begin(), all.end());

    int count = 0;
    while (count < all.size())
    {
        auto curr = all[count];
        auto push_elem = [&](nn_workflow_item_t* item) {
                if (visited.count(item) != 0) return;
                all.push_back(item);
                visited.insert(item);
            };

        for(auto index = 0u; index < curr->use_count; ++index)
            push_elem(curr->use[index].item);
        ++count;
    }
    return all;
}

std::vector<nn_workflow_item_t*> flow_items_in_execution_order(nn_workflow_t* workflow)
{
    auto todo = all_items(workflow);
    std::unordered_map<nn_workflow_item_t*, std::vector<nn_workflow_item_t*>> inputs;
    for (auto item : todo)
        for (auto i = 0u; i < item->use_count; ++i)
            inputs[item->use[i].item].push_back(item);

    decltype(todo) ret;
    while(!todo.empty())
    {
        auto is_not_done = [todo](nn_workflow_item_t* item) {
                return todo.end() != std::find(todo.begin(), todo.end(), item);
            };
        auto it = std::remove_if(todo.begin(), todo.end(),
            [&](nn_workflow_item_t* item) {
                auto all_inputs_done =
                    (0 == std::count_if(inputs[item].begin(), inputs[item].end(), is_not_done));
                if (all_inputs_done) ret.push_back(item);
                return all_inputs_done;
            });
        todo.erase(it, todo.end());
    }
    return ret;
}

std::string str(NN_WORK_ITEM_TYPE type)
{
    switch (type)
    {
    case NN_WORK_ITEM_TYPE_INPUT: return "NN_WORK_ITEM_TYPE_INPUT";
    case NN_WORK_ITEM_TYPE_OUTPUT: return "NN_WORK_ITEM_TYPE_OUTPUT";
    case NN_WORK_ITEM_TYPE_VIEW: return "NN_WORK_ITEM_TYPE_VIEW";
    case NN_WORK_ITEM_TYPE_LOCAL_CONNECTIVITY: return "NN_WORK_ITEM_TYPE_LOCAL_CONNECTIVITY";
    case NN_WORK_ITEM_TYPE_CONVOLUTION: return "NN_WORK_ITEM_TYPE_CONVOLUTION";
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED: return "NN_WORK_ITEM_TYPE_FULLY_CONNECTED";
    case NN_WORK_ITEM_TYPE_POOLING: return "NN_WORK_ITEM_TYPE_POOLING";
    case NN_WORK_ITEM_TYPE_NORMALIZATION: return "NN_WORK_ITEM_TYPE_NORMALIZATION";
    case NN_WORK_ITEM_TYPE_SOFTMAX: return "NN_WORK_ITEM_TYPE_SOFTMAX";
    case NN_WORK_ITEM_TYPE_MERGE: return "NN_WORK_ITEM_TYPE_MERGE";
    case NN_WORK_ITEM_TYPE_ARITHMETIC: return "NN_WORK_ITEM_TYPE_ARITHMETIC";
    case NN_WORK_ITEM_TYPE_RELU: return "NN_WORK_ITEM_TYPE_RELU";
    case NN_WORK_ITEM_TYPE_RELU_1D: return "NN_WORK_ITEM_TYPE_RELU_1D";
    case NN_WORK_ITEM_TYPE_RELU_3D: return "NN_WORK_ITEM_TYPE_RELU_3D";
    case NN_WORK_ITEM_TYPE_SOFTMAX_LOSS: return "NN_WORK_ITEM_TYPE_SOFTMAX_LOSS";
    case NN_WORK_ITEM_TYPE_SOFTMAX_LOSS_BACKPROP: return "NN_WORK_ITEM_TYPE_SOFTMAX_LOSS_BACKPROP";
    case NN_WORK_ITEM_TYPE_AVERAGE_DELTAS: return "NN_WORK_ITEM_TYPE_AVERAGE_DELTAS";
    case NN_WORK_ITEM_TYPE_DROPOUT: return "NN_WORK_ITEM_TYPE_DROPOUT";
    case NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP: return "NN_WORK_ITEM_TYPE_DROPOUT_BACKPROP";
    case NN_WORK_ITEM_TYPE_LOSS_FUNCTION: return "NN_WORK_ITEM_TYPE_LOSS_FUNCTION";
    case NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS: return "NN_WORK_ITEM_TYPE_UPDATE_ARGUMENTS";
    case NN_WORK_ITEM_TYPE_RELU_BACKPROP: return "NN_WORK_ITEM_TYPE_RELU_BACKPROP";
    case NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP: return "NN_WORK_ITEM_TYPE_CONVOLUTION_BACKPROP";
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP: return "NN_WORK_ITEM_TYPE_FULLY_CONNECTED_BACKPROP";
    case NN_WORK_ITEM_TYPE_POOLING_BACKPROP: return "NN_WORK_ITEM_TYPE_POOLING_BACKPROP";
    case NN_WORK_ITEM_TYPE_NORMALIZATION_BACKPROP: return "NN_WORK_ITEM_TYPE_NORMALIZATION_BACKPROP";
    case NN_WORK_ITEM_TYPE_SOFTMAX_BACKPROP: return "NN_WORK_ITEM_TYPE_SOFTMAX_BACKPROP";
    case NN_WORK_ITEM_TYPE_RELU_1D_BACKPROP: return "NN_WORK_ITEM_TYPE_RELU_1D_BACKPROP";
    case NN_WORK_ITEM_TYPE_RELU_3D_BACKPROP: return "NN_WORK_ITEM_TYPE_RELU_3D_BACKPROP";
    case NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT: return "NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT";
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN: return "NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN";
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN: return "NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN";
    case NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT: return "NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT";
    case NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT: return "NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT";
    case NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT: return "NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT";
    case NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN: return "NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN";
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: return "NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2";
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT: return "NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT";
    case NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT: return "NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT";
    default: return "unknown";
    }
}

template <typename T_Flow>
void nn_merge_layer_if_beneficial(
    nn_workload_item_t* load_item,
    T_Flow&             all_items)
{
    if (load_item->type != NN_WORK_ITEM_TYPE_ARITHMETIC) return;

    std::vector<nn_workload_item_t*> used_by;
    for (auto elem : all_items)
        for (auto input_descr : elem.first->input)
            if (input_descr.item == load_item)
                used_by.push_back(elem.first);

    if (used_by.size() != 1) return;
    auto next_item = used_by.front();

    if (next_item->type != NN_WORK_ITEM_TYPE_CONVOLUTION) return;

    auto conv_primitive = dynamic_cast<layer::convolution_f32*>(next_item->primitive);
    if (not conv_primitive) return;
    if (not conv_primitive->uses_data_only()) return;

    auto arith_primitive = dynamic_cast<layer::arithmetic_f32*>(load_item->primitive);
    if (not arith_primitive) return;
    if (not arith_primitive->is_linear()) return;

    auto translated_params = arith_primitive->get_input_feat_periodic(
        {load_item->parameters.begin(), load_item->parameters.end()});
    
    if (translated_params.empty()) return;

    conv_primitive->update_bias_by_linear_factors(
        translated_params,
        {next_item->parameters.begin(), next_item->parameters.end()});

    auto it = std::find_if(all_items.begin(), all_items.end(),
        [&](decltype(all_items.front()) elem){ return elem.first == load_item; });
    assert(it != all_items.end());

    next_item->input = load_item->input;
    all_items.erase(it);
}

} //namespace

/* compile workflow into workload */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_compile_0_function(
    nn_workload_t         **workload,       /* resulting workload */
    nn_device_t            *device,         /* device context */
    nn_workflow_t          *workflow,       /* workflow to be compiled */
    NN_WORKLOAD_DATA_TYPE  *input_format,   /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE  *output_format,  /* array containing formats of outputs */
    uint32_t                batch           /* batch size for compilation */
    )
{
    if(!workload || !device || !workflow) return NN_API_STATUS_ERROR_INVALID_POINTER;

    for(auto index=0u; index<workflow->input_count; ++index)
        if(workflow->input[index] ->type!=NN_WORK_ITEM_TYPE_INPUT)
            return NN_API_STATUS_ERROR_INVALID_WORKFLOW; // TODO: more granular error code here
    for(auto index=0u; index<workflow->output_count; ++index)
        if(workflow->output[index]->type!=NN_WORK_ITEM_TYPE_OUTPUT)
            return NN_API_STATUS_ERROR_INVALID_WORKFLOW; // TODO: more granular error code here
    try
    {
        nn_workload_t aux = {
            device,
            workflow->input_count,
            workflow->output_count,
            new NN_WORKLOAD_DATA_TYPE[workflow->input_count],
            new NN_WORKLOAD_DATA_TYPE[workflow->output_count],
            batch
        };
        auto workload_opaque = new nn_workload_opaque_t(aux);
        std::copy(input_format, input_format + workflow->input_count, workload_opaque->input_format);
        std::copy(output_format, output_format + workflow->output_count, workload_opaque->output_format);

        auto whole_flow = flow_items_in_execution_order(workflow);
        std::vector<std::pair<nn_workload_item_t*, nn_workflow_item_t*>> flow_and_load;
        std::transform(whole_flow.begin(), whole_flow.end(), std::back_inserter(flow_and_load),
            [](nn_workflow_item_t* flow_item){
                return std::make_pair(new nn_workload_item_t(), flow_item); });

        auto find_load_item = [&](nn_workflow_item_t* flow_item) -> nn_workload_item_t* {
                auto it = std::find_if(flow_and_load.begin(), flow_and_load.end(),
                    [&](decltype(flow_and_load.front()) elem){ return elem.second == flow_item; });
                if (it == flow_and_load.end()) return nullptr;
                return it->first;
            };
        for (auto& elem : flow_and_load)
            nn_workflow_compile_0_function_copy_item(
                elem.first,
                elem.second,
                workflow,
                workload_opaque,
                batch,
                find_load_item,
                reinterpret_cast<nn_device_internal*>(device));
        
        auto flow_and_load_merged = flow_and_load;
        for (auto& elem : flow_and_load)
            nn_merge_layer_if_beneficial(elem.first, flow_and_load_merged);

        auto flow_and_load_with_conversions = flow_and_load_merged;
        for (auto& elem : flow_and_load_merged)
            nn_workflow_compile_add_batch_block_conversions(
                elem.first, batch, flow_and_load_with_conversions, reinterpret_cast<nn_device_internal*>(device));

        auto flow_and_load_with_all_conversions = flow_and_load_with_conversions;
        for (auto& elem : flow_and_load_with_conversions)
            nn_workflow_compile_0_function_add_conversions(
                elem.first, elem.second, batch, input_format, output_format, flow_and_load_with_all_conversions);


        for(auto index = 0u; index < workflow->input_count; ++index)
            workload_opaque->input.push_back(find_load_item(workflow->input[index]));

        for(auto index = 0u; index < workflow->output_count; ++index)
            workload_opaque->output.push_back(find_load_item(workflow->output[index]));

        std::transform(flow_and_load_with_all_conversions.begin(), flow_and_load_with_all_conversions.end(),
                       std::back_inserter(workload_opaque->order_of_execution),
                       [](std::pair<nn_workload_item_t*, nn_workflow_item_t*> elem){ return elem.first; });

        // Now create param list.
        for(auto item : workload_opaque->order_of_execution)
        {
            auto save_param = [item, &workload_opaque](uint32_t param_id)
            {
                workload_opaque->params.push_back(
                    nn_workload_params
                    {
                        nullptr,
                        item->type,
                        (uint32_t)workload_opaque->param_sizes.back().size(),
                        nullptr,
                        item->parameters[param_id]
                    });
            };

            switch(item->type)
            {
            case NN_WORK_ITEM_TYPE_CONVOLUTION:
            {
                // Create weight data.
                workload_opaque->param_sizes.push_back(
                {
                    item->parameters[0]->parent->lengths.t[NN_DATA_COORD_x],
                    item->parameters[0]->parent->lengths.t[NN_DATA_COORD_y],
                    item->parameters[0]->parent->lengths.t[NN_DATA_COORD_z],
                    item->parameters[0]->parent->lengths.t[NN_DATA_COORD_p] * item->parameters[0]->parent->lengths.t[NN_DATA_COORD_q]
                });
                workload_opaque->param_names.push_back(item->name + "_weights");
                save_param(0);

                // Create bias data.
                workload_opaque->param_sizes.push_back(
                {
                    item->parameters[1]->parent->lengths.t[NN_DATA_COORD_x]
                });
                workload_opaque->param_names.push_back(item->name + "_biases");
                save_param(1);

                break;
            }
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED:
            {
                // Create weight data.
                auto primitive = static_cast<layer::fully_connected_f32*>(item->primitive);

                // Check if it had 3D or 1D input size.
                if(primitive->get_has_3d_input())
                {
                    workload_opaque->param_sizes.push_back(
                    {
                        primitive->get_input_size_x(),
                        primitive->get_input_size_y(),
                        primitive->get_input_size_z(),
                        primitive->get_output_size()
                    });
                }
                else
                {
                    workload_opaque->param_sizes.push_back(
                    {
                        primitive->get_input_size(),
                        primitive->get_output_size()
                    });
                }
                workload_opaque->param_names.push_back(item->name + "_weights");
                save_param(0);

                // Create bias data.
                workload_opaque->param_sizes.push_back(
                {
                    item->parameters[1]->parent->lengths.t[NN_DATA_COORD_x]
                });
                workload_opaque->param_names.push_back(item->name + "_biases");
                save_param(1);
                break;
            }
            default:
                break;
            }
        }

        // Update pointers in C-like param structure.
        uint32_t index = 0;
        for(auto& param : workload_opaque->params)
        {
            param.name = workload_opaque->param_names[index].c_str();
            param.sizes = workload_opaque->param_sizes[index].data();

            ++index;
        }

        //call prepare_forward
        for(auto item : workload_opaque->order_of_execution)
        {
            if (not item->primitive) continue;
            std::vector<const nn_workload_data_t *> inputs;
            for(auto& input_descriptor : item->input)
                inputs.push_back(input_descriptor.get_data_view());
            std::vector<const nn_workload_data_t *> params(item->parameters.begin(), item->parameters.end());
            item->primitive->prepare_forward(inputs, params, item->output);
        }

        // set result
        *workload = workload_opaque;
    }
    catch( std::exception &e )
    {
        std::cerr << e.what();
        throw;
    }
    catch (...)
    {
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

    if (!workload_public || !input || (!output && workload_public->output_count))
        return NN_API_STATUS_ERROR_INVALID_POINTER;
    else {
        try {
            *status = NN_API_WORK_IN_PROGRESS;
            auto workload_opaque = static_cast<nn_workload_opaque_t *>(workload_public);
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

                    // If you want inputs to be copied don't assign them to output (buffers will override each other)
                    // This situation take place only when merge is next layer after inputs
                    if( !item->arguments.input.copy_on_merge ) {
                        // If after input there isn't merge, do standard way
                        item->output[0]->parent->data_buffer = item_input->buffer;  // TODO validate if input have same lengths and layout as one given during compilation.
                    }
                    break;
                }
                case NN_WORK_ITEM_TYPE_OUTPUT: {
                    // Copy result to workload output.
                    auto item_output = reinterpret_cast<nn_data_t*>(output[item->arguments.output.index]);

                    item->output[0]->parent->data_buffer = item_output->buffer;
                    nn_workload_data_copy(item->output[0], item->input[0].get_data_view());
                    break;
                }
                case NN_WORK_ITEM_TYPE_MERGE: {
                    uint32_t z_offset = 0;

                    for( uint32_t i = 0; i < item->input.size() ; ++i ) {
                        if( NN_WORK_ITEM_TYPE_INPUT == item->input[i].item->type
                        && item->input[i].item->arguments.input.copy_on_merge
                        ) {

                            // get input buffer from inputs table
                            auto item_input = reinterpret_cast< nn_data_t* >(input[item->input[i].item->arguments.input.index]);

                            auto dim = item_input->dimension;
                            assert( 4 == dim );
                            auto size_ptr = item_input->size;
                            uint32_t input_buf_x  = static_cast<uint32_t>( *(size_ptr + 1)),
                                     input_buf_y  = static_cast<uint32_t>( *(size_ptr + 2)),
                                     input_buf_z  = static_cast<uint32_t>( *(size_ptr + 0)),
                                     input_buf_n  = static_cast<uint32_t>( *(size_ptr + 3));

                            auto destination = item->output[0];
                            auto source = item->input[i].item->output[0];

                            unsigned int source_sizes      [NN_DATA_COORD_MAX + 1],
                                         destination_sizes [NN_DATA_COORD_MAX + 1];

                            for( int i = 0 ; i < NN_DATA_COORD_MAX+1 ; ++i){
                                source_sizes[i]      = source->view_end.t[i]      - source->view_begin.t[i]      + 1;
                                destination_sizes[i] = destination->view_end.t[i] - destination->view_begin.t[i] + 1;
                            }

                            // check if destination size is >= for z merge
                            assert(    destination_sizes[NN_DATA_COORD_n] == source_sizes[NN_DATA_COORD_n]
                                    && destination_sizes[NN_DATA_COORD_x] == source_sizes[NN_DATA_COORD_x]
                                    && destination_sizes[NN_DATA_COORD_y] == source_sizes[NN_DATA_COORD_y]
                                    && destination_sizes[NN_DATA_COORD_z] >= source_sizes[NN_DATA_COORD_z]
                                    && destination_sizes[NN_DATA_COORD_p] == source_sizes[NN_DATA_COORD_p]
                                    && destination_sizes[NN_DATA_COORD_q] == source_sizes[NN_DATA_COORD_q] );

                            // check if z is not out of bounds
                            assert( destination->view_end.t[NN_DATA_COORD_z] + 1 >= source_sizes[NN_DATA_COORD_z] + z_offset );

                            // check if input_buffer's sizes are equal to source_sizes
                            assert(    input_buf_z == source_sizes[NN_DATA_COORD_z]
                                    && input_buf_x == source_sizes[NN_DATA_COORD_x]
                                    && input_buf_y == source_sizes[NN_DATA_COORD_y]
                                    && input_buf_n == source_sizes[NN_DATA_COORD_n] );

                            auto input_buf = static_cast< float* >(item_input->buffer);

                            for( uint32_t p = 0; p < source_sizes[NN_DATA_COORD_p]; p++ )
                            for( uint32_t q = 0; q < source_sizes[NN_DATA_COORD_q]; q++ )
                            for( uint32_t n = 0; n < source_sizes[NN_DATA_COORD_n]; n++ )
                            for( uint32_t z = 0; z < source_sizes[NN_DATA_COORD_z]; z++ )
                            for( uint32_t y = 0; y < source_sizes[NN_DATA_COORD_y]; y++ )
                            for( uint32_t x = 0; x < source_sizes[NN_DATA_COORD_x]; x++ ) {
                                auto tmp = input_buf[ z +
                                                        source_sizes[NN_DATA_COORD_z] * x +
                                                        source_sizes[NN_DATA_COORD_z] * source_sizes[NN_DATA_COORD_x] * y +
                                                        source_sizes[NN_DATA_COORD_z] * source_sizes[NN_DATA_COORD_x] * source_sizes[NN_DATA_COORD_y] * n +
                                                        source_sizes[NN_DATA_COORD_z] * source_sizes[NN_DATA_COORD_x] * source_sizes[NN_DATA_COORD_y] * source_sizes[NN_DATA_COORD_n] * p +
                                                        source_sizes[NN_DATA_COORD_z] * source_sizes[NN_DATA_COORD_x] * source_sizes[NN_DATA_COORD_y] * source_sizes[NN_DATA_COORD_n] * source_sizes[NN_DATA_COORD_p] * q ];

                                nn_workload_data_get<float>( destination, n, x, y, z + z_offset, p, q ) = tmp;
                            }

                            z_offset += source_sizes[NN_DATA_COORD_z];
                          }
                    }
                    break;
                }
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
                case NN_WORK_ITEM_TYPE_SOFTMAX_LOSS_BACKPROP: {
                    layer::run_softmax_loss_backward(item);
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
                    layer::run_parameter_update(item, static_cast<nn_device_internal*>(workload_public->device));
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
        }
        catch(NN_API_STATUS status)
        {
            return status;
        }
        catch( std::exception &e )
        {
            std::cerr << e.what();
            throw;
        }
        catch(...)
        {
            return NN_API_STATUS_ERROR_OUT_OF_MEMORY;
        }
    }
    return NN_API_STATUS_OK;
}

#if ENABLE_WORKLOAD_PROFILING
void nn_workload_print_profiling_data(nn_workload_opaque_t* workload_opaque)
{
    printf("\n------------profiling--------------\n");
    printf("%20s %20s %10s %10s\n", "layer_type", "layer_name", "cycles", "");
    printf("%20s %20s %10s %10s\n", "", "", "minimum", "average");

    assert(workload_opaque->order_of_execution.size() != 0); // Empty workload (?).

    for (const auto &item : workload_opaque->order_of_execution)
    {
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
            case NN_WORK_ITEM_TYPE_SOFTMAX_LOSS: return "softmax_loss";
            case NN_WORK_ITEM_TYPE_SOFTMAX_LOSS_BACKPROP: return "softmax_loss_backprop";
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
            auto workload_opaque = static_cast<nn_workload_opaque_t*>(workload_public);

#if ENABLE_WORKLOAD_PROFILING
            nn_workload_print_profiling_data(workload_opaque);
#endif

            for (auto& item : workload_opaque->order_of_execution)
            {
                for(auto& parameter : item->parameters)
                {
                    delete parameter;
                    parameter = nullptr;
                }

                for(auto& output : item->output)
                {
                    delete output;
                    output = nullptr;
                }

                delete item->primitive;
                item->primitive = nullptr;

                delete item;
                item = nullptr;
            }

            delete workload_opaque->input_format;
            delete workload_opaque->output_format;
            delete workload_opaque;
        }
        catch( std::exception &e ) {
            std::cerr << e.what();
            throw;
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

/* returns list of parameters found in workload along with their sizes */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workload_query_param_0_function(
    nn_workload_t          *workload_public, /* workload to be queried */
    nn_workload_params    **params,          /* returns list of parameters found in workload along with their sizes */
    uint32_t               *num_params       /* number of params returned */
    )
{
    nn_workload_opaque_t *workload_opaque = static_cast<nn_workload_opaque_t *>(workload_public);
    *params = &workload_opaque->params[0];
    *num_params = static_cast<uint32_t>(workload_opaque->params.size());

    return NN_API_STATUS_OK;
}

NN_API_STATUS NN_API_CALL_CONVENTION nn_workload_recover_param_0_function(
    nn_workload_t          *workload_public,  /* workload to be queried */
    char                   *param_name,       /* name of parameter to recover */
    nn_data                *data              /* data will be returned to this pointer */
    )
{
    auto workload_opaque = static_cast<nn_workload_opaque_t*>(workload_public);
    auto& data_wrapper = *static_cast<nn::data<float>*>(data);

    try
    {
        for(auto& param : workload_opaque->params)
        {
            if(strcmp(param_name,param.name) == 0)
            {   // Found item in table.
                std::string name(param.name);
                auto& internal_data_wrapper = *static_cast<nn::workload_data<nn::layout_f32>*>(param.handler);
                auto& internal_lengths = internal_data_wrapper.parent->lengths.t;

                if(name.find("_weights")!=std::string::npos)
                {   // Weights.
                    if(param.type == NN_WORK_ITEM_TYPE_CONVOLUTION)
                    {   // Convolution sliced weights.
                        auto slice_size = internal_lengths[NN_DATA_COORD_p];

                        for(uint32_t x = 0; x < data_wrapper.size[0]; ++x)
                            for(uint32_t y = 0; y < data_wrapper.size[1]; ++y)
                                for(uint32_t z = 0; z < data_wrapper.size[2]; ++z)
                                    for(uint32_t o = 0; o < data_wrapper.size[3]; ++o)
                                        data_wrapper(x, y, z, o) = internal_data_wrapper(0, x, y, z, o%slice_size, o/slice_size);
                    }
                    else if(param.type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED)
                    {   // Fully connected.
                        if(param.dimensions == 2)
                        {   // 1D.
                            if(workload_public->batch != 1)
                            {   // Sliced weights.
                                auto slice_size = internal_lengths[NN_DATA_COORD_p];

                                for(uint32_t x = 0; x < data_wrapper.size[0]; ++x)
                                    for(uint32_t y = 0; y < data_wrapper.size[1]; ++y)
                                        data_wrapper(x, y) = internal_data_wrapper(0, x, 0, 0, y%slice_size, y/slice_size);

                            }
                            else
                            {   // Non-sliced weights. Batch 1.
                                for(uint32_t x = 0; x < data_wrapper.size[0]; ++x)
                                    for(uint32_t y = 0; y < data_wrapper.size[1]; ++y)
                                        data_wrapper(x, y) = internal_data_wrapper(0, x, y, 0, 0, 0);
                            }
                        }
                        else if(param.dimensions == 4)
                        {   // 3D.
                            if(workload_public->batch != 1)
                            {   // Sliced weights.
                                auto slice_size = internal_lengths[NN_DATA_COORD_p];

                                // Create 3d view.
                                nn::workload_data<nn::layout_f32> internal_data_wrapper_3d_view(
                                    internal_data_wrapper.parent->data_buffer,
                                    {
                                        internal_lengths[NN_DATA_COORD_n],
                                        (uint32_t)data_wrapper.size[0],
                                        (uint32_t)data_wrapper.size[1],
                                        (uint32_t)data_wrapper.size[2],
                                        internal_lengths[NN_DATA_COORD_p],
                                        internal_lengths[NN_DATA_COORD_q]
                                    },
                                    nn::layout_t<nn::layout_pzxyqn_f32>::layout); // P-ZXY-Q -> P-I-Q

                                for(uint32_t x = 0; x < data_wrapper.size[0]; ++x)
                                    for(uint32_t y = 0; y < data_wrapper.size[1]; ++y)
                                        for(uint32_t z = 0; z < data_wrapper.size[2]; ++z)
                                            for(uint32_t o = 0; o < data_wrapper.size[3]; ++o)
                                                data_wrapper(x, y, z, o) = internal_data_wrapper_3d_view(0, x, y, z, o%slice_size, o/slice_size);
                            }
                            else
                            {   // Non-sliced weights. Batch 1.

                                // Create 3d view.
                                nn::workload_data<nn::layout_f32> internal_data_wrapper_3d_view(
                                    internal_data_wrapper.parent->data_buffer,
                                    {
                                        internal_lengths[NN_DATA_COORD_n],
                                        (uint32_t)data_wrapper.size[0],
                                        (uint32_t)data_wrapper.size[1],
                                        (uint32_t)data_wrapper.size[2],
                                        (uint32_t)data_wrapper.size[3],
                                        internal_lengths[NN_DATA_COORD_q]
                                    },
                                    nn::layout_t<nn::layout_pzxyqn_f32>::layout); // O-ZXY -> O-I

                                for(uint32_t x = 0; x < data_wrapper.size[0]; ++x)
                                    for(uint32_t y = 0; y < data_wrapper.size[1]; ++y)
                                        for(uint32_t z = 0; z < data_wrapper.size[2]; ++z)
                                            for(uint32_t o = 0; o < data_wrapper.size[3]; ++o)
                                                data_wrapper(x, y, z, o) = internal_data_wrapper_3d_view(0, x, y, z, o, 0);
                            }
                        }
                    }
                    else
                        throw std::invalid_argument("recover_param: unknown format");
                }
                else if(name.find("_biases")!=std::string::npos)
                {   // Biases.
                    if(internal_data_wrapper.parent->layout == nn::layout_t<nn::layout_nxyzpq_f32>::layout && param.dimensions == 1)
                    {
                        for(uint32_t x = 0; x < internal_lengths[NN_DATA_COORD_x]; ++x)
                            data_wrapper(x) = internal_data_wrapper(0, x, 0, 0, 0, 0);
                    }
                    else
                        throw std::invalid_argument("recover_param: unknown format");
                }
                else
                    throw std::invalid_argument("recover_param: unknown format");

                break;
            }
        }
    }
    catch(...)
    {
        return NN_API_STATUS_ERROR_STATUS_CODE_NOT_FOUND;
    }

    return NN_API_STATUS_OK;
}

NN_API_STATUS NN_API_CALL_CONVENTION nn_set_use_jit_primitives_0_function(int flag)
{
    nn::use_asmjit_primitives = (flag == 1);
    return NN_API_STATUS_OK;
}

