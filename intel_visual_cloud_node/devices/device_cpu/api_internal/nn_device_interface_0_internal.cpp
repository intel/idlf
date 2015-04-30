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

#include "../../api/nn_device_interface_0.h"
#include "../../common/nn_workload_data.h"
#include "../core/layer_convolution_avx2.h"
#include "../core/layer_convolution_pooling_avx2.h"
#include "../core/layer_fully_connected_avx2.h"
#include "../core/layer_softmax_avx2.h"
#include "../core/layer_pooling_avx2.h"
#include "../core/layer_normalization_avx2.h"
#include "../core/layer_convert_data_layout.h"
#include "../core/layers_fixedpoint.h"
#include "../core/layer_arithmetic_operation.h"
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
    case NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT: item_name = "conv_layout";break;
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
    case NN_WORK_ITEM_TYPE_VIEW: item_name = "view";break;
    case NN_WORK_ITEM_TYPE_INPUT: item_name = "input";break;
    case NN_WORK_ITEM_TYPE_OUTPUT: item_name = "output";break;
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
        const auto output = static_cast<nn::nn_workload_data_t<float>*>(item->output);
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

        size_t padding_w = (flow_item->output_format.format_3d.size[0] - 1) * arguments.stride[0] - input_w + arguments.weights->size[0];
        size_t padding_h = (flow_item->output_format.format_3d.size[1] - 1) * arguments.stride[1] - input_h + arguments.weights->size[1];

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

        size_t padding_w = ((flow_item->output_format.format_3d.size[0] - 1) * pooling_stride + pooling_size - 1) *
                               arguments.stride[0] -
                           input_w + arguments.weights->size[0];
        size_t padding_h = ((flow_item->output_format.format_3d.size[1] - 1) * pooling_stride + pooling_size - 1) *
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

        size_t padding_w = (flow_item->output_format.format_3d.size[0] - 1) * arguments.stride[0] - input_w + arguments.weights->size[0];
        size_t padding_h = (flow_item->output_format.format_3d.size[1] - 1) * arguments.stride[1] - input_h + arguments.weights->size[1];

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

        size_t padding_w = ((flow_item->output_format.format_3d.size[0] - 1) * pooling_stride + pooling_size - 1) *
                               arguments.stride[0] -
                           input_w + arguments.weights->size[0];
        size_t padding_h = ((flow_item->output_format.format_3d.size[1] - 1) * pooling_stride + pooling_size - 1) *
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
    const size_t output_w = flow_item->output_format.format_3d.size[0];
    const size_t output_h = flow_item->output_format.format_3d.size[1];

    for(size_t it_use = 0; it_use < flow_item->use_count; ++it_use){
        auto& use_item = flow_item->use[it_use];
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
                                                     nn_device_internal *device) {
    switch (load_item->type) {
    case NN_WORK_ITEM_TYPE_CONVOLUTION: {
        auto &args = flow_item->arguments.forward_convolution;
        assert((flow_item->output_format.format >= NN_DATA_FORMAT_3D ? flow_item->output_format.format_3d.size[2]
                                                                     : 1) == args.weights->size[3]);
        assert(args.padding == NN_PADDING_MODE_DATA_OR_ZERO);

        load_item->primitive = layer::convolution_f32::create(
            args.weights->size[0],
            args.weights->size[1],
            args.weights->size[2],
            args.weights->size[3],
            flow_item->output_format.format_1d.size[0],
            flow_item->output_format.format >= NN_DATA_FORMAT_2D ? flow_item->output_format.format_2d.size[1] : 1,
            args.center_offset[0],
            args.center_offset[1],
            args.stride[0],
            args.stride[1],
            args.activation,
            batch,
            reinterpret_cast<nn_device_t *>(device));
        break;
    }
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: {
        auto &args = flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2;
        assert((flow_item->output_format.format >= NN_DATA_FORMAT_3D ? flow_item->output_format.format_3d.size[2]
                                                                     : 1) == args.weights->size[3]);
        assert(args.padding == NN_PADDING_MODE_DATA_OR_ZERO);

        load_item->primitive = layer::convolution_pooling_f32_2x2stride2::create(
            args.weights->size[0],
            args.weights->size[1],
            args.weights->size[2],
            args.weights->size[3],
            flow_item->output_format.format_1d.size[0],
            flow_item->output_format.format >= NN_DATA_FORMAT_2D ? flow_item->output_format.format_2d.size[1] : 1,
            args.center_offset[0],
            args.center_offset[1],
            args.stride[0],
            args.stride[1],
            args.activation,
            batch,
            reinterpret_cast<nn_device_t *>(device));
        break;
    }
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED: {
        auto &args = flow_item->arguments.forward_fully_connected;

        assert(flow_item->input[0]->output_format.format == NN_DATA_FORMAT_1D ||
               flow_item->input[0]->output_format.format == NN_DATA_FORMAT_3D);

        bool use_3d_input = flow_item->input[0]->output_format.format == NN_DATA_FORMAT_3D;

        load_item->primitive = layer::fully_connected_f32::create(
            use_3d_input ? args.weights->size[0] * args.weights->size[1] * args.weights->size[2]
                         : args.weights->size[0],
            use_3d_input ? args.weights->size[3] : args.weights->size[1],
            args.activation,
            batch,
            device);
        break;
    }
    case NN_WORK_ITEM_TYPE_POOLING: {
        auto &args = flow_item->arguments.forward_pooling;

        assert(flow_item->input[0]->output_format.format >= NN_DATA_FORMAT_2D);
        assert(flow_item->output_format.format >= NN_DATA_FORMAT_2D);
        assert(get_format_size<2>(flow_item->input[0]->output_format) ==
               get_format_size<2>(flow_item->output_format)); // input and output have same depth

        load_item->primitive = layer::pooling_f32::create(args.mode,
                                                          args.size[0],
                                                          args.size[1],
                                                          args.stride[0],
                                                          args.stride[1],
                                                          get_format_size<2>(flow_item->output_format),
                                                          get_format_size<0>(flow_item->output_format),
                                                          get_format_size<1>(flow_item->output_format),
                                                          batch,
                                                          device);
        break;
    }
    case NN_WORK_ITEM_TYPE_ARITHMETIC: {
        auto &args = flow_item->arguments.forward_arithmetic;
        load_item->primitive = layer::arithmetic_f32::create(get_format_size<0>(flow_item->output_format),
                                                             get_format_size<1>(flow_item->output_format),
                                                             get_format_size<2>(flow_item->output_format),
                                                             args.arithmetic_function,
                                                             batch,
                                                             device);
        break;
    }
    case NN_WORK_ITEM_TYPE_NORMALIZATION: {
        auto &args = flow_item->arguments.forward_normalization;
        switch (args.normalization.mode) {
        case NN_NORMALIZATION_MODE_LINEAR_SINGLE: {
            load_item->primitive =
                layer::normalization_elementwise_linear_f32::create(args.normalization.alpha,
                                                                    args.normalization.beta,
                                                                    get_format_size<0>(flow_item->output_format),
                                                                    get_format_size<1>(flow_item->output_format),
                                                                    get_format_size<2>(flow_item->output_format),
                                                                    batch,
                                                                    device);
            break;
        }
        case NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS: {
            load_item->primitive =
                layer::normalization_response_across_maps_f32::create(args.normalization.alpha,
                                                                      args.normalization.beta,
                                                                      args.normalization.k,
                                                                      args.normalization.n,
                                                                      get_format_size<0>(flow_item->output_format),
                                                                      get_format_size<1>(flow_item->output_format),
                                                                      get_format_size<2>(flow_item->output_format),
                                                                      batch,
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
        load_item->primitive = layer::softmax_f32::create(get_format_size<0>(flow_item->output_format), batch, device);
        break;
    }
    }
}

void nn_workflow_compile_0_function_copy_item(
             nn_workload_item_t *load_item,
             nn_workflow_item_t *flow_item,
             uint32_t batch,
             std::map<nn_workflow_item_t *, nn_workload_item_t *> &flow_to_work,
             nn_device_internal *device
             ){

            // copy name
            load_item->name = flow_item->name;

            // copy type & create nn_workload_data_t
            load_item->type = flow_item->type;

            // setup primitive handle
            nn_workflow_compile_0_function_create_primitive(load_item, flow_item, batch, device);


            // calculate needed output buffer paddings
            size_t padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0;
            nn_workflow_compile_0_function_calculate_output_padding(flow_item, padding_left, padding_right, padding_top, padding_bottom);

            // setup output buffer
            switch(load_item->type) {
            case NN_WORK_ITEM_TYPE_INPUT:
            case NN_WORK_ITEM_TYPE_OUTPUT: {
                load_item->output = nullptr;
                break;
            }
            case NN_WORK_ITEM_TYPE_VIEW: {
                auto& origin = flow_item->arguments.view.origin;

                nn::nn_workload_data_t<float> *input_data = reinterpret_cast<nn::nn_workload_data_t<float> *>(flow_to_work[flow_item->input[0]]->output);
                nn_workload_data_coords_t start(0, origin[0], origin[1], origin[2] / input_data->parent->lengths.t[NN_DATA_COORD_p], 0, 0);
                nn_workload_data_coords_t end(
                    batch - 1,
                    origin[0] +  flow_item->output_format.format_1d.size[0] - 1,
                    origin[1] + (flow_item->output_format.format>=NN_DATA_FORMAT_2D ? flow_item->output_format.format_2d.size[1] : 1) - 1,
                    (origin[2] + (flow_item->output_format.format >= NN_DATA_FORMAT_3D ? flow_item->output_format.format_3d.size[2] : 1)) / input_data->parent->lengths.t[NN_DATA_COORD_p] - 1,
                    input_data->view_end.t[NN_DATA_COORD_p] - input_data->view_begin.t[NN_DATA_COORD_p],
                    0
                );

                load_item->output = new nn::nn_workload_data_t<float>(*input_data, start, end);
                break;
            }
            case NN_WORK_ITEM_TYPE_MERGE: {
                uint16_t axis = flow_item->arguments.forward_merge.axis; // x = 0, y = 1, z = 2

                auto input_data = flow_to_work[flow_item->input[0]]->output;
                nn_workload_data_layout_t previous_layout = input_data->parent->layout;

                uint32_t x_size = input_data->parent->lengths.t[1];
                uint32_t y_size = input_data->parent->lengths.t[2];
                uint32_t z_size = input_data->parent->lengths.t[3];
                for (int index = 1; index < flow_item->input_count; index++)
                {
                    auto input_data_local = flow_to_work[flow_item->input[index]]->output;
                    if (memcmp(&(input_data->parent->layout), &previous_layout, sizeof(nn_workload_data_layout_t)) != 0)
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
                load_item->output = new nn_workload_data_t;
                nn_workload_data_placement_create(load_item->output, nullptr, &size, &previous_layout);

                nn_workload_data_coords_t start_coords(0, padding_left, padding_top, 0, 0, 0);

                nn_workload_data_coords_t end_coords(
                    input_data->parent->lengths.t[0] - 1,
                    x_size - padding_right - 1,
                    y_size - padding_bottom - 1,
                    z_size - 1,
                    input_data->parent->lengths.t[4] - 1,
                    input_data->parent->lengths.t[5] - 1
                );

                uint32_t x_position = 0, y_position = 0, z_position = 0;

                // pin to the input buffers
                for (int index = 0; index < flow_item->input_count; index++)
                {
                    input_data = flow_to_work[flow_item->input[index]]->output;

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

                    delete flow_to_work[flow_item->input[index]]->output;
                    nn_workload_data_t *merge_output = load_item->output;
                    flow_to_work[flow_item->input[index]]->output = nn_workload_data_create_view(merge_output, &start_coords, &end_coords);
                }

                break;
            }
            case NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN: {
                nn_workload_data_layout_t layout = {
                    { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                    { 0, 0, 0, 0, 0, 0 }, // alignment
                    { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
                    NN_DATATYPE_INT16
                };

                const uint32_t z_block = 8;
                const uint32_t z_size = flow_item->output_format.format >= NN_DATA_FORMAT_3D ? flow_item->output_format.format_3d.size[2] : 1;
                nn_workload_data_coords_t size = {
                    batch,
                    flow_item->output_format.format_1d.size[0],
                    flow_item->output_format.format>=NN_DATA_FORMAT_2D ? flow_item->output_format.format_2d.size[1] : 1,
                    (z_size - 1) / z_block + 1,
                    z_block,
                    1
                };

                load_item->output = new nn::nn_workload_data_t<int16_t>(size, layout, padding_left, padding_right, padding_top, padding_bottom);
                break;
            }
            case NN_WORK_ITEM_TYPE_SOFTMAX: {
                load_item->output = static_cast<layer::softmax_f32 *>(load_item->primitive)->create_output();
                break;
            }
            case NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT: {
                nn_workload_data_layout_t layout = {
                    { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                    { 0, 0, 0, 0, 0, 0 }, // alignment
                    { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
                      NN_DATATYPE_FLOAT
                };
                nn_workload_data_coords_t size = {
                    batch,
                    flow_item->output_format.format_1d.size[0],
                    flow_item->output_format.format >= NN_DATA_FORMAT_2D ? flow_item->output_format.format_2d.size[1] : 1,
                    flow_item->output_format.format >= NN_DATA_FORMAT_3D ? flow_item->output_format.format_3d.size[2] : 1,
                    1, 1
                };

                load_item->output = new nn::nn_workload_data_t<float>(size, layout);
                break;
            }
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED:{
                load_item->output = static_cast<layer::fully_connected_f32 *>(load_item->primitive)->create_output();
                break;
            }
            case NN_WORK_ITEM_TYPE_ARITHMETIC:
            case NN_WORK_ITEM_TYPE_CONVOLUTION:
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2:
            case NN_WORK_ITEM_TYPE_POOLING:
            case NN_WORK_ITEM_TYPE_NORMALIZATION: {
                // views broken in arithmetic and element wise normalization
                if (load_item->type == NN_WORK_ITEM_TYPE_ARITHMETIC ||
                    (load_item->type == NN_WORK_ITEM_TYPE_NORMALIZATION &&
                     flow_item->arguments.forward_normalization.normalization.mode ==
                         NN_NORMALIZATION_MODE_LINEAR_SINGLE)) {
                    if (flow_item->use[0]->type == NN_WORK_ITEM_TYPE_MERGE ||
                        flow_item->input[0]->type == NN_WORK_ITEM_TYPE_VIEW)
                        throw NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;
                    if (padding_left != 0 || padding_right != 0 || padding_top != 0 || padding_bottom != 0)
                        throw NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;
                }

                load_item->output = static_cast<layer::helper_zxyn_f32::primitive_zxyn_f32_base *>(load_item->primitive)
                                        ->create_output(padding_left, padding_right, padding_top, padding_bottom);
                break;
            }
            case NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT:
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT:
            case NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT: {
                nn_workload_data_layout_t layout = {
                    { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                    { 0, 0, 0, 0, 0, 0 }, // alignment
                    { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
                    NN_DATATYPE_INT16
                };

                const uint32_t OFMOutBlock = 8;

                nn_workload_data_coords_t size = { batch,
                                                    flow_item->output_format.format == NN_DATA_FORMAT_3D
                                                        ? flow_item->output_format.format_3d.size[0]
                                                        : 1,
                                                    flow_item->output_format.format == NN_DATA_FORMAT_3D
                                                        ? flow_item->output_format.format_3d.size[1]
                                                        : 1,
                                                    flow_item->output_format.format == NN_DATA_FORMAT_3D
                                                        ? flow_item->output_format.format_3d.size[2] / OFMOutBlock
                                                        : flow_item->output_format.format_1d.size[0] / OFMOutBlock,
                                                    OFMOutBlock,
                                                    1};

                load_item->output = new nn::nn_workload_data_t<std::int16_t>(size, layout, padding_left, padding_right, padding_top, padding_bottom);
                break;
            }
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN: {
                nn_workload_data_layout_t layout = {
                    {0, 0, 0, 0, 0, 0}, // tile in log2(size)
                    {0, 0, 0, 0, 0, 0}, // alignment
                    {NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q}, // ordering
                    NN_DATATYPE_INT16};

                const uint32_t OutBlock = 2;
                nn_workload_data_coords_t size = {
                    batch,
                    1,
                    1,
                    flow_item->output_format.format_1d.size[0] / OutBlock,
                    OutBlock,
                    1
                };

                load_item->output = new nn::nn_workload_data_t<std::int16_t>(size, layout);

                break;
            }
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN: {
                nn_workload_data_layout_t layout = {
                    {0, 0, 0, 0, 0, 0}, // tile in log2(size)
                    {0, 0, 0, 0, 0, 0}, // alignment
                    {NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q}, // ordering
                    NN_DATATYPE_INT32};

                const uint32_t OutBlock = 2;
                nn_workload_data_coords_t size = {
                    batch,
                    1,
                    1,
                    flow_item->output_format.format_1d.size[0] / OutBlock,
                    OutBlock,
                    1
                };

                load_item->output = new nn::nn_workload_data_t<std::int32_t>(size, layout);

                break;
            }
            case NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT: {
                nn_workload_data_layout_t layout = {{0, 0, 0, 0, 0, 0}, // tile in log2(size)
                                                    {0, 0, 0, 0, 0, 0}, // alignment
                                                    {NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q}, // ordering
                                                    NN_DATATYPE_INT16};

                const uint32_t out_block = 4;

                nn_workload_data_coords_t size = {
                    batch,
                    flow_item->output_format.format_3d.size[0],
                    flow_item->output_format.format_3d.size[1],
                    (flow_item->output_format.format_3d.size[2] - 1) / out_block + 1,
                    out_block,
                    1
                };

                load_item->output = new nn::nn_workload_data_t<std::int16_t>(size, layout, padding_left, padding_right, padding_top, padding_bottom);
                break;
            }
            default:
                // If this assert fires it meant that new workflow item type was added, but its support
                // was not added to compile function.
                // This switch must contain *all* workflow items on the API.
                assert(0);
            } // switch
            
            // copy arguments, fill buffers
            assert(sizeof(load_item->arguments) >= sizeof(flow_item->arguments));
            std::memcpy(&load_item->arguments, &flow_item->arguments, sizeof(load_item->arguments));

            switch(load_item->type) {
            case NN_WORK_ITEM_TYPE_CONVOLUTION: {
                load_item->arguments.forward_convolution.biases =
                    static_cast<layer::convolution_f32 *>(load_item->primitive)
                    ->create_bias(*nn::data_cast<float, 1>(flow_item->arguments.forward_convolution.biases));

                load_item->arguments.forward_convolution.weights =
                    static_cast<layer::convolution_f32 *>(load_item->primitive)
                    ->create_weights(*nn::data_cast<float, 4>(flow_item->arguments.forward_convolution.weights));
                break;
            }
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: {
                load_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.biases =
                    static_cast<layer::convolution_f32 *>(load_item->primitive)
                        ->create_bias(*nn::data_cast<float, 1>(
                                          flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.biases));

                load_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights =
                    static_cast<layer::convolution_f32 *>(load_item->primitive)
                        ->create_weights(
                            *nn::data_cast<float, 4>(
                                flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights));
                break;
            }
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED: {
                load_item->arguments.forward_fully_connected.biases =
                    static_cast<layer::fully_connected_f32 *>(load_item->primitive)
                        ->create_bias(*nn::data_cast<float, 1>(flow_item->arguments.forward_fully_connected.biases));

                if (flow_item->input[0]->output_format.format == NN_DATA_FORMAT_3D) {
                    // input is 3d, require 4d weights
                    load_item->arguments.forward_fully_connected.weights =
                        static_cast<layer::fully_connected_f32 *>(load_item->primitive)
                            ->create_weights(
                                *nn::data_cast<float, 4>(flow_item->arguments.forward_fully_connected.weights));
                }else{
                    // input is 1d, require 2d weights
                    assert(flow_item->input[0]->output_format.format == NN_DATA_FORMAT_1D);
                          
                    load_item->arguments.forward_fully_connected.weights =
                        static_cast<layer::fully_connected_f32 *>(load_item->primitive)
                            ->create_weights(
                                *nn::data_cast<float, 2>(flow_item->arguments.forward_fully_connected.weights));
                }
                break;
            }
            case NN_WORK_ITEM_TYPE_ARITHMETIC: {
                const auto input = flow_item->input[0]->output_format;
                // check sizes  (x,y,z of factor must be equal to x,y,z of input)
                if (get_format_size<0>(input) != get_format_size<0>(flow_item->output_format) ||
                    get_format_size<1>(input) != get_format_size<1>(flow_item->output_format) ||
                    get_format_size<2>(input) != get_format_size<2>(flow_item->output_format))
                    throw NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;
                load_item->arguments.forward_arithmetic.factor =
                    static_cast<layer::arithmetic_f32 *>(load_item->primitive)
                        ->create_factor(*nn::data_cast<float, 0>(flow_item->arguments.forward_arithmetic.factor));
                break;
            }
            case NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT:
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT: {
                {
                    // biases
                    auto flow_biases = nn::data_cast<int32_t,1>(flow_item->arguments.forward_convolution_int16_fixedpoint.biases);
                    //TODO: validate bias format
                    nn_workload_data_layout_t layout = {
                        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                        { 0, 0, 0, 0, 0, 0 }, // alignment
                        { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
                        NN_DATATYPE_INT32
                    };
                    nn_workload_data_coords_t size = { 1, 1, 1, static_cast<uint32_t>(flow_biases->size[0]), 1, 1 };
                    auto *load_biases = new nn::nn_workload_data_t<int32_t>(size, layout);
                    load_item->arguments.forward_convolution_fixedpoint.biases = load_biases;
                    for(auto index=0u; index<load_biases->get_length(3); ++index) {
                        //             n, x, y, z      p  q  =                n, x,     y, z, p, q
                        (*load_biases)(0, 0, 0, index, 0, 0) = flow_biases->at(index);
                    }
                }
                { // weights
                    auto flow_weights = nn::data_cast<int16_t, 4>(flow_item->arguments.forward_convolution_int16_fixedpoint.weights);
                    //TODO: validate weight format
                    nn_workload_data_layout_t layout = {
                        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                        { 0, 0, 0, 0, 0, 0 }, // alignment
                        { NN_DATA_COORD_y, NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_q }, // ordering
                        NN_DATATYPE_INT16
                    };

                    const unsigned int OFMpBlock = 2;
                    const unsigned int OFMBlock = flow_item->output_format.format == NN_DATA_FORMAT_1D ? 384 : 32;

                    nn_workload_data_coords_t size = {static_cast<uint32_t>(flow_weights->size[0]),
                                                      static_cast<uint32_t>(flow_weights->size[1]),
                                                      OFMpBlock,
                                                      static_cast<uint32_t>(flow_weights->size[2] - 1) / OFMpBlock + 1,
                                                      OFMBlock,
                                                      static_cast<uint32_t>(flow_weights->size[3] - 1) / OFMBlock + 1};

                    auto load_weights = new nn::nn_workload_data_t<std::int16_t>(size, layout);
                    load_item->arguments.forward_convolution_fixedpoint.weights = load_weights;
                    for (auto q = 0u; q < size.t[5]; ++q)
                        for (auto p = 0u; p < size.t[4]; ++p)
                            for (auto z = 0u; z < size.t[3]; ++z)
                                for (auto y = 0u; y < size.t[2]; ++y)
                                    for (auto x = 0u; x < size.t[1]; ++x)
                                        for (auto n = 0u; n < size.t[0]; ++n)
                                            (*load_weights)(n, x, y, z, p, q) =
                                                (z * OFMpBlock + y < flow_weights->size[2])
                                                    ? flow_weights->at( n, x, z *OFMpBlock + y, q * OFMBlock + p)
                                                    : 0;
                }
                break;
            }
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN:
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN: {
                {
                    // biases
                    auto flow_biases = nn::data_cast<int32_t, 1>(
                        flow_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN
                             ? flow_item->arguments.fully_connected_forward_i16qn_i16qn.biases
                             : flow_item->arguments.fully_connected_forward_i16qn_i32qn.biases);
                    //TODO: validate bias format
                    nn_workload_data_layout_t layout = {
                        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                        { 0, 0, 0, 0, 0, 0 }, // alignment
                        { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
                        NN_DATATYPE_INT32
                    };
                    nn_workload_data_coords_t size = { 1, static_cast<uint32_t>(flow_biases->size[0]), 1, 1, 1, 1 };
                    auto *load_biases = new nn::nn_workload_data_t<std::int32_t>(size, layout);
                    for(auto index=0u; index<load_biases->get_length(1); ++index) {
                        //             n, x, y, z      p  q  =                n, x,     y, z, p, q
                        (*load_biases)(0, index, 0, 0, 0, 0) = flow_biases->at(index);
                    }

                    (flow_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN
                         ? load_item->arguments.fully_connected_forward_i16qn_i16qn.biases
                         : load_item->arguments.fully_connected_forward_i16qn_i32qn.biases) = load_biases;
                }
                { // weights
                    auto tmp_flow_weights = nn::data_cast<int16_t, 0>(
                        flow_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN
                             ? flow_item->arguments.fully_connected_forward_i16qn_i16qn.weights
                             : flow_item->arguments.fully_connected_forward_i16qn_i32qn.weights);

                    assert(tmp_flow_weights->dimension == 2 || tmp_flow_weights->dimension == 4);
                    nn::data<int16_t,2> *flow_weights;
                    if(tmp_flow_weights->dimension == 4){
                        flow_weights = new nn::data<int16_t, 2>(static_cast<int16_t *>(tmp_flow_weights->buffer),
                                                                tmp_flow_weights->size[0] * tmp_flow_weights->size[1] *
                                                                    tmp_flow_weights->size[2],
                                                                tmp_flow_weights->size[3]);
                    }else if(tmp_flow_weights->dimension == 2){
                        flow_weights = new nn::data<int16_t, 2>(static_cast<int16_t *>(tmp_flow_weights->buffer),
                                                                tmp_flow_weights->size[0],
                                                                tmp_flow_weights->size[1]);
                    }

                    //TODO: validate weight format
                    nn_workload_data_layout_t layout = {
                        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                        { 0, 0, 0, 0, 0, 0 }, // alignment
                        { NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
                        NN_DATATYPE_INT16
                    };

                    const unsigned int IFMBlock = 2;
                    const unsigned int OFMBlock = 8;

                    nn_workload_data_coords_t size = {1,
                                                      IFMBlock,
                                                      static_cast<uint32_t>(flow_weights->size[0]) / IFMBlock,
                                                      OFMBlock,
                                                      static_cast<uint32_t>(flow_weights->size[1]) / OFMBlock,
                                                      1};

                    auto load_weights = new nn::nn_workload_data_t<std::int16_t>(size, layout);
                    for (std::uint32_t p = 0; p < size.t[4]; ++p)
                        for (std::uint32_t z = 0; z < size.t[3]; ++z)
                            for (std::uint32_t y = 0; y < size.t[2]; ++y)
                                for (std::uint32_t x = 0; x < size.t[1]; ++x)
                                    (*load_weights)(0, x, y, z, p, 0) =
                                        flow_weights->at(y * IFMBlock + x, p * OFMBlock + z);

                    (load_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN
                         ? load_item->arguments.fully_connected_forward_i16qn_i16qn.weights
                         : load_item->arguments.fully_connected_forward_i16qn_i32qn.weights) = load_weights;

                    delete flow_weights;
                }
                break;
            }
            default:
                // This is the case when all workflow item arguments are empty or do not contain buffers.
                ;
            }

            // copying inputs
            load_item->input.resize(flow_item->input_count);
            for(auto index=0u; index<flow_item->input_count; ++index) {
                assert(flow_to_work.find(flow_item->input[index])!=flow_to_work.end());
                load_item->input[index] = flow_to_work[flow_item->input[index]];
            }

            // copying uses
            load_item->use.resize(flow_item->use_count);
            for(auto index=0u; index<flow_item->use_count; ++index) {
                assert(flow_to_work.find(flow_item->use[index])!=flow_to_work.end());
                load_item->use[index] = flow_to_work[flow_item->use[index]];
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
            conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            conversion->arguments.convert_data_layout.type = 2;

            nn_workload_data_layout_t layout = {{0, 0, 0, 0, 0, 0}, // tile in log2(size)
                                                {0, 0, 0, 0, 0, 0}, // alignment
                                                {NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q}, // ordering
                                                NN_DATATYPE_INT32};

            const uint32_t OutBlock = 2;
            nn_workload_data_coords_t size = {
                batch,
                1,
                1,
                flow_item->output_format.format_1d.size[0] / OutBlock,
                OutBlock,
                1};

            conversion->output = new nn::nn_workload_data_t<std::int32_t>(size, layout);
            conversion->input.push_back(load_item);
            return conversion;
        };

        if (load_item->output != nullptr && load_item->output->parent->layout.ordering.t[0] != NN_DATA_COORD_n) {
            for (auto &next_load_item : load_item->use) {
                if (next_load_item->type == NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT) {
                    auto type2_conversion = init_type2_conversion();
                    type2_conversion->name = std::string("convert_layout2_before_") + next_load_item->name;

                    for (auto &useinput : next_load_item->input)
                    if (useinput == load_item) {
                        useinput = type2_conversion;
                        type2_conversion->use.push_back(next_load_item);
                    }

                    next_load_item = type2_conversion;
                }
            }
        }

        // add type3 conversions
        auto init_type3_conversion = [&batch, &flow_item, &load_item] {
            auto conversion = new nn_workload_item_t;
            conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            conversion->arguments.convert_data_layout.type = 3;

            nn_workload_data_layout_t layout = {{0, 0, 0, 0, 0, 0}, // tile in log2(size)
                                                {0, 0, 0, 0, 0, 0}, // alignment
                                                {NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q}, // ordering
                                                NN_DATATYPE_INT16};

            const uint32_t OutBlock = 2;
            nn_workload_data_coords_t size = {
                batch,
                1,
                1,
                flow_item->output_format.format_3d.size[0] * flow_item->output_format.format_3d.size[1] * flow_item->output_format.format_3d.size[2] / OutBlock,
                OutBlock,
                1};

            conversion->output = new nn::nn_workload_data_t<std::int16_t>(size, layout);
            conversion->input.push_back(load_item);
            return conversion;
        };

        if (load_item->output != nullptr && load_item->output->parent->layout.ordering.t[1] != NN_DATA_COORD_n) {
            for (auto &next_load_item : load_item->use) {
                if (next_load_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN ||
                    next_load_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN) {
                    auto type3_conversion = init_type3_conversion();
                    type3_conversion->name = std::string("convert_layout3_before_") + next_load_item->name;

                    for (auto &useinput : next_load_item->input)
                    if (useinput == load_item) {
                        useinput = type3_conversion;
                        type3_conversion->use.push_back(next_load_item);
                    }

                    next_load_item = type3_conversion;
                }
            }
        }

        // Add type4 conversions - float batch8 conversion between conv and fc
        auto init_type4_conversion = [&batch, &flow_item, &load_item]
        {
            auto conversion = new nn_workload_item_t;
            conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            conversion->arguments.convert_data_layout.type = 4;

            nn_workload_data_layout_t layout = {{0, 0, 0, 0, 0, 0}, // tile in log2(size)
                                                {0, 0, 0, 0, 0, 0}, // alignment
                                                {NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_p, NN_DATA_COORD_q}, // ordering
                                                NN_DATATYPE_FLOAT};

            nn_workload_data_coords_t size = {
                batch,
                flow_item->output_format.format_1d.size[0],
                flow_item->output_format.format >= NN_DATA_FORMAT_2D ? flow_item->output_format.format_2d.size[1] : 1,
                flow_item->output_format.format >= NN_DATA_FORMAT_3D ? flow_item->output_format.format_3d.size[2] : 1,
                1,
                1 };

            conversion->output = new nn::nn_workload_data_t<float>(size, layout);
            conversion->input.push_back(load_item);
            return conversion;
        };

        if (load_item->output != nullptr &&
            load_item->output->parent->layout.ordering.t[0] != NN_DATA_COORD_n &&
            (load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION ||
             load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2 ||
             load_item->type == NN_WORK_ITEM_TYPE_POOLING))
        {
            for (auto &next_load_item : load_item->use)
            {
                if (next_load_item->output->parent->layout.ordering.t[0] == NN_DATA_COORD_n &&
                    next_load_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED)
                {
                    auto type4_conversion = init_type4_conversion();
                    type4_conversion->name = std::string("convert_layout4_before_") + next_load_item->name;

                    for (auto &useinput : next_load_item->input)
                    if (useinput == load_item)
                    {
                        useinput = type4_conversion;
                        type4_conversion->use.push_back(next_load_item);
                    }

                    next_load_item = type4_conversion;
                }
            }
        }
    }
    else { // batch == 1
        // add type0 conversions
        auto init_type0_conversion = [&batch, &flow_item, &load_item] {
            auto conversion = new nn_workload_item_t;
            conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
            conversion->arguments.convert_data_layout.type = 0;

            nn_workload_data_layout_t layout = {{0, 0, 0, 0, 0, 0}, // tile in log2(size)
                                                {0, 0, 0, 0, 0, 0}, // alignment
                                                {NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q}, // ordering
                                                NN_DATATYPE_INT16};

            const uint32_t OutBlock = 2;
            nn_workload_data_coords_t size = {
                batch,
                1,
                1,
                flow_item->output_format.format_3d.size[0] * flow_item->output_format.format_3d.size[1] * flow_item->output_format.format_3d.size[2] / OutBlock,
                OutBlock,
                1};

            conversion->output = new nn::nn_workload_data_t<std::int16_t>(size, layout);
            conversion->input.push_back(load_item);
            return conversion;
        };

        if (load_item->output != nullptr && flow_item->output_format.format == NN_DATA_FORMAT_3D) {
            for (auto &next_load_item : load_item->use) {
                if (next_load_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN ||
                    next_load_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN) {
                    auto type3_conversion = init_type0_conversion();
                    type3_conversion->name = std::string("convert_layout3_before_") + next_load_item->name;

                    for (auto &useinput : next_load_item->input)
                    if (useinput == load_item) {
                        useinput = type3_conversion;
                        type3_conversion->use.push_back(next_load_item);
                    }

                    next_load_item = type3_conversion;
                }
            }
        }
    }

    auto init_type5_conversion = [&batch, &flow_item, &load_item]
    {
        auto conversion = new nn_workload_item_t;
        conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
        conversion->arguments.convert_data_layout.type = 5;

        nn_workload_data_layout_t layout = {{0, 0, 0, 0, 0, 0}, // tile in log2(size)
                                            {0, 0, 0, 0, 0, 0}, // alignment
                                            {NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q}, // ordering
                                            NN_DATATYPE_INT16};

        uint32_t z_size = flow_item->output_format.format >= NN_DATA_FORMAT_3D ? flow_item->output_format.format_3d.size[2] : 1;
        uint32_t z_block = z_size > 4 ? 8 : 4;

        nn_workload_data_coords_t size = {
            batch,
            flow_item->output_format.format_1d.size[0],
            flow_item->output_format.format >= NN_DATA_FORMAT_2D ? flow_item->output_format.format_2d.size[1] : 1,
            (z_size - 1) / z_block + 1,
            z_block,
            1 };

        conversion->output = new nn::nn_workload_data_t<float>(size, layout);
        conversion->input.push_back(load_item);
        return conversion;
    };

    if (load_item->type == NN_WORK_ITEM_TYPE_INPUT) {
        for (auto &next_load_item : load_item->use) {
            if (next_load_item->type == NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN ||
                next_load_item->type == NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT ||
                next_load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT ||
                next_load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT) {
                assert(input_format[load_item->arguments.input.index] == NN_WORKLOAD_DATA_TYPE_I16_ZXY); // TODO support other formats
                auto type5_conversion = init_type5_conversion();
                type5_conversion->name = std::string("convert_layout5_before_") + next_load_item->name;

                for (auto &useinput : next_load_item->input)
                if (useinput == load_item) {
                    useinput = type5_conversion;
                    type5_conversion->use.push_back(next_load_item);
                }

                next_load_item = type5_conversion;
            }
        }
    }

    auto init_type6_conversion = [&batch, &flow_item, &load_item]
    {
        auto conversion = new nn_workload_item_t;
        conversion->type = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT;
        conversion->arguments.convert_data_layout.type = 6;

        nn_workload_data_layout_t layout = {{0, 0, 0, 0, 0, 0}, // tile in log2(size)
                                            {0, 0, 0, 0, 0, 0}, // alignment
                                            {NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q}, // ordering
                                            NN_DATATYPE_INT16};

        nn_workload_data_coords_t size = {
            batch,
            flow_item->output_format.format_1d.size[0],
            flow_item->output_format.format >= NN_DATA_FORMAT_2D ? flow_item->output_format.format_2d.size[1] : 1,
            flow_item->output_format.format >= NN_DATA_FORMAT_3D ? flow_item->output_format.format_3d.size[2] : 1,
            1,
            1 };

        conversion->output = new nn::nn_workload_data_t<float>(size, layout);
        conversion->input.push_back(load_item);
        return conversion;
    };

    if ((load_item->type == NN_WORK_ITEM_TYPE_MERGE && load_item->input[0]->output->parent->layout.data_type == NN_DATATYPE_INT16) ||
        load_item->type == NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN ||
        load_item->type == NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT ||
        load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT ||
        load_item->type == NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT) {
        for (auto &next_load_item : load_item->use) {
            if (next_load_item->type == NN_WORK_ITEM_TYPE_OUTPUT) {
                auto type6_conversion = init_type6_conversion();
                type6_conversion->name = std::string("convert_layout6_before_") + next_load_item->name;

                for (auto &useinput : next_load_item->input)
                if (useinput == load_item) {
                    useinput = type6_conversion;
                    type6_conversion->use.push_back(next_load_item);
                }

                next_load_item = type6_conversion;
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
    if(workflow->input_count!=1)                return NN_API_STATUS_ERROR_INVALID_INPUT_COUNT;
    if(workflow->output_count!=1)               return NN_API_STATUS_ERROR_INVALID_OUTPUT_COUNT;

    if(!workflow->input || !workflow->output)   return NN_API_STATUS_ERROR_INVALID_WORKFLOW;
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
        *const_cast<NN_WORKLOAD_DATA_TYPE **>(&workload_public->output_format) = reinterpret_cast<NN_WORKLOAD_DATA_TYPE *>(buffer);
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
                        todo.push(flow_item->use[index]);
                }
            }
        }

        { // now for every workflow item there's a workload item
            std::queue<nn_workflow_item_t *> todo;
            std::set<nn_workflow_item_t *> done;
            for(auto index=0u; index<workflow->input_count; ++index)
                todo.push(workflow->input[index]);
            while(!todo.empty()) {
                nn_workflow_item_t *flow_item = todo.front();
              todo.pop();
                if(done.find(flow_item)==done.end()) {
                    done.insert(flow_item);
                    nn_workload_item_t *load_item = flow_to_work[flow_item];
                    nn_workflow_compile_0_function_copy_item(load_item, flow_item, batch, flow_to_work, reinterpret_cast<nn_device_internal*>(device));
                    for(auto index=0u; index<flow_item->use_count; ++index)
                        todo.push(flow_item->use[index]);
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
                    todo.push(flow_item->use[index]);
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
            std::queue<nn_workload_item_t *> todo;
            std::set<nn_workload_item_t *> done;
            for(auto output_item : workload_opaque->output) {
                todo.push(output_item);
            }
            while(!todo.empty()) {
                nn_workload_item_t *load_item = todo.front();
                todo.pop();
                if(done.find(load_item)==done.end()) {
                    done.insert(load_item);
                    workload_opaque->order_of_execution.push_front(load_item);
                    for(auto input_item : load_item->input)
                        todo.push(input_item);
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
    if(!workload_public || !input || !output) return NN_API_STATUS_ERROR_INVALID_POINTER;
    else {
        try {
            // layout for inputs & outputs
            auto get_workload_layout = [](NN_WORKLOAD_DATA_TYPE type) -> nn_workload_data_layout_t {
                switch (type) {
                    case NN_WORKLOAD_DATA_TYPE_F32_1D:
                    case NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_F32_2D:
                    case NN_WORKLOAD_DATA_TYPE_F32_2D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_F32_3D:
                    case NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH:
                        return nn_workload_data_layout_t{ { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                        { 0, 0, 0, 0, 0, 0 }, // alignment
                        { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n },
                        NN_DATATYPE_FLOAT };

                    case NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_F32_ZXY:
                        return nn_workload_data_layout_t{ { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                        { 0, 0, 0, 0, 0, 0 }, // alignment
                        { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q },
                        NN_DATATYPE_FLOAT };

                    case NN_WORKLOAD_DATA_TYPE_I16_1D:
                    case NN_WORKLOAD_DATA_TYPE_I16_1D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_I16_3D:
                    case NN_WORKLOAD_DATA_TYPE_I16_3D_BATCH:
                        return nn_workload_data_layout_t{ { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                        { 0, 0, 0, 0, 0, 0 }, // alignment
                        { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n },
                        NN_DATATYPE_INT16 };

                    case NN_WORKLOAD_DATA_TYPE_I16_ZXY:
                    case NN_WORKLOAD_DATA_TYPE_I16_ZXY_BATCH:
                        return nn_workload_data_layout_t{ { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                        { 0, 0, 0, 0, 0, 0 }, // alignment
                        { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q },
                        NN_DATATYPE_INT16 };

                    case NN_WORKLOAD_DATA_TYPE_I32_1D:
                    case NN_WORKLOAD_DATA_TYPE_I32_1D_BATCH:
                        return nn_workload_data_layout_t{ { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                        { 0, 0, 0, 0, 0, 0 }, // alignment
                        { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n },
                        NN_DATATYPE_INT32 };

                    default:
                        throw std::out_of_range("unsupported data type");
                }
            };

            // calculates 6D size from nn::data, returns it as nn_workload_data_coords_t
            auto calculate_size = [](uint32_t batch, NN_WORKLOAD_DATA_TYPE type, nn_data_t *data) -> nn_workload_data_coords_t {
                uint32_t size_n=batch, size_x=1, size_y=1, size_z=1, size_p=1, size_q=1;
                switch(type) {
                case NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_F32_3D:
                    case NN_WORKLOAD_DATA_TYPE_I16_3D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_I16_3D:
                    size_z = data->size[2];
                    // fall through
                    case NN_WORKLOAD_DATA_TYPE_F32_2D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_F32_2D:
                    size_y = data->size[1];
                    // fall through
                case NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH:
                case NN_WORKLOAD_DATA_TYPE_F32_1D:
                    case NN_WORKLOAD_DATA_TYPE_I16_1D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_I16_1D:
                    case NN_WORKLOAD_DATA_TYPE_I32_1D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_I32_1D:
                    size_x = data->size[0];
                    break;
                    case NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_F32_ZXY:
                    case NN_WORKLOAD_DATA_TYPE_I16_ZXY_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_I16_ZXY:
                        size_z = data->size[0];
                        size_x = data->size[1];
                        size_y = data->size[2];
                        break;
                default:
                    assert(0);
                }
                switch (type) {
                case NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_I16_3D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_I16_ZXY_BATCH:
                    size_n = data->size[3];
                    break;
                    case NN_WORKLOAD_DATA_TYPE_F32_2D_BATCH:
                    size_n = data->size[2];
                    break;
                case NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_I16_1D_BATCH:
                    case NN_WORKLOAD_DATA_TYPE_I32_1D_BATCH:
                    size_n = data->size[1];
                    break;
                default:
                    ;
                }
                return nn_workload_data_coords_t{size_n, size_x, size_y, size_z, size_p, size_q};
            };

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
                    // This item will contain view to input data. Needs to be released after execution.
                    auto index = item->arguments.input.index;
                    auto item_input = reinterpret_cast<nn_data_t*>(input[index]);
                    auto item_input_format = workload_public->input_format[index];
                    auto item_input_size = calculate_size(workload_public->batch, item_input_format, item_input);
                    auto item_input_layout = get_workload_layout(item_input_format);
                    item->output = new nn::nn_workload_data_t<float /* NOTE: this type is disregarded in this case */ >(item_input->buffer, item_input_size, item_input_layout);
                    break;
                }
                case NN_WORK_ITEM_TYPE_OUTPUT: {
                    // Copy result to workload output.
                    auto index = item->arguments.output.index;
                    auto item_output = reinterpret_cast<nn_data_t*>(output[index]);
                    auto item_output_format = workload_public->output_format[index];
                    auto item_output_size = calculate_size(workload_public->batch, item_output_format, item_output);
                    auto item_output_layout = get_workload_layout(item_output_format);
                    auto workload_output_wrapper = new nn::nn_workload_data_t<float /* NOTE: this type is disregarded in this case */ >(item_output->buffer, item_output_size, item_output_layout);
                    nn_workload_data_copy(workload_output_wrapper, item->input[0]->output);
                    delete workload_output_wrapper;
                    break;
                }
                case NN_WORK_ITEM_TYPE_MERGE:
                case NN_WORK_ITEM_TYPE_VIEW: {
                    // Nothing to do here - it's just limiting access to input by attaching view to it.
                    break;
                }
                case NN_WORK_ITEM_TYPE_ARITHMETIC: {
                    layer::wrapper_arithmetic_operation_work_item(item, reinterpret_cast<nn_device_internal*>(workload_public->device));
                    break;
                }
                case NN_WORK_ITEM_TYPE_CONVOLUTION: {
                    // convolution
                    layer::run_multithreaded_convolve_work_item(item, reinterpret_cast<nn_device_internal*>(workload_public->device));
                    break;
                }
                case NN_WORK_ITEM_TYPE_FULLY_CONNECTED: {
                    layer::wrapper_fully_connected_work_item(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_POOLING: {
                    layer::wrapper_pooling_work_item(item, reinterpret_cast<nn_device_internal*>(workload_public->device));
                    break;
                }
                case NN_WORK_ITEM_TYPE_NORMALIZATION: {
                    layer::wrapper_normalization_work_item(item, reinterpret_cast<nn_device_internal*>(workload_public->device));
                    break;
                }
                case NN_WORK_ITEM_TYPE_SOFTMAX: {
                    layer::wrapper_softmax_work_item(item, reinterpret_cast<nn_device_internal*>(workload_public->device));
                    break;
                }
                case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: {
                    layer::run_multithreaded_convolve_maxpooling2x2_stride2x2_work_item(item, reinterpret_cast<nn_device_internal*>(workload_public->device));
                    break;
                }
                case NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT: {
                    int16_fixedpoint::run_multithreaded_convolve_fixedpoint_work_item(item, reinterpret_cast<nn_device_internal*>(workload_public->device));
                    break;
                }
                case NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT: {
                    int16_fixedpoint::run_pooling_work_item(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN: {
                    int16_fixedpoint::wrapper_lrn_fixedpoint_work_item(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN:
                case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN: {
                    int16_fixedpoint::run_multithreaded_fully_connected_fixedpoint_work_item(item, reinterpret_cast<nn_device_internal*>(workload_public->device));
                    break;
                }
                case NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT: {
                    int16_fixedpoint::run_softmax_int32_float_work_item(item, reinterpret_cast<nn_device_internal*>(workload_public->device));
                    break;
                }
                case NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT: {
                    int16_fixedpoint::run_convert_float_to_int16_fp_work_item(item);
                    break;
                }
                case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT: {
                    int16_fixedpoint::run_multithreaded_convolve_pooling_fixedpoint_work_item(item, reinterpret_cast<nn_device_internal*>(workload_public->device));
                    break;
                }
                case NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT: {
                    layer::run_convert_to_data_layout_work_item(item);
                    break;
                }
                default:
                    NN_UNREACHABLE_CODE;
                } // switch

#if ENABLE_WORKLOAD_PROFILING
                auto t1 = __rdtsc();
                workload_opaque->profiling_data.work_item_cycles[item].push_back(t1 - t0);
#endif

#if ENABLE_WORKLOAD_MONITORING
                nn_workload_item_data_marshaling(item, ++item_count);
#endif // ENABLE_WORKLOAD_MONITORING
            }

            // remove all created views
            for(auto item : workload_opaque->order_of_execution) {
                if(item->type==NN_WORK_ITEM_TYPE_INPUT)
                    delete item->output;
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
    for (const auto &item : workload_opaque->order_of_execution) {
        const auto &times = workload_opaque->profiling_data.work_item_cycles[item];
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        uint64_t minimum = *std::min_element(times.begin(), times.end());

        auto item_name = [](nn_workload_item* item){
            switch (item->type){
            case NN_WORK_ITEM_TYPE_NORMALIZATION: return "norm";
            case NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT: return "conv_layout";
            case NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT: return "conv_float2int";
            case NN_WORK_ITEM_TYPE_CONVOLUTION: return "cnn_f32";
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: return "cnn+pool2x2_f32";
            case NN_WORK_ITEM_TYPE_POOLING: return "pooling_f32";
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED: return "fc_f32";
            case NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT: return "cnn_i16";
            case NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT: return "pool_i16";
            case NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN: return "lrn_i16";
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT: return "cnn+pool2x2_i16";
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN:
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN: return "fc_i16";
            case NN_WORK_ITEM_TYPE_SOFTMAX: return "softmax_f32";
            case NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT: return "softmax_i16";
            case NN_WORK_ITEM_TYPE_MERGE: return "merge";
            case NN_WORK_ITEM_TYPE_ARITHMETIC: return "arithmetic";
            case NN_WORK_ITEM_TYPE_VIEW: return "view";
            case NN_WORK_ITEM_TYPE_INPUT: return "input";
            case NN_WORK_ITEM_TYPE_OUTPUT: return "output";
            default: return "";
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
                    for(auto element : load_item->use) todo.push(element);
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
