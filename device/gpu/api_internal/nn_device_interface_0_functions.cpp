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
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <ostream>
#include <iostream>
#include <stack>
#include <vector>
#include <set>
#include <queue>
#include <map>
#include "device/api/nn_device_interface_0.h"
#include "device/common/nn_workload_data.h"
#include "device/gpu/core/layers_opencl.h"
#include "device/gpu/api_internal/nn_device_interface_0_internal.h"

//#define DUMP_LAYERS

#ifdef DUMP_LAYERS
#include <string>
#include <fstream>
#include <iomanip>

auto layer_name = [] ( nn_gpu_workload_item& item ){
    switch( item.type ){
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
    case NN_WORK_ITEM_TYPE_SOFTMAX:
    case NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT: return "softmax";
    case NN_WORK_ITEM_TYPE_MERGE: return "merge";
    case NN_WORK_ITEM_TYPE_ARITHMETIC: return "arithmetic";
    case NN_WORK_ITEM_TYPE_VIEW: return "view";
    case NN_WORK_ITEM_TYPE_INPUT: return "input";
    case NN_WORK_ITEM_TYPE_OUTPUT: return "output";
    default: return "";
    }
};

void nn_data_marshaling_to_txt( nn_cl_data &data, std::string filename, nn_device_t *const device )
{
    std::fstream file;

    // TODO: Make dumping of images supported
    if(data.parent->cl_buffer[0] == nullptr)
        return;

    file.open( "dumps\\" + filename, std::ios::out | std::ios::trunc );

    cl_int err;
    float* ptr = static_cast<float*>(
        clEnqueueMapBuffer(
            reinterpret_cast<device_gpu::ocl_toolkit *>(device)->get_command_queue()(),
            (*data.parent->cl_buffer[0])(),
            true,
            CL_MEM_READ_ONLY,
            0,
            data.parent->buffer_aligned_size,
            0,
            nullptr,
            nullptr,
            &err));

    if (err != CL_SUCCESS)
        THROW_ERROR(err, "Error in mapping buffer at OUTPUT memcpy.");

    nn::workload_data<> temp( ptr, data.parent->lengths, data.parent->layout );
    nn::workload_data<> view( temp, data.view_begin, data.view_end );

    auto view_size = view.get_length( );

    if( file.is_open( ) ) {
        for( auto n = 0u; n < view_size.t[0]; ++n ) {
            for( auto z = 0u; z < view_size.t[3]; ++z ) {
                file << "n=" << n << " z=" << z << std::endl;
                for( auto y = 0u; y < view_size.t[2]; ++y ) {
                    for( auto x = 0u; x < view_size.t[1]; ++x ) {
                        file << std::setprecision( 6 ) << view( n, x, y, z, 0, 0 ) << " ";
                    }
                    file << std::endl;
                }
                file << std::endl;
            }
        }

    }
    else
        std::cout << "file access denied" << std::endl;

    clEnqueueUnmapMemObject(
        reinterpret_cast<device_gpu::ocl_toolkit *>( device )->get_command_queue( )( ),
        ( *data.parent->cl_buffer[0] )( ),
        ptr,
        0,
        nullptr,
        nullptr );


    file.close( );
}

#endif

namespace handy_routines
{
///////////////////////////////////////////////////////////////////////////////////////////////////
bool use_fully_connected_8x8(uint_least32_t num_inputs, uint_least32_t num_outputs, uint32_t batch, nn_device_t *device)
{
    return((num_inputs % 8 == 0) && (num_outputs % 8 == 0) &&
           (batch % 8 == 0) &&
           (num_inputs*num_outputs*sizeof(float) <= reinterpret_cast< device_gpu::ocl_toolkit * >(device)->m_max_buffer_size));
}
///////////////////////////////////////////////////////////////////////////////////////////////////
//Routine checking if corresposing load_item to given flow_item should have output data encapsulated in CL image
bool is_image_as_output_needed(nn_workflow_item_t * flow_item, uint32_t batch, nn_device_t *device)
{
    uint_least32_t input_width, input_height, input_depth;
    uint_least32_t num_outputs;

    // If next Layer is Fully connected and is to use images as input
    // then make current layer to produce image as output
    if( (flow_item->use_count >= 1) && (flow_item->use[0].item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED)) {

        input_width  = flow_item->output_format[0].format_1d.size[0];
        input_height = flow_item->output_format[0].format >= NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1;
        input_depth = flow_item->output_format[0].format >= NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1;

        num_outputs = ( flow_item->use[0].item->output_format[0].format_1d.size[0] *
              ( flow_item->use[0].item->output_format[0].format >=
                NN_DATA_FORMAT_2D ? flow_item->use[0].item->output_format[0].format_2d.size[1] : 1 ) *
              ( flow_item->use[0].item->output_format[0].format >=
                NN_DATA_FORMAT_3D ? flow_item->use[0].item->output_format[0].format_3d.size[2] : 1 ) );

        return use_fully_connected_8x8(input_width * input_height * input_depth, num_outputs, batch, device);
    }
    return false;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool check_layout( nn_workload_data_layout_t & candidate, nn_workload_data_layout_t & pattern )
{
    return ( ( memcmp( &pattern, &candidate, sizeof( nn_workload_data_layout_t ) ) == 0 ) ? true : false );
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void nn_workflow_compile_0_function_prepare_ocl_kernels(
    nn_device_t        *device,         /* device context */
    NN_WORKLOAD_DATA_TYPE  output_layout,
    NN_WORKLOAD_DATA_TYPE  input_layout,
    nn_gpu_workload_item_t *load_item,
    nn_workflow_item_t     *flow_item,
    uint32_t               batch )
{
    switch( load_item->type )
    {
    case NN_WORK_ITEM_TYPE_INPUT:
        // No OCL kernel for this one
        break;
    case NN_WORK_ITEM_TYPE_VIEW:
        // No OCL kernel for this one
        break;
    case NN_WORK_ITEM_TYPE_OUTPUT:
        // No OCL kernel for this one
        // TODO: consider using OCL for that
        break;
    case NN_WORK_ITEM_TYPE_ARITHMETIC:
    {
        assert( flow_item->input_count < 2 );
        auto& input = flow_item->input[0];

        uint_least32_t total_input_width = input.item->output_format[input.index].format_1d.size[0];
        uint_least32_t total_input_height = input.item->output_format[input.index].format >=
                                            NN_DATA_FORMAT_2D ? input.item->output_format[input.index].format_2d.size[1] : 1;
        uint_least32_t total_input_depth = input.item->output_format[input.index].format >=
                                           NN_DATA_FORMAT_3D ? input.item->output_format[input.index].format_3d.size[2] : 1;

        reinterpret_cast< device_gpu::ocl_toolkit * >( device )->prepare_arithmetic_kernel( output_layout,
                                                                                            input_layout,
                                                                                            total_input_width,
                                                                                            total_input_height,
                                                                                            total_input_depth,
                                                                                            flow_item->arguments.forward_arithmetic.arithmetic_function,
                                                                                            batch );
        break;
    }
    case NN_WORK_ITEM_TYPE_NORMALIZATION:
    {
        assert( flow_item->input_count < 2 );

        uint_least32_t input_depth;
        if( load_item->input[0]->type == NN_WORK_ITEM_TYPE_INPUT )
        {
            auto& input = flow_item->input[0];
            input_depth = input.item->output_format[input.index].format >=
                          NN_DATA_FORMAT_3D ? input.item->output_format[input.index].format_3d.size[2] : 1;
        }
        else
        {
            input_depth =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_z ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_z ] +
                1;
        }

        NN_NORMALIZATION_MODE normalization_mode =
            flow_item->arguments.forward_normalization.normalization.mode;

        switch( normalization_mode )
        {
        case NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS:
        {
            const uint_least32_t normalization_size =
                flow_item->arguments.forward_normalization.normalization.n;
            const uint_least32_t k =
                flow_item->arguments.forward_normalization.normalization.k;
            const float alpha =
                flow_item->arguments.forward_normalization.normalization.alpha;
            const float beta =
                flow_item->arguments.forward_normalization.normalization.beta;

            reinterpret_cast< device_gpu::ocl_toolkit * >( device )->prepare_normalization_kernel( input_depth,
                                                                                                   k,
                                                                                                   alpha,
                                                                                                   beta,
                                                                                                   normalization_size );
        }
        break;

        case NN_NORMALIZATION_MODE_LINEAR_SINGLE:
        {
            uint_least32_t total_input_width = flow_item->input[0].item->output_format[flow_item->input[0].index].format_1d.size[0];
            uint_least32_t total_input_height = flow_item->input[0].item->output_format[flow_item->input[0].index].format >=
                                                NN_DATA_FORMAT_2D ? flow_item->input[0].item->output_format[flow_item->input[0].index].format_2d.size[1] : 1;
            uint_least32_t total_input_depth = flow_item->input[0].item->output_format[flow_item->input[0].index].format >=
                                               NN_DATA_FORMAT_3D ? flow_item->input[0].item->output_format[flow_item->input[0].index].format_3d.size[2] : 1;

            reinterpret_cast< device_gpu::ocl_toolkit * >( device )->prepare_norm_linear_single_kernel(
                                        output_layout,
                                        input_layout,
                                        total_input_width,
                                        total_input_height,
                                        total_input_depth,
                                        flow_item->arguments.forward_normalization.normalization.alpha,
                                        flow_item->arguments.forward_normalization.normalization.beta);
            break;
        }
        default:
            DBG_PRINTF( "Normalization mode not supported!\n" );
            assert( 0 );
            break;
        }
        break;
    }

    case NN_WORK_ITEM_TYPE_SOFTMAX:
    {
        assert( flow_item->input_count < 2 );

        uint_least32_t input_width;
        if( load_item->input[0]->type == NN_WORK_ITEM_TYPE_INPUT )
        {
            input_width = flow_item->input[0].item->output_format[flow_item->input[0].index].format_1d.size[0];
        }
        else
        {
            input_width =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_x ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_x ] +
                1;
        }

        reinterpret_cast< device_gpu::ocl_toolkit * >( device )->prepare_softmax_kernel( input_width, batch );
        break;
    }

    case NN_WORK_ITEM_TYPE_POOLING:
    {
        assert( flow_item->input_count < 2 );

        uint_least32_t input_width, input_height, input_depth, input_start_x, input_start_y, input_end_x, input_end_y;

        if( load_item->input[0]->type == NN_WORK_ITEM_TYPE_INPUT )
        {
            auto& input = flow_item->input[0];
            input_width = input.item->output_format[input.index].format_1d.size[0];
            input_height = input.item->output_format[input.index].format >=
                           NN_DATA_FORMAT_2D ? input.item->output_format[input.index].format_2d.size[1] : 1;

            input_depth = input.item->output_format[input.index].format >=
                           NN_DATA_FORMAT_2D ? input.item->output_format[input.index].format_3d.size[2] : 1;

            // Currently nn_data does not provide interface for views
            input_start_x = 0;
            input_start_y = 0;
            input_end_x   = input_width - 1;
            input_end_y   = input_height - 1;
        }
        else
        {
            input_width =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_x ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_x ] +
                1;
            input_height =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_y ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_y ] +
                1;
            input_depth =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_z ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_z ] +
                1;

            input_start_x = load_item->input[0]->output->view_begin.t[ NN_DATA_COORD_x ];
            input_start_y = load_item->input[0]->output->view_begin.t[ NN_DATA_COORD_y ];
            input_end_x   = load_item->input[0]->output->view_end.t[ NN_DATA_COORD_x ];
            input_end_y   = load_item->input[0]->output->view_end.t[ NN_DATA_COORD_y ];

        }
        uint_least32_t output_start_x = load_item->output->view_begin.t[ NN_DATA_COORD_x ];
        uint_least32_t output_start_y = load_item->output->view_begin.t[ NN_DATA_COORD_y ];
        uint_least32_t size_x         = flow_item->arguments.forward_pooling.size[0];
        uint_least32_t stride_x       = flow_item->arguments.forward_pooling.stride[0];

        reinterpret_cast< device_gpu::ocl_toolkit * >( device )->prepare_pooling_kernel(
                                            load_item->output->parent->cl_buffer[0] == nullptr,
                                            input_width,
                                            input_height,
                                            input_depth,
                                            input_start_x,
                                            input_start_y,
                                            input_end_x,
                                            input_end_y,
                                            output_start_x,
                                            output_start_y,
                                            size_x,
                                            stride_x );
        break;
    }
    case NN_WORK_ITEM_TYPE_FULLY_CONNECTED:
    {
        // one input so far is allowed
        assert( flow_item->input_count < 2 );

        // if input node to fully_connected is input workflow item then output is to be read from workflow
        // format
        // as load_item of input got invalid size of output (it is set to proper one at execute stage of
        // workload
        uint_least32_t input_width, input_height, input_depth;
        if( load_item->input[0]->type == NN_WORK_ITEM_TYPE_INPUT )
        {
            auto& input = flow_item->input[0];
            input_width = input.item->output_format[input.index].format_1d.size[0];
            input_height = input.item->output_format[input.index].format >=
                           NN_DATA_FORMAT_2D ? input.item->output_format[input.index].format_2d.size[1] : 1;
            input_depth = input.item->output_format[input.index].format >=
                          NN_DATA_FORMAT_3D ? input.item->output_format[input.index].format_3d.size[2] : 1;
        }
        else
        {
            input_width =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_x ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_x ] +
                1;
            input_height =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_y ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_y ] +
                1;
            input_depth =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_z ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_z ] +
                1;
        }
        uint_least32_t num_outputs =
            ( flow_item->output_format[0].format_1d.size[0] *
              ( flow_item->output_format[0].format >=
                NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1 ) *
              ( flow_item->output_format[0].format >=
                NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1 ) );

        NN_ACTIVATION_FUNCTION activation_function =
            flow_item->arguments.forward_fully_connected.activation.function;

        reinterpret_cast< device_gpu::ocl_toolkit * >( device )->prepare_fully_connected_kernel(
            load_item->output->parent->cl_buffer[0] == nullptr,
            input_width * input_height * input_depth, num_outputs, batch,
            activation_function );

        auto num_inputs = input_width * input_height * input_depth;


        break;
    }
    case NN_WORK_ITEM_TYPE_CONVOLUTION:
    {
        // one input so far is allowed
        assert( flow_item->input_count < 2 );

        // if input node to convolution is input workflow item then output is to be read from workflow format
        // as load_item of input got invalid size of output (it is set to proper one at execute stage of
        // workload
        uint_least32_t total_input_width, total_input_height, total_input_depth, input_width, input_height, input_depth, input_start_x, input_start_y, input_start_z, input_end_x,
                       input_end_y, input_end_z;
        if( load_item->input[0]->type == NN_WORK_ITEM_TYPE_INPUT )
        {
            auto& input = flow_item->input[0];
            input_width = input.item->output_format[input.index].format_1d.size[0];
            input_height = input.item->output_format[input.index].format >=
                           NN_DATA_FORMAT_2D ? input.item->output_format[input.index].format_2d.size[1] : 1;
            input_depth = input.item->output_format[input.index].format >=
                          NN_DATA_FORMAT_3D ? input.item->output_format[input.index].format_3d.size[2] : 1;

            // Currently nn_data does not provide interface for views
            input_start_x = 0;
            input_start_y = 0;
            input_start_z = 0;
            input_end_x   = input_width - 1;
            input_end_y   = input_height - 1;
            input_end_y   = input_depth - 1;
            total_input_width = input_width;
            total_input_height = input_height;
            total_input_depth = input_depth;
        }
        else
        {
            total_input_width = load_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_x];
            total_input_height= load_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_y];
            total_input_depth = load_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_z];

            input_width =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_x ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_x ] +
                1;
            input_height =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_y ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_y ] +
                1;
            input_depth =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_z ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_z ] +
                1;

            input_start_x = load_item->input[0]->output->view_begin.t[ NN_DATA_COORD_x ];
            input_start_y = load_item->input[0]->output->view_begin.t[ NN_DATA_COORD_y ];
            input_start_z = load_item->input[0]->output->view_begin.t[ NN_DATA_COORD_z ];
            input_end_x   = load_item->input[0]->output->view_end.t[ NN_DATA_COORD_x ];
            input_end_y   = load_item->input[0]->output->view_end.t[ NN_DATA_COORD_y ];
            input_end_z   = load_item->input[0]->output->view_end.t[ NN_DATA_COORD_z ];
        }

        uint_least32_t total_output_depth =
            load_item->output_view->parent->lengths.t[NN_DATA_COORD_z];

        uint_least32_t output_depth =
            load_item->output_view->view_end.t[NN_DATA_COORD_z] -
            load_item->output_view->view_begin.t[NN_DATA_COORD_z] +
            1;
        uint_least32_t output_height =
            load_item->output_view->view_end.t[NN_DATA_COORD_y] -
            load_item->output_view->view_begin.t[NN_DATA_COORD_y] +
            1;
        uint_least32_t output_width =
            load_item->output_view->view_end.t[NN_DATA_COORD_x] -
            load_item->output_view->view_begin.t[NN_DATA_COORD_x] +
            1;

        // TODO: Perhaps end of view should be read as well
        uint_least32_t output_start_x = load_item->output->view_begin.t[ NN_DATA_COORD_x ];
        uint_least32_t output_start_y = load_item->output->view_begin.t[ NN_DATA_COORD_y ];
        uint_least32_t output_start_z = load_item->output->view_begin.t[ NN_DATA_COORD_z ];
        uint_least32_t output_end_x   = load_item->output->view_end.t[ NN_DATA_COORD_x ];
        uint_least32_t output_end_y   = load_item->output->view_end.t[ NN_DATA_COORD_y ];
        uint_least32_t output_end_z   = load_item->output->view_end.t[ NN_DATA_COORD_z ];

        // TODO: make sure it is 3D buffer
        uint_least32_t weights_width  = flow_item->arguments.forward_convolution.weights->size[0];
        uint_least32_t weights_height = flow_item->arguments.forward_convolution.weights->size[1];
        uint_least32_t weights_depth  = flow_item->arguments.forward_convolution.weights->size[2];

        uint_least32_t         stride_x            = flow_item->arguments.forward_convolution.stride[0];
        uint_least32_t         stride_y            = flow_item->arguments.forward_convolution.stride[1];
        uint_least32_t         center_x            = flow_item->arguments.forward_convolution.center_offset[0];
        uint_least32_t         center_y            = flow_item->arguments.forward_convolution.center_offset[1];
        NN_ACTIVATION_FUNCTION activation_function =
            flow_item->arguments.forward_convolution.activation.function;

        uint_least32_t num_batches =
            load_item->output->view_end.t[NN_DATA_COORD_n] -
            load_item->output->view_begin.t[NN_DATA_COORD_n] + 1;

        // This assumes simetrical padding. TODO: make it more flexible
        auto output_buffer_offset = (load_item->output_h_pad_for_next_layer / 2) * load_item->output->parent->lengths.t[NN_DATA_COORD_x]
            + (load_item->output_w_pad_for_next_layer / 2);

        reinterpret_cast< device_gpu::ocl_toolkit * >( device )->prepare_conv_kernel( load_item->output->parent->cl_buffer[0] == nullptr,
                                                                                      output_width,
                                                                                      output_height,
                                                                                      output_start_z,
                                                                                      total_output_depth,
                                                                                      total_input_width,
                                                                                      total_input_height,
                                                                                      total_input_depth,
                                                                                      input_width,
                                                                                      input_height,
                                                                                      input_depth,
                                                                                      input_start_x,
                                                                                      input_start_y,
                                                                                      input_start_z,
                                                                                      weights_width,
                                                                                      weights_height,
                                                                                      weights_depth,
                                                                                      output_depth,
                                                                                      stride_x,
                                                                                      stride_y,
                                                                                      activation_function,
                                                                                      num_batches,
                                                                                      output_buffer_offset,
                                                                                      load_item->output_w_pad_for_next_layer,
                                                                                      load_item->output_h_pad_for_next_layer
                                                                                      );

        uint32_t kernel_batch = reinterpret_cast< device_gpu::ocl_toolkit * >(device)->get_batch(
                                           load_item->output->parent->cl_buffer[0] == nullptr,
                                                                                   output_width, //Padding is not be mentioned in OCL kernel preparation
                                                                                   output_height,
                                                                                   output_start_z,
                                                                                   total_output_depth,
                                                                                   total_input_width,
                                                                                   total_input_height,
                                                                                   total_input_depth,
                                                                                   input_width,
                                                                                   input_height,
                                                                                   input_depth,
                                                                                   input_start_x,
                                                                                   input_start_y,
                                                                                   input_start_z,
                                                                                   weights_width,
                                                                                   weights_height,
                                                                                   weights_depth,
                                                                                   output_depth,
                                                                                   stride_x,
                                                                                   stride_y,
                                                                                   activation_function,
                                                                                   num_batches,
                                                                                   output_buffer_offset,
                                                                                   load_item->output_w_pad_for_next_layer,
                                                                                   load_item->output_h_pad_for_next_layer
                                                                                   );

        for (uint32_t image = 0; image < batch; image += kernel_batch)
        {
            load_item->input[0]->output->add_data_subbuffer(
                0,
                image * total_input_width * total_input_height * total_input_depth * sizeof(float),
                kernel_batch * total_input_width * total_input_height * total_input_depth * sizeof(float));
        }

    }
    break;
    case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2:
    {
        // one input so far is allowed
        assert( flow_item->input_count < 2 );

        // if input node to convolution is input workflow item then output is to be read from workflow format
        // as load_item of input got invalid size of output (it is set to proper one at execute stage of
        // workload
        uint_least32_t input_width, input_height, input_depth;
        if( load_item->input[0]->type == NN_WORK_ITEM_TYPE_INPUT )
        {
            auto& input = flow_item->input[0];
            input_width = input.item->output_format[input.index].format_1d.size[0];
            input_height = input.item->output_format[input.index].format >=
                           NN_DATA_FORMAT_2D ? input.item->output_format[input.index].format_2d.size[1] : 1;
            input_depth = input.item->output_format[input.index].format >=
                          NN_DATA_FORMAT_3D ? input.item->output_format[input.index].format_3d.size[2] : 1;
        }
        else
        {
            input_width =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_x ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_x ] +
                1;
            input_height =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_y ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_y ] +
                1;
            input_depth =
                load_item->input[0]->output->view_end.t[NN_DATA_COORD_z ] -
                load_item->input[0]->output->view_begin.t[NN_DATA_COORD_z ] +
                1;
        }

        uint_least32_t output_depth =
            load_item->output_view->view_end.t[NN_DATA_COORD_z] -
            load_item->output_view->view_begin.t[NN_DATA_COORD_z] +
            1;
        uint_least32_t output_height =
            load_item->output_view->view_end.t[NN_DATA_COORD_y] -
            load_item->output_view->view_begin.t[NN_DATA_COORD_y] +
            1;
        uint_least32_t output_width =
            load_item->output_view->view_end.t[NN_DATA_COORD_x] -
            load_item->output_view->view_begin.t[ NN_DATA_COORD_x ] +
            1;
        // TODO: make sure it is 3D buffer

        uint_least32_t weights_width =
            flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights->size[0];
        uint_least32_t weights_height =
            flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights->size[1];
        uint_least32_t weights_depth =
            flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights->size[2];

        auto stride_x            = flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.stride[0];
        auto stride_y            = flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.stride[1];
        auto center_x            = flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.center_offset[0];
        auto center_y            = flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.center_offset[1];
        auto activation_function =
            flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.activation.function;

        // This assumes simetrical padding. TODO: make it more flexible
        auto output_buffer_offset = (load_item->output_h_pad_for_next_layer / 2) * load_item->output->parent->lengths.t[NN_DATA_COORD_x]
            + (load_item->output_w_pad_for_next_layer / 2);

        reinterpret_cast< device_gpu::ocl_toolkit * >( device )->prepare_conv_maxpool_kernel( output_width,
                                                                                              output_height,
                                                                                              input_width,
                                                                                              input_height,
                                                                                              input_depth,
                                                                                              weights_width,
                                                                                              weights_height,
                                                                                              weights_depth,
                                                                                              output_depth,
                                                                                              stride_x,
                                                                                              stride_y,
                                                                                              activation_function,
                                                                                              2,
                                                                                              2,
                                                                                              NN_POOLING_MODE_MAX,
                                                                                              output_buffer_offset,
                                                                                              load_item->output_w_pad_for_next_layer,
                                                                                              load_item->output_h_pad_for_next_layer
                                                                                              );

        for (uint32_t image = 0; image < batch; image += 1)
        {
            load_item->input[0]->output->add_data_subbuffer(
                0,
                image * input_depth * input_height * input_width * sizeof(float),
                1 * input_depth * input_height * input_width * sizeof(float));
        }


        break;
    }
    default:
        DBG_PRINTF( "Add offline compilation for this one!\n" );
        break;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
NN_API_STATUS nn_workload_item_normalization_same_map_start(
    nn_device_t    *const context,
    nn_gpu_workload_item *const work_item)
{
    uint_least32_t num_batches =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_n ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_n
        ] +
        1;

    uint_least32_t input_feature_map_width =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_x ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_x ] +
        1;

    uint_least32_t input_feature_map_height =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_y ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_y ] +
        1;
    uint_least32_t num_input_feature_maps =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_z ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_z ] +
        1;

    auto output_buffer_offset = ( work_item->output_h_pad_for_next_layer / 2 ) * work_item->output->parent->lengths.t[NN_DATA_COORD_x]
        + ( work_item->output_w_pad_for_next_layer / 2 );

    reinterpret_cast< device_gpu::ocl_toolkit * >( context )->normalize(
        work_item->output,
        work_item->input[0]->output,
        num_batches,
        num_input_feature_maps,
        input_feature_map_width,
        input_feature_map_height,
        work_item->arguments.forward_normalization.normalization.k,
        work_item->arguments.forward_normalization.normalization.alpha,
        work_item->arguments.forward_normalization.normalization.beta,
        work_item->arguments.forward_normalization.normalization.n,
        work_item->arguments.forward_normalization.normalization.mode,
        work_item->output->parent->buffer_size,
        output_buffer_offset,
        work_item->output_w_pad_for_next_layer,
        work_item->output_h_pad_for_next_layer
        );
    return NN_API_STATUS_OK;

}
///////////////////////////////////////////////////////////////////////////////////////////////////
NN_API_STATUS nn_workload_item_output_start( nn_device_t *const context, nn_gpu_workload_item *const work_item )            /* work item to be started */
{
    assert(work_item->input.size() == 1);    //TODO: so far one input is supported

    // Make sure all previous workload_items are computeed (finished)
    reinterpret_cast< device_gpu::ocl_toolkit * >( context )->finish();

/*    ( workload_data_cast<nn::layout_f32>( work_item->output ) )->copy( (work_item->input[0]->output_view != nullptr) ?
                                                                                      *work_item->input[0]->output_view :
                                                                                      *workload_data_cast<nn::layout_f32>(work_item->input[0]->output) );*/

    return NN_API_STATUS_OK;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
NN_API_STATUS nn_workload_item_softmax_start( nn_device_t *const context, nn_gpu_workload_item *const work_item )            /* work item to be started */
{
    assert( work_item->input.size() == 1 );    //TODO: so far one input is supported

    // Take output of previous (input) workload item as input of softmax(current) one
    uint_least32_t num_batches =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_n ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_n
        ] +
        1;

    uint_least32_t num_samples =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_x ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_x
        ] + 1;

    reinterpret_cast< device_gpu::ocl_toolkit * >( context )->softmax(
        work_item->output,
        work_item->input[0]->output,
        num_samples,
        num_batches );
    return NN_API_STATUS_OK;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
static NN_API_STATUS nn_workload_item_norm_linear_single_start(
    nn_device_t    *const       context,
    NN_WORKLOAD_DATA_TYPE       output_layout,
    NN_WORKLOAD_DATA_TYPE       input_layout,
    nn_gpu_workload_item *const work_item )
{
    uint_least32_t total_input_width  = work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_x];
    uint_least32_t total_input_height = work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_y];
    uint_least32_t total_input_depth  = work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_z];
    uint_least32_t total_batch        = work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_n];

    // Run proper convolution depending if bias was used to output initialization or not
    reinterpret_cast< device_gpu::ocl_toolkit * >( context )->norm_linear_single(
        work_item->output,
        work_item->input[0]->output,
        output_layout,
        input_layout,
        total_input_depth,
        total_input_width,
        total_input_height,
        work_item->arguments.forward_normalization.normalization.alpha,
        work_item->arguments.forward_normalization.normalization.beta,
        total_batch
        );
    return NN_API_STATUS_OK;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
NN_API_STATUS nn_workload_item_normalization_start( nn_device_t *const          context,
                                                    NN_WORKLOAD_DATA_TYPE       output_layout,
                                                    NN_WORKLOAD_DATA_TYPE       input_layout,
                                                    nn_gpu_workload_item *const work_item )
{
    assert(work_item->input.size() == 1);    //TODO: so far one input is supported

    switch (work_item->arguments.forward_normalization.normalization.mode)
    {
    case NN_NORMALIZATION_MODE_LINEAR_SINGLE:
        return nn_workload_item_norm_linear_single_start( context, output_layout, input_layout, work_item );
        break;

    case NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS:
        return nn_workload_item_normalization_same_map_start(context, work_item);
        break;
    case NN_NORMALIZATION_MODE_CONTRAST:
    case NN_NORMALIZATION_MODE_RESPONSE_SAME_MAP:
    default:
        // TODO
        assert(0);
        return NN_API_STATUS_ERROR_INVALID_WORK_ITEM_TYPE;
        break;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
NN_API_STATUS nn_workload_item_full_connectivity_start(
    nn_device_t    *const context,
    nn_gpu_workload_item *const work_item)
{
    assert( work_item->input.size() == 1 );    //TODO: so far one input is supported

    uint_least32_t num_batches =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_n ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_n
        ] +
        1;

    uint_least32_t input_width =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_x ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_x ] +
        1;
    uint_least32_t input_height =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_y ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_y ] +
        1;
    uint_least32_t input_depth =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_z ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_z ] +
        1;

    uint_least32_t output_width =
        work_item->output->view_end.t[NN_DATA_COORD_x ] -
        work_item->output->view_begin.t[ NN_DATA_COORD_x ] +
        1;

    uint_least32_t weights_width =
        work_item->arguments.forward_fully_connected.weights->view_end.t[NN_DATA_COORD_x ] -
        work_item->arguments.forward_fully_connected.weights->view_begin.t[NN_DATA_COORD_x
        ] +
        1;
    uint_least32_t weights_height =
        work_item->arguments.forward_fully_connected.weights->view_end.t[NN_DATA_COORD_y ] -
        work_item->arguments.forward_fully_connected.weights->view_begin.t[NN_DATA_COORD_y
        ] +
        1;
    uint_least32_t weights_depth =
        work_item->arguments.forward_fully_connected.weights->view_end.t[NN_DATA_COORD_z ] -
        work_item->arguments.forward_fully_connected.weights->view_begin.t[NN_DATA_COORD_z
        ] +
        1;

    uint_least32_t weights_p_size =
        work_item->arguments.forward_fully_connected.weights->view_end.t[NN_DATA_COORD_p ] -
        work_item->arguments.forward_fully_connected.weights->view_begin.t[NN_DATA_COORD_p
        ] +
        1;

    // Run proper convolution depending if bias was used to output initialization or not
    reinterpret_cast<device_gpu::ocl_toolkit *>(context)->fully_connect(
        work_item->output,
        work_item->input[0]->output,
        work_item->arguments.forward_fully_connected.weights,
        work_item->arguments.forward_fully_connected.biases,
        output_width,
        input_width,
        input_height,
        input_depth,
        weights_width * weights_height * weights_depth * weights_p_size,
        num_batches,
        work_item->arguments.forward_fully_connected.activation.function );
    return NN_API_STATUS_OK;

}
///////////////////////////////////////////////////////////////////////////////////////////////////
NN_API_STATUS  nn_workload_item_arithmetic_start( nn_device_t *const          context,
                                                  NN_WORKLOAD_DATA_TYPE       output_layout,
                                                  NN_WORKLOAD_DATA_TYPE       input_layout,
                                                  nn_gpu_workload_item *const work_item )
{
    uint_least32_t num_batches =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_n ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_n ] + 1;

    uint_least32_t input_width =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_x ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_x ] +
        1;
    uint_least32_t input_height =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_y ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_y ] +
        1;
    uint_least32_t input_depth =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_z ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_z ] +
        1;

    reinterpret_cast< device_gpu::ocl_toolkit * >( context )->arithmetize(
        work_item->output,
        work_item->input[0]->output,
        work_item->arguments.forward_arithmetic.factor,
        output_layout,
        input_layout,
        input_depth,
        input_width,
        input_height,
        work_item->arguments.forward_arithmetic.arithmetic_func,
        num_batches );

    return NN_API_STATUS_OK;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
NN_API_STATUS nn_workload_item_convolution_start(
    nn_device_t    *const context,
    nn_gpu_workload_item *const work_item)
{
    uint_least32_t num_batches =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_n ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_n
        ] +
        1;

    uint_least32_t input_width =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_x ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_x ] +
        1;
    uint_least32_t input_height =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_y ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_y ] +
        1;
    uint_least32_t input_depth =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_z ] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_z ] +
        1;

    uint_least32_t input_start_x = work_item->input[0]->output->view_begin.t[ NN_DATA_COORD_x ];
    uint_least32_t input_start_y = work_item->input[0]->output->view_begin.t[ NN_DATA_COORD_y ];
    uint_least32_t input_start_z = work_item->input[0]->output->view_begin.t[ NN_DATA_COORD_z ];

    uint_least32_t output_width =
        work_item->output_view->view_end.t[ NN_DATA_COORD_x ] -
        work_item->output_view->view_begin.t[ NN_DATA_COORD_x ] +
        1;
    uint_least32_t output_height =
        work_item->output_view->view_end.t[ NN_DATA_COORD_y ] -
        work_item->output_view->view_begin.t[ NN_DATA_COORD_y ] +
        1;
    uint_least32_t output_depth =
        work_item->output_view->view_end.t[ NN_DATA_COORD_z ] -
        work_item->output_view->view_begin.t[ NN_DATA_COORD_z ] +
        1;

    uint_least32_t output_start_z = work_item->output->view_begin.t[ NN_DATA_COORD_z ];

    uint_least32_t weights_width =
        work_item->arguments.forward_convolution.weights->view_end.t[NN_DATA_COORD_x ] -
        work_item->arguments.forward_convolution.weights->view_begin.t[NN_DATA_COORD_x
        ] +
        1;
    uint_least32_t weights_height =
        work_item->arguments.forward_convolution.weights->view_end.t[NN_DATA_COORD_y ] -
        work_item->arguments.forward_convolution.weights->view_begin.t[NN_DATA_COORD_y
        ] +
        1;
    uint_least32_t weights_depth =
        work_item->arguments.forward_convolution.weights->view_end.t[NN_DATA_COORD_z ] -
        work_item->arguments.forward_convolution.weights->view_begin.t[NN_DATA_COORD_z
        ] +
        1;


    auto output_buffer_offset = ( work_item->output_h_pad_for_next_layer / 2 ) * work_item->output->parent->lengths.t[NN_DATA_COORD_x]
        + ( work_item->output_w_pad_for_next_layer / 2 );

    // Run proper convolution depending if bias was used to output initialization or not
    reinterpret_cast< device_gpu::ocl_toolkit * >( context )->convolve(
        work_item->output,
        work_item->input[0]->output,
        work_item->arguments.forward_convolution.weights,
        work_item->arguments.forward_convolution.biases,
        work_item->output->parent->lengths.t[NN_DATA_COORD_z],  //Total buffer depth
        output_width,
        output_height,
        output_depth,
        work_item->output->parent->buffer_size,
        output_start_z,
        work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_x],
        work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_y],
        work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_z],
        input_width,
        input_height,
        input_depth,
        input_start_x,
        input_start_y,
        input_start_z,
        weights_width,
        weights_height,
        weights_depth,
        num_batches,
        work_item->arguments.forward_convolution.stride[0],
        work_item->arguments.forward_convolution.stride[1],
        work_item->arguments.forward_convolution.activation.function,
        output_buffer_offset,
        work_item->output_w_pad_for_next_layer,
        work_item->output_h_pad_for_next_layer
        );
    return NN_API_STATUS_OK;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
NN_API_STATUS nn_workload_item_conv_maxpool_start(
    nn_device_t    *const context,
    nn_gpu_workload_item *const work_item)
{
    uint_least32_t num_batches =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_n] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_n] + 1;

    uint_least32_t input_width =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_x] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_x] + 1;
    uint_least32_t input_height =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_y] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_y] + 1;
    uint_least32_t input_depth =
        work_item->input[0]->output->view_end.t[NN_DATA_COORD_z] -
        work_item->input[0]->output->view_begin.t[NN_DATA_COORD_z] + 1;

    uint_least32_t output_width =
        work_item->output_view->view_end.t[ NN_DATA_COORD_x ] -
        work_item->output_view->view_begin.t[ NN_DATA_COORD_x ] + 1;
    uint_least32_t output_height =
        work_item->output_view->view_end.t[ NN_DATA_COORD_y ] -
        work_item->output_view->view_begin.t[ NN_DATA_COORD_y ] + 1;
    uint_least32_t output_depth =
        work_item->output_view->view_end.t[ NN_DATA_COORD_z ] -
        work_item->output_view->view_begin.t[ NN_DATA_COORD_z ] + 1;

    uint_least32_t weights_width =
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2 .weights->view_end.t[NN_DATA_COORD_x] -
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights->view_begin.t[NN_DATA_COORD_x] + 1;
    uint_least32_t weights_height =
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights->view_end.t[NN_DATA_COORD_y] -
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights->view_begin.t[NN_DATA_COORD_y] + 1;
    uint_least32_t weights_depth =
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights->view_end.t[NN_DATA_COORD_z] -
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights->view_begin.t[NN_DATA_COORD_z] + 1;

    // Run proper convolution depending if bias was used to output initialization or not
    reinterpret_cast< device_gpu::ocl_toolkit * >(context)->convolve_maxpool(
        work_item->output,
        work_item->input[0]->output,
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights,
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.biases,
        output_width,
        output_height,
        output_depth,
        work_item->output->parent->buffer_size,
        input_width,
        input_height,
        input_depth,
        weights_width,
        weights_height,
        weights_depth,
        num_batches,
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.stride[0],
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.stride[1],
        work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.activation.function,
        2, 2, 2, 2,
        NN_POOLING_MODE_MAX
        );

    return NN_API_STATUS_OK;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
NN_API_STATUS nn_workload_item_pooling_start( nn_device_t *const context, nn_gpu_workload_item *const work_item )            /* work item to be started */
{
    assert( work_item->input.size() == 1 );    //TODO: so far one input is supported

    uint_least32_t num_batches       = work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_n];
    uint_least32_t input_width       = work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_x];
    uint_least32_t input_height      = work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_y];
    uint_least32_t input_depth       = work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_z];
    uint_least32_t output_width      = work_item->output->parent->lengths.t[NN_DATA_COORD_x];
    uint_least32_t output_height     = work_item->output->parent->lengths.t[NN_DATA_COORD_y];
    uint_least32_t output_depth      = work_item->output->parent->lengths.t[NN_DATA_COORD_z];

    cl::NDRange output_start_offset( work_item->output->view_begin.t[ NN_DATA_COORD_x ],
                                     work_item->output->view_begin.t[ NN_DATA_COORD_y ],
                                     work_item->output->view_begin.t[ NN_DATA_COORD_z ] );
    cl::NDRange output_end_offset( work_item->output->view_end.t[ NN_DATA_COORD_x ],
                                   work_item->output->view_end.t[ NN_DATA_COORD_y ],
                                   work_item->output->view_end.t[ NN_DATA_COORD_z ] );

    cl::NDRange input_start_offset( work_item->input[0]->output->view_begin.t[ NN_DATA_COORD_x ],
                                    work_item->input[0]->output->view_begin.t[ NN_DATA_COORD_y ],
                                    work_item->input[0]->output->view_begin.t[ NN_DATA_COORD_z ] );
    cl::NDRange input_end_offset( work_item->input[0]->output->view_end.t[ NN_DATA_COORD_x ],
                                  work_item->input[0]->output->view_end.t[ NN_DATA_COORD_y ],
                                  work_item->input[0]->output->view_end.t[ NN_DATA_COORD_z ] );

    uint_least32_t output_batch_start = work_item->output->view_begin.t[ NN_DATA_COORD_n ];
    uint_least32_t output_batch_end   = work_item->output->view_end.t[ NN_DATA_COORD_n ];
    uint_least32_t input_batch_start  = work_item->input[0]->output->view_begin.t[ NN_DATA_COORD_n ];
    uint_least32_t input_batch_end    = work_item->input[0]->output->view_end.t[ NN_DATA_COORD_n ];
    reinterpret_cast< device_gpu::ocl_toolkit * >( context )->max_pool(
        work_item->output,
        work_item->input[0]->output,
        output_start_offset,
        output_end_offset,
        input_start_offset,
        input_end_offset,
        output_batch_start,
        output_batch_end,
        input_batch_start,
        input_batch_end,
        output_width,
        output_height,
        output_depth,
        input_width,
        input_height,
        input_depth,
        num_batches,
        work_item->arguments.forward_pooling.stride[0],
        work_item->arguments.forward_pooling.stride[1],
        work_item->arguments.forward_pooling.size[0],
        work_item->arguments.forward_pooling.size[1],
        work_item->arguments.forward_pooling.mode );
    return NN_API_STATUS_OK;
}
};
///////////////////////////////////////////////////////////////////////////////////////////////////

/* query workflow for metrics of compilation variants */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_metrics_query_0x0_function(
    nn_workflow_metrics_array_t **array,/* resulting array of variants */
    nn_device_t        *device,         /* device context */
    nn_workflow_t      *workflow        /* workflow to be querried */
    )
{
    assert(0);  //TODO: Implement function
    return NN_API_STATUS_ERROR_OTHER;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

/* delete array of workload metrics */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_metrics_delete_0x0_function(
    nn_workflow_metrics_array_t *array  /* array to delete */
    )
{

    assert(0);  //TODO: Implement function
    return NN_API_STATUS_ERROR_OTHER;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

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

            if (temp_left_padding > left_padding) left_padding = temp_left_padding;
            if (temp_right_padding > right_padding) right_padding = temp_right_padding;
            if (temp_top_padding > top_padding) top_padding = temp_top_padding;
            if (temp_bottom_padding > bottom_padding) bottom_padding = temp_bottom_padding;
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

    for (size_t it_use = 0; it_use < flow_item->use_count; ++it_use){
        auto& use_item = flow_item->use[it_use].item;
        if (use_item->type == NN_WORK_ITEM_TYPE_VIEW)
        {
            nn_workflow_compile_0_function_calculate_output_padding(use_item, left_padding, right_padding, top_padding, bottom_padding);
        }
        else
        {
            nn_workflow_compile_0_function_update_output_padding_for_use<NN_WORK_ITEM_TYPE_CONVOLUTION>(use_item, output_w, output_h, left_padding, right_padding, top_padding, bottom_padding);
            nn_workflow_compile_0_function_update_output_padding_for_use<NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2>(use_item, output_w, output_h, left_padding, right_padding, top_padding, bottom_padding);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/* compile workflow into workload */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_compile_0x0_function(
    nn_workload_t     **workload,       /* resulting workload */
    nn_device_t        *device,         /* device context */
    nn_workflow_t      *workflow,       /* workflow to be compiled */
    NN_WORKLOAD_DATA_TYPE       *input_format,   /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE       *output_format,  /* array containing formats of outputs */
    uint32_t            batch           /* batch size for compilation */
    )
{
    if( !workload || !device || !workflow )
        return NN_API_STATUS_ERROR_INVALID_POINTER;

    //TODO: Add support for many inputs & outputs
    if( workflow->input_count != 1 )
        return NN_API_STATUS_ERROR_INVALID_INPUT_COUNT;

    if( workflow->output_count != 1 )
        return NN_API_STATUS_ERROR_INVALID_OUTPUT_COUNT;

    if( !workflow->input || !workflow->output )
        return NN_API_STATUS_ERROR_INVALID_WORKFLOW;

    // TODO: more granular error code here
    for( auto index = 0u; index < workflow->input_count; ++index )
        if( workflow->input[index]->type != NN_WORK_ITEM_TYPE_INPUT )
            return NN_API_STATUS_ERROR_INVALID_WORKFLOW;
    // TODO: more granular error code here
    for( auto index = 0u; index < workflow->output_count; ++index )
        if( workflow->output[index]->type != NN_WORK_ITEM_TYPE_OUTPUT )
            return NN_API_STATUS_ERROR_INVALID_WORKFLOW;

    // 1. Alocation of workload object
    try {
        nn_gpu_workload *gpu_workload   = new nn_gpu_workload;
        nn_workload_t   *dummy_workload =
            reinterpret_cast< nn_workload * >( new  char[sizeof( nn_workload_t )] );

        // This made my remaining faith in C++ vanished
        *( const_cast< nn_device_t ** >( &( dummy_workload->device ) ) )         = device;
        *( const_cast< uint32_t * >( &( dummy_workload->input_count ) ) )        = 1;
        *( const_cast< uint32_t * >( &( dummy_workload->output_count ) ) )       = 1;
        *( const_cast< NN_WORKLOAD_DATA_TYPE ** >( &( dummy_workload->input_format ) ) )  = new NN_WORKLOAD_DATA_TYPE;
        const_cast< NN_WORKLOAD_DATA_TYPE * >( dummy_workload->input_format )[0]          = input_format[0];
        *( const_cast< NN_WORKLOAD_DATA_TYPE ** >( &( dummy_workload->output_format ) ) ) = new NN_WORKLOAD_DATA_TYPE;
        const_cast< NN_WORKLOAD_DATA_TYPE * >( dummy_workload->output_format )[0]         = output_format[0];
        *const_cast<uint32_t *>(&dummy_workload->batch) = batch;

        memcpy( gpu_workload->nn_workload_placeholder, dummy_workload, sizeof( nn_workload ) );
        delete[] reinterpret_cast< char * >( dummy_workload );
        gpu_workload->m_workload_items.clear();

        //TODO: Make sure that first member of struct is aligned with beginning of struct itself
        *workload = reinterpret_cast< nn_workload * >( gpu_workload->nn_workload_placeholder );

        // lookup for matching workflow items to workload items
        std::map< nn_workflow_item_t *, nn_gpu_workload_item_t * > flow_to_work;

        // lambda for copying arguments between workflow and workload items
        auto copy_item =
        [device, input_format, output_format, batch, &flow_to_work]( nn_gpu_workload_item_t * load_item, nn_workflow_item_t * flow_item ){

            nn_workload_data_layout_t layout = nn::layout_t<nn::layout_xyzpqn_f32>::layout;
            // copy type & create nn_workload_data_t
            load_item->type = flow_item->type;

            // calculate needed output buffer paddings
            size_t padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0;
            nn_workflow_compile_0_function_calculate_output_padding(flow_item, padding_left, padding_right, padding_top, padding_bottom);
            load_item->output_w_pad_for_next_layer = padding_left + padding_right;
            load_item->output_h_pad_for_next_layer = padding_top + padding_bottom;

            switch( load_item->type )
            {
            case NN_WORK_ITEM_TYPE_INPUT:
                load_item->arguments.output.index = 0;
                {
                    nn_workload_data_coords_t size = {
                        batch,
                        flow_item->output_format[0].format_1d.size[0],
                        flow_item->output_format[0].format >=
                        NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
                        flow_item->output_format[0].format >=
                        NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1,
                        1, 1
                    };

                    nn::workload_data<> temp(size, layout);
                    if(handy_routines::is_image_as_output_needed(flow_item,batch,device)) {
                        // image
                        load_item->output = new nn_cl_data(
                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                        CL_MEM_READ_WRITE,
                        &temp,
                        nullptr,
                        true,
                        size.t[1]*size.t[2]*size.t[3], // width*height*depth == num_inputs of next layer
                        batch);
                    } else {
                        // If next layer is convolution or convolution_pooling (merged one) then make input buffer here
                        // as well as subbuffers in ocl kernels preparation stage
                        // TODO: This step can be deferred to execute as well
                   //     if((flow_item->use[0]->type == NN_WORK_ITEM_TYPE_CONVOLUTION) || (flow_item->use[0]->type == NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2))
                   //     {
                            load_item->output = new nn_cl_data(
                            reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                            CL_MEM_READ_WRITE,
                            &temp);
                   //     } else {
                   //         load_item->output = nullptr;
                   //     }
                    }
                }

                break;
            case NN_WORK_ITEM_TYPE_OUTPUT:
                load_item->output                 = nullptr;
                load_item->arguments.output.index = 0;
                break;

            case NN_WORK_ITEM_TYPE_LOCAL_CONNECTIVITY: {
                throw NN_API_STATUS_ERROR_INVALID_WORK_ITEM_TYPE;
            }
            case NN_WORK_ITEM_TYPE_CONVOLUTION:
            case NN_WORK_ITEM_TYPE_ARITHMETIC:
            case NN_WORK_ITEM_TYPE_POOLING:
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED:
            case NN_WORK_ITEM_TYPE_NORMALIZATION:
            case NN_WORK_ITEM_TYPE_SOFTMAX:
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: {
                nn_workload_data_coords_t size = {
                    batch,
                    flow_item->output_format[0].format_1d.size[ 0 ] + static_cast<uint32_t>( padding_left + padding_right ),
                    flow_item->output_format[0].format >=
                    NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] + static_cast<uint32_t>( padding_top + padding_bottom )  : 1,
                    flow_item->output_format[0].format >=
                    NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1,
                    1, 1
                };

                nn::workload_data<> temp(size, layout);

                if(handy_routines::is_image_as_output_needed(flow_item,batch,device)) {
                    // image
                    load_item->output = new nn_cl_data(
                    reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                    CL_MEM_READ_WRITE,
                    &temp,
                    nullptr,
                    true,
                    size.t[1]*size.t[2]*size.t[3], // width*height*depth == num_inputs of next layer
                    batch);
                } else {
                    load_item->output = new nn_cl_data(
                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                        CL_MEM_READ_WRITE,
                        &temp);
                }

                if (padding_left != 0 || padding_right != 0 || padding_top != 0 || padding_bottom != 0)
                {
                    nn_workload_data_coords_t start_coords( 0, padding_left, padding_top, 0, 0, 0 );

                    nn_workload_data_coords_t end_coords(
                        load_item->output->parent->lengths.t[ 0 ] - 1,
                        load_item->output->parent->lengths.t[ 1 ] - padding_right - 1,
                        load_item->output->parent->lengths.t[ 2 ] - padding_bottom - 1,
                        load_item->output->parent->lengths.t[ 3 ] - 1,
                        load_item->output->parent->lengths.t[ 4 ] - 1,
                        load_item->output->parent->lengths.t[ 5 ] - 1
                        );

                    load_item->output_view.reset( new nn_cl_data( *load_item->output, start_coords, end_coords ) );
                }
                else
                {

                    load_item->output_view.reset( new nn_cl_data( *load_item->output,
                                                                  load_item->output->view_begin, load_item->output->view_end ) );
                }

                break;
            }
            case NN_WORK_ITEM_TYPE_VIEW:
            {
                auto& origin = flow_item->arguments.view.origin;
                nn_workload_data_coords_t start(0, origin[0], origin[1], origin[2], 0, 0);
                nn_workload_data_coords_t end(
                batch - 1,
                origin[0] +  flow_item->output_format[0].format_1d.size[0] - 1 + padding_left + padding_right,
                origin[1] + (flow_item->output_format[0].format>=NN_DATA_FORMAT_2D ?
                 flow_item->output_format[0].format_2d.size[1] : 1) - 1 + padding_top + padding_bottom,
                origin[2] + (flow_item->output_format[0].format>=NN_DATA_FORMAT_3D ?
                 flow_item->output_format[0].format_3d.size[2] : 1) - 1,
                0, 0
                );
                // If INPUT node is precedessor of VIEW then output of INPUT is not known
                if(flow_to_work[flow_item->input[0].item]->type != NN_WORK_ITEM_TYPE_INPUT) {

                    load_item->output = new nn_cl_data(
                        *flow_to_work[flow_item->input[0].item]->output,
                        start,
                        end);

                    // For debug only - output_view is convenient to verify correctness of activations between layers
                    nn_workload_data_coords_t start_coords( 0, padding_left, padding_top, 0, 0, 0 );
                    nn_workload_data_coords_t end_coords(
                        load_item->output->view_end.t[0] - load_item->output->view_begin.t[0],
                        load_item->output->view_end.t[1] - load_item->output->view_begin.t[1] - padding_right,
                        load_item->output->view_end.t[2] - load_item->output->view_begin.t[2] - padding_bottom,
                        load_item->output->view_end.t[3] - load_item->output->view_begin.t[3],
                        load_item->output->view_end.t[4] - load_item->output->view_begin.t[4],
                        load_item->output->view_end.t[5] - load_item->output->view_begin.t[5]
                    );

                    load_item->output_view.reset( new nn_cl_data( *load_item->output, start_coords, end_coords ) );


                } else {
                    //TODO: Modify this code when VIEWs on INPUT are supported
                    nn_workload_data_coords_t size = {
                        1,
                        flow_item->output_format[0].format_1d.size[0],
                        flow_item->output_format[0].format >=
                        NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1,
                        flow_item->output_format[0].format >=
                        NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1,
                        1, 1
                    };
                    // for INPUT node we do all intitialization in execute
                    nn::workload_data<> temp(size, layout);
                    load_item->output = new nn_cl_data(
                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                        CL_MEM_READ_WRITE,
                        &temp);
                }

                break;
            }
            // TODO: Move all of this stuff to some separate function
            case NN_WORK_ITEM_TYPE_MERGE: {
                uint16_t axis = flow_item->arguments.forward_merge.axis; // x = 0, y = 1, z = 2

                auto input_data = flow_to_work[flow_item->input[0].item]->output;
                nn_workload_data_layout_t previous_layout = input_data->parent->layout;

                // Calculate total size needed for all input buffers of merge layer

                uint16_t x_size = input_data->parent->lengths.t[1];
                uint16_t y_size = input_data->parent->lengths.t[2];
                uint16_t z_size = input_data->parent->lengths.t[3];
                for (int index = 1; index < flow_item->input_count; index++)
                {
                    auto input_data_local = flow_to_work[flow_item->input[index].item]->output;
                    if (input_data->parent->layout!=previous_layout)
                        assert(0);

                    if (axis == 0) {
                        x_size += input_data_local->parent->lengths.t[1];
                    } else if (axis == 1) {
                        y_size += input_data_local->parent->lengths.t[2];
                    } else if (axis == 2) {
                        z_size += input_data_local->parent->lengths.t[3];
                    }
                }

                nn_workload_data_coords_t size(
                    input_data->parent->lengths.t[0],
                    x_size,
                    y_size,
                    z_size,
                    input_data->parent->lengths.t[4],
                    input_data->parent->lengths.t[5]
                );

                // allocate
                //load_item->output = new nn_workload_data_t;
                //nn_workload_data_placement_create(load_item->output, nullptr, &size, &previous_layout);

                nn::workload_data<> temp(size, previous_layout);
                load_item->output = new nn_cl_data(
                    reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                    CL_MEM_READ_WRITE,
                    &temp);

                nn_workload_data_coords_t start_coords(0, 0, 0, 0, 0, 0);

                nn_workload_data_coords_t end_coords(
                    input_data->parent->lengths.t[0] - 1,
                    x_size - 1,
                    y_size - 1,
                    z_size - 1,
                    input_data->parent->lengths.t[4] - 1,
                    input_data->parent->lengths.t[5] - 1
                );

                x_size = 0;
                y_size = 0;
                z_size = 0;

                // pin to the input buffers
                for (int index = 0; index < flow_item->input_count; index++)
                {
                    input_data = flow_to_work[flow_item->input[index].item]->output;

                    if (axis == 0)
                    {
                        start_coords.t[1] = x_size;
                        x_size += input_data->parent->lengths.t[1];
                        end_coords.t[1] = x_size - 1;
                    }
                    else if (axis == 1)
                    {
                        start_coords.t[2] = y_size;
                        y_size += input_data->parent->lengths.t[2];
                        end_coords.t[2] = y_size - 1;
                    }
                    else if (axis == 2)
                    {
                        start_coords.t[3] = z_size;
                        z_size += input_data->parent->lengths.t[3];
                        end_coords.t[3] = z_size - 1;
                    }

                    delete flow_to_work[flow_item->input[index].item]->output;
                    flow_to_work[flow_item->input[index].item]->output = new nn_cl_data(*load_item->output, start_coords, end_coords);

                    start_coords.t[1] = flow_to_work[flow_item->input[index].item]->output_view->view_begin.t[1];
                    start_coords.t[2] = flow_to_work[flow_item->input[index].item]->output_view->view_begin.t[2];

                    end_coords.t[1] = flow_to_work[flow_item->input[index].item]->output_view->view_end.t[1];
                    end_coords.t[2] = flow_to_work[flow_item->input[index].item]->output_view->view_end.t[2];


                    flow_to_work[flow_item->input[index].item]->output_view.reset(new nn_cl_data(*load_item->output, start_coords, end_coords));

                    //TODO: Instead of recompiling it is enough to find compiled kernel and change runtime params eg. GWS

                    // Having changed the output we may need to adjust EnqueuNDRange offset here for all inputs of merge layer
                    handy_routines::nn_workflow_compile_0_function_prepare_ocl_kernels( device, NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH, input_format[0], flow_to_work[flow_item->input[index].item], flow_item->input[index].item, batch);
                }

                break;
            }

            default:
                DBG_PRINTF( "Some layer is not implemented or invalid layer type passed\n" );
                assert( 0 ); //TODO: Implement this layer
                throw NN_API_STATUS_ERROR_INVALID_WORK_ITEM_TYPE;
            } // switch

            // copy arguments, fill buffers
            std::memcpy( &load_item->arguments, &flow_item->arguments, sizeof( load_item->arguments ) );
            switch( load_item->type )
            {
            case NN_WORK_ITEM_TYPE_ARITHMETIC:
            {
                // only 3D format of arithmetic layer is supported so far
                auto flow_factor = nn::data_cast<float, 3>(flow_item->arguments.forward_arithmetic.factor);
                // Each factor data is used to manipulate every single image (batch)
                nn_workload_data_coords_t size = { 1, static_cast<uint32_t>(flow_factor->size[0])
                                                    , static_cast<uint32_t>(flow_factor->size[1])
                                                    , static_cast<uint32_t>(flow_factor->size[2])
                                                    , 1, 1 };

                // convert nn_data to nn_workload_data
                nn::workload_data<> temp(flow_factor->buffer, size, layout);

                load_item->arguments.forward_arithmetic.factor = new nn_cl_data(
                    reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    &temp,
                    temp.parent->data_buffer);
                break;
            }

            case NN_WORK_ITEM_TYPE_CONVOLUTION: {
                { // biases are given as 1 dimensional data (continuous area)
                    auto flow_biases = nn::data_cast<float, 1>(flow_item->arguments.forward_convolution.biases);
                    //TODO: validate bias format
                    nn_workload_data_coords_t size = {1, 1, 1, 1, static_cast<uint32_t>(flow_biases->size[0]), 1};

                    nn::workload_data<nn::layout_f32> temp(size, layout);
                    for (size_t index = 0u; index<size.t[4]; ++index)
                        temp(0, 0, 0, 0, index, 0) = flow_biases->at(index);

                    load_item->arguments.forward_convolution.biases = new nn_cl_data(
                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        &temp,
                        temp.parent->data_buffer);
                }
                { // weights
                    auto flow_weights = nn::data_cast<float, 4>(flow_item->arguments.forward_convolution.weights);
                    //TODO: validate weight format
                    nn_workload_data_coords_t size = { 1, static_cast<uint32_t>(flow_weights->size[0])
                                                        , static_cast<uint32_t>(flow_weights->size[1])
                                                        , static_cast<uint32_t>(flow_weights->size[2])
                                                        , static_cast<uint32_t>(flow_weights->size[3]), 1 };
                    nn::workload_data<nn::layout_f32> temp(size, layout);

                    auto stride_x = flow_item->arguments.forward_convolution.stride[0];
                    auto stride_y = flow_item->arguments.forward_convolution.stride[1];

                    auto output_width  = flow_item->output_format[0].format_3d.size[0];
                    auto output_height = flow_item->output_format[0].format_3d.size[1];
                    auto output_depth  = flow_item->output_format[0].format_3d.size[2];
                    auto kernel_width  = flow_weights->size[0];
                    auto kernel_height = flow_weights->size[1];
                    auto input_depth   = flow_weights->size[2];

                    bool inv_mode = false; // TODO: better name?

                    auto swizzle_weights = [&](uint32_t weight_interleave, bool inv_mode)
                    {
                        auto src = static_cast<float *>( flow_weights->buffer );
                        auto dst = static_cast<float *>( temp.parent->data_buffer );

                        auto src_stride_kr = kernel_width;
                        auto src_stride_i  = kernel_width * kernel_height;
                        auto src_stride_o  = src_stride_i * input_depth;

                        if( inv_mode )
                        {
                            for( auto out_depth_tile = 0u; out_depth_tile < ( output_depth / weight_interleave ); out_depth_tile++ )
                                for (auto id = 0u; id < input_depth; id++)
                                    for (auto kr = 0u; kr < kernel_height; kr++)
                                        for (auto kc = 0u; kc < kernel_width; kc++)
                                            for (auto od = 0u; od < weight_interleave; od++)
                                                *( dst++ ) = src[kc + src_stride_kr*kr + src_stride_i*id + src_stride_o*( out_depth_tile * weight_interleave + od )];
                        }
                        else
                        {
                            for( auto id = 0u; id < input_depth; id++ )
                                for (auto out_depth_tile = 0u; out_depth_tile < (output_depth / weight_interleave); out_depth_tile++)
                                    for (auto kr = 0u; kr < kernel_height; kr++)
                                        for (auto kc = 0u; kc < kernel_width; kc++)
                                            for (auto od = 0u; od < weight_interleave; od++)
                                                *( dst++ ) = src[kc + src_stride_kr*kr + src_stride_i*id + src_stride_o*( out_depth_tile * weight_interleave + od )];
                        }


                    };

                    // TODO: This should be integrated with kernel choice and compilation in layer_convolution_opencl.cpp
                    if ((11 == kernel_width) && (11 == kernel_height) && (4 == stride_x) && (4 == stride_y)
                        && (56 == output_width) && (56 == output_height) && (output_depth % 4 == 0))
                    {
                        // specific for OverFeat C1
                        swizzle_weights(4, true);
                    }
                    else if ((11 == kernel_width) && (11 == kernel_height) && (4 == stride_x) && (4 == stride_y)
                        && (output_depth % 16 == 0))
                    {
                        // specific for AlexNet C1
                        swizzle_weights(16, true);
                    }
                    else if ((5 == kernel_width) && (5 == kernel_height) && (1 == stride_x) && (1 == stride_y)
                        && (24 == output_width) && (24 == output_height) && (output_depth % 4 == 0))
                    {
                        // specific for OverFeat C2
                        swizzle_weights(4, false);
                    }
                    else if ((5 == kernel_width) && (5 == kernel_height) && (1 == stride_x) && (1 == stride_y)
                        && (output_depth % 16 == 0))
                    {
                        // specific for AlexNet C2
                        swizzle_weights(16, true);
                    }
                    else if ((3 == kernel_width) && (3 == kernel_height) && (1 == stride_x) && (1 == stride_y)
                        && (12 == output_height) && (12 == output_height) && (output_depth % 8 == 0))
                    {
                        // specific for OverFeat C3 C4 C5
                        if (output_depth <= 512)
                            swizzle_weights(4, false); // performs better
                        else
                            swizzle_weights(8, false);
                    }
                    else if ((3 == kernel_width) && (3 == kernel_height) && (1 == stride_x) && (1 == stride_y)
                        && (output_depth % 16 == 0))
                    {
                        // specific for AlexNet C3 C4 C5
                        if( ( output_depth * batch ) > 256 )
                            swizzle_weights( 8, true );
                        else
                            swizzle_weights(16, true);
                    }
                    else if ((6 == kernel_width) && (6 == kernel_height) && (1 == stride_x) && (1 == stride_y)
                        && ((output_depth*output_width / 4) % 256 == 0))
                    {
                        // specific for OverFeat C6
                        swizzle_weights(4, false);
                    }
                    else
                    {
                        for(auto p=0u; p<size.t[4]; ++p)
                            for(auto z=0u; z<size.t[3]; ++z)
                                for(auto y=0u; y<size.t[2]; ++y)
                                    for(auto x=0u; x<size.t[1]; ++x)
                                        temp(0, x, y, z, p, 0) = flow_weights->at(x, y, z, p);
                    }

                    load_item->arguments.forward_convolution.weights = new nn_cl_data(
                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        &temp,
                        temp.parent->data_buffer);
                }

                break;
            }

            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED: {
                { // biases
                    auto flow_biases = nn::data_cast<float, 1>(flow_item->arguments.forward_fully_connected.biases);
                    //TODO: validate bias format
                    nn_workload_data_coords_t size = { 1, static_cast<uint32_t>(flow_biases->size[0]), 1, 1, 1, 1 };
                    nn::workload_data<nn::layout_f32> temp(size, layout);
                    for(size_t x=0u; x<size.t[1]; ++x)
                        temp(0, x, 0, 0, 0, 0) = flow_biases->at(x);

                    load_item->arguments.forward_fully_connected.biases = new nn_cl_data(
                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        &temp,
                        temp.parent->data_buffer);
                }
                { // weights

                    switch(flow_item->arguments.forward_fully_connected.weights->dimension)
                    {
                        case 4:
                        {
                            auto flow_weights = nn::data_cast<float, 4>(flow_item->arguments.forward_fully_connected.weights);
                            nn_workload_data_coords_t size = { 1, static_cast<uint32_t>(flow_weights->size[0]),
                                                                  static_cast<uint32_t>(flow_weights->size[1]),
                                                                  static_cast<uint32_t>(flow_weights->size[2]),
                                                                  static_cast<uint32_t>(flow_weights->size[3]), 1};

                            nn_workload_data_layout_t layout_fc = layout;

                            // use lambda here
                            auto fc_inputs = flow_weights->size[0] * flow_weights->size[1] * flow_weights->size[2];
                            auto fc_outputs = flow_weights->size[3];
                            auto total_num_weights = fc_inputs * fc_outputs;

                            if( handy_routines::use_fully_connected_8x8(fc_inputs, fc_outputs, batch, device ) )
                            {
                                // special ordering for "fully_connected_8x8" kernel
                                layout_fc.ordering = { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y,
                                    NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q };
                            }

                            nn::workload_data<> temp( size, layout_fc );

                            auto src = static_cast<float *>( flow_weights->buffer );
                            auto dst = static_cast<float *>( temp.parent->data_buffer );

                            if( handy_routines::use_fully_connected_8x8( fc_inputs, fc_outputs, batch, device ) )
                            {
                                auto src_stride_o = fc_inputs;
                                for( size_t i = 0u; i < fc_inputs; ++i )
                                    for( size_t o = 0u; o < fc_outputs; ++o )
                                        *( dst++ ) = src[i + o * src_stride_o];
                            }
                            else
                            {
                                memcpy( dst, src, total_num_weights * sizeof( float ) );
                            }

                            if (total_num_weights*sizeof(float) > reinterpret_cast<device_gpu::ocl_toolkit*>(device)->m_max_buffer_size)
                            {
                                auto& input = flow_item->input[0];
                                auto num_inputs =
                                    input.item->output_format[input.index].format_3d.size[0] *
                                    input.item->output_format[input.index].format_3d.size[1] *
                                    input.item->output_format[input.index].format_3d.size[2];

                                auto num_outputs = load_item->output->parent->lengths.t[NN_DATA_COORD_x];
                                uint_least32_t weights2_neuron_idx = num_outputs / 2;
                                auto size_weights2 = (total_num_weights - weights2_neuron_idx*num_inputs);
                                auto offset_weights2 = weights2_neuron_idx*num_inputs;

                                std::vector<nn_cl_data_fragment> fragments;
                                fragments.push_back(nn_cl_data_fragment{ temp.parent->data_buffer, offset_weights2*sizeof(float) });
                                fragments.push_back(nn_cl_data_fragment{ (float*)temp.parent->data_buffer + offset_weights2, size_weights2*sizeof(float) });

                                load_item->arguments.forward_fully_connected.weights = new nn_cl_data(
                                    reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    &temp,
                                    temp.parent->data_buffer,
                                    false,
                                    0,
                                    0,
                                    true,
                                    &fragments);
                            }
                            else
                            {
                                //TODO: Make it a separate routine
                                uint_least32_t input_width, input_height, input_depth;
                                auto& input = flow_item->input[0];
                                input_width = input.item->output_format[input.index].format_1d.size[0];
                                input_height = input.item->output_format[input.index].format >=
                                               NN_DATA_FORMAT_2D ? input.item->output_format[input.index].format_2d.size[1] : 1;
                                input_depth = input.item->output_format[input.index].format >=
                                              NN_DATA_FORMAT_3D ? input.item->output_format[input.index].format_3d.size[2] : 1;

                                //TODO: check if this works
                                uint_least32_t num_outputs =
                                    ( flow_item->output_format[0].format_1d.size[0] *
                                      ( flow_item->output_format[0].format >=
                                        NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1 ) *
                                      ( flow_item->output_format[0].format >=
                                        NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1 ) );

                                auto num_inputs = input_width * input_height * input_depth;

                                if (handy_routines::use_fully_connected_8x8(num_inputs, num_outputs, batch, device))
                                {
                                    load_item->arguments.forward_fully_connected.weights = new nn_cl_data(
                                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        &temp,
                                        temp.parent->data_buffer,
                                        true,
                                        num_outputs,
                                        num_inputs);
                                } else {
                                    load_item->arguments.forward_fully_connected.weights = new nn_cl_data(
                                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        &temp,
                                        temp.parent->data_buffer);
                                }
                            }
                        break;
                        }
                        case 2:
                        {
                            auto flow_weights = nn::data_cast<float, 2>(flow_item->arguments.forward_fully_connected.weights);

                              //TODO: validate weight format
                            nn_workload_data_coords_t size = { 1, static_cast<uint32_t>(flow_weights->size[0])
                                                                , static_cast<uint32_t>(flow_weights->size[1]), 1, 1, 1};

                            nn_workload_data_layout_t layout_fc = layout;
                            if( handy_routines::use_fully_connected_8x8( flow_weights->size[0], flow_weights->size[1], batch, device ) )
                            {
                                // special ordering for "fully_connected_8x8" kernel
                                layout_fc.ordering = { NN_DATA_COORD_y, NN_DATA_COORD_x, NN_DATA_COORD_z,
                                                       NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_q };
                            }
                            nn::workload_data<> temp( size, layout_fc );

                            auto src = static_cast<float *>( flow_weights->buffer );
                            auto dst = static_cast<float *>( temp.parent->data_buffer );

                            if( handy_routines::use_fully_connected_8x8( flow_weights->size[0], flow_weights->size[1], batch, device ) )
                            {
                                auto src_stride_y = flow_weights->size[0];
                                auto size_x = size.t[1];
                                auto size_y = size.t[2];
                                for( size_t x = 0u; x < size_x; ++x )
                                    for( size_t y = 0u; y < size_y; ++y )
                                        *( dst++ ) = src[x + y * src_stride_y];
                            }
                            else
                            {
                                memcpy( dst, src, size.t[1] * size.t[2] * sizeof(float) );
                            }

                            auto total_num_weights = size.t[NN_DATA_COORD_x] * size.t[NN_DATA_COORD_y] * size.t[NN_DATA_COORD_z] * size.t[NN_DATA_COORD_p];
                            if (total_num_weights*sizeof(float) > reinterpret_cast<device_gpu::ocl_toolkit*>(device)->m_max_buffer_size)
                            {
                                auto& input = flow_item->input[0];
                                auto num_inputs =
                                    input.item->output_format[input.index].format_3d.size[0] *
                                    input.item->output_format[input.index].format_3d.size[1] *
                                    input.item->output_format[input.index].format_3d.size[2];

                                auto num_outputs = load_item->output->parent->lengths.t[NN_DATA_COORD_x];
                                uint_least32_t weights2_neuron_idx = num_outputs / 2;
                                auto size_weights2 = (total_num_weights - weights2_neuron_idx*num_inputs);
                                auto offset_weights2 = weights2_neuron_idx*num_inputs;

                                std::vector<nn_cl_data_fragment> fragments;
                                fragments.push_back(nn_cl_data_fragment{ temp.parent->data_buffer, offset_weights2*sizeof(float) });
                                fragments.push_back(nn_cl_data_fragment{ (float*)temp.parent->data_buffer + offset_weights2, size_weights2*sizeof(float) });

                                load_item->arguments.forward_fully_connected.weights = new nn_cl_data(
                                    reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    &temp,
                                    temp.parent->data_buffer,
                                    false,      // Do not create image
                                    0,          // image params - not used
                                    0,          // image params - not used
                                    true,
                                    &fragments);
                            }
                            else
                            {
                                //TODO: Make it a separate routine
                                uint_least32_t input_width, input_height, input_depth;
                                auto& input = flow_item->input[0];

                                input_width = input.item->output_format[input.index].format_1d.size[0];
                                input_height = input.item->output_format[input.index].format >=
                                                NN_DATA_FORMAT_2D ? input.item->output_format[input.index].format_2d.size[1] : 1;
                                input_depth = input.item->output_format[input.index].format >=
                                                NN_DATA_FORMAT_3D ? input.item->output_format[input.index].format_3d.size[2] : 1;

                                //TODO: check if this works
                                uint_least32_t num_outputs =
                                    ( flow_item->output_format[0].format_1d.size[0] *
                                      ( flow_item->output_format[0].format >=
                                        NN_DATA_FORMAT_2D ? flow_item->output_format[0].format_2d.size[1] : 1 ) *
                                      ( flow_item->output_format[0].format >=
                                        NN_DATA_FORMAT_3D ? flow_item->output_format[0].format_3d.size[2] : 1 ) );

                                auto num_inputs = input_width * input_height * input_depth;

                                if (handy_routines::use_fully_connected_8x8(num_inputs, num_outputs, batch, device))
                                {
                                    load_item->arguments.forward_fully_connected.weights = new nn_cl_data(
                                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        &temp,
                                        temp.parent->data_buffer,
                                        true,
                                        num_outputs,
                                        num_inputs);

                                } else {
                                    load_item->arguments.forward_fully_connected.weights = new nn_cl_data(
                                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        &temp,
                                        temp.parent->data_buffer);
                                }
                            }
                        break;
                        }
                        default:
                            assert( 0 ); // We cannot have dimensions of weights diffrent from 2 or 4
                            throw NN_API_STATUS_ERROR_INVALID_WORK_ITEM_TYPE;
                        break;
                    }

                    break;
                }
            }
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2: {
                { // biases
                    auto flow_biases = nn::data_cast<float,1>(flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.biases);
                    //TODO: validate bias format
                    nn_workload_data_coords_t size = {1, 1, 1, 1, static_cast<uint32_t>(flow_biases->size[0]), 1};
                    nn::workload_data<nn::layout_f32> temp(size, layout);
                    for(size_t index=0u; index<size.t[4]; ++index)
                        temp(0, 0, 0, 0, index, 0) = flow_biases->at(index);

                    load_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.biases = new nn_cl_data(
                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        &temp,
                        temp.parent->data_buffer);
                }
                { // weights
                    auto flow_weights = nn::data_cast<float, 4>(flow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights);
                    //TODO: validate weight format
                    nn_workload_data_coords_t size = { 1, static_cast<uint32_t>(flow_weights->size[0])
                                                        , static_cast<uint32_t>(flow_weights->size[1])
                                                        , static_cast<uint32_t>(flow_weights->size[2])
                                                        , static_cast<uint32_t>(flow_weights->size[3]), 1 };

                    nn::workload_data<nn::layout_f32> temp(size, layout);
                    for(auto p=0u; p<size.t[4]; ++p)
                        for(auto z=0u; z<size.t[3]; ++z)
                            for(auto y=0u; y<size.t[2]; ++y)
                                for(auto x=0u; x<size.t[1]; ++x)
                                    temp(0,x,y,z,p,0) = flow_weights->at(x,y,z,p);

                    load_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights = new nn_cl_data(
                        reinterpret_cast<device_gpu::ocl_toolkit*>(device),
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        &temp,
                        temp.parent->data_buffer);
                }
                break;
            }
            default:
                // This is the case when all workflow item arguments are empty or do not contain buffers.
                ;
            }
            // copying inputs
            for( auto index = 0u; index < flow_item->input_count; ++index )
            {
                assert( flow_to_work.find( flow_item->input[index].item ) != flow_to_work.end() );
                load_item->input.push_back( flow_to_work[flow_item->input[index].item] );
            }

            // copying uses
            for( auto index = 0u; index < flow_item->use_count; ++index )
            {
                assert( flow_to_work.find( flow_item->use[index].item ) != flow_to_work.end() );
                load_item->use.push_back( flow_to_work[flow_item->use[index].item] );
            }

            // 3. Generation of OpenCL kernels
            handy_routines::nn_workflow_compile_0_function_prepare_ocl_kernels( device, NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH, input_format[0], load_item, flow_item, batch);


        }; // end of lambda

        {   // traverse workflow items and create workload items
            std::queue<nn_workflow_item_t *> todo;
            for( auto index = 0u; index < workflow->input_count; ++index )
                todo.push( workflow->input[index] );
            while( !todo.empty() )
            {
                nn_workflow_item_t *flow_item = todo.front();
                todo.pop();
                if( flow_to_work.find( flow_item ) == flow_to_work.end() )
                {
                    flow_to_work[flow_item] = new nn_gpu_workload_item_t;
                    for( auto index = 0u; index < flow_item->use_count; ++index )
                        todo.push( flow_item->use[index].item );
                }
            }
        }

        { // now for every workflow item there's a workload item
            std::queue<nn_workflow_item_t *> todo;
            std::set< nn_workflow_item_t * >   done;
            for( auto index = 0u; index < workflow->input_count; ++index )
                todo.push( workflow->input[index] );
            while( !todo.empty() )
            {
                nn_workflow_item_t *flow_item = todo.front();
                todo.pop();
                if( done.find( flow_item ) == done.end() )
                {
                    done.insert( flow_item );
                    nn_gpu_workload_item_t *load_item = flow_to_work[flow_item];
                    copy_item( load_item, flow_item );
                    gpu_workload->m_workload_items.push_back( load_item ); // TODO: Won't work for cyclic graphs
                    for( auto index = 0u; index < flow_item->use_count; ++index )
                        todo.push( flow_item->use[index].item );
                }
            }
        }
    }
    catch( device_gpu::runtime_error err )
    {
        std::cerr << err.what( ) << std::endl;
        return NN_API_STATUS_ERROR_OUT_OF_MEMORY;
    }
    catch( ... ) {
        return NN_API_STATUS_ERROR_OUT_OF_MEMORY;
    }

    return NN_API_STATUS_OK;
}
//////////////////////////////////////////////////////////////////////////////////////////////////
/* executes workload with given inputs & outputs */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workload_execute_0x0_function(
    nn_workload_t      *workload,       /* workload to be started */
    void *             *input,          /* array of pointers with input data;  format is in workload->input_format */
    void *             *output,         /* array of pointers with output data; format is in workload->output_format */
    NN_API_STATUS      *status          /* asynchronous status */
    )
{
    nn_gpu_workload *gpu_workload = reinterpret_cast< nn_gpu_workload * >( workload );

    assert( gpu_workload->m_workload_items.size() > 0 );   ///There has to be some workload items to have execute running

    if( status != nullptr )
    {
        *status = NN_API_WORK_IN_PROGRESS;
    }

    // attach input & output
    nn::data<float,0> *workload_input_data  = reinterpret_cast<nn::data<float,0> **>(input)[0];
    nn::data<float,0> *workload_output_data = reinterpret_cast<nn::data<float,0> **>(output)[0];

    assert(workload->input_count==1);
    assert(workload->output_count==1);

    // layout for inputs & outputs
    auto get_workload_layout =
        [](NN_WORKLOAD_DATA_TYPE type) -> nn_workload_data_layout_t {
            switch (type) {
                    case NN_WORKLOAD_DATA_TYPE_F32_1D:
                    case NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH:
                        return nn::layout_t<nn::layout_xyzpqn_f32>::layout;
            case NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH:
            case NN_WORKLOAD_DATA_TYPE_F32_ZXY:
                return nn::layout_t<nn::layout_zxynpq_f32>::layout;
            default:
                return nn::layout_t<nn::layout_xyzpnq_f32>::layout;
            }
    };

    auto calculate_size = [](uint32_t batch, NN_WORKLOAD_DATA_TYPE type, nn::data<float,0> *data) -> nn_workload_data_coords_t {
        uint32_t size_n=batch, size_x=1, size_y=1, size_z=1, size_p=1, size_q=1;
        switch(type) {
            case NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH:
            case NN_WORKLOAD_DATA_TYPE_F32_3D:
                size_z = data->size[2];
                // fall through
            case NN_WORKLOAD_DATA_TYPE_F32_2D_BATCH:
            case NN_WORKLOAD_DATA_TYPE_F32_2D:
                size_y = data->size[1];
                // fall through
            case NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH:
            case NN_WORKLOAD_DATA_TYPE_F32_1D:
                size_x = data->size[0];
                break;
            case NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH:
            case NN_WORKLOAD_DATA_TYPE_F32_ZXY:
                size_z = data->size[0];
                size_x = data->size[1];
                size_y = data->size[2];
                break;
            default:
                assert(0);
        }
        switch(type) {
            case NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH:
            case NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH:
                size_n = data->size[3];
                break;
            case NN_WORKLOAD_DATA_TYPE_F32_2D_BATCH:
                size_n = data->size[2];
                break;
            case NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH:
                size_n = data->size[1];
                break;
            default:
                ;
        }
        return nn_workload_data_coords_t{size_n, size_x, size_y, size_z, size_p, size_q};
    };

    // specify dimensions of input, output
    nn_workload_data_coords_t workload_input_size = calculate_size(workload->batch, workload->input_format[0], workload_input_data);

    // specify dimensions of input, output
    nn_workload_data_coords_t workload_output_size = calculate_size(workload->batch, workload->output_format[0], workload_output_data);

    auto input_layout = get_workload_layout(workload->input_format[0]);
    auto output_layout = get_workload_layout(workload->output_format[0]);

    // Here I do recognize which type we are of format and then I do conversion or not
    // 1. If layout are diffrent among input and gpu_layout then we need to convert
    // 2. We need to create new nn_data with our layout and make a copy of source input
    // 4. If no conversion is required then do not perform one
    // 5. Delete our no longer needed workload data
    using auto_delete_workload_data = std::unique_ptr<nn::workload_data<>>;
    auto_delete_workload_data workload_input(new nn::workload_data<>(workload_input_data->buffer, workload_input_size, input_layout));
    auto_delete_workload_data workload_output(new nn::workload_data<>(workload_output_data->buffer, workload_output_size, output_layout));

    // Check if VIEW is the second node if that is the case then create nn_data for that view
    if(gpu_workload->m_workload_items[1]->type == NN_WORK_ITEM_TYPE_VIEW) {
            auto& origin = gpu_workload->m_workload_items[1]->arguments.view.origin;
            // --> TEMPORARY HACK
            nn_workload_data_coords_t start(0, origin[0], origin[1], origin[2], 0, 0);
            unsigned int view_width = gpu_workload->m_workload_items[1]->output->parent->lengths.t[NN_DATA_COORD_x];
            unsigned int view_height = gpu_workload->m_workload_items[1]->output->parent->lengths.t[NN_DATA_COORD_y];
            unsigned int view_depth = gpu_workload->m_workload_items[1]->output->parent->lengths.t[NN_DATA_COORD_z];
            delete gpu_workload->m_workload_items[1]->output;
            // <-- TEMPORARY HACK
            nn_workload_data_coords_t end(
            0,
            origin[0] + view_width - 1,
            origin[1] + view_height - 1,
            origin[2] + view_depth - 1,
            0, 0
            );
            // If INPUT node is precedessor of VIEW then output of INPUT is not known
            gpu_workload->m_workload_items[1]->output = new nn_cl_data(*gpu_workload->m_workload_items[0]->output, start, end);
    }

    // Execute sequencially all workload items
    try
    {
        for( auto it = gpu_workload->m_workload_items.begin(); it < gpu_workload->m_workload_items.end(); ++it )
        {

            switch( ( *it )->type )
            {
            case NN_WORK_ITEM_TYPE_INPUT:
            {
                    cl_int err = CL_SUCCESS;
                    // Next Layer may expect image as input so then output of this layer has to be image
                    // and then we map existing image to copy user input into it

                    // If no buffer/image exists then we create one using USE_MEM_HOST_PTR
                    if(gpu_workload->m_workload_items[0]->output == nullptr)
                    {
                        gpu_workload->m_workload_items[0]->output = new nn_cl_data(
                        reinterpret_cast<device_gpu::ocl_toolkit *>(workload->device),
                        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                        workload_input.get(),
                        workload_input->parent->data_buffer);

                    } else if(gpu_workload->m_workload_items[0]->output->parent->cl_buffer[0] != nullptr) {
                        err = clEnqueueWriteBuffer(
                                reinterpret_cast<device_gpu::ocl_toolkit *>(workload->device)->get_command_queue()(),
                                (*gpu_workload->m_workload_items[0]->output->parent->cl_buffer[0])(),
                                CL_FALSE,
                                0,
                                workload_input_data->count()*sizeof(float),
                                workload_input_data->buffer,
                                0,
                                nullptr,
                                nullptr);
                    } else {

                        // If image here is a nullptr then we are doing something very wrong
                        assert(gpu_workload->m_workload_items[0]->output->parent->cl_image[0] != nullptr);

                        // Region is (width*height*depth,num_batches,1)
                        size_t origin[3] = {0,0,0};
                        size_t image_width =(*it)->output->parent->lengths.t[NN_DATA_COORD_x]*(*it)->output->parent->lengths.t[NN_DATA_COORD_y]*(*it)->output->parent->lengths.t[NN_DATA_COORD_z];
                        size_t image_height = (*it)->output->parent->lengths.t[NN_DATA_COORD_n];
                        size_t region[3] = {image_width, image_height, 1};

                        err = clEnqueueWriteImage(
                                reinterpret_cast<device_gpu::ocl_toolkit *>(workload->device)->get_command_queue()(),
                                (*gpu_workload->m_workload_items[0]->output->parent->cl_image[0])(),
                                CL_FALSE,
                                origin,
                                region,
                                0,
                                0,
                                workload_input_data->buffer,
                                0,
                                nullptr,
                                nullptr);
                    }
                    if (err != CL_SUCCESS) {
                        THROW_ERROR(err, "Error in loading input data  into INPUT load_item.");
                    }
                }
                break;
            case NN_WORK_ITEM_TYPE_VIEW:
            case NN_WORK_ITEM_TYPE_MERGE:
                // Nothing here to be done
                break;
            case NN_WORK_ITEM_TYPE_OUTPUT:
                // Copy from output work_item to
                {
                    reinterpret_cast< device_gpu::ocl_toolkit * >(workload->device)->finish();

                    auto& previous_item_output = (*it)->input[0]->output;

                    cl_int err;
                    float* ptr = static_cast<float*>(
                        clEnqueueMapBuffer(
                            reinterpret_cast<device_gpu::ocl_toolkit *>(workload->device)->get_command_queue()(),
                            (*previous_item_output->parent->cl_buffer[0])(),
                            true,
                            CL_MEM_READ_ONLY,
                            0,
                            previous_item_output->parent->buffer_aligned_size,
                            0,
                            nullptr,
                            nullptr,
                            &err));

                    if (err != CL_SUCCESS)
                        THROW_ERROR(err, "Error in mapping buffer at OUTPUT memcpy.");

                    nn::workload_data<> temp(ptr, previous_item_output->parent->lengths, previous_item_output->parent->layout);
                    nn::workload_data<> view(temp, workload_output->view_begin, workload_output->view_end);

                    *workload_output = view;

                    clEnqueueUnmapMemObject(
                        reinterpret_cast<device_gpu::ocl_toolkit *>(workload->device)->get_command_queue()(),
                        (*previous_item_output->parent->cl_buffer[0])(),
                        ptr,
                        0,
                        nullptr,
                        nullptr);
                }
                break;
            case NN_WORK_ITEM_TYPE_CONVOLUTION:
            {
                handy_routines::nn_workload_item_convolution_start(workload->device, (*it));
                break;
            }
            case NN_WORK_ITEM_TYPE_ARITHMETIC:
            {
                handy_routines::nn_workload_item_arithmetic_start(workload->device, NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH , workload->input_format[0], (*it));
                break;
            }
            case NN_WORK_ITEM_TYPE_FULLY_CONNECTED:
                handy_routines::nn_workload_item_full_connectivity_start( workload->device, ( *it ) );
                break;
            case NN_WORK_ITEM_TYPE_POOLING:
                handy_routines::nn_workload_item_pooling_start( workload->device, ( *it ) );
                break;
            case NN_WORK_ITEM_TYPE_SOFTMAX:
                handy_routines::nn_workload_item_softmax_start( workload->device, ( *it ) );
                break;
            case NN_WORK_ITEM_TYPE_NORMALIZATION:
                handy_routines::nn_workload_item_normalization_start( workload->device, NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH , workload->input_format[0], ( *it ) );
                break;
            case NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2:
                handy_routines::nn_workload_item_conv_maxpool_start( workload->device, (*it) );
                break;
            default:
                DBG_PRINTF( "Error: Unsupported Layer for device GPU\n" );
                assert( 0 );
                if( status != nullptr )
                {
                    *status = NN_API_STATUS_ERROR_OTHER;
                }
                throw NN_API_STATUS_ERROR_INVALID_WORK_ITEM_TYPE;
            }


#ifdef DUMP_LAYERS
            reinterpret_cast< device_gpu::ocl_toolkit * >( workload->device )->finish( );
            static auto layer = 0;

            if( ( *it )->output_view  != nullptr)
            {
                std::string filename = "layer_" + std::to_string( layer ) + "_type_" + std::to_string( ( *it )->type ) + "_" + layer_name(**it) + "_output_padded.txt";
                nn_data_marshaling_to_txt(  *( *it )->output , filename, workload->device );

                filename = "layer_" + std::to_string( layer ) + "_type_" + std::to_string( ( *it )->type ) + "_" + layer_name( **it ) + "_output.txt";
                nn_data_marshaling_to_txt( *( *it )->output_view, filename, workload->device );
            }
            else if( ( *it )->output != nullptr )
            {
                std::string filename = "layer_" + std::to_string( layer ) + "_type_" + std::to_string( ( *it )->type ) + "_" + layer_name( **it ) + "_output.txt";
                nn_data_marshaling_to_txt( *( *it )->output, filename, workload->device );
            }

            layer++;
#endif // #ifdef DUMP_LAYERS

            // This is example of tools used for debugging:
            //  - casting output of our data into one matching CPU device
            //  - modifying output of selected layer
            //  Note: It all does work when execution on GPU took place eg. finish was called.
            #if 0
            reinterpret_cast< device_gpu::ocl_toolkit * >( workload->device )->finish();
            nn_workload_data_layout_t cpu_layout = {
                { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                { 0, 0, 0, 0, 0, 0 }, // alignment
                nn::workload_data<>::layout.nxyzpq, // ordering (FF and softmax got it N,X as oppose to Z,X,Y)
                NN_DATATYPE_FLOAT
            };
            nn_workload_data_coords_t cpu_size =
            {
                (*it)->output->parent->lengths.t[0],
                (*it)->output->parent->lengths.t[1],
                (*it)->output->parent->lengths.t[2],
                (*it)->output->parent->lengths.t[3],
                (*it)->output->parent->lengths.t[4],
                (*it)->output->parent->lengths.t[5]
            };
            if( (*it)->type == NN_WORK_ITEM_TYPE_NORMALIZATION )  {
                ;
                //for(int z = 0; z< 3; ++z)  {
                //for(int j = 0; j < 231; ++j) {
                    //for(int k = 0; k < 231; ++k) {
                       //nn_workload_data_set_float32( (*it)->output,z + j*100.0f + k*10.0f,0,k,j,z,0,0);
                    //}
                //}
                //}
            }

            if(  ((*it)->type == NN_WORK_ITEM_TYPE_SOFTMAX) )  {
                nn::workload_data<> *cpu_output = new nn::workload_data<>( cpu_size, cpu_layout );
                nn::workload_data<> *gpu_output  = workload_data_cast<nn::layout_f32>( (*it)->output);
                nn_workload_data_copy( gpu_output, cpu_output);
                delete cpu_output;
            }
            //if( ((*it)->type == NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2) || ((*it)->type == NN_WORK_ITEM_TYPE_CONVOLUTION) )  {
            //if(  ((*it)->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED) )  {
            #endif
        }
    }
    catch ( device_gpu::runtime_error err )
    {
        //TODO: Return proper error when compilation of Kernel failed!
        std::cerr << err.what() << std::endl;
        return NN_API_STATUS_ERROR_OUT_OF_MEMORY;
    }
    catch ( NN_API_STATUS err ) {
        return err;
    }
    catch ( ... ) {
        assert( 0 );                                // This should not happen. If we ever
        return NN_API_STATUS_ERROR_OUT_OF_MEMORY;     // throw wrong type of object.  It should be caught here
    }
    // TODO: Make it asynchronous. I should really set it from inside of device implementation and allow
    // this function to return (non blocking call)
    if( status != nullptr )
    {
        *status = NN_API_WORK_FINISHED;
    }
    return NN_API_STATUS_OK;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

/* delete workload */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workload_delete_0x0_function(
    nn_workload_t       *workload       /* workload to be deleted */
    )
{
    if(workload == nullptr)
    {
        return NN_API_STATUS_ERROR_INVALID_POINTER;
    }
    else
    {
        // Delete workload items that are contained in given workload
        nn_gpu_workload* gpu_workload = reinterpret_cast<nn_gpu_workload*>(workload);

        for(auto it = gpu_workload->m_workload_items.begin(); it < gpu_workload->m_workload_items.end(); ++it) {
            delete (*it);
        }
        gpu_workload->m_workload_items.clear();

        // TODO: When more than input, output is possible then make suitable changes
        assert( workload->input_count == 1 );
        assert( workload->output_count == 1 );

        // Free area for input & output formats
        delete *( const_cast< NN_WORKLOAD_DATA_TYPE ** >( &( workload->input_format ) ) );
        delete *( const_cast< NN_WORKLOAD_DATA_TYPE ** >( &( workload->output_format ) ) );

        delete gpu_workload;

        return NN_API_STATUS_OK;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
/* validate parameters of work_item for this particular device */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_item_validate_0x0_function(
    nn_device_t        *device,         /* target device */
    nn_workflow_item_t *work_item       /* work item to be validated */
    )
{
    if( work_item == nullptr )
    {
        return NN_API_STATUS_ERROR_INVALID_POINTER;
    }

    //TODO: Dependencies checking

    return NN_API_STATUS_OK;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

NN_API_STATUS NN_API_CALL_CONVENTION nn_device_parameter_get_0x0_function(
    nn_device_t        *device,         /* target context */
    NN_PARAMETER        parameter,      /* parameter to get */
    void               *buffer,         /* buffer to store result to */
    uint32_t            size            /* size of buffer */
    )
{
    assert(0);
    return NN_API_STATUS_OK;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

NN_API_STATUS NN_API_CALL_CONVENTION nn_device_parameter_set_0x0_function(
    nn_device_t        *device,         /* target context */
    NN_PARAMETER        parameter,      /* parameter to set */
    void               *buffer,         /* buffer with argument */
    uint32_t            size            /* size of buffer */
    )
{
    assert(0);
    return NN_API_STATUS_OK;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

NN_API_STATUS NN_API_CALL_CONVENTION nn_translate_api_status_0x0_function(
    NN_API_STATUS       status,          /* status code to translate */
    char*              *brief,           /* one-line explanation */
    char*              *detailed         /* multi-line explanation */
    )
{
    brief = nullptr;
    detailed = nullptr;

    return NN_API_STATUS_ERROR_STATUS_CODE_NOT_FOUND;
}
