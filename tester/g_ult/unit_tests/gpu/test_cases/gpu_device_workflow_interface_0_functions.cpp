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
#include <memory>
#include <malloc.h>
#include <cassert>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include "gtest/gtest.h"


#include "device/api/nn_device_api.h"
#include "device/common/nn_workload_data.h"
#include "device/api/nn_device_interface_0.h"
#include "device/gpu/api_internal/nn_device_interface_0_functions.h"
#include "device/gpu/api_internal/nn_device_interface_0_internal.h"

#include "tester/g_ult/unit_tests/gpu/common.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
void create_input_workflow_item(
    const nn_device_interface_0_t &di,
    nn_workflow_item_t *&input_workflow_item,
    uint_least32_t                num_input_feature_maps,
    uint_least32_t                input_feature_map_width,
    uint_least32_t                input_feature_map_height)
{
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_create_function( &input_workflow_item, 0, nullptr, 1 ) );
    input_workflow_item->type                            = NN_WORK_ITEM_TYPE_INPUT;
    input_workflow_item->output_format[0].format            = NN_DATA_FORMAT_3D;
    input_workflow_item->output_format[0].format_3d.size[0] = input_feature_map_width;
    input_workflow_item->output_format[0].format_3d.size[1] = input_feature_map_height;
    input_workflow_item->output_format[0].format_3d.size[2] = num_input_feature_maps;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void create_output_workflow_item(
    const nn_device_interface_0_t &di,
    nn_workflow_item_t *          &output_workflow_item,
    nn_workflow_item_t *          &input_workflow_item,
    uint_least32_t                output_width,
    uint_least32_t                output_height,
    uint_least32_t                output_depth )
{
    nn_workflow_use_descriptor_t desc = { input_workflow_item, 0 };
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&output_workflow_item, 1, &desc, 1));
    output_workflow_item->type                            = NN_WORK_ITEM_TYPE_OUTPUT;
    output_workflow_item->output_format[0].format            = NN_DATA_FORMAT_3D;
    output_workflow_item->output_format[0].format_3d.size[0] = output_width;
    output_workflow_item->output_format[0].format_3d.size[1] = output_height;
    output_workflow_item->output_format[0].format_3d.size[2] = output_depth;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void create_merge_workflow_item( const nn_device_interface_0_t &di,
                                    nn_workflow_item_t *          &merge_workflow_item,
                                    uint16_t                      num_inputs,
                                    nn_workflow_item_t *          *input_workflow_items,       //TODO: make it an array of inputs for merge layer
                                    uint16_t                      merge_axis                                // TODO: other axes support (so far only Z axis is supported)
)
{
    assert(num_inputs == 2);
    nn_workflow_use_descriptor_t desc[] = { { input_workflow_items[0], 0 }, { input_workflow_items[1], 0 } };
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&merge_workflow_item, num_inputs, desc, 1));


    merge_workflow_item->type = NN_WORK_ITEM_TYPE_MERGE;
    merge_workflow_item->arguments.forward_merge.axis = merge_axis;  
    merge_workflow_item->output_format[0].format = NN_DATA_FORMAT_3D;

    // Here we declare 3d sizes of to be done merge layer 
    
    assert(merge_axis == 2);    //TODO: support for other axes
    uint_least32_t out_width = input_workflow_items[0]->output_format[0].format_3d.size[0]; 
    uint_least32_t out_height = input_workflow_items[0]->output_format[0].format_3d.size[1];  
    uint_least32_t out_depth =  0;
    for(uint16_t i =0 ; i< num_inputs; ++i) {
       out_depth +=  input_workflow_items[i]->output_format[0].format_3d.size[2]; 
    }
   
    merge_workflow_item->output_format[0].format_3d = nn_output_format_3d{ { out_width,out_height, out_depth }  };
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void create_convolution_workflow_item(
    const nn_device_interface_0_t &di,
    nn_workflow_item_t *          &conv_workflow_item,
    nn_workflow_item_t *          &input_workflow_item,
    uint_least32_t                output_width,
    uint_least32_t                output_height,
    uint_least32_t                output_depth,
    float                         *weights,
    size_t                        *weight_coords,
    float                         *biases,
    size_t                        *bias_coords,
    uint_least32_t                kernel_stride_x,
    uint_least32_t                kernel_stride_y,
    uint_least32_t                center_x,
    uint_least32_t                center_y,
    NN_ACTIVATION_FUNCTION        activation_function)
{
    nn_workflow_use_descriptor_t desc = { input_workflow_item, 0 };
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_create_function( &conv_workflow_item, 1, &desc, 1 ) );
    conv_workflow_item->type                                              = NN_WORK_ITEM_TYPE_CONVOLUTION;
    conv_workflow_item->output_format[0].format                              = NN_DATA_FORMAT_3D;
    conv_workflow_item->output_format[0].format_3d.size[0]                   = output_width;
    conv_workflow_item->output_format[0].format_3d.size[1]                   = output_height;
    conv_workflow_item->output_format[0].format_3d.size[2]                   = output_depth;
    conv_workflow_item->arguments.forward_convolution.padding             = NN_PADDING_MODE_DATA_OR_ZERO;
    conv_workflow_item->arguments.forward_convolution.activation.function = activation_function;
    conv_workflow_item->arguments.forward_convolution.center_offset[0]    = center_x;
    conv_workflow_item->arguments.forward_convolution.center_offset[1]    = center_y;
    conv_workflow_item->arguments.forward_convolution.stride[0]           = kernel_stride_x;
    conv_workflow_item->arguments.forward_convolution.stride[1]           = kernel_stride_y;

    // Wrap weights data up with nn_data
    nn::data<float, 4> weight_data(weights, weight_coords, 4);
    conv_workflow_item->arguments.forward_convolution.weights = new nn::data<float, 4>(weight_data);

    // Bias packaging into nn_data
    nn::data<float, 1> bias_data(biases, bias_coords, 1);
    conv_workflow_item->arguments.forward_convolution.biases = new nn::data<float, 1>(bias_data);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void create_merged_convolution_maxpool_workflow_item(
    const nn_device_interface_0_t &di,
    nn_workflow_item_t *          &workflow_item,
    nn_workflow_item_t *          &input_workflow_item,
    uint_least32_t                output_width,
    uint_least32_t                output_height,
    uint_least32_t                output_depth,
    float                         *weights,
    size_t                        *weight_coords,
    float                         *biases,
    size_t                        *bias_coords,
    uint_least32_t                kernel_stride_x,
    uint_least32_t                kernel_stride_y,
    uint_least32_t                pool_stride_x,
    uint_least32_t                pool_stride_y,
    uint_least32_t                pool_size_x,
    uint_least32_t                pool_size_y,
    uint_least32_t                center_x,
    uint_least32_t                center_y,
    NN_ACTIVATION_FUNCTION        activation_function)
{
    assert(pool_stride_x == 2);
    assert(pool_stride_y == 2);
    assert(pool_size_x == 2);
    assert(pool_size_y == 2);
    nn_workflow_use_descriptor_t desc = { input_workflow_item, 0 };
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&workflow_item, 1, &desc, 1));
    workflow_item->type = NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2;
    workflow_item->output_format[0].format = NN_DATA_FORMAT_3D;
    workflow_item->output_format[0].format_3d.size[0] = output_width;
    workflow_item->output_format[0].format_3d.size[1] = output_height;
    workflow_item->output_format[0].format_3d.size[2] = output_depth;
    workflow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.padding = NN_PADDING_MODE_NONE;
    workflow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.activation.function = activation_function;
    workflow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.center_offset[0] = center_x; 
    workflow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.center_offset[1] = center_y; 
    workflow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.stride[0] = kernel_stride_x;
    workflow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.stride[1] = kernel_stride_y;

    // Wrap weights data up with nn_data
    nn::data<float, 4> weight_data(weights, weight_coords, 4);
    workflow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights = new nn::data<float, 4>(weight_data);

    // Bias packaging into nn_data
    nn::data<float, 1> bias_data(biases, bias_coords, 1);
    workflow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.biases = new nn::data<float, 1>(bias_data);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void create_fully_connected_workflow_item(
    const nn_device_interface_0_t &di,
    nn_workflow_item_t *          &fully_connected_workflow_item,
    nn_workflow_item_t *          &input_workflow_item,
    uint_least32_t                output_width,
    float                         *weights,
    size_t                        *weight_coords,
    unsigned short                weights_dimensionality,      // num of weight_coords , currently 2 and 4 are supported 
    float                         *biases,
    size_t                        *bias_coords,
    NN_ACTIVATION_FUNCTION        activation_function)
{
    nn_workflow_use_descriptor_t desc = { input_workflow_item, 0 };
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_item_create_function( &fully_connected_workflow_item, 1, &desc, 1 ) );
    fully_connected_workflow_item->type =
        NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
    fully_connected_workflow_item->output_format[0].format                                  = NN_DATA_FORMAT_3D;
    fully_connected_workflow_item->output_format[0].format_3d.size[0]                       = output_width;
    fully_connected_workflow_item->output_format[0].format_3d.size[1]                       = 1;
    fully_connected_workflow_item->output_format[0].format_3d.size[2]                       = 1;
    fully_connected_workflow_item->arguments.forward_fully_connected.activation.function = activation_function;

    switch(weights_dimensionality)
    {
        case 2:
        {
            // Wrap weights data up with nn_data
            nn::data<float, 2> weight_data(weights, weight_coords, 2);
            fully_connected_workflow_item->arguments.forward_fully_connected.weights = new nn::data<float, 2>(weight_data);
            break;
        }
        case 4:
        {
            // Wrap weights data up with nn_data
            nn::data<float, 4> weight_data(weights, weight_coords, 4);
            fully_connected_workflow_item->arguments.forward_fully_connected.weights = new nn::data<float, 4>(weight_data);
            break;
        }
        default:
        {
            assert(0);
            printf("Error: Weights dimensionality can be only 2 or 4\n");
            return;
        }
    }

    // Bias packaging into nn_data
    nn::data<float, 1> bias_data(biases, bias_coords, 1);
    fully_connected_workflow_item->arguments.forward_fully_connected.biases = new nn::data<float, 1>(bias_data);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void create_pooling_workflow_item(
    const nn_device_interface_0_t &di,
    nn_workflow_item_t *          &pooling_workflow_item,
    nn_workflow_item_t *          &input_workflow_item,
    const uint_least32_t          output_width,
    const uint_least32_t          output_height,
    const uint_least32_t          output_depth,
    const uint_least32_t          window_size,
    const uint_least32_t          stride_x,
    const uint_least32_t          stride_y )
{
    nn_workflow_use_descriptor_t desc = { input_workflow_item, 0 };
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_item_create_function( &pooling_workflow_item, 1, &desc, 1 ) );

    pooling_workflow_item->type = NN_WORK_ITEM_TYPE_POOLING;
    pooling_workflow_item->output_format[0].format = NN_DATA_FORMAT_3D;
    pooling_workflow_item->output_format[0].format_3d.size[0] = output_width;
    pooling_workflow_item->output_format[0].format_3d.size[1] = output_height;
    pooling_workflow_item->output_format[0].format_3d.size[2] = output_depth;
    pooling_workflow_item->arguments.forward_pooling.stride[0] = stride_x;
    pooling_workflow_item->arguments.forward_pooling.stride[1] = stride_y;
    pooling_workflow_item->arguments.forward_pooling.size[0] = window_size;
    pooling_workflow_item->arguments.forward_pooling.size[1] = window_size;
    pooling_workflow_item->arguments.forward_pooling.mode = NN_POOLING_MODE_MAX;

}
///////////////////////////////////////////////////////////////////////////////////////////////////
void create_view_workflow_item(
    const nn_device_interface_0_t &di,
    nn_workflow_item_t *          &view_workflow_item,
    nn_workflow_item_t *          &input_workflow_item,
    nn_workload_data_coords_t     output_view_begin,
    nn_workload_data_coords_t     output_view_end )
{
    nn_workflow_use_descriptor_t desc = { input_workflow_item, 0 };
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_item_create_function( &view_workflow_item, 1, &desc, 1 ) );
    view_workflow_item->type                            = NN_WORK_ITEM_TYPE_VIEW;
    view_workflow_item->output_format[0].format            = NN_DATA_FORMAT_3D;
    view_workflow_item->output_format[0].format_3d.size[0] = output_view_end.t[NN_DATA_COORD_x] -
                                                          output_view_begin.t[NN_DATA_COORD_x] + 1;
    view_workflow_item->output_format[0].format_3d.size[1] = output_view_end.t[NN_DATA_COORD_y] -
                                                          output_view_begin.t[NN_DATA_COORD_y] + 1;
    view_workflow_item->output_format[0].format_3d.size[2] = output_view_end.t[NN_DATA_COORD_z] -
                                                          output_view_begin.t[NN_DATA_COORD_z] + 1;
    view_workflow_item->arguments.view.origin[0] = output_view_begin.t[NN_DATA_COORD_x];
    view_workflow_item->arguments.view.origin[1] = output_view_begin.t[NN_DATA_COORD_y];
    view_workflow_item->arguments.view.origin[2] = output_view_begin.t[NN_DATA_COORD_z];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void create_softmax_workflow_item(
    const nn_device_interface_0_t &di,
    nn_workflow_item_t *          &softmax_workflow_item,
    nn_workflow_item_t *          &input_workflow_item,
    const uint_least32_t          num_samples )
{
    nn_workflow_use_descriptor_t desc = { input_workflow_item, 0 };
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_item_create_function( &softmax_workflow_item, 1, &desc, 1 ) );
    softmax_workflow_item->type = NN_WORK_ITEM_TYPE_SOFTMAX;
    softmax_workflow_item->output_format[0].format = NN_DATA_FORMAT_3D;
    softmax_workflow_item->output_format[0].format_3d.size[0] = num_samples;
    softmax_workflow_item->output_format[0].format_3d.size[1] = 1;
    softmax_workflow_item->output_format[0].format_3d.size[2] = 1;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void create_arithmetic_workflow_item( 
     const nn_device_interface_0_t &di,
     nn_workflow_item_t * &arithmetic_workflow_item,
     nn_workflow_item_t * &input_workflow_item,
     const uint_least32_t output_width,
     const uint_least32_t output_height,
     const uint_least32_t output_depth,
     float                *const factor,
     size_t               *factor_coords,
     NN_ARITHMETIC_FUNCTION arithmetic_function )
{
    nn_workflow_use_descriptor_t desc = { input_workflow_item, 0 };
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_item_create_function( &arithmetic_workflow_item, 1, &desc, 1 ) );
    arithmetic_workflow_item->type = NN_WORK_ITEM_TYPE_ARITHMETIC;
    arithmetic_workflow_item->output_format[0].format = NN_DATA_FORMAT_3D;
    arithmetic_workflow_item->output_format[0].format_3d.size[0] = output_width;
    arithmetic_workflow_item->output_format[0].format_3d.size[1] = output_height;
    arithmetic_workflow_item->output_format[0].format_3d.size[2] = output_depth;
    arithmetic_workflow_item->arguments.forward_arithmetic.arithmetic_function = arithmetic_function;

    // Wrap factor data up with nn_data
    nn::data<float, 3> factor_data(factor, factor_coords, 3);
    arithmetic_workflow_item->arguments.forward_arithmetic.factor = new nn::data<float, 3>(factor_data);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void create_normalization_workflow_item(
    const nn_device_interface_0_t &di,
    nn_workflow_item_t *          &normalization_workflow_item,
    nn_workflow_item_t *          &input_workflow_item,
    const uint_least32_t          output_width,
    const uint_least32_t          output_height,
    const uint_least32_t          output_depth,
    const uint_least32_t          normalization_size,                            // normalization area
    const uint_least32_t          k,                                             //hyper parameter k
    const float                   alpha,                                //hyper parameter alpha
    const float                   beta,                                 //hyper parameter k
    NN_NORMALIZATION_MODE         normalization_mode)
{
    nn_workflow_use_descriptor_t desc = { input_workflow_item, 0 };
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_item_create_function( &normalization_workflow_item, 1, &desc, 1 ) );
    normalization_workflow_item->type                                                = NN_WORK_ITEM_TYPE_NORMALIZATION;
    normalization_workflow_item->output_format[0].format                                = NN_DATA_FORMAT_3D;
    normalization_workflow_item->output_format[0].format_3d.size[0]                     = output_width;
    normalization_workflow_item->output_format[0].format_3d.size[1]                     = output_height;
    normalization_workflow_item->output_format[0].format_3d.size[2]                     = output_depth;
    normalization_workflow_item->arguments.forward_normalization.normalization.mode  = normalization_mode;
    normalization_workflow_item->arguments.forward_normalization.normalization.n     = normalization_size;
    normalization_workflow_item->arguments.forward_normalization.normalization.k     = k;
    normalization_workflow_item->arguments.forward_normalization.normalization.alpha = alpha;
    normalization_workflow_item->arguments.forward_normalization.normalization.beta  = beta;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_arithmetic_workflow_test( const nn_device_interface_0_t &di,
                             const uint_least32_t          num_input_feature_maps,
                             const uint_least32_t          input_feature_map_width,
                             const uint_least32_t          input_feature_map_height,
                             const uint_least32_t          num_batches,
                             NN_ARITHMETIC_FUNCTION        arithmetic_function )  // which operation to execute on two buffers
{
    uint_least32_t output_width  = input_feature_map_width;
    uint_least32_t output_height = input_feature_map_height;
    uint_least32_t output_depth  = num_input_feature_maps;

    // Specify layout
    nn_workload_data_layout_t output_layout = nn::workload_data<float>::layout.xyzpnq;

    // specify dimensions of input, output
    size_t output_coords[4] = { input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches };
    size_t factor_coords[3] = { input_feature_map_width, input_feature_map_height, num_input_feature_maps };



    // Input generation (input feature maps to have normalization run on it)
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         num_batches );

    float *factor = nullptr;
    generate_input_data( factor, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         1 );


    float *cpu_outputs;
    init_data( cpu_outputs,
               num_batches * input_feature_map_width * input_feature_map_height * num_input_feature_maps,
               0.0f );

    float *gpu_outputs;
    init_data( gpu_outputs,
               num_batches * input_feature_map_width * input_feature_map_height * num_input_feature_maps,
               0.0f );

    // Activation function
    fp_func_arithmetic arithmetic_func = nullptr;
    switch( arithmetic_function )
    {
    case NN_ARITHMETIC_FUNCTION_NONE:
        arithmetic_func = none2f;
        break;
    case NN_ARITHMETIC_FUNCTION_ADDITION:
        arithmetic_func = add2f;
        break;
    case NN_ARITHMETIC_FUNCTION_SUBTRACTION:
        arithmetic_func = sub2f;
        break;
    case NN_ARITHMETIC_FUNCTION_MULTIPLICATION:
        arithmetic_func = mul2f;
        break;
    case NN_ARITHMETIC_FUNCTION_DIVISION:
        arithmetic_func = div2f;
        break;
    default:
        printf( "Error: Not supported arithmetic function chosen: %d\n", arithmetic_function );
        assert( 0 );
        break;
    }

    arithmetic_ref( cpu_outputs, input, factor, num_input_feature_maps, input_feature_map_width, input_feature_map_height, num_batches, arithmetic_func );

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di,
                                input_workflow_item,
                                num_input_feature_maps,
                                input_feature_map_width,
                                input_feature_map_height );

    // 2. Arithmetic workflow_item
    nn_workflow_item_t *arithmetic_workflow_item;

    create_arithmetic_workflow_item( di,
                                     arithmetic_workflow_item,
                                     input_workflow_item,
                                     output_width,
                                     output_height,
                                     output_depth,
                                     factor,
                                     factor_coords,
                                     arithmetic_function );
    
    //3. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di, output_workflow_item, arithmetic_workflow_item, output_width, 1, 1 );

    // 4. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &io_format, &io_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    using io_data = std::unique_ptr<nn::data<float, 0>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data(new nn::data<float, 0>(input, output_coords, 4));
    execute_outputs[0] = io_data(new nn::data<float, 0>(gpu_outputs, output_coords, 4));

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );
    delete reinterpret_cast<nn::data<float, 3>*>(arithmetic_workflow_item->arguments.forward_arithmetic.factor);
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( arithmetic_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );
#ifdef __linux__
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( input );
    input = nullptr;
    free( factor );
    factor = nullptr;
#else
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( input );
    input = nullptr;
    _aligned_free( factor );
    factor = nullptr;
#endif //__linux__

    return true;

}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_normalization_workflow_test( const nn_device_interface_0_t &di,
                             const uint_least32_t          num_batches,
                             const uint_least32_t          num_input_feature_maps,
                             const uint_least32_t          input_feature_map_width,
                             const uint_least32_t          input_feature_map_height,
                             const uint_least32_t          normalization_size,   // normalization area
                             const uint_least32_t          k,                    //hyper parameter k
                             const float                   alpha,       //hyper parameter alpha
                             const float                   beta,        //hyper parameter k
                             NN_NORMALIZATION_MODE         normalization_mode )  // mode of naormalization
{
    
    uint_least32_t output_width  = input_feature_map_width;
    uint_least32_t output_height = input_feature_map_height;
    uint_least32_t output_depth  = num_input_feature_maps;


    //TODO: Try changing layout so that depth of input feature map comes first
    // Specify layout
    nn_workload_data_layout_t output_layout = nn::workload_data<float>::layout.xyzpnq;

    // specify dimensions of input, output
    size_t output_coords[4] = { input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches };

    // Input generation (input feature maps to have normalization run on it)
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         num_batches );

    float *cpu_outputs;
    init_data( cpu_outputs,
               num_batches * input_feature_map_width * input_feature_map_height * num_input_feature_maps,
               0.0f );

    float *gpu_outputs;
    init_data( gpu_outputs,
               num_batches * input_feature_map_width * input_feature_map_height * num_input_feature_maps,
               0.0f );

    normalize_ref( cpu_outputs, input, num_batches, num_input_feature_maps, input_feature_map_width,
                   input_feature_map_height, normalization_size, k, alpha, beta, normalization_mode );

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di,
                                input_workflow_item,
                                num_input_feature_maps,
                                input_feature_map_width,
                                input_feature_map_height );

    // 2. Normalization workflow_item
    nn_workflow_item_t *normalization_workflow_item;

    create_normalization_workflow_item( di,
                                        normalization_workflow_item,
                                        input_workflow_item,
                                        output_width,
                                        output_height,
                                        output_depth,
                                        normalization_size,
                                        k,
                                        alpha,
                                        beta,
                                        normalization_mode );

    //3. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di, output_workflow_item, normalization_workflow_item, output_width, 1,
                                 1 );
    // 4. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &io_format, &io_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    using io_data = std::unique_ptr<nn::data<float, 0>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data(new nn::data<float, 0>(input, output_coords, 4));
    execute_outputs[0] = io_data(new nn::data<float, 0>(gpu_outputs, output_coords, 4));

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    nn_workload_data_coords_t output_view_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t output_view_end(num_batches - 1, output_width - 1, output_height - 1, output_depth - 1, 0, 0);

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( normalization_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );
#ifdef __linux__
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( input );
    input = nullptr;
#else
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( input );
    input = nullptr;
#endif //__linux__

    return true;

}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_normalization_view_convolution_test( const nn_device_interface_0_t &di)
{
    uint_least32_t num_input_feature_maps   = 3;
    uint_least32_t input_feature_map_width  = 256;
    uint_least32_t input_feature_map_height = 256;
    uint_least32_t normalize_output_width   = input_feature_map_width;
    uint_least32_t normalize_output_height  = input_feature_map_height;
    uint_least32_t normalize_output_depth   = num_input_feature_maps;

    uint_least32_t view_output_width   = 227;
    uint_least32_t view_output_height  = 227;
    uint_least32_t view_output_depth   = 3; 

    uint_least32_t         kernel_width        = 11;
    uint_least32_t         kernel_height       = 11;
    uint_least32_t         kernel_stride_x     = 4;
    uint_least32_t         kernel_stride_y     = 4;
    uint_least32_t         conv_output_depth   = 96;
    uint_least32_t         conv_output_width   = ( ( view_output_width - kernel_width ) / kernel_stride_x + 1 );
    uint_least32_t         conv_output_height  = ( ( view_output_height - kernel_height ) / kernel_stride_y + 1 );
    uint_least32_t         center_x            = 0;
    uint_least32_t         center_y            = 0;
    NN_ACTIVATION_FUNCTION activation_function = NN_ACTIVATION_FUNCTION_RELU;
    NN_NORMALIZATION_MODE  normalization_mode  = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
    const uint_least32_t   k                   = 2;
    const uint_least32_t   normalization_size  = 5;
    const float            alpha               = 0.0001f;
    const float            beta                = 0.75;
    const uint_least32_t   num_batches         = 2;
    const uint32_t origin[3] = {14,14,0};


    // Specify layout
    nn_workload_data_layout_t output_layout = nn::workload_data<float>::layout.xyzpnq;

    // specify dimensions of input, output
    size_t input_coords[4] = {input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches};
    size_t output_coords[4] = { conv_output_width, conv_output_height, conv_output_depth, num_batches };
    size_t weight_coords[4] = {kernel_width, kernel_height, normalize_output_depth, conv_output_depth};       // Half of depth of weights to be used
    size_t bias_coords[1] = {conv_output_depth};

    // Input generation (input feature maps to have normalization run on it)
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         num_batches );

    float *normalize_cpu_outputs;
    init_data( normalize_cpu_outputs,
               num_batches * input_feature_map_width * input_feature_map_height * num_input_feature_maps,
               0.0f );

    // Generate Filter Data
    float *filters = nullptr;
    generate_filter_data( filters,
                          kernel_width,
                          kernel_height,
                          normalize_output_depth,   
                          conv_output_depth );

    float *biases = nullptr;
    init_data( biases, conv_output_width * conv_output_height * conv_output_depth, 10.0f );

    normalize_ref( normalize_cpu_outputs, input, num_batches, num_input_feature_maps, input_feature_map_width,
                   input_feature_map_height, normalization_size, k, alpha, beta, normalization_mode );

    float *cpu_outputs = nullptr;
    float *gpu_outputs = nullptr;
    init_data( gpu_outputs, conv_output_width * conv_output_height * conv_output_depth * num_batches, 0.0f );
    init_data( cpu_outputs, conv_output_width * conv_output_height * conv_output_depth * num_batches, 0.0f );

    // Activation function
    fp_func_activ activ_func = nullptr;
    switch( activation_function )
    {
    case NN_ACTIVATION_FUNCTION_NONE:
        activ_func = none;
        break;
    case NN_ACTIVATION_FUNCTION_TANH:
        activ_func = mytanh;
        break;
    case NN_ACTIVATION_FUNCTION_RELU:
        activ_func = relu;
        break;
    case NN_ACTIVATION_FUNCTION_SOFTPLUS:
        activ_func = softplus;
        break;
    default:
        printf( "Error: Not supported activation function chosen: %d\n", activation_function );
        assert( 0 );
        break;
    }

    nn_workload_data_coords_t conv_input_view_begin( 0, origin[0], origin[1], origin[2], 0, 0 );
    nn_workload_data_coords_t conv_input_view_end( num_batches - 1, origin[0] + view_output_width - 1, origin[1] + view_output_height - 1, origin[2] + view_output_depth - 1, 0, 0 );
    nn_workload_data_coords_t conv_output_view_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t conv_output_view_end( num_batches - 1, conv_output_width - 1, conv_output_height - 1, conv_output_depth - 1, 0, 0 );

    // Run reference first convolution
    convolve_ref( activ_func,
                  cpu_outputs,
                  normalize_cpu_outputs,
                  filters,
                  biases,
                  conv_output_view_begin,
                  conv_output_view_end,
                  conv_input_view_begin,
                  conv_input_view_end,
                  conv_output_width,
                  conv_output_height,
                  conv_output_depth,
                  normalize_output_width,
                  normalize_output_height,
                  normalize_output_depth,
                  kernel_width,
                  kernel_height,
                  normalize_output_depth,
                  kernel_stride_x,
                  kernel_stride_y,
                  center_x,        // center offset x
                  center_y,        // center offset y
                  num_batches );

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di,
                                input_workflow_item,
                                num_input_feature_maps,
                                input_feature_map_width,
                                input_feature_map_height );

    // 2. Normalization workflow_item
    nn_workflow_item_t *normalization_workflow_item;

    create_normalization_workflow_item( di,
                                        normalization_workflow_item,
                                        input_workflow_item,
                                        normalize_output_width,
                                        normalize_output_height,
                                        normalize_output_depth,
                                        normalization_size,
                                        k,
                                        alpha,
                                        beta,
                                        normalization_mode );
    //3. View workflow Item

    nn_workflow_item_t *view_workflow_item;

    nn_workload_data_coords_t view_to_normalization_begin( 0, origin[0], origin[1], origin[2], 0, 0 );
    nn_workload_data_coords_t view_to_normalization_end( num_batches - 1, origin[0] + view_output_width - 1, origin[1] + view_output_height - 1, origin[2] + view_output_depth - 1, 0, 0 );

    create_view_workflow_item( di,
                               view_workflow_item,
                               normalization_workflow_item,
                               view_to_normalization_begin,
                               view_to_normalization_end );

    // 4. Convolution workflow Items working on views

    nn_workflow_item_t *conv_workflow_item;

    create_convolution_workflow_item( di,
                                      conv_workflow_item,
                                      view_workflow_item,
                                      conv_output_width,
                                      conv_output_height,
                                      conv_output_depth,
                                      filters,
                                      weight_coords,
                                      biases,
                                      bias_coords,
                                      kernel_stride_x,
                                      kernel_stride_y,
                                      center_x,        // center offset x
                                      center_y,        // center offset y
                                      activation_function );

    //6. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di, output_workflow_item, conv_workflow_item, conv_output_width, conv_output_height, conv_output_depth );

    // 7. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload           *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &io_format, &io_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    using   io_data = std::unique_ptr< nn::data< float, 0 > >;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data( new nn::data< float, 0 >( input, input_coords, 4 ) );
    execute_outputs[0] = io_data( new nn::data< float, 0 >( gpu_outputs, output_coords, 4 ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( normalization_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( view_workflow_item  ));
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( conv_workflow_item  ));
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );

#ifdef __linux__
    free( normalize_cpu_outputs);
    normalize_cpu_outputs = nullptr;
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( input );
    input = nullptr;
    free( biases );
    biases = nullptr;
    free( filters );
    filters = nullptr;
#else
    _aligned_free(normalize_cpu_outputs); 
    normalize_cpu_outputs = nullptr;
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( input );
    input = nullptr;
    _aligned_free( biases );
    biases = nullptr;
    _aligned_free( filters );
    filters = nullptr;
#endif //__linux__

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_normalization_convolution_split_test( const nn_device_interface_0_t &di)
{
    uint_least32_t num_input_feature_maps   = 96;
    uint_least32_t input_feature_map_width  = 31;
    uint_least32_t input_feature_map_height = 31;
    uint_least32_t normalize_output_width   = input_feature_map_width;
    uint_least32_t normalize_output_height  = input_feature_map_height;
    uint_least32_t normalize_output_depth   = num_input_feature_maps;
    uint_least32_t kernel_width             = 5;
    uint_least32_t kernel_height            = 5;
    uint_least32_t kernel_stride_x          = 1;
    uint_least32_t kernel_stride_y          = 1;
    uint_least32_t conv_output_depth        = 256;
    uint_least32_t conv_output_width        =
        ( ( normalize_output_width - kernel_width ) / kernel_stride_x + 1 );
    uint_least32_t conv_output_height =
        ( ( normalize_output_height - kernel_height ) / kernel_stride_y + 1 );
    uint_least32_t         center_x            = 0;
    uint_least32_t         center_y            = 0;
    NN_ACTIVATION_FUNCTION activation_function = NN_ACTIVATION_FUNCTION_RELU;
    NN_NORMALIZATION_MODE  normalization_mode  = NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS;
    const uint_least32_t   k                   = 2;
    const uint_least32_t   normalization_size  = 5;
    const float            alpha               = 0.0001f;
    const float            beta                = 0.75;
    const uint_least32_t   num_batches         = 2;

    //TODO: zero-padding support !!!

    // Specify layout
    nn_workload_data_layout_t output_layout = nn::workload_data<float>::layout.xyzpnq;

    // specify dimensions of input, output
    size_t input_coords[4] = {input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches};
    size_t output_coords[4] = { conv_output_width, conv_output_height, conv_output_depth, num_batches };
    size_t weight_coords[4] = {kernel_width, kernel_height, normalize_output_depth/2, conv_output_depth/2};       // Half of depth of weights to be used
    size_t bias_coords[1] = {conv_output_depth/2};

    // Input generation (input feature maps to have normalization run on it)
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         num_batches );

    float *normalize_cpu_outputs;
    init_data( normalize_cpu_outputs,
               num_batches * input_feature_map_width * input_feature_map_height * num_input_feature_maps,
               0.0f );

    // Generate Filter Data
    float *filters = nullptr;
    generate_filter_data( filters,
                          kernel_width,
                          kernel_height,
                          normalize_output_depth/2,   //TODO: This is twice as much depth as really used for every filter
                          conv_output_depth/2 );

    // TODO: Enable Biases!!! Make it random values
    float *biases = nullptr;
    init_data( biases, conv_output_width * conv_output_height * conv_output_depth, 100.0f );

    normalize_ref( normalize_cpu_outputs, input, num_batches, num_input_feature_maps, input_feature_map_width,
                   input_feature_map_height, normalization_size, k, alpha, beta, normalization_mode );

    float *cpu_outputs = nullptr;
    float *gpu_outputs = nullptr;
    init_data( gpu_outputs, conv_output_width * conv_output_height * conv_output_depth * num_batches, 0.0f );
    init_data( cpu_outputs, conv_output_width * conv_output_height * conv_output_depth * num_batches, 0.0f );

    // Activation function
    fp_func_activ activ_func = nullptr;
    switch( activation_function )
    {
    case NN_ACTIVATION_FUNCTION_NONE:
        activ_func = none;
        break;
    case NN_ACTIVATION_FUNCTION_TANH:
        activ_func = mytanh;
        break;
    case NN_ACTIVATION_FUNCTION_RELU:
        activ_func = relu;
        break;
    case NN_ACTIVATION_FUNCTION_SOFTPLUS:
        activ_func = softplus;
        break;
    default:
        printf( "Error: Not supported activation function chosen: %d\n", activation_function );
        assert( 0 );
        break;
    }
    // First convolution will work on first half of feature maps
    // and is using only first half of filters
    nn_workload_data_coords_t first_conv_input_view_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t first_conv_input_view_end( num_batches - 1, input_feature_map_width - 1, input_feature_map_height - 1, num_input_feature_maps/2 - 1, 0, 0 );
    nn_workload_data_coords_t first_conv_output_view_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t first_conv_output_view_end( num_batches - 1, conv_output_width - 1, conv_output_height - 1, conv_output_depth/2 - 1, 0, 0 );

    // Run reference first convolution
    convolve_ref( activ_func,
                  cpu_outputs,
                  normalize_cpu_outputs,
                  filters,
                  biases,
                  first_conv_output_view_begin,
                  first_conv_output_view_end,
                  first_conv_input_view_begin,
                  first_conv_input_view_end,
                  conv_output_width,
                  conv_output_height,
                  conv_output_depth,
                  normalize_output_width,
                  normalize_output_height,
                  normalize_output_depth,
                  kernel_width,
                  kernel_height,
                  normalize_output_depth/2,
                  kernel_stride_x,
                  kernel_stride_y,
                  center_x,        // center offset x
                  center_y,        // center offset y
                  num_batches );

    // Second convolution will work on second half of feature maps
    // and is using only first half of filters
    nn_workload_data_coords_t second_conv_input_view_begin( 0, 0, 0, num_input_feature_maps/2, 0, 0 );
    nn_workload_data_coords_t second_conv_input_view_end( num_batches - 1, input_feature_map_width - 1, input_feature_map_height - 1, num_input_feature_maps - 1, 0, 0 );
    nn_workload_data_coords_t second_conv_output_view_begin( 0, 0, 0, conv_output_depth/2, 0, 0 );
    nn_workload_data_coords_t second_conv_output_view_end( num_batches - 1, conv_output_width - 1, conv_output_height - 1, conv_output_depth - 1, 0, 0 );

    convolve_ref( activ_func,
                  cpu_outputs,
                  normalize_cpu_outputs,
                  filters,
                  biases,
                  second_conv_output_view_begin,
                  second_conv_output_view_end,
                  second_conv_input_view_begin,
                  second_conv_input_view_end,
                  conv_output_width,
                  conv_output_height,
                  conv_output_depth,
                  normalize_output_width,
                  normalize_output_height,
                  normalize_output_depth,
                  kernel_width,
                  kernel_height,
                  normalize_output_depth/2,
                  kernel_stride_x,
                  kernel_stride_y,
                  center_x,        // center offset x
                  center_y,        // center offset y
                  num_batches );

    // Make reference convolution here on whole buffer
    // Then compare results with gpu splittetd solution

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di,
                                input_workflow_item,
                                num_input_feature_maps,
                                input_feature_map_width,
                                input_feature_map_height );

    // 2. Normalization workflow_item
    nn_workflow_item_t *normalization_workflow_item;

    create_normalization_workflow_item( di,
                                        normalization_workflow_item,
                                        input_workflow_item,
                                        normalize_output_width,
                                        normalize_output_height,
                                        normalize_output_depth,
                                        normalization_size,
                                        k,
                                        alpha,
                                        beta,
                                        normalization_mode );

    // 3. View workflow_item
    // Create two views to normalization layer. First will allow following convolution to process first half of feature
    // maps
    // second View will allow following convolution to process second half of feature maps

    nn_workflow_item_t *first_view_workflow_item;

    nn_workload_data_coords_t first_view_to_normalization_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t first_view_to_normalization_end( num_batches - 1, normalize_output_width - 1, normalize_output_height - 1, normalize_output_depth / 2 - 1, 0, 0 );

    create_view_workflow_item( di,
                               first_view_workflow_item,
                               normalization_workflow_item,
                               first_view_to_normalization_begin,
                               first_view_to_normalization_end );


    nn_workflow_item_t *second_view_workflow_item;

    nn_workload_data_coords_t second_view_to_normalization_begin( 0, 0, 0, normalize_output_depth / 2, 0, 0 );
    nn_workload_data_coords_t second_view_to_normalization_end( num_batches - 1, normalize_output_width - 1, normalize_output_height - 1, normalize_output_depth - 1, 0, 0 );

    create_view_workflow_item( di,
                               second_view_workflow_item,
                               normalization_workflow_item,
                               second_view_to_normalization_begin,
                               second_view_to_normalization_end );


    // 4. Convolution workflow Items working on views

    nn_workflow_item_t *first_conv_workflow_item;

    create_convolution_workflow_item( di,
                                      first_conv_workflow_item,
                                      first_view_workflow_item,
                                      conv_output_width,
                                      conv_output_height,
                                      conv_output_depth/2,  //This convolution is processing first half of filters
                                      filters,
                                      weight_coords,
                                      biases,
                                      bias_coords,
                                      kernel_stride_x,
                                      kernel_stride_y,
                                      center_x,        // center offset x
                                      center_y,        // center offset y
                                      activation_function );


    nn_workflow_item_t *second_conv_workflow_item;

    create_convolution_workflow_item( di,
                                      second_conv_workflow_item,
                                      second_view_workflow_item,
                                      conv_output_width,
                                      conv_output_height,
                                      conv_output_depth/2,  //This convolution is processing second half of filters
                                      filters,
                                      weight_coords,
                                      biases,
                                      bias_coords,
                                      kernel_stride_x,
                                      kernel_stride_y,
                                      center_x,        // center offset x
                                      center_y,        // center offset y
                                      activation_function );
    // 5. Merging workflow item
    nn_workflow_item_t *merge_workflow_item;
    nn_workflow_item_t *convolution_work_flow_items[2] = {first_conv_workflow_item, second_conv_workflow_item};

    create_merge_workflow_item( di, merge_workflow_item, 2, convolution_work_flow_items, 2 );

    //6. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di, output_workflow_item, merge_workflow_item, conv_output_width, conv_output_height,
                                 conv_output_depth );
    // 7. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload           *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &io_format, &io_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    using   io_data = std::unique_ptr< nn::data< float, 0 > >;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data( new nn::data< float, 0 >( input, input_coords, 4 ) );
    execute_outputs[0] = io_data( new nn::data< float, 0 >( gpu_outputs, output_coords, 4 ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( normalization_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( first_view_workflow_item  ));
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( second_view_workflow_item ));
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( first_conv_workflow_item  ));
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( second_conv_workflow_item ));
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( merge_workflow_item       ));
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );

    //TODO: change it to unique_ptrs

#ifdef __linux__
    free( normalize_cpu_outputs);
    normalize_cpu_outputs = nullptr;
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( input );
    input = nullptr;
    free( biases );
    biases = nullptr;
    free( filters );
    filters = nullptr;
#else
    _aligned_free(normalize_cpu_outputs); 
    normalize_cpu_outputs = nullptr;
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( input );
    input = nullptr;
    _aligned_free( biases );
    biases = nullptr;
    _aligned_free( filters );
    filters = nullptr;
#endif //__linux__

    return true;


}
///////////////////////////////////////////////////////////////////////////////////////////////////
#if 0 // those tests are diabled because currently there's no container passed to execute function that supports views
bool run_views( const nn_device_interface_0_t &di)
{
    // Run : input -> view -> output  
    const uint_least32_t num_batches = 1;  
    uint_least32_t num_input_feature_maps   = 100;
    uint_least32_t input_feature_map_width  = 100;
    uint_least32_t input_feature_map_height = 100;

    // Specify layout
    nn_workload_data_layout_t input_output_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        nn::workload_data<float>::layout.xyzpnq,
        NN_DATATYPE_FLOAT
    };

    nn_workload_data_coords_t output_view_begin( 0, 1, 1, 0, 0, 0 );
    nn_workload_data_coords_t output_view_end( 0, 1, 11, 0, 0, 0 );

    // specify dimensions of input 
    nn_workload_data_coords_t input_coords =
    {
        num_batches,
        input_feature_map_width,
        input_feature_map_height,
        num_input_feature_maps,
        1,
        1
    };

    // specify dimensions of output 
    nn_workload_data_coords_t output_coords =
    {
        num_batches,
        output_view_end.t[NN_DATA_COORD_x] - output_view_begin.t[NN_DATA_COORD_x] + 1,
        output_view_end.t[NN_DATA_COORD_y] - output_view_begin.t[NN_DATA_COORD_y] + 1,
        output_view_end.t[NN_DATA_COORD_z] - output_view_begin.t[NN_DATA_COORD_z] + 1,
        1,
        1
    };


    // Input generation (input feature maps to have pooling run on it)
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches );

    float *cpu_outputs;
    init_data( cpu_outputs,
               num_batches * input_feature_map_width * input_feature_map_height * num_input_feature_maps,
               0.0f );

    float *gpu_outputs;
    init_data( gpu_outputs,
               num_batches * input_feature_map_width * input_feature_map_height * num_input_feature_maps,
               0.0f );


    view_ref(cpu_outputs,input,output_view_begin,output_view_end,input_feature_map_width, input_feature_map_height, num_input_feature_maps);

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di,
                                input_workflow_item,
                                num_input_feature_maps,
                                input_feature_map_width,
                                input_feature_map_height );

    // 2. Normalization workflow_item
    nn_workflow_item_t *view_workflow_item;

    create_view_workflow_item( di,
                               view_workflow_item,
                               input_workflow_item,
                               output_view_begin,
                               output_view_end );

    //3. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di,
                                 output_workflow_item,
                                 view_workflow_item,
                                 output_view_end.t[NN_DATA_COORD_x] - output_view_begin.t[NN_DATA_COORD_x] + 1,
                                 output_view_end.t[NN_DATA_COORD_y] - output_view_begin.t[NN_DATA_COORD_y] + 1,
                                 output_view_end.t[NN_DATA_COORD_z] - output_view_begin.t[NN_DATA_COORD_z] + 1 );

    // 4. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_DATA_TYPE io_format = NN_DATA_TYPE_DATA_T;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &io_format, &io_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    std::unique_ptr< nn::workload_data< float > > execute_inputs[1];
    std::unique_ptr< nn::workload_data< float > > execute_outputs[1];

    execute_inputs[0]  = create_nn_workload_data_using_buffer( input, input_output_layout, input_coords );
    execute_outputs[0] = create_nn_workload_data_using_buffer( gpu_outputs, input_output_layout, output_coords );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs, output_view_begin, output_view_end, input_feature_map_width, input_feature_map_height, num_input_feature_maps, 1, num_batches ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( view_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );
#ifdef __linux__
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( input );
    input = nullptr;
#else
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( input );
    input = nullptr;
#endif //__linux__

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_pooling_view_workflow_test( const nn_device_interface_0_t &di)
{
    // Run : input -> pooling -> view -> output  
    uint_least32_t num_input_feature_maps   = 1;
    uint_least32_t input_feature_map_width  = 4;
    uint_least32_t input_feature_map_height = 4;
    uint_least32_t num_batches              = 1;
    uint_least32_t window_size              = 2;
    uint_least32_t stride_x                 = 2;
    uint_least32_t stride_y                 = 2;

    // pooling operates on two dimensional feature maps
    // so output feadture maps depth is equal to input feature maps depth
    uint_least32_t output_width  = ( ( input_feature_map_width - window_size ) / stride_x + 1 );
    uint_least32_t output_height = ( ( input_feature_map_height - window_size ) / stride_y + 1 );
    uint_least32_t output_depth  = num_input_feature_maps;

    // Specify layout
    nn_workload_data_layout_t input_output_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        nn::workload_data<float>::layout.xyzpnq,
        NN_DATATYPE_FLOAT
    };

    //// Let's define a views to input and output for pooling operation
    nn_workload_data_coords_t input_view_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t input_view_end( 0, 1, 1, 0, 0, 0 );
    nn_workload_data_coords_t output_view_begin( 0, 1, 1, 0, 0, 0 );
    nn_workload_data_coords_t output_view_end( 0, 1, 1, 0, 0, 0 );

    // specify dimensions of input and output buffers
    nn_workload_data_coords_t input_coords =
    {
        num_batches,
        input_feature_map_width,
        input_feature_map_height,
        num_input_feature_maps,
        1,
        1
    };

    // specify dimensions of output 
    nn_workload_data_coords_t output_coords =
    {
        num_batches,
        output_view_end.t[NN_DATA_COORD_x] - output_view_begin.t[NN_DATA_COORD_x] + 1,
        output_view_end.t[NN_DATA_COORD_y] - output_view_begin.t[NN_DATA_COORD_y] + 1,
        output_view_end.t[NN_DATA_COORD_z] - output_view_begin.t[NN_DATA_COORD_z] + 1,
        1,
        1
    };

    // Input generation (input feature maps to have pooling run on it)
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         num_batches );

    float *cpu_outputs;
    init_data( cpu_outputs, output_width * output_height * output_depth * num_batches, 0.0f );

    float *gpu_outputs;
    init_data( gpu_outputs, output_width * output_height * output_depth * num_batches, 0.0f );

    pool_ref( cpu_outputs,
              input,
              output_view_begin,
              output_view_end,
              input_view_begin,
              input_view_end,
              output_width,
              output_height,
              output_depth,
              input_feature_map_width,
              input_feature_map_height,
              num_input_feature_maps,
              window_size,
              stride_x,
              stride_y,
              num_batches );

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di,
                                input_workflow_item,
                                num_input_feature_maps,
                                input_feature_map_width,
                                input_feature_map_height );

    // 2. View workflow_item
    nn_workflow_item_t *view_workflow_item;

    create_view_workflow_item( di,
                               view_workflow_item,
                               input_workflow_item,
                               input_view_begin,
                               input_view_end );

    // 3. Pooling workflow item 
    nn_workflow_item_t *pooling_workflow_item;

    // We need to know output of pooling sizes
    uint_least32_t output_pooling_width  = ( ( input_view_end.t[NN_DATA_COORD_x] - input_view_begin.t[NN_DATA_COORD_x] + 1 - window_size ) / stride_x + 1 );
    uint_least32_t output_pooling_height = ( ( input_view_end.t[NN_DATA_COORD_y] - input_view_begin.t[NN_DATA_COORD_y] + 1 - window_size ) / stride_y + 1 );
    uint_least32_t output_pooling_depth  = num_input_feature_maps;

    create_pooling_workflow_item( di,
                                  pooling_workflow_item,
                                  view_workflow_item,
                                  output_pooling_width,
                                  output_pooling_height,
                                  output_pooling_depth,
                                  window_size,
                                  stride_x,
                                  stride_y );

    // 4. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di,
                                 output_workflow_item,
                                 pooling_workflow_item,
                                 output_view_end.t[NN_DATA_COORD_x] - output_view_begin.t[NN_DATA_COORD_x] + 1,
                                 output_view_end.t[NN_DATA_COORD_y] - output_view_begin.t[NN_DATA_COORD_y] + 1,
                                 output_view_end.t[NN_DATA_COORD_z] - output_view_begin.t[NN_DATA_COORD_z] + 1 );

    // 5. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_DATA_TYPE io_format = NN_DATA_TYPE_DATA_T;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &io_format, &io_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    std::unique_ptr< nn::workload_data< float > > execute_inputs[1];
    std::unique_ptr< nn::workload_data< float > > execute_outputs[1];

    execute_inputs[0]  = create_nn_workload_data_using_buffer( input, input_output_layout, input_coords );
    execute_outputs[0] = create_nn_workload_data_using_buffer( gpu_outputs, input_output_layout, output_coords );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs, output_view_begin, output_view_end, output_width, output_height, output_depth, 1, num_batches ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( view_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( pooling_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );

#ifdef __linux__
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( input );
    input = nullptr;
#else
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( input );
    input = nullptr;
#endif //__linux__

    return true;
}

#endif



///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_softmax_workflow_test( const nn_device_interface_0_t &di,
                       uint_least32_t                num_samples,
                       uint_least32_t                num_batches) // length of input to be  processed (softmax normalize)
{
    // Specify layout of softmax workload
    nn_workload_data_layout_t output_layout = nn::workload_data<float>::layout.xyzpnq;

    // specify dimensions of input, output
    size_t output_coords[2] = {num_samples, num_batches};

    // Input generation (input feature maps to have pooling run on it)
    float *input = nullptr;
    generate_input_data( input, num_samples, 1, 1, num_batches );

    // length of output is the same as input

    float *cpu_outputs;
    init_data( cpu_outputs, num_samples * num_batches, 0.0f );

    float *gpu_outputs;
    init_data( gpu_outputs, num_samples * num_batches, 0.0f );

    softmax_ref( cpu_outputs, input, num_samples, num_batches );

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di,
                                input_workflow_item,
                                1,
                                num_samples,
                                1 );

    // 2. Fully_connected workflow_item
    nn_workflow_item_t *softmax_workflow_item;

    create_softmax_workflow_item( di,
                                  softmax_workflow_item,
                                  input_workflow_item,
                                  num_samples );

    //3. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di, output_workflow_item, softmax_workflow_item, num_samples, 1,
                                 1 );
    // 4. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &io_format, &io_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    using io_data = std::unique_ptr<nn::data<float>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data(new nn::data<float>(input, output_coords, 2));
    execute_outputs[0] = io_data(new nn::data<float>(gpu_outputs, output_coords, 2));

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    nn_workload_data_coords_t output_view_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t output_view_end(num_batches - 1, num_samples - 1, 0, 0, 0, 0);

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( softmax_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );

#ifdef __linux__
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( input );
    input = nullptr;
#else
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( input );
    input = nullptr;
#endif //__linux__

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_pooling_workflow_test( const nn_device_interface_0_t &di,
                       uint_least32_t                num_input_feature_maps,
                       uint_least32_t                input_feature_map_width,
                       uint_least32_t                input_feature_map_height,
                       const uint_least32_t          window_size, // pooling window is of size n x n , window_size is
                                                                  // equal to this n
                       const uint_least32_t          stride_x,  // stride in x dimension (should be at least equal to
                                                                // window_size to have non overlapping mode)
                       const uint_least32_t          stride_y,  // stride in y dimension (should be at least equal to
                                                                // window_size to have non overlapping mode)
                       uint_least32_t                num_batches ) 
                                                                
{
    // pooling operates on two dimensional feature maps
    // so output feadture maps depth is equal to input feature maps depth
    uint_least32_t output_width  = ( ( input_feature_map_width - window_size ) / stride_x + 1 );
    uint_least32_t output_height = ( ( input_feature_map_height - window_size ) / stride_y + 1 );
    uint_least32_t output_depth  = num_input_feature_maps;

#if 0
    // Specify layouts of pooling buffers
    nn_workload_data_layout_t input_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        nn::workload_data<float>::layout.xyzpnq, // ordering
        NN_DATATYPE_FLOAT
    };
    nn_workload_data_layout_t output_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        nn::workload_data<float>::layout.xyzpnq, // ordering
        NN_DATATYPE_FLOAT
    };
#endif

    // specify dimensions of input and output buffers
    size_t input_coords[4]  = {input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches};
    size_t output_coords[4] = {output_width, output_height, num_input_feature_maps, num_batches};

    // Input generation (input feature maps to have pooling run on it)
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         num_batches );

    float *cpu_outputs;
    init_data( cpu_outputs, output_width * output_height * output_depth * num_batches, 0.0f );

    float *gpu_outputs;
    init_data( gpu_outputs, output_width * output_height * output_depth * num_batches, 0.0f );

    nn_workload_data_coords_t input_view_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t output_view_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t input_view_end(num_batches - 1, input_feature_map_width-1, input_feature_map_height-1, num_input_feature_maps-1, 0, 0);
    nn_workload_data_coords_t output_view_end(num_batches - 1, output_width - 1, output_height - 1, output_depth - 1, 0, 0);

    pool_ref( cpu_outputs,
              input,
              output_view_begin,
              output_view_end,
              input_view_begin,
              input_view_end,
              output_width,
              output_height,
              output_depth,
              input_feature_map_width,
              input_feature_map_height,
              num_input_feature_maps,
              window_size,
              stride_x,
              stride_y,
              num_batches );

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di,
                                input_workflow_item,
                                num_input_feature_maps,
                                input_feature_map_width,
                                input_feature_map_height );

    // 2. Pooling workflow_item
    nn_workflow_item_t *pooling_workflow_item;

    create_pooling_workflow_item( di,
                                  pooling_workflow_item,
                                  input_workflow_item,
                                  output_width,
                                  output_height,
                                  output_depth,
                                  window_size,
                                  stride_x,
                                  stride_y );

    //3. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di, output_workflow_item, pooling_workflow_item, output_width, 1,
                                 1 );

    // 4. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &io_format, &io_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    using io_data = std::unique_ptr<nn::data<float, 0>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data(new nn::data<float, 0>(input, input_coords, 4));
    execute_outputs[0] = io_data(new nn::data<float, 0>(gpu_outputs, output_coords, 4));

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs ) );

    //Releasing data
    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( pooling_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );

#ifdef __linux__
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( input );
    input = nullptr;
#else
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( input );
    input = nullptr;
#endif //__linux__

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_pooling_fully_connected_workflow_test( 
        const nn_device_interface_0_t &di,
        uint_least32_t                num_output_feature_maps,  // num_output_feature_maps (number of neurons in FC layer)
        NN_ACTIVATION_FUNCTION        activation_function,      //(Activation function to be used for FC layer)
        uint_least32_t                num_input_feature_maps,   //(entry for pooling)
        uint_least32_t                input_feature_map_width,  //(entry for pooling)
        uint_least32_t                input_feature_map_height, //(entry for pooling)
        const uint_least32_t          window_size, // pooling window is of size n x n ,(entry for pooling)
        const uint_least32_t          stride_x,  // stride in x dimension (entry for pooling)
        const uint_least32_t          stride_y,  // stride in y dimension (entry for pooling)
        uint_least32_t                num_batches)
{

    // pooling operates on two dimensional feature maps
    // so output feadture maps depth is equal to input feature maps depth
    uint_least32_t pooling_output_width  = ( ( input_feature_map_width - window_size ) / stride_x + 1 );
    uint_least32_t pooling_output_height = ( ( input_feature_map_height - window_size ) / stride_y + 1 );
    uint_least32_t pooling_output_depth  = num_input_feature_maps;

    // Specify layout
    nn_workload_data_layout_t output_layout = nn::workload_data<float>::layout.xyzpnq;

    // specify dimensions of input, output
    size_t input_coords[4] = {input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches};
    size_t output_coords[2] = {num_output_feature_maps, num_batches};
    size_t weight_coords[4] = {pooling_output_width, pooling_output_height, pooling_output_depth, num_output_feature_maps };
    size_t bias_coords[1] = {num_output_feature_maps};

    // Input generation (input feature maps to have normalization run on it)
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         num_batches );

    float *pooling_cpu_outputs;
    init_data( pooling_cpu_outputs, num_batches*pooling_output_width*pooling_output_height*pooling_output_depth, 0.0f );

    float *biases         = nullptr;
    float *cpu_outputs    = nullptr;
    float *gpu_outputs    = nullptr;

    init_data( biases, num_output_feature_maps , 10.0f );
    init_data( gpu_outputs, num_output_feature_maps * num_batches, 0.0f );
    init_data( cpu_outputs, num_output_feature_maps * num_batches, 0.0f );

    // Generate Filter Data
    float *filters = nullptr;
    generate_filter_data( filters, pooling_output_width, pooling_output_height, pooling_output_depth, num_output_feature_maps );

    //// Let's define a views to input and output for pooling operation
    nn_workload_data_coords_t input_view_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t input_view_end( input_coords[3] - 1, input_coords[0] - 1, input_coords[1] - 1, input_coords[2] - 1, 0, 0 );
    nn_workload_data_coords_t output_view_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t output_view_end( num_batches - 1, pooling_output_width - 1, pooling_output_height - 1, pooling_output_depth - 1, 0, 0 );

    pool_ref( pooling_cpu_outputs,
              input,
              output_view_begin,
              output_view_end,
              input_view_begin,
              input_view_end,
              pooling_output_width,
              pooling_output_height,
              pooling_output_depth,
              input_feature_map_width,
              input_feature_map_height,
              num_input_feature_maps,
              window_size,
              stride_x,
              stride_y,
              num_batches );

    // Activation function
    fp_func_activ activ_func = nullptr;
    switch( activation_function )
    {
    case NN_ACTIVATION_FUNCTION_NONE:
        activ_func = none;
        break;
    case NN_ACTIVATION_FUNCTION_TANH:
        activ_func = mytanh;
        break;
    case NN_ACTIVATION_FUNCTION_RELU:
        activ_func = relu;
        break;
    case NN_ACTIVATION_FUNCTION_SOFTPLUS:
        activ_func = softplus;
        break;
    default:
        printf( "Error: Not supported activation function chosen: %d\n", activation_function );
        assert( 0 );
        break;
    }

    // Run reference fully connected (needed for comparison)
    fully_connect_ref( activ_func,
                       cpu_outputs,
                       pooling_cpu_outputs, // pooling's output is an input to fully connected
                       filters,
                       biases,
                       num_output_feature_maps,
                       pooling_output_width,
                       pooling_output_height,
                       pooling_output_depth,
                       num_batches );

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di,
                                input_workflow_item,
                                num_input_feature_maps,
                                input_feature_map_width,
                                input_feature_map_height );

    // 2. Pooling workflow_item
    nn_workflow_item_t *pooling_workflow_item;

    create_pooling_workflow_item( di,
                                  pooling_workflow_item,
                                  input_workflow_item,
                                  pooling_output_width,
                                  pooling_output_height,
                                  pooling_output_depth,
                                  window_size,
                                  stride_x,
                                  stride_y );

    // 3. Fully_connected workflow_item
    nn_workflow_item_t *fully_connect_workflow_item;

    create_fully_connected_workflow_item( di,
                                      fully_connect_workflow_item,
                                      pooling_workflow_item,
                                      num_output_feature_maps,
                                      filters,
                                      weight_coords,
                                      4,
                                      biases,
                                      bias_coords,
                                      activation_function );

    // 4. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di, output_workflow_item, fully_connect_workflow_item, num_output_feature_maps, 1, 1 );

    // 4. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE i_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
    NN_WORKLOAD_DATA_TYPE o_format = NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &i_format, &o_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    using io_data = std::unique_ptr<nn::data<float>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data(new nn::data<float, 0>(input, input_coords, 4));
    execute_outputs[0] = io_data(new nn::data<float, 0>(gpu_outputs, output_coords, 2));

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs ) );

    //Releasing data
    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( pooling_workflow_item ) );

    delete reinterpret_cast<nn::data<float, 1>*>(fully_connect_workflow_item->arguments.forward_fully_connected.biases);
    delete reinterpret_cast<nn::data<float, 4>*>(fully_connect_workflow_item->arguments.forward_fully_connected.weights);
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( fully_connect_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );

#ifdef __linux__
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free(pooling_cpu_outputs);
    pooling_cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( filters );
    filters = nullptr;
    free( biases );
    biases = nullptr;
    free( input );
    input = nullptr;
#else
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free(pooling_cpu_outputs);
    pooling_cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( filters );
    filters = nullptr;
    _aligned_free( biases );
    biases = nullptr;
    _aligned_free( input );
    input = nullptr;
#endif //__linux__

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_fully_connected_workflow_test(
    const nn_device_interface_0_t &di,
    uint_least32_t                num_output_feature_maps,
    uint_least32_t                num_input_feature_maps,
    uint_least32_t                input_feature_map_width,
    uint_least32_t                input_feature_map_height,
    uint_least32_t                num_batches,
    NN_ACTIVATION_FUNCTION        activation_function)
{
    // Output of FC layer is just bunch of outputs (one per each neuron)
    // so I declare it like that, that depth is equal to number requested
    // output feature maps
    uint_least32_t output_width  = num_output_feature_maps;

#if 0
    // layouts of weights, outputs and inputs of fully connected layers
    nn_workload_data_layout_t all_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        nn::workload_data<float>::layout.xyzpnq, // ordering
        NN_DATATYPE_FLOAT
    };
#endif

    // specify dimensions of input, output and weights
    size_t  input_coords[4] = {input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches};
    size_t output_coords[2] = {num_output_feature_maps, num_batches};
    size_t weight_coords[2] = { input_feature_map_width*input_feature_map_height*num_input_feature_maps, num_output_feature_maps };
    size_t   bias_coords[1] = { num_output_feature_maps };

    // Input generation
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         num_batches );

    // In Fully connected layer , number of input is equal to number of weights (not including bias)
    // Generate Filter Data
    float *filters = nullptr;
    generate_filter_data( filters,
                          input_feature_map_width,
                          input_feature_map_height,
                          num_input_feature_maps,
                          num_output_feature_maps );

    // cpu_outputs and gpu_outputs are to be filled in with biases
    // or they can exists separatly
    float init_output_val = 0.0;        //No biases in output then output is initialized with zeros
    float *biases         = nullptr;
    float *cpu_outputs    = nullptr;
    float *gpu_outputs    = nullptr;

    // Biases exists as separate entity (each neuron got it own bias value)
    init_data( biases, output_width , 1.0f );
    init_data( gpu_outputs, output_width * num_batches, 0.0f );
    init_data( cpu_outputs, output_width * num_batches, 0.0f );

    // Activation function
    fp_func_activ activ_func = nullptr;
    switch( activation_function )
    {
    case NN_ACTIVATION_FUNCTION_NONE:
        activ_func = none;
        break;
    case NN_ACTIVATION_FUNCTION_TANH:
        activ_func = mytanh;
        break;
    case NN_ACTIVATION_FUNCTION_RELU:
        activ_func = relu;
        break;
    case NN_ACTIVATION_FUNCTION_SOFTPLUS:
        activ_func = softplus;
        break;
    default:
        printf( "Error: Not supported activation function chosen: %d\n", activation_function );
        assert( 0 );
        break;
    }

    // Run reference convolving (needed for comparison)
    fully_connect_ref( activ_func,
                       cpu_outputs,
                       input,
                       filters,
                       biases,
                       output_width ,
                       input_feature_map_width,
                       input_feature_map_height,
                       num_input_feature_maps,
                       num_batches );

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di, input_workflow_item, num_input_feature_maps, input_feature_map_width, input_feature_map_height);

    // 2. Fully_connected workflow_item
    nn_workflow_item_t *fully_connect_workflow_item;

    create_fully_connected_workflow_item( di,
                                      fully_connect_workflow_item,
                                      input_workflow_item,
                                      output_width,
                                      filters,
                                      weight_coords,
                                      2,  
                                      biases,
                                      bias_coords,
                                      activation_function );

    //3. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di, output_workflow_item, fully_connect_workflow_item, output_width, 1,
                                 1 );

    // 4. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE i_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
    NN_WORKLOAD_DATA_TYPE o_format = NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &i_format, &o_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    using io_data = std::unique_ptr<nn::data<float>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data(new nn::data<float, 0>(input, input_coords, 4));
    execute_outputs[0] = io_data(new nn::data<float, 0>(gpu_outputs, output_coords, 2));

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    nn_workload_data_coords_t output_view_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t output_view_end(num_batches - 1, output_width - 1, 0, 0, 0, 0);

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs ) );

    //Releasing data
    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );

    // Release bias and weights of workflow item (ask architect if it is meant to be done that way
    delete reinterpret_cast<nn::data<float, 1>*>(fully_connect_workflow_item->arguments.forward_fully_connected.biases);
    delete reinterpret_cast<nn::data<float, 1>*>(fully_connect_workflow_item->arguments.forward_fully_connected.weights);
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( fully_connect_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );

#ifdef __linux__
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( filters );
    filters = nullptr;
    free( biases );
    biases = nullptr;
    free( input );
    input = nullptr;
#else
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( filters );
    filters = nullptr;
    _aligned_free( biases );
    biases = nullptr;
    _aligned_free( input );
    input = nullptr;
#endif //__linux__

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_double_fully_connected_workflow_test(
    const nn_device_interface_0_t &di,
    uint_least32_t                num_second_output_feature_maps,  // num_output_feature_maps (number of neurons in second FF layer)
    NN_ACTIVATION_FUNCTION        second_activation_function,
    uint_least32_t                num_first_output_feature_maps,          // num_output_feature_maps (number of neurons in first  FF layer)
    uint_least32_t                num_input_feature_maps,
    uint_least32_t                input_feature_map_width,
    uint_least32_t                input_feature_map_height,
    uint_least32_t                num_batches,
    NN_ACTIVATION_FUNCTION        first_activation_function)
{
    // Output of FC layer is just bunch of outputs (one per each neuron)
    // so I declare it like that, that depth is equal to number requested
    // output feature maps
    uint_least32_t second_output_width  = num_second_output_feature_maps;
    uint_least32_t first_output_width  = num_first_output_feature_maps;

    // specify dimensions of input, output and weights
    size_t input_coords[4] = {input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches};
    size_t output_coords[2] = { num_second_output_feature_maps, num_batches };
    size_t weight_first_coords[2] = { input_feature_map_width*input_feature_map_height*num_input_feature_maps, num_first_output_feature_maps };
    size_t weight_second_coords[2] = { num_first_output_feature_maps, num_second_output_feature_maps };
    size_t bias_first_coords[1] = { num_first_output_feature_maps };
    size_t bias_second_coords[1] = { num_second_output_feature_maps };

    // Input generation
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         num_batches );

    // In Fully connected layer , number of input is equal to number of weights (not including bias)
    // Generate Filter Data
    float *filters_first = nullptr;
    generate_filter_data( filters_first,
                          input_feature_map_width,
                          input_feature_map_height,
                          num_input_feature_maps,
                          num_first_output_feature_maps);

    float *filters_second = nullptr;
    generate_filter_data( filters_second,
                          num_first_output_feature_maps,        // Second FF's input is just 1D bunch of input values
                          1,        
                          1,
                          num_second_output_feature_maps);

    float *biases_first  = nullptr;
    float *biases_second = nullptr;
    float *cpu_first_outputs   = nullptr;
    float *cpu_second_outputs   = nullptr;
    float *gpu_outputs   = nullptr;

    // Biases exists as separate entity (each neuron got it own bias value)
    init_data( biases_first, num_first_output_feature_maps , 12.0f );
    init_data( biases_second, num_second_output_feature_maps , 13.0f );
    init_data( gpu_outputs, num_second_output_feature_maps * num_batches, 0.0f );
    init_data( cpu_first_outputs, num_first_output_feature_maps * num_batches, 0.0f );
    init_data( cpu_second_outputs, num_second_output_feature_maps * num_batches, 0.0f );

    // Activation function of first Fully connected layer
    fp_func_activ activ_first_func = nullptr;
    switch( first_activation_function )
    {
    case NN_ACTIVATION_FUNCTION_NONE:
        activ_first_func = none;
        break;
    case NN_ACTIVATION_FUNCTION_TANH:
        activ_first_func = mytanh;
        break;
    case NN_ACTIVATION_FUNCTION_RELU:
        activ_first_func = relu;
        break;
    case NN_ACTIVATION_FUNCTION_SOFTPLUS:
        activ_first_func = softplus;
        break;
    default:
        printf( "Error: Not supported activation function chosen: %d\n", first_activation_function );
        assert( 0 );
        break;
    }

    // Run reference first fully_connected feed forward (needed for comparison)
    fully_connect_ref( activ_first_func,
                       cpu_first_outputs,
                       input,
                       filters_first,
                       biases_first,
                       num_first_output_feature_maps,
                       input_feature_map_width,
                       input_feature_map_height,
                       num_input_feature_maps,
                       num_batches );

    // Activation function of second Fully connected layer
    fp_func_activ activ_second_func = nullptr;
    switch( second_activation_function )
    {
    case NN_ACTIVATION_FUNCTION_NONE:
        activ_second_func = none;
        break;
    case NN_ACTIVATION_FUNCTION_TANH:
        activ_second_func = mytanh;
        break;
    case NN_ACTIVATION_FUNCTION_RELU:
        activ_second_func = relu;
        break;
    case NN_ACTIVATION_FUNCTION_SOFTPLUS:
        activ_second_func = softplus;
        break;
    default:
        printf( "Error: Not supported activation function chosen: %d\n", second_activation_function );
        assert( 0 );
        break;
    }

    // Run reference second fully_connected feed forward (needed for comparison)
    fully_connect_ref( activ_second_func,
                       cpu_second_outputs,
                       cpu_first_outputs,
                       filters_second,
                       biases_second,
                       num_second_output_feature_maps,
                       num_first_output_feature_maps,
                       1,
                       1,
                       num_batches );


    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di, input_workflow_item, num_input_feature_maps, input_feature_map_width, input_feature_map_height);

    // 2. Fully_connected workflow_item
    nn_workflow_item_t *fully_connect_first_workflow_item;

    create_fully_connected_workflow_item( di,
                                      fully_connect_first_workflow_item,
                                      input_workflow_item,
                                      num_first_output_feature_maps,
                                      filters_first,
                                      weight_first_coords,
                                      2,  
                                      biases_first,
                                      bias_first_coords,
                                      first_activation_function );

    // 3. Fully_connected workflow_item
    nn_workflow_item_t *fully_connect_second_workflow_item;

    create_fully_connected_workflow_item( di,
                                      fully_connect_second_workflow_item,
                                      fully_connect_first_workflow_item,
                                      num_second_output_feature_maps,
                                      filters_second,
                                      weight_second_coords,
                                      2,  
                                      biases_second,
                                      bias_second_coords,
                                      second_activation_function );

    // 4. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di, output_workflow_item, fully_connect_second_workflow_item, num_second_output_feature_maps, 1, 1 );

    // 4. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE i_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
    NN_WORKLOAD_DATA_TYPE o_format = NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &i_format, &o_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    using io_data = std::unique_ptr<nn::data<float>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data(new nn::data<float, 0>(input, input_coords, 4));
    execute_outputs[0] = io_data(new nn::data<float, 0>(gpu_outputs, output_coords, 2));

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_second_outputs ) );

    //Releasing data
    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );

    // Release bias and weights of workflow item (ask architect if it is meant to be done that way
    delete reinterpret_cast<nn::data<float, 1>*>(fully_connect_first_workflow_item->arguments.forward_fully_connected.biases);
    delete reinterpret_cast<nn::data<float, 2>*>(fully_connect_first_workflow_item->arguments.forward_fully_connected.weights);
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( fully_connect_first_workflow_item ) );

    // Release bias and weights of workflow item (ask architect if it is meant to be done that way
    delete reinterpret_cast<nn::data<float, 1>*>(fully_connect_second_workflow_item->arguments.forward_fully_connected.biases);
    delete reinterpret_cast<nn::data<float, 2>*>(fully_connect_second_workflow_item->arguments.forward_fully_connected.weights);
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( fully_connect_second_workflow_item ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );

#ifdef __linux__
    free( cpu_first_outputs );
    cpu_first_outputs = nullptr;
    free( cpu_second_outputs );
    cpu_second_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( filters_first );
    filters_first = nullptr;
    free( filters_second );
    filters_second = nullptr;
    free( biases_first );
    biases_first = nullptr;
    free( biases_second );
    biases_second = nullptr;
    free( input );
    input = nullptr;
#else
    _aligned_free( cpu_first_outputs );
    cpu_first_outputs = nullptr;
    _aligned_free( cpu_second_outputs );
    cpu_second_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( filters_first );
    filters_first = nullptr;
    _aligned_free( filters_second );
    filters_second = nullptr;
    _aligned_free( biases_first );
    biases_first = nullptr;
    _aligned_free( biases_second );
    biases_second = nullptr;
    _aligned_free( input );
    input = nullptr;
#endif //__linux__

    return true;

}
///////////////////////////////////////////////////////////////////////////////////////////////////
// This function is to run whole process (from creation of workfile through comppilation, to workload execution
bool run_convolve_workflow_test( const nn_device_interface_0_t &di,
    uint_least32_t                num_output_feature_maps,
    uint_least32_t                output_width, 
    uint_least32_t                output_height,
    uint_least32_t                num_input_feature_maps,
    uint_least32_t                input_feature_map_width,
    uint_least32_t                input_feature_map_height,
    uint_least32_t                kernel_width,
    uint_least32_t                kernel_height,
    uint_least32_t                kernel_stride_x,
    uint_least32_t                kernel_stride_y,
    uint_least32_t                num_batches,
    NN_ACTIVATION_FUNCTION        activation_function  )
{
    uint_least32_t output_depth  = num_output_feature_maps;
    uint_least32_t nonpadded_width = ( ( input_feature_map_width - kernel_width ) / kernel_stride_x + 1 );
    uint_least32_t nonpadded_height = ( ( input_feature_map_height - kernel_height ) / kernel_stride_y + 1 );
    uint_least32_t center_x, center_y;
    // Calculate center_x center_y to have padding symetrially located on input area
    if((output_width == 0)&&(output_height == 0)) {
        output_width  = nonpadded_width; 
        output_height = nonpadded_height; 
        center_x = 0;
        center_y = 0;
    } else {
        center_x = (output_width - nonpadded_width)* kernel_stride_x / 2;
        center_y = (output_height - nonpadded_height)*kernel_stride_y/ 2;
    } 

#if 0
    // Specify layout
    nn_workload_data_layout_t all_layout = {
        { 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0 },
        nn::workload_data<float>::layout.xyzpnq,
        NN_DATATYPE_FLOAT
    };
#endif

    size_t   bias_coords[1] = {num_output_feature_maps};
    size_t  input_coords[4] = {input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches};
    size_t output_coords[4] = {output_width, output_height, num_output_feature_maps, num_batches};
    size_t weight_coords[4] = {kernel_width, kernel_height, num_input_feature_maps, num_output_feature_maps};

    // Input generation
    float *input = nullptr;
    generate_input_data( input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
                         num_batches );

    // Generate Filter Data
    float *filters = nullptr;
    generate_filter_data( filters,
                          kernel_width,
                          kernel_height,
                          num_input_feature_maps,
                          num_output_feature_maps );

    float *biases = nullptr;
    init_data( biases, output_width * output_height * output_depth, 100.0f );

    float *cpu_outputs = nullptr;
    float *gpu_outputs = nullptr;
    init_data( gpu_outputs, output_width * output_height * output_depth * num_batches, 0.0f );
    init_data( cpu_outputs, output_width * output_height * output_depth * num_batches, 0.0f );

    // Activation function
    fp_func_activ activ_func = nullptr;
    switch( activation_function )
    {
    case NN_ACTIVATION_FUNCTION_NONE:
        activ_func = none;
        break;
    case NN_ACTIVATION_FUNCTION_TANH:
        activ_func = mytanh;
        break;
    case NN_ACTIVATION_FUNCTION_RELU:
        activ_func = relu;
        break;
    case NN_ACTIVATION_FUNCTION_SOFTPLUS:
        activ_func = softplus;
        break;
    default:
        printf( "Error: Not supported activation function chosen: %d\n", activation_function );
        assert( 0 );
        break;
    }

    nn_workload_data_coords_t conv_input_view_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t conv_input_view_end( num_batches - 1, input_feature_map_width - 1, input_feature_map_height - 1, num_input_feature_maps - 1, 0, 0 );
    nn_workload_data_coords_t conv_output_view_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t conv_output_view_end( num_batches - 1, output_width - 1, output_height - 1, output_depth - 1, 0, 0 );

    // Run reference convolving (needed for comparison)
    convolve_ref( activ_func,
                  cpu_outputs,
                  input,
                  filters,
                  biases,
                  conv_output_view_begin,
                  conv_output_view_end,
                  conv_input_view_begin,
                  conv_input_view_end,
                  output_width,
                  output_height,
                  output_depth,
                  input_feature_map_width,
                  input_feature_map_height,
                  num_input_feature_maps,
                  kernel_width,
                  kernel_height,
                  num_input_feature_maps,
                  kernel_stride_x,
                  kernel_stride_y,
                  center_x,        // center offset x
                  center_y,        // center offset y
                  num_batches );

    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item( di,
                                input_workflow_item,
                                num_input_feature_maps,
                                input_feature_map_width,
                                input_feature_map_height );

    // 2. Convolution workflow_item
    nn_workflow_item_t *conv_workflow_item;

    create_convolution_workflow_item( di,
                                      conv_workflow_item,
                                      input_workflow_item,
                                      output_width,
                                      output_height,
                                      output_depth,
                                      filters,
                                      weight_coords,
                                      biases,
                                      bias_coords,
                                      kernel_stride_x,
                                      kernel_stride_y,
                                      center_x,        // center offset x
                                      center_y,        // center offset y
                                      activation_function );

    //3. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item( di, output_workflow_item, conv_workflow_item, output_width, output_height,
                                 output_depth );

    // 4. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_create_function( &test_workflow, 1, 1 ) );
    test_workflow->input[0]  = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
    EXPECT_EQ( NN_API_STATUS_OK,
               di.workflow_compile_function( &workload, di.device, test_workflow, &io_format, &io_format,
                                             num_batches ) );

    // Here we Execute compiled workload
    using io_data = std::unique_ptr<nn::data<float, 0>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data(new nn::data<float, 0>(input, input_coords, 4));
    execute_outputs[0] = io_data(new nn::data<float, 0>(gpu_outputs, output_coords, 4));

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( workload,
                                                               ( void ** )execute_inputs,
                                                               ( void ** )execute_outputs, nullptr ) );

    nn_workload_data_coords_t output_view_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t output_view_end(num_batches - 1, output_width - 1, output_height - 1, output_depth - 1, 0, 0);

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs ) );

    //Releasing data
    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function( workload ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_delete_function( test_workflow ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( input_workflow_item ) );
    // Release bias and weights of workflow item (ask architect if it is meant to be done that way
    delete reinterpret_cast<nn::data<float, 1>*>(conv_workflow_item->arguments.forward_convolution.biases);
    delete reinterpret_cast<nn::data<float, 1>*>(conv_workflow_item->arguments.forward_convolution.weights);
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( conv_workflow_item ) );
    EXPECT_EQ( NN_API_STATUS_OK, di.workflow_item_delete_function( output_workflow_item ) );

#ifdef __linux__
    free( cpu_outputs );
    cpu_outputs = nullptr;
    free( gpu_outputs );
    gpu_outputs = nullptr;
    free( filters );
    filters = nullptr;
    free( biases );
    biases = nullptr;
    free( input );
    input = nullptr;
#else
    _aligned_free( cpu_outputs );
    cpu_outputs = nullptr;
    _aligned_free( gpu_outputs );
    gpu_outputs = nullptr;
    _aligned_free( filters );
    filters = nullptr;
    _aligned_free( biases );
    biases = nullptr;
    _aligned_free( input );
    input = nullptr;
#endif //__linux__

    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// This function is to run whole process (from creation of workfile through comppilation, to workload execution
bool run_merged_convolve_maxpool_workflow_test(const nn_device_interface_0_t &di,
    uint_least32_t                num_output_feature_maps,
    uint_least32_t                output_width, 
    uint_least32_t                output_height,
    uint_least32_t                num_input_feature_maps,
    uint_least32_t                input_feature_map_width,
    uint_least32_t                input_feature_map_height,
    uint_least32_t                kernel_width,
    uint_least32_t                kernel_height,
    uint_least32_t                kernel_stride_x,
    uint_least32_t                kernel_stride_y,
    uint_least32_t                pool_stride_x,
    uint_least32_t                pool_stride_y,
    uint_least32_t                pool_size_x,
    uint_least32_t                pool_size_y,
    uint_least32_t                num_batches,
    NN_ACTIVATION_FUNCTION        activation_function)
{
    uint_least32_t conv_output_width;
    uint_least32_t conv_output_height;
    uint_least32_t pool_output_width; 
    uint_least32_t pool_output_height;
    uint_least32_t output_depth = num_output_feature_maps;

    // calculate how sizes of nopadded convolution
    uint_least32_t nonpadded_conv_width = ( ( input_feature_map_width - kernel_width ) / kernel_stride_x + 1 );
    uint_least32_t nonpadded_conv_height = ( ( input_feature_map_height - kernel_height ) / kernel_stride_y + 1 );
    uint_least32_t center_x, center_y;
    // Calculate center_x center_y to have padding symetrially located on input area
    if((output_width == 0)&&(output_height == 0)) {
        conv_output_width = nonpadded_conv_width; 
        conv_output_height = nonpadded_conv_height;

        pool_output_width = ((nonpadded_conv_width - pool_size_x) / pool_stride_x + 1);
        pool_output_height = ((nonpadded_conv_height - pool_size_y) / pool_stride_y + 1);
        center_x = 0;
        center_y = 0;
    } else {
        pool_output_width = output_width; 
        pool_output_height = output_height;
        conv_output_width = (pool_output_width - 1)*pool_stride_x + pool_size_x;
        conv_output_height = (pool_output_height - 1)*pool_stride_y + pool_size_y;

        center_x = (conv_output_width- nonpadded_conv_width)* kernel_stride_x / 2;
        center_y = (conv_output_height - nonpadded_conv_height)*kernel_stride_y/ 2;
    } 

    // Specify layout
    nn_workload_data_layout_t all_layout = nn::workload_data<float>::layout.xyzpnq;

    // Input generation
    float *input = nullptr;
    generate_input_data(input, input_feature_map_width, input_feature_map_height, num_input_feature_maps,
        num_batches);

    // Generate Filter Data
    float *filters = nullptr;
    generate_filter_data(filters,
        kernel_width,
        kernel_height,
        num_input_feature_maps,
        num_output_feature_maps);

    float *biases = nullptr;
    init_data(biases, conv_output_width * conv_output_height * output_depth, 1.0f);

    float *conv_cpu_outputs = nullptr;
    float *pool_cpu_outputs = nullptr;
    float *gpu_outputs = nullptr;
    init_data(gpu_outputs, pool_output_width * pool_output_height * output_depth * num_batches, 0.0f);
    init_data(conv_cpu_outputs, conv_output_width * conv_output_height * output_depth * num_batches, 0.0f);
    init_data(pool_cpu_outputs, pool_output_width * pool_output_height * output_depth * num_batches, 0.0f);

    // Activation function
    fp_func_activ activ_func = nullptr;
    switch (activation_function)
    {
    case NN_ACTIVATION_FUNCTION_NONE:
        activ_func = none;
        break;
    case NN_ACTIVATION_FUNCTION_TANH:
        activ_func = mytanh;
        break;
    case NN_ACTIVATION_FUNCTION_RELU:
        activ_func = relu;
        break;
    case NN_ACTIVATION_FUNCTION_SOFTPLUS:
        activ_func = softplus;
        break;
    default:
        printf("Error: Not supported activation function chosen: %d\n", activation_function);
        assert(0);
        break;
    }

    nn_workload_data_coords_t conv_input_view_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t conv_input_view_end( num_batches - 1, input_feature_map_width - 1, input_feature_map_height - 1, num_input_feature_maps - 1, 0, 0 );
    nn_workload_data_coords_t conv_output_view_begin( 0, 0, 0, 0, 0, 0 );
    nn_workload_data_coords_t conv_output_view_end( num_batches - 1, conv_output_width - 1, conv_output_height - 1, output_depth - 1, 0, 0 );

    // Run reference convolving (needed for comparison)
    convolve_ref(activ_func,
        conv_cpu_outputs,
        input,
        filters,
        biases,
        conv_output_view_begin,
        conv_output_view_end,
        conv_input_view_begin,
        conv_input_view_end,
        conv_output_width,
        conv_output_height,
        output_depth,
        input_feature_map_width,
        input_feature_map_height,
        num_input_feature_maps,
        kernel_width,
        kernel_height,
        num_input_feature_maps,
        kernel_stride_x,
        kernel_stride_y,
        center_x,        // center offset x
        center_y,        // center offset y
        num_batches);
    
    // now put the data through max pooling layer

    //This is view needed for 
    nn_workload_data_coords_t input_view_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t output_view_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t input_view_end(num_batches - 1, conv_output_width-1, conv_output_height-1, num_input_feature_maps-1, 0, 0);
    nn_workload_data_coords_t output_view_end(num_batches - 1, pool_output_width - 1, pool_output_height - 1, output_depth - 1, 0, 0);

    pool_ref(pool_cpu_outputs,                // outputs,
        conv_cpu_outputs,                     // inputs,
        output_view_begin,
        output_view_end,
        input_view_begin,
        input_view_end,
        pool_output_width,                    // outputs_width,
        pool_output_height,                   // outputs_height,
        output_depth,                         // outputs_depth,
        conv_output_width,                    // inputs_width,
        conv_output_height,                   // inputs_height,
        output_depth,                         // inputs_depth,
        pool_size_x,                          // window_size,
        pool_stride_x,                        // stride_x,
        pool_stride_y,                        // stride_y,
        num_batches);                         // num_batches 


    // 1. Input workflow_item
    nn_workflow_item_t *input_workflow_item;

    create_input_workflow_item(di,
        input_workflow_item,
        num_input_feature_maps,
        input_feature_map_width,
        input_feature_map_height);

    // 2. Convolution workflow_item
    nn_workflow_item_t *convmaxpool_workflow_item;

    size_t weight_size[4] = { kernel_width, kernel_height, num_input_feature_maps, num_output_feature_maps, };
    size_t bias_size[1] = { num_output_feature_maps };



    create_merged_convolution_maxpool_workflow_item(di,
        convmaxpool_workflow_item,
        input_workflow_item,
        pool_output_width,
        pool_output_height,
        output_depth,
        filters,
        weight_size,
        biases,
        bias_size,
        kernel_stride_x,
        kernel_stride_y,
        pool_stride_x,                  // pool_stride_x,
        pool_stride_y,                  // pool_stride_y,
        pool_size_x,                    // pool_size_x,
        pool_size_y,                    // pool_size_y   
        center_x,                       // center offset x
        center_y,                       // center offset y
        activation_function);

    //3. Output workflow_item
    nn_workflow_item_t *output_workflow_item;

    create_output_workflow_item(di, output_workflow_item, convmaxpool_workflow_item, pool_output_width, pool_output_height,
        output_depth);

    // 4. Workflow itself
    nn_workflow *test_workflow;
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&test_workflow, 1, 1));
    test_workflow->input[0] = input_workflow_item;
    test_workflow->output[0] = output_workflow_item;

    nn_workload  *workload = nullptr;
    NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH;
    EXPECT_EQ(NN_API_STATUS_OK,
        di.workflow_compile_function(&workload, di.device, test_workflow, &io_format, &io_format,
        num_batches));

    size_t pool_output_size[4] = { pool_output_width, pool_output_height, num_output_feature_maps, num_batches };
    size_t pool_input_size[4] = { input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches };



    // Here we Execute compiled workload
    using io_data = std::unique_ptr<nn::data<float>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data(new nn::data<float>(input, pool_input_size, 4));
    execute_outputs[0] = io_data(new nn::data<float>(gpu_outputs, pool_output_size, 4));

    EXPECT_EQ(NN_API_STATUS_OK, di.workload_execute_function(workload,
        (void **)execute_inputs,
        (void **)execute_outputs, nullptr));

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], pool_cpu_outputs ) );

    //Releasing data
    EXPECT_EQ(NN_API_STATUS_OK, di.workload_delete_function(workload));

    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(test_workflow));

    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input_workflow_item));
    // Release bias and weights of workflow item (ask architect if it is meant to be done that way
    delete convmaxpool_workflow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.biases;
    delete convmaxpool_workflow_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2.weights;
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(convmaxpool_workflow_item));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(output_workflow_item));

#ifdef __linux__
    free(conv_cpu_outputs);
    conv_cpu_outputs = nullptr;
    free(pool_cpu_outputs);
    pool_cpu_outputs = nullptr;
    free(gpu_outputs);
    gpu_outputs = nullptr;
    free(filters);
    filters = nullptr;
    free(biases);
    biases = nullptr;
    free(input);
    input = nullptr;
#else
    _aligned_free(conv_cpu_outputs);
    conv_cpu_outputs = nullptr;
    _aligned_free(pool_cpu_outputs);
    pool_cpu_outputs = nullptr;
    _aligned_free(gpu_outputs);
    gpu_outputs = nullptr;
    _aligned_free(filters);
    filters = nullptr;
    _aligned_free(biases);
    biases = nullptr;
    _aligned_free(input);
    input = nullptr;
#endif //__linux__

    return true;
}

TEST(gpu_device_workflow_interface_0, multi_pass_interface_test)
{
    for (uint32_t iteration = 0; iteration < 10; ++iteration)
    {
        // 1. Initialize and check if interface 0 is supported
        nn_device_description_t dd;

        EXPECT_EQ(0, nn_device_load(&dd));              // non-zero return code on valid call
        EXPECT_EQ(0, dd.version_first);                 // First supported interface version should be 0

        // 2. Get interface 0
        const uint16_t          interface_version = 0;
        nn_device_interface_0_t di;
        if ((interface_version >= dd.version_first) && (interface_version <= dd.version_last))
        {
            EXPECT_EQ(0, nn_device_interface_open(interface_version, &di));
            EXPECT_EQ(interface_version, di.version);           // returned version matches requested
            EXPECT_NE(nullptr, di.device);                      // non-null device returned
            EXPECT_NE(nullptr, di.workflow_item_validate_function); // non-null function pointer returned
            EXPECT_NE(nullptr, di.workflow_item_delete_function);   // non-null function pointer returned
            EXPECT_NE(nullptr, di.parameter_set_function);      // non-null function pointer returned
            EXPECT_NE(nullptr, di.parameter_get_function);      // non-null function pointer returned
            //assert(0);//TODO: Add other interface functions  just above
        }
        EXPECT_EQ(true, run_convolve_workflow_test(di, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, NN_ACTIVATION_FUNCTION_NONE));

        EXPECT_EQ(0, nn_device_interface_close(&di));       // successful close of interface
        //// nn_device_unload
        EXPECT_EQ(0, nn_device_unload()); // successful unload
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
TEST( gpu_device_workflow_interface_0, full_api_usage_test )
{

    // 1. Initialize and check if interface 0 is supported
    nn_device_description_t dd;

    EXPECT_EQ( 0, nn_device_load( &dd ) );              // non-zero return code on valid call
    EXPECT_EQ( 0, dd.version_first );                 // First supported interface version should be 0

    // 2. Get interface 0
    const uint16_t          interface_version = 0;
    nn_device_interface_0_t di;
    if( ( interface_version >= dd.version_first ) && ( interface_version <= dd.version_last ) )
    {
        EXPECT_EQ( 0, nn_device_interface_open( interface_version, &di) );
        EXPECT_EQ( interface_version, di.version );           // returned version matches requested
        EXPECT_NE( nullptr, di.device );                      // non-null device returned
        EXPECT_NE( nullptr, di.workflow_item_validate_function ); // non-null function pointer returned
        EXPECT_NE( nullptr, di.workflow_item_delete_function );   // non-null function pointer returned
        EXPECT_NE( nullptr, di.parameter_set_function );      // non-null function pointer returned
        EXPECT_NE( nullptr, di.parameter_get_function );      // non-null function pointer returned
    //assert(0);//TODO: Add other interface functions  just above
    }

#if 0 // those tests are diabled because currently there's no container passed to execute function that supports views
    EXPECT_EQ( true, run_views(di) );
    EXPECT_EQ( true, run_pooling_view_workflow_test(di) );
#endif

    EXPECT_EQ( true, run_convolve_workflow_test(di,1,0,0,1,1,1,1,1,1,1,2,NN_ACTIVATION_FUNCTION_NONE) );
    //EXPECT_EQ( true, run_convolve_workflow_test(di,512,12,12,256,12,12,3,3,1,1,1,NN_ACTIVATION_FUNCTION_RELU) );
    EXPECT_EQ( true, run_convolve_workflow_test( di, 10, 0,0, 23, 100, 80, 15, 15, 2, 3, 1, NN_ACTIVATION_FUNCTION_SOFTPLUS) );
    EXPECT_EQ( true, run_convolve_workflow_test( di, 96, 0,0, 3, 224, 224, 11, 11, 4, 4, 1, NN_ACTIVATION_FUNCTION_RELU)); //C1 of AlexK architecture
    EXPECT_EQ( true, run_convolve_workflow_test( di, 256, 0,0, 48, 61, 61, 5, 5, 1, 1, 1, NN_ACTIVATION_FUNCTION_RELU));   //C2 of AlexK architecture

    EXPECT_EQ( true, run_convolve_workflow_test( di, 10, 0,0, 1, 1, 1, 1, 1, 1, 1, 32, NN_ACTIVATION_FUNCTION_TANH));

    const auto groups = 2;

    printf("---- AlexK C1 (GPU Efficiency) --->\n");
    EXPECT_EQ(true, run_convolve_workflow_test(di, 96,0,0, 3, 227, 227, 11, 11, 4, 4, 1, NN_ACTIVATION_FUNCTION_NONE));
    printf("<--- AlexK C1 (GPU Efficiency) ---\n");
    
    printf("---- AlexK C2 (GPU Efficiency) --->\n");
    EXPECT_EQ( true, run_convolve_workflow_test( di, 256 / groups, 0, 0, 96 / groups, 31, 31, 5, 5, 1, 1,1, NN_ACTIVATION_FUNCTION_NONE ) );
    printf("<--- AlexK C2 (GPU Efficiency) ---\n");

    printf("---- AlexK C3 (GPU Efficiency) --->\n");
    EXPECT_EQ(true, run_convolve_workflow_test(di, 384, 0, 0, 256, 15, 15, 3, 3, 1, 1, 1, NN_ACTIVATION_FUNCTION_NONE));
    printf("<--- AlexK C3 (GPU Efficiency) ---\n");

    printf("---- AlexK C4 (GPU Efficiency) --->\n");
    EXPECT_EQ( true, run_convolve_workflow_test( di, 384 / groups, 0, 0, 384 / groups, 15, 15, 3, 3, 1, 1, 1, NN_ACTIVATION_FUNCTION_NONE ) );
    printf("<--- AlexK C4 (GPU Efficiency) ---\n");
    
    printf("---- AlexK C5 (GPU Efficiency) --->\n");
    EXPECT_EQ( true, run_convolve_workflow_test( di, 256 / groups, 0, 0, 384 / groups, 15, 15, 3, 3, 1, 1, 1, NN_ACTIVATION_FUNCTION_NONE ) );
    printf("<--- AlexK C5 (GPU Efficiency) ---\n");
  
    printf( "---- AlexK C1 (GPU Efficiency) batch --->\n" );
    EXPECT_EQ( true, run_convolve_workflow_test( di, 96, 0, 0, 3, 227, 227, 11, 11, 4, 4, 2, NN_ACTIVATION_FUNCTION_RELU ) );
    printf( "<--- AlexK C1 (GPU Efficiency) ---\n" );
  
    printf( "---- AlexK C2 (GPU Efficiency) batch --->\n" );
    EXPECT_EQ( true, run_convolve_workflow_test( di, 256 / groups, 0, 0, 96 / groups, 31, 31, 5, 5, 1, 1, 2, NN_ACTIVATION_FUNCTION_RELU ) );
    printf( "<--- AlexK C2 (GPU Efficiency) ---\n" );
    
    printf( "---- AlexK C3 (GPU Efficiency) batch --->\n" );
    EXPECT_EQ( true, run_convolve_workflow_test( di, 384, 0, 0, 256, 15, 15, 3, 3, 1, 1, 2, NN_ACTIVATION_FUNCTION_RELU ) );
    printf( "<--- AlexK C3 (GPU Efficiency) ---\n" );
    
    printf( "---- AlexK C4 (GPU Efficiency) batch --->\n" );
    EXPECT_EQ( true, run_convolve_workflow_test( di, 384 / groups, 0, 0, 384 / groups, 15, 15, 3, 3, 1, 1, 2, NN_ACTIVATION_FUNCTION_RELU ) );
    printf( "<--- AlexK C4 (GPU Efficiency) ---\n" );
    
    printf( "---- AlexK C5 (GPU Efficiency) batch --->\n" );
    EXPECT_EQ( true, run_convolve_workflow_test( di, 256 / groups, 0, 0, 384 / groups, 15, 15, 3, 3, 1, 1, 2, NN_ACTIVATION_FUNCTION_RELU ) );
    printf( "<--- AlexK C5 (GPU Efficiency) ---\n" );

    printf("---- OverFeat C1 (GPU Efficiency) --->\n");
    EXPECT_EQ(true, run_convolve_workflow_test(di, 96,0,0, 3, 231, 231, 11, 11, 4, 4, 3, NN_ACTIVATION_FUNCTION_NONE)); //C1 of OverFeat architecture
    printf("<--- OverFeat C1 (GPU Efficiency) ---\n");

    printf("---- OverFeat C2 (GPU Efficiency) --->\n");
    EXPECT_EQ(true, run_convolve_workflow_test(di, 256,0,0, 96, 28, 28, 5, 5, 1, 1, 3, NN_ACTIVATION_FUNCTION_NONE));   //C2 of OverFeat architecture
    printf("<--- OverFeat C2 (GPU Efficiency) ---\n");

    printf("---- OverFeat C3 (GPU Efficiency) --->\n");
    EXPECT_EQ(true, run_convolve_workflow_test(di, 512,0,0, 256, 14, 14, 3, 3, 1, 1, 3, NN_ACTIVATION_FUNCTION_NONE));   //C3 of OverFeat architecture
    printf("<--- OverFeat C3 (GPU Efficiency) ---\n");

    printf("---- OverFeat C4 (GPU Efficiency) --->\n");
    EXPECT_EQ(true, run_convolve_workflow_test(di, 1024,0,0, 512, 14, 14, 3, 3, 1, 1, 3, NN_ACTIVATION_FUNCTION_NONE));   //C4 of OverFeat architecture
    printf("<--- OverFeat C4 (GPU Efficiency) ---\n");

    printf("---- OverFeat C5 (GPU Efficiency) --->\n");
    EXPECT_EQ(true, run_convolve_workflow_test(di, 1024,0,0, 1024, 14, 14, 3, 3, 1, 1, 3, NN_ACTIVATION_FUNCTION_NONE));  //C5 of OverFeat architecture
    printf("<--- OverFeat C5 (GPU Efficiency) ---\n");
    
    //TODO: Enable this layer
    //printf("---- OverFeat C6 (GPU Efficiency) --->\n");
    //EXPECT_EQ(true, run_convolve_workflow_test(di, 3072, 0,0, 1024, 6, 6, 6, 6, 1, 1, 3, NN_ACTIVATION_FUNCTION_NONE));    //C6 of OverFeat architecture
    //printf("<--- OverFeat C6 (GPU Efficiency) ---\n");

    // Normalized area is divided (along Z) on two halves and then separate convolutions are made on those halves , then merged layer is invoked to gather results into one buffer
    EXPECT_EQ( true, run_normalization_convolution_split_test(di));

    EXPECT_EQ( true, run_arithmetic_workflow_test( di, 3, 256, 256, 32, NN_ARITHMETIC_FUNCTION_SUBTRACTION )); 

    EXPECT_EQ( true, run_normalization_view_convolution_test(di));

    EXPECT_EQ( true, run_pooling_workflow_test( di,   // interface

                                       1,    // num_input_feature_maps,
                                       2,    // input_feature_map_width,
                                       2,    // input_feature_map_height,
                                       2,    // window_size, pooling window is of size n x n , window_size is  equal to this n
                                       2,    // stride in x dimension (should be at least equal to window_size to have non overlapping mode)
                                       2,    // stride in y dimension (should be at least equal to window_size to have non overlapping mode)
                                       2 ) );// num_batches

    EXPECT_EQ( true, run_pooling_workflow_test( di,  // interface
                                       1,   // num_input_feature_maps,
                                       224,   // input_feature_map_width,
                                       224,   // input_feature_map_height,
                                       3,   // window_size, pooling window is of size n x n , window_size is  equal to this n
                                       2,   // stride in x dimension (should be at least equal to window_size to have non overlapping mode)
                                       2,   // stride in y dimension (should be at least equal to window_size to have non overlapping mode)
                                       1) );// num_batches

    EXPECT_EQ( true, run_pooling_workflow_test( di,  // interface
                                       96,   // num_input_feature_maps,
                                       27,   // input_feature_map_width,
                                       27,   // input_feature_map_height,
                                       3,   // window_size, pooling window is of size n x n , window_size is  equal to this n
                                       2,   // stride in x dimension (should be at least equal to window_size to have non overlapping mode)
                                       2,   // stride in y dimension (should be at least equal to window_size to have non overlapping mode)
                                       32) );// num_batches

    EXPECT_EQ( true, run_pooling_fully_connected_workflow_test( di,  // interface
                                       256,                             // num_output_feature_maps (number of neurons in FC layer)
                                       NN_ACTIVATION_FUNCTION_RELU,     //(Activation function to be used for FC layer)
                                       256,  // num_input_feature_maps,  (entry for pooling)
                                       13,   // input_feature_map_width, (entry for pooling)
                                       13,   // input_feature_map_height, (entry for pooling)
                                       3,   // window_size, pooling window is of size n x n , window_size is  equal to this n
                                       2,   // stride in x dimension (should be at least equal to window_size to have non overlapping mode)
                                       2,   // stride in y dimension (should be at least equal to window_size to have non overlapping mode)
                                       32) );// num_batches

    EXPECT_EQ( true, run_fully_connected_workflow_test( di,         // interface
                                                         2,         // num_output_feature_maps (number of neurons in FC layer)
                                                         1,         // num_input_feature_maps,
                                                         1,         // input_feature_map_width,
                                                         1,         // input_feature_map_height,
                                                         1,         // num_batches
                                                         NN_ACTIVATION_FUNCTION_NONE) );     // Activation function
    // CaffeNet fc7->fc8 (2nd and 3rd fully connected layers)
    EXPECT_EQ( true, run_double_fully_connected_workflow_test( di,         // interface
                                                        1000,       // num_output_feature_maps (number of neurons in second FF layer)
                                                        NN_ACTIVATION_FUNCTION_NONE,    // Activation function of second FF layer
                                                        4096,       // num_output_feature_maps (number of neurons in first  FF layer)
                                                        4096,       // num_input_feature_maps,
                                                        1,          // input_feature_map_width,
                                                        1,          // input_feature_map_height,
                                                        32,         // num_batches
                                                        NN_ACTIVATION_FUNCTION_RELU));     // Activation function of first FF layer

    // CaffeNet fc6 (1st fully connected)
    EXPECT_EQ( true, run_fully_connected_workflow_test( di,         // interface
                                                        4096,       // num_output_feature_maps (number of neurons in FC layer)
                                                        256,       // num_input_feature_maps,
                                                        6,          // input_feature_map_width,
                                                        6,          // input_feature_map_height,
                                                        1,         // num_batches
                                                        NN_ACTIVATION_FUNCTION_RELU));     // Activation function
    // CaffeNet fc7 (2nd fully connected)
    EXPECT_EQ( true, run_fully_connected_workflow_test( di,         // interface
                                                        4096,       // num_output_feature_maps (number of neurons in FC layer)
                                                        4096,       // num_input_feature_maps,
                                                        1,          // input_feature_map_width,
                                                        1,          // input_feature_map_height,
                                                        1,         // num_batches
                                                        NN_ACTIVATION_FUNCTION_RELU));     // Activation function
    
    // CaffeNet fc8 (3rd fully connected)
    EXPECT_EQ( true, run_fully_connected_workflow_test( di,         // interface
                                                        1000,       // num_output_feature_maps (number of neurons in FC layer)
                                                        4096,       // num_input_feature_maps,
                                                        1,          // input_feature_map_width,
                                                        1,          // input_feature_map_height,
                                                        1,         // num_batches
                                                        NN_ACTIVATION_FUNCTION_RELU));     // Activation function

    // Overfeat fc6
    EXPECT_EQ( true, run_fully_connected_workflow_test( di,          // interface
                                                        3072,        // num_output_feature_maps (number of neurons in FC layer)
                                                        1024,        // num_input_feature_maps,
                                                        6,           // input_feature_map_width,
                                                        6,           // input_feature_map_height,
                                                        1,           // num_batches
                                                        NN_ACTIVATION_FUNCTION_NONE) );     // Activation function
    // Overfeat F7 layer
    EXPECT_EQ( true, run_fully_connected_workflow_test( di,        // interface
                                                        4096,      // num_output_feature_maps (number of neurons in FC layer)
                                                        1,         // num_input_feature_maps,
                                                        64,        // input_feature_map_width,
                                                        48,        // input_feature_map_height,
                                                        1,         // num_batches
                                                        NN_ACTIVATION_FUNCTION_RELU) );     // Activation function
    EXPECT_EQ( true, run_softmax_workflow_test( di,   // interface
                                       10,  // length of input to be  processed (softmax normalize)
                                       3) );

    EXPECT_EQ( true, run_softmax_workflow_test( di,   // interface
                                       20,  // length of input to be  processed (softmax normalize)
                                       3) );

    // ImageNet final layers (overfeat, AlexK anything to have outputs normalized for 1000 classes)
    EXPECT_EQ( true, run_softmax_workflow_test( di,  // interface
                                       1000, // length of input to be  processed (softmax normalize)
                                       2
                                       ) );

    //Local Reponse normalization test along with AlexK topology LRN params
    EXPECT_EQ( true, run_normalization_workflow_test( di,   // interface
                                             2,           // num batches
                                             30,       // num_input_feature_maps,
                                             224,       // input_feature_map_width,
                                             224,       // input_feature_map_height,
                                             5,       // normalization area size is of size n (normalization for feature map i goes from i - n/2 to i + n/2) //5 
                                             2,       // hyper parameter : k    //2
                                             0.0001f,    // hyper parameter : Alpha  //0.0001f 
                                             0.75f,   // hyper parameter : Beta     //0.75 
                                             NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS ) );

    EXPECT_EQ(true, run_normalization_workflow_test(di,       //interface
                                           2,       // num batches
                                           30,      //num_input_feature_maps,
                                           224,     // input_feature_map_width,
                                           224,     // input_feature_map_height,
                                           0,       // normalization area size - not used for NN_NORMALIZATION_MODE_LINEAR_SINGLE
                                           0,       // hyper parameter : k - not used for NN_NORMALIZATION_MODE_LINEAR_SINGLE
                                           0.0001f, // coefficient a 
                                           0.75f,   // coefficient b
                                           NN_NORMALIZATION_MODE_LINEAR_SINGLE));

    EXPECT_EQ(true, run_normalization_workflow_test(di,       // interface
                                           2,        // num batches
                                           3,        // num_input_feature_maps,
                                           200,      // input_feature_map_width,
                                           200,      // input_feature_map_height,
                                           0,        // normalization area size - not used for NN_NORMALIZATION_MODE_LINEAR_SINGLE
                                           0,        // hyper parameter : k - not used for NN_NORMALIZATION_MODE_LINEAR_SINGLE
                                           1000.99f, // coefficient a 
                                           15000.00f,  // coefficient b
                                           NN_NORMALIZATION_MODE_LINEAR_SINGLE));

    EXPECT_EQ(true, run_merged_convolve_maxpool_workflow_test(
        di,     // interface
        96,     // num_output_feature_maps, <1024>
        28,      // output width (0 means that output is to be calculated based on other params: input and kernel
        28,      // output height
        3,      // num_input_feature_maps, <1024>
        231,    // input_feature_map_width,
        231,    // input_feature_map_height,
        11,     // kernel_width,
        11,     // kernel_height,
        4,      // kernel_stride_x,
        4,      // kernel_stride_y,
        2,      // pool_stride_x,
        2,      // pool_stride_y,
        2,      // pool_size_x,
        2,      // pool_size_y,
        1,      // num_batches,
        NN_ACTIVATION_FUNCTION_RELU      // activation_function
        ));

    EXPECT_EQ(true, run_merged_convolve_maxpool_workflow_test(
        di,     // interface
        1024,     // num_output_feature_maps, <1024>
        0,      // output width (0 means that output is to be calculated based on other params: input and kernel
        0,      // output height
        1024,      // num_input_feature_maps, <1024>
        12,    // input_feature_map_width,
        12,    // input_feature_map_height,
        3,     // kernel_width,
        3,     // kernel_height,
        1,      // kernel_stride_x,
        1,      // kernel_stride_y,
        2,      // pool_stride_x,
        2,      // pool_stride_y,
        2,      // pool_size_x,
        2,      // pool_size_y,
        1,      // num_batches,
        NN_ACTIVATION_FUNCTION_RELU      // activation_function
        ));


    EXPECT_EQ(true, run_merged_convolve_maxpool_workflow_test(
        di,     // interface
        6,     // num_output_feature_maps,
        0,      // output width (0 means that output is to be calculated based on other params: input and kernel
        0,      // output height
        1,      // num_input_feature_maps,
        224,    // input_feature_map_width,
        224,    // input_feature_map_height,
        11,     // kernel_width,
        11,     // kernel_height,
        4,      // kernel_stride_x,
        4,      // kernel_stride_y,
        2,      // pool_stride_x,
        2,      // pool_stride_y,
        2,      // pool_size_x,
        2,      // pool_size_y,
        3,      // num_batches,
        NN_ACTIVATION_FUNCTION_RELU      // activation_function
        ));

    EXPECT_EQ(true, run_merged_convolve_maxpool_workflow_test(
        di,     // interface
        256,    // num_output_feature_maps,
        0,      // output width (0 means that output is to be calculated based on other params: input and kernel
        0,      // output height
        48,     // num_input_feature_maps,
        61,     // input_feature_map_width,
        61,     // input_feature_map_height,
        5,      // kernel_width,
        5,      // kernel_height,
        1,      // kernel_stride_x,
        1,      // kernel_stride_y,
        2,      // pool_stride_x,
        2,      // pool_stride_y,
        2,      // pool_size_x,
        2,      // pool_size_y,
        3,      // num_batches,
        NN_ACTIVATION_FUNCTION_RELU      // activation_function
        ));
    EXPECT_EQ( 0, nn_device_interface_close( &di ) );       // successful close of interface
    //// nn_device_unload
    EXPECT_EQ( 0, nn_device_unload() ); // successful unload
}


