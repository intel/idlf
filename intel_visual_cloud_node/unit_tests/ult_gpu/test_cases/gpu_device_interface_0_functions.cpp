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


#include "../../devices/api/nn_device_api.h"
#include "../../devices/common/nn_workload_data_cpp_wrapper.h"
#include "../../devices/api/nn_device_interface_0.h"
#include "../../devices/device_gpu/api_internal/nn_device_interface_0_functions.h"
#include "../../devices/device_gpu/api_internal/nn_device_interface_0_internal.h"
#include "../common.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
bool initialize_input_workload_item( nn_gpu_workload_item * &work_item )
{
    work_item                        = new nn_gpu_workload_item();
    work_item->type                  = NN_WORK_ITEM_TYPE_INPUT;
    work_item->arguments.input.index = 0; //???
    work_item->output                = nullptr;

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool initialize_layer_workload_item( nn_gpu_workload_item * &layer_workload_item,
                                     nn_gpu_workload_item * &input_workload_item,
                                     nn_workload_data_layout_t       &output_layout,
                                     nn_workload_data_coords_t       &output_coords)
{
    layer_workload_item              = new nn_gpu_workload_item();
    layer_workload_item->input.push_back(input_workload_item);
    layer_workload_item->output = new nn::nn_workload_data_t< float >( output_coords, output_layout );
    layer_workload_item->output_view.reset( new nn::nn_workload_data_t< float >( *reinterpret_cast< nn::nn_workload_data_t< float > * >( layer_workload_item->output ),
                                                                                  layer_workload_item->output->view_begin, layer_workload_item->output->view_end ) );

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool initialize_output_workload_item( nn_gpu_workload_item * &output_workload_item,
                                      nn_gpu_workload_item * &input_workload_item )
{
    output_workload_item              = new nn_gpu_workload_item();
    output_workload_item->type        = NN_WORK_ITEM_TYPE_OUTPUT;
    output_workload_item->input.push_back(input_workload_item);

    output_workload_item->output = nullptr;    //to be filled in at the execute stage

    output_workload_item->arguments.output.index = 0;

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool create_workload_using_workload_items( const nn_device_interface_0_t &di,
                                           nn_gpu_workload *             &gpu_workload,
                                           uint32_t                       batch,
                                           NN_WORKLOAD_DATA_TYPE          input_type,
                                           NN_WORKLOAD_DATA_TYPE          output_type,
                                           nn_gpu_workload_item          *first_workload_item,
                                           nn_gpu_workload_item          *second_workload_item,
                                           nn_gpu_workload_item          *third_workload_item )
{
    gpu_workload = new ( std::nothrow ) nn_gpu_workload;
    nn_workload_t *dummy_workload = reinterpret_cast< nn_workload * >( new char[sizeof( nn_workload_t )] );

    // This made my remaining faith in C++ vanished
    *( const_cast< nn_device_t ** >( &( dummy_workload->device ) ) )         = di.device;
    *( const_cast< uint32_t * >( &( dummy_workload->input_count ) ) )        = 1;
    *( const_cast< uint32_t * >( &( dummy_workload->output_count ) ) )       = 1;
    *( const_cast< NN_WORKLOAD_DATA_TYPE ** >( &( dummy_workload->input_format ) ) )  = new NN_WORKLOAD_DATA_TYPE;
    const_cast< NN_WORKLOAD_DATA_TYPE * >( dummy_workload->input_format )[0]          = input_type;
    *( const_cast< NN_WORKLOAD_DATA_TYPE ** >( &( dummy_workload->output_format ) ) ) = new NN_WORKLOAD_DATA_TYPE;
    const_cast< NN_WORKLOAD_DATA_TYPE * >( dummy_workload->output_format )[0]         = output_type;
    *const_cast<uint32_t *>(&dummy_workload->batch) = batch;

    memcpy( gpu_workload->nn_workload_placeholder, dummy_workload, sizeof( nn_workload ) );
    gpu_workload->m_workload_items.clear();
    gpu_workload->m_workload_items.push_back( first_workload_item );
    gpu_workload->m_workload_items.push_back( second_workload_item );
    gpu_workload->m_workload_items.push_back( third_workload_item );

    delete[] reinterpret_cast< char * >(dummy_workload);

    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool run_softmax_test( const nn_device_interface_0_t &di,
                       uint_least32_t                num_samples,
                       uint_least32_t                num_batches) // length of input to be  processed (softmax normalize)
{
    // Input generation (input feature maps to have pooling run on it)
    float *input = nullptr;
    generate_input_data( input, num_samples, 1, 1, num_batches );

    // length of output is the same as input

    float *cpu_outputs;
    init_data( cpu_outputs, num_samples * num_batches, 0.0f );

    float *gpu_outputs;
    init_data( gpu_outputs, num_samples * num_batches, 0.0f );

    softmax_ref( cpu_outputs, input, num_samples, num_batches );

    // First workload item is input one (entity producing input data)
    nn_gpu_workload_item *input_workload_item = nullptr;
    initialize_input_workload_item( input_workload_item);

    // Specify layout of softmax workload
    nn_workload_data_layout_t workload_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_q },
        NN_DATATYPE_FLOAT
    };

    // specify dimensions of input, output
    nn_workload_data_coords_t workload_coords =
    {
        num_batches,
        num_samples,
        1,
        1,
        1,
        1
    };

    size_t output_coords[2] = {num_samples, num_batches};

    // Now create softmax workload_item giving as input input_workload_item
    nn_gpu_workload_item *softmax_workload_item = nullptr;
    initialize_layer_workload_item( softmax_workload_item, input_workload_item, workload_layout, workload_coords );
    softmax_workload_item->type        = NN_WORK_ITEM_TYPE_SOFTMAX;

    // Now create output workload_item giving softmax workload item as precedessor
    nn_gpu_workload_item *output_workload_item = nullptr;
    initialize_output_workload_item( output_workload_item, softmax_workload_item );

    // Make a workload using two above created workload_items
    nn_gpu_workload *gpu_workload = nullptr;
    create_workload_using_workload_items( di, gpu_workload, num_batches, NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH, NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH, input_workload_item, softmax_workload_item, output_workload_item );

    using io_data = std::unique_ptr<nn::data<float, 0>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    execute_inputs[0]  = io_data(new nn::data<float, 0>(input, output_coords, 2));
    execute_outputs[0] = io_data(new nn::data<float, 0>(gpu_outputs, output_coords, 2));

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( ( nn_workload * )gpu_workload,
                                             ( void ** )execute_inputs,
                                             ( void ** )execute_outputs, nullptr ) );

    nn_workload_data_coords_t output_view_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t output_view_end(num_batches - 1, num_samples - 1, 0, 0, 0, 0);

    // Compare CPU(reference) output with the one returned by GPU
    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs ) );

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function(( nn_workload * )gpu_workload));

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
bool run_convolve_test(
    const nn_device_interface_0_t &di,
    uint_least32_t                num_output_feature_maps,
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

    uint_least32_t output_width  = ( ( input_feature_map_width - kernel_width ) / kernel_stride_x + 1 );
    uint_least32_t output_height = ( ( input_feature_map_height - kernel_height ) / kernel_stride_y + 1 );
    uint_least32_t output_depth  = num_output_feature_maps;

    // cpu_outputs and gpu_outputs are filled in with biases
    // so as such biases do not exist as separate entity
    float init_output_val = 0.0;        //No biases in output then output is initialized with zeros
    float *biases         = nullptr;
    float *cpu_outputs = nullptr;
    float *gpu_outputs = nullptr;

    // Biases exists as separate entity (each neuron got it own bias value)
    init_data( biases, output_width * output_height * output_depth, 1.0f );
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
                  0,        // center offset x
                  0,        // center offset y
                  num_batches );



    // First workload item is input one (entity producing input data)
    nn_gpu_workload_item *input_workload_item = nullptr;
    initialize_input_workload_item( input_workload_item);

    // Specify layout
    nn_workload_data_layout_t input_output_weights_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_FLOAT
    };

    // specify dimensions of input, output and weights
    nn_workload_data_coords_t input_coords =
    {
        num_batches,
        input_feature_map_width,
        input_feature_map_height,
        num_input_feature_maps,
        1,
        1
    };

    nn_workload_data_coords_t output_coords =
    {
        num_batches,
        output_width,
        output_height,
        num_output_feature_maps,
        1,
        1
    };

    nn_workload_data_coords_t weight_coords =
    {
        1,
        kernel_width,
        kernel_height,
        num_input_feature_maps,
        num_output_feature_maps,
        1
    };

    // Now create convolution workload_item giving as input input_workload_item
    nn_gpu_workload_item *convolution_workload_item = nullptr;
    initialize_layer_workload_item( convolution_workload_item, input_workload_item, input_output_weights_layout, output_coords);
    convolution_workload_item->type = NN_WORK_ITEM_TYPE_CONVOLUTION;
    convolution_workload_item->arguments.forward_convolution.padding = NN_PADDING_MODE_NONE;
    convolution_workload_item->arguments.forward_convolution.stride[0] = kernel_stride_x;
    convolution_workload_item->arguments.forward_convolution.stride[1] = kernel_stride_y;
    convolution_workload_item->arguments.forward_convolution.center_offset[0] = 0;
    convolution_workload_item->arguments.forward_convolution.center_offset[1] = 0;
    convolution_workload_item->arguments.forward_convolution.activation.function = activation_function; 

    nn::nn_workload_data_t< float > *weight_data = new nn::nn_workload_data_t< float >( filters, weight_coords, input_output_weights_layout );

    convolution_workload_item->arguments.forward_convolution.weights = new nn::nn_workload_data_t< float >( weight_coords, input_output_weights_layout );
    nn_workload_data_copy( weight_data, convolution_workload_item->arguments.forward_convolution.weights );
    delete weight_data; //release temporary buffers

    nn_workload_data_coords_t bias_coords =
    {
        1,
        1,
        1,
        1,
        num_output_feature_maps,
        1
    };

    nn::nn_workload_data_t< float > *bias_data = new nn::nn_workload_data_t< float >(biases, bias_coords, input_output_weights_layout);
    convolution_workload_item->arguments.forward_convolution.biases = new nn::nn_workload_data_t< float >( bias_coords, input_output_weights_layout );
    nn_workload_data_copy( bias_data, convolution_workload_item->arguments.forward_convolution.biases );
    delete bias_data;   //release temporary buffers

    // Now create output workload_item giving softmax workload item as precedessor
    nn_gpu_workload_item *output_workload_item = nullptr;
    initialize_output_workload_item( output_workload_item, convolution_workload_item );

    // Make a workload using two above created workload_items
    nn_gpu_workload *gpu_workload = nullptr;
    create_workload_using_workload_items( di, gpu_workload, num_batches, NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH, NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH, input_workload_item, convolution_workload_item, output_workload_item );

    using io_data = std::unique_ptr<nn::data<float, 0>>;
    io_data execute_inputs[1];
    io_data execute_outputs[1];

    // specify dimensions of input, output and weights
    size_t execution_input_size[4] = {input_feature_map_width, input_feature_map_height, num_input_feature_maps, num_batches};
    size_t execution_output_size[4] = {output_width, output_height, num_output_feature_maps, num_batches};

    execute_inputs[0]  = io_data(new nn::data<float, 0>(input, execution_input_size, 4));
    execute_outputs[0] = io_data(new nn::data<float, 0>(gpu_outputs, execution_output_size, 4));

    EXPECT_EQ( NN_API_STATUS_OK, di.workload_execute_function( ( nn_workload * )gpu_workload,
                                             ( void ** )execute_inputs,
                                             ( void ** )execute_outputs, nullptr ) );

    EXPECT_EQ( true, verify_output( execute_outputs[0], cpu_outputs ) );


    EXPECT_EQ( NN_API_STATUS_OK, di.workload_delete_function(( nn_workload * )gpu_workload));

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
TEST( gpu_device_load_and_unload, callbacks_test )
{
    // Run in a loop: load, validate,work_item, unload
    for( int i = 0; i < 2; ++i )
    {
        // nn_device_load
        EXPECT_LT( nn_device_load( nullptr ), 0 );  // no error on invalid pointer

        nn_device_description_t dd;
        const auto              invalid_first = static_cast< decltype( dd.version_first ) >( -1 );
        const auto              invalid_last  = static_cast< decltype( dd.version_first ) >( -2 );
        dd.version_first = invalid_first;
        dd.version_last  = invalid_last;

        EXPECT_EQ( 0, nn_device_load( &dd ) );      // non-zero return code on valid call
        EXPECT_NE( invalid_first, dd.version_first ); // nn_device_description_t::version_first is incorrect
        EXPECT_NE( invalid_last, dd.version_last ); // nn_device_description_t::version_last is incorrect
        EXPECT_LE( dd.version_first, dd.version_last ); // nn_device_description_t::version_first is greater than
                                                        // ::version_last

        // nn_device_interface_open & close
        {
            uint8_t buffer[4096];

            // nn_device_interface_open parameter validation
            EXPECT_GT( 0, nn_device_interface_open( invalid_last, buffer ) ); // no error on invalid
                                                                                            // version
            EXPECT_GT( 0, nn_device_interface_open( dd.version_first, nullptr) ); // no error on invalid
                                                                                                 // buffer
            // nn_device_interface_close parameter validation
            EXPECT_GT( 0, nn_device_interface_close( nullptr ) );                           // no error on invalid
                                                                                            // interface pointer
        }

        { // interface version 0
            const uint16_t          interface_version = 0;
            nn_device_interface_0_t di;
            if( ( interface_version >= dd.version_first ) && ( interface_version <= dd.version_last ) )
            {
                EXPECT_EQ( 0, nn_device_interface_open( interface_version, &di ) );
                EXPECT_EQ( interface_version, di.version );     // returned version matches requested
                EXPECT_NE( nullptr, di.device );                // non-null device returned
                EXPECT_NE( nullptr, di.workflow_item_validate_function ); // non-null function pointer returned
                EXPECT_NE( nullptr, di.workflow_item_delete_function ); // non-null function pointer returned
                EXPECT_NE( nullptr, di.parameter_set_function ); // non-null function pointer returned
                EXPECT_NE( nullptr, di.parameter_get_function ); // non-null function pointer returned
            }

            EXPECT_EQ( true, run_convolve_test( di, 1, 1, 2, 2, 1, 1, 1, 1, 1, NN_ACTIVATION_FUNCTION_NONE ) );
            EXPECT_EQ( 0, nn_device_interface_close( &di ) ); // successful close of interface
        }


        // nn_device_unload
        EXPECT_EQ( 0, nn_device_unload() ); // successful unload
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
TEST( gpu_device_interface_0, start_work_item_test )
{

    // 1. Initialize and check if interface 0 is supported
    nn_device_description_t dd;
    const auto              invalid_first = static_cast< decltype( dd.version_first ) >( -1 );
    const auto              invalid_last  = static_cast< decltype( dd.version_first ) >( -2 );
    dd.version_first = invalid_first;
    dd.version_last  = invalid_last;


    EXPECT_EQ( 0, nn_device_load( &dd ) );              // non-zero return code on valid call

    EXPECT_NE( invalid_first, dd.version_first );     // nn_device_description_t::version_first is incorrect
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

    EXPECT_EQ( NN_API_STATUS_ERROR_INVALID_POINTER, di.workflow_item_validate_function( di.device, nullptr ) );    // test invalid pointer case

    EXPECT_EQ( true, run_softmax_test( di,   // interface
                                       10,  // length of input to be  processed (softmax normalize)
                                       3) );

    EXPECT_EQ( 0, nn_device_interface_close( &di ) );       // successful close of interface
    //// nn_device_unload
    EXPECT_EQ( 0, nn_device_unload() ); // successful unload
}

