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
#include <cassert>
#include <vector>
#include <utility>
#include <random>
#include <string>
#include <memory>
#include <malloc.h>
#include "device/gpu/api_internal/nn_device_interface_0_internal.h"
#include "layers_opencl.h"

namespace device_gpu
{


//build options :-Darithmetic_function(x,y)=(x) -DNUM_BATCHES;
//get_global_id(0) is single float to be modulated within a given batch. 
//

static std::string kernelSource = R"( 
__kernel void arithmetic(__global float* output,  __global const float* input, __global const float* factor)
{
    // Calculate offset of data (input and corressponding output) to be processed (first batch)
#ifdef CONVERSION_ZXY_TO_3D
    unsigned int z = get_global_id(0) % TOTAL_INPUT_DEPTH;
    unsigned int x = (get_global_id(0) / TOTAL_INPUT_DEPTH) % TOTAL_INPUT_WIDTH;
    unsigned int y =  get_global_id(0) / (TOTAL_INPUT_DEPTH*TOTAL_INPUT_WIDTH);
    unsigned input_offset = get_global_id(0);
    unsigned output_offset = z*TOTAL_INPUT_WIDTH*TOTAL_INPUT_HEIGHT + y*TOTAL_INPUT_WIDTH + x;
#else
    unsigned input_output_offset = get_global_id(0);    // No conversion scenario can use same value for input and output offset
#endif

    //.. factor data is the same for all batches
#ifdef CONVERSION_ZXY_TO_3D
    float factor_value = factor[output_offset]; // factor data is given as X,Y,Z 
#else
    float factor_value = factor[input_output_offset]; // factor data is given as X,Y,Z 
#endif
    for(unsigned int i = 0 ; i < NUM_BATCHES; ++i) {  // test short vs int
#ifdef CONVERSION_ZXY_TO_3D
        output[output_offset] = arithmetic_function(input[input_offset],factor_value);
        input_offset += get_global_size(0);  // iterate over batches
        output_offset += get_global_size(0);  // iterate over batches
#else
        output[input_output_offset] = arithmetic_function(input[input_output_offset],factor_value);
        input_output_offset += get_global_size(0);  // iterate over batches
#endif
    }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
bool operator < ( const arithmetic_kernel_key &A, const arithmetic_kernel_key &B )
{

    if( A.m_conv_to_perform < B.m_conv_to_perform ) 
    {
        return true;
    }

    if( ( A.m_conv_to_perform == B.m_conv_to_perform ) && (A.m_arithmetic_function < B.m_arithmetic_function) ) 
    {
        return true;
    }

    if( ( A.m_conv_to_perform == B.m_conv_to_perform ) && (A.m_arithmetic_function < B.m_arithmetic_function) ) 
    {
        return true;
    }

    if( ( A.m_conv_to_perform == B.m_conv_to_perform ) &&
        (A.m_arithmetic_function == B.m_arithmetic_function) &&
        (A.m_total_input_width < B.m_total_input_width) ) 
    {
        return true;
    }

    if( (A.m_conv_to_perform == B.m_conv_to_perform ) &&
        (A.m_arithmetic_function == B.m_arithmetic_function) &&
        (A.m_total_input_width == B.m_total_input_width) &&
        (A.m_total_input_height < B.m_total_input_height) ) 
    {
        return true;
    }

    if( (A.m_conv_to_perform == B.m_conv_to_perform ) &&
        (A.m_arithmetic_function == B.m_arithmetic_function) &&
        (A.m_total_input_width == B.m_total_input_width) &&
        (A.m_total_input_height == B.m_total_input_height) &&
        (A.m_total_input_depth < B.m_total_input_depth) ) 
    {
        return true;
    }

    if( ( A.m_conv_to_perform == B.m_conv_to_perform ) &&
        (A.m_arithmetic_function == B.m_arithmetic_function) &&
        (A.m_total_input_width == B.m_total_input_width) &&
        (A.m_total_input_height == B.m_total_input_height) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        (A.m_num_batches < B.m_num_batches) ) 
    {
        return true;
    }

    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
arithmetic_kernel_key ocl_toolkit::prepare_arithmetic_kernel(
    NN_WORKLOAD_DATA_TYPE  output_layout,
    NN_WORKLOAD_DATA_TYPE  input_layout,
    const uint_least32_t   total_input_width,
    const uint_least32_t   total_input_height,
    const uint_least32_t   total_input_depth,
    NN_ARITHMETIC_FUNCTION arithmetic_function,
    unsigned int           num_batches )
{
    std::map< arithmetic_kernel_key, std::unique_ptr< cl::Kernel > >::iterator kit;

    // Here we decide about conversion to be taken if any
    Conversion conv_to_perform = Conversion::NO_CONVERSION;
    switch( input_layout )
    {
    case NN_WORKLOAD_DATA_TYPE_F32_ZXY:           /* nn_data_t, 3D float32: 3D signal (Z, X, Y) */
    case NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH:     /* nn_data_t, 3D float32: 3D signal (Z, X, Y) */
        switch( output_layout )
        {
        case NN_WORKLOAD_DATA_TYPE_F32_3D:                /* nn_data_t, 3D float32: 3D signal (X, Y, Z) */
        case NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH:          /* nn_data_t, 4D float32: sequence of 3D signals */
            conv_to_perform = Conversion::CONVERSION_ZXY_TO_3D;
            break;
        }
        break;

    default:
        break;
    }

    // Check if we have desired kernel (corresponding to requested dimensions)
    // Search for kernel in container
    arithmetic_kernel_key arithmetic_kernel_key_to_use( conv_to_perform,
                                                        arithmetic_function,
                                                        total_input_width,
                                                        total_input_height,
                                                        total_input_depth,
                                                        num_batches );
    kit = m_arithmetic_kernels.find( arithmetic_kernel_key_to_use );

    // If we do not have such a kernel...
    if( kit == m_arithmetic_kernels.end() )
    {
        // ...then make it
        DBG_PRINTF( "compiling arithmetic kernel\n" );

        // Prepare additional compilation args for building program
        std::string extra_compile_args = "";

        // Activation function
        switch( arithmetic_function )
        {
        case NN_ARITHMETIC_FUNCTION_NONE:
            extra_compile_args += " -Darithmetic_function(x,y)=(x)";
            break;
        case NN_ARITHMETIC_FUNCTION_ADDITION:
            extra_compile_args += " -Darithmetic_function(x,y)=(x+y)";
            break;
        case NN_ARITHMETIC_FUNCTION_SUBTRACTION:
            extra_compile_args += " -Darithmetic_function(x,y)=(x-y)";
            break;
        case NN_ARITHMETIC_FUNCTION_MULTIPLICATION:
            extra_compile_args += " -Darithmetic_function(x,y)=(x*y)";
            break;
        case NN_ARITHMETIC_FUNCTION_DIVISION:
            extra_compile_args += " -Darithmetic_function(x,y)=(x/y)";
            break;
        default:
            printf( "Error: Not supported arithmetic function chosen: %d\n", arithmetic_function );
            assert( 0 );
            break;
        }

        switch( conv_to_perform )
        {
        case Conversion::CONVERSION_ZXY_TO_3D:
            extra_compile_args += " -DCONVERSION_ZXY_TO_3D=1";
            break;
        default:
            break;
        }

        extra_compile_args += " -DTOTAL_INPUT_WIDTH=" + std::to_string( total_input_width );
        extra_compile_args += " -DTOTAL_INPUT_HEIGHT=" + std::to_string( total_input_height );
        extra_compile_args += " -DTOTAL_INPUT_DEPTH=" + std::to_string( total_input_depth );
        extra_compile_args += " -DNUM_BATCHES=" + std::to_string( num_batches );

#ifndef DONT_USE_FAST_RELAXED_MATH
        extra_compile_args += " -cl-fast-relaxed-math";
#endif

        std::vector<std::string> kernels(1, kernelSource);

        std::string kernel_name =
            "arithmetic";
        std::pair< std::map< arithmetic_kernel_key, std::unique_ptr< cl::Kernel > >::iterator, bool > ret;
        ret =
            m_arithmetic_kernels.insert( std::pair< arithmetic_kernel_key,
                                                    std::unique_ptr< cl::Kernel > >( arithmetic_kernel_key_to_use,
                                                                                     make_kernels( kernels,
                                                                                                   kernel_name,
                                                                                                   extra_compile_args ) ) );

        // ret.second == false means we are inserting element with key that already exists
        assert( ret.second == true );
    }
    else
    {
        DBG_PRINTF( "reusing existing arithmetic kernel\n" );
    }

    return arithmetic_kernel_key_to_use;
}
////////////////////////////////////////////////////////////////////////////////////////////
void ocl_toolkit::arithmetize( nn_cl_data             *output,
                               nn_cl_data             *input,
                               nn_cl_data             *factor,
                               NN_WORKLOAD_DATA_TYPE  output_layout,
                               NN_WORKLOAD_DATA_TYPE  input_layout,
                               const uint_least32_t   num_input_feature_maps,
                               const uint_least32_t   input_feature_map_width,
                               const uint_least32_t   input_feature_map_height,
                               NN_ARITHMETIC_FUNCTION arithmetic_function,
                               const uint_least32_t   num_batches )
{
    // Get Kernel for a job
    std::map< arithmetic_kernel_key, std::unique_ptr< cl::Kernel > >::iterator kit =
        m_arithmetic_kernels.find( prepare_arithmetic_kernel( output_layout, input_layout, input_feature_map_width,
                                                              input_feature_map_height, num_input_feature_maps,
                                                              arithmetic_function, num_batches ) );

    // If needed kernel was not there then its creation failed for some reason
    assert( kit != m_arithmetic_kernels.end() );

    // Set input and output as args of OpenCL arithmetic kernel
    int retVal = 0;
    retVal = ( kit->second )->setArg( 0, *output->parent->cl_buffer[0] );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL arithmetic kernel argument idx: 0 failed with error: " );
    }

    retVal = ( kit->second )->setArg( 1, *input->parent->cl_buffer[0] );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL arithmetic kernel argument idx: 1 failed with error: " );
    }

    retVal = ( kit->second )->setArg( 2, *factor->parent->cl_buffer[0] );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL arithmetic kernel argument idx: 2 failed with error: " );
    }

    cl::NDRange offset( 0, 0, 0 );
#if defined( DEBUG )
    // data is  dynamically allocated
    // and pointer to it is passed to as data to callback mechanism
    // after using callback function will free dynamic allocation
    exec_struct *psc = new exec_struct;
    psc->name = "arithmetic";
    // for every float of input data there is one multiplication
    psc->num_fmads  = num_input_feature_maps * input_feature_map_width * input_feature_map_height * num_batches;
    psc->time_event = new cl::Event;

    retVal = m_queue->enqueueNDRangeKernel( *( kit->second ),
                                            offset,
                                            cl::NDRange( num_input_feature_maps * input_feature_map_width *
                                                         input_feature_map_height ),
                                            cl::NullRange, 0, psc->time_event ); // PROFILING
    psc->time_event->setCallback( CL_COMPLETE, &exec_completed, ( void * )psc );
#else
    retVal = m_queue->enqueueNDRangeKernel( *( kit->second ),
                                            offset,
                                            cl::NDRange( num_input_feature_maps * input_feature_map_width *
                                                         input_feature_map_height ),
                                            cl::NullRange );
#endif

    //TODO: Enable more optimal arithmetic kernel
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal,
                     " Error executing OpenCL enqueueNDRange for arithmetic kernel. Call failed with error: " );
    }

}

} //namespace device_gpu
