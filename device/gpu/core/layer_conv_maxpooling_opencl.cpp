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
#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <random>
#include <string>
#include <memory>
#include <malloc.h>
#include "device/gpu/api_internal/nn_device_interface_0_internal.h"
#include "layers_opencl.h"

namespace device_gpu
{
    static std::string kernelSource = R"(
        __kernel void merged_convolve_maxpoolND(__global floatx* output, __global floatx* input, filter_qualifier floatx* filter, __global floatx* biases, unsigned int batch_output_offset)
    {
        const unsigned int x_pool = get_global_id(0);
        const unsigned int y_pool = get_global_id(1);
        const unsigned int z = get_global_id(2);

        const unsigned int x_conv = x_pool * POOL_WINDOW_SIZE;
        const unsigned int y_conv = y_pool * POOL_WINDOW_SIZE;

        const unsigned int filter_size = FILTER_DEPTH*FILTER_HEIGHT*FILTER_WIDTH;

        float cur_max;
        const float bias = biases[z];
        for (unsigned x = x_conv;  x < (x_conv + POOL_WINDOW_SIZE); x++)
        {
            for (unsigned y = y_conv; y < (y_conv + POOL_WINDOW_SIZE); y++)
            {
                unsigned int filter_offset = z * filter_size;
                floatx dotProd = 0.0f;
                unsigned int input_offset = STRIDEY * y * INPUT_WIDTH + x * STRIDEX;

                for (unsigned int k = 0; k< FILTER_DEPTH; ++k) {
                    for (unsigned int j = 0; j< FILTER_HEIGHT; ++j) {
                        for (unsigned int i = 0; i< FILTER_WIDTH; ++i) {
                            floatx signal;
                            signal = input[input_offset];
                            dotProd += signal * filter[filter_offset];
                            ++input_offset;
                            ++filter_offset;
                        }
                        input_offset += INPUT_WIDTH - FILTER_WIDTH;
                    }
                    input_offset += (INPUT_HEIGHT - FILTER_HEIGHT)*INPUT_WIDTH;
                }
                if(x == x_conv && y == y_conv)
                    cur_max = activation_function(dotProd + bias);
                else
                    cur_max = max(cur_max, activation_function(dotProd + bias));
            }
        }

    const unsigned int pool_output_stride = (OUTPUT_WIDTH + OWPAD) * (OUTPUT_HEIGHT + OHPAD);

    output[batch_output_offset + OUT_BUFF_OFFSET + z * pool_output_stride + y_pool*(OUTPUT_WIDTH + OWPAD) + x_pool] = cur_max;
})";

bool operator < ( const conv_maxpool_kernel_key &A, const conv_maxpool_kernel_key &B )
{
    if( A.m_input_width < B.m_input_width )
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) && ( A.m_input_height < B.m_input_height ) )
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth < B.m_input_depth ) )
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_filter_width < B.m_filter_width ) )
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height < B.m_filter_height ) )
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth < B.m_filter_depth ) )
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth == B.m_filter_depth ) &&
        ( A.m_num_filters < B.m_num_filters ) )
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth == B.m_filter_depth ) &&
        ( A.m_num_filters == B.m_num_filters ) &&
        ( A.m_stride_x < B.m_stride_x ) )
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth == B.m_filter_depth ) &&
        ( A.m_num_filters == B.m_num_filters ) &&
        ( A.m_stride_x == B.m_stride_x ) &&
        ( A.m_stride_y < B.m_stride_y ) )
    {
        return true;
    }


    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth == B.m_filter_depth ) &&
        ( A.m_num_filters == B.m_num_filters ) &&
        ( A.m_stride_x == B.m_stride_x ) &&
        ( A.m_stride_y == B.m_stride_y ) &&
        ( A.m_activation_function < B.m_activation_function ) )
    {
        return true;
    }

    if ( ( A.m_input_width == B.m_input_width ) &&
         ( A.m_input_height == B.m_input_height ) &&
         ( A.m_input_depth == B.m_input_depth ) &&
         ( A.m_filter_width == B.m_filter_width ) &&
         ( A.m_filter_height == B.m_filter_height ) &&
         ( A.m_filter_depth == B.m_filter_depth ) &&
         ( A.m_num_filters == B.m_num_filters ) &&
         ( A.m_stride_x == B.m_stride_x ) &&
         ( A.m_stride_y == B.m_stride_y ) &&
         ( A.m_activation_function == B.m_activation_function ) &&
         ( A.m_pool_window_size < B.m_pool_window_size )
        )
    {
        return true;
    }

    if ( ( A.m_input_width == B.m_input_width) &&
         ( A.m_input_height == B.m_input_height) &&
         ( A.m_input_depth == B.m_input_depth) &&
         ( A.m_filter_width == B.m_filter_width) &&
         ( A.m_filter_height == B.m_filter_height) &&
         ( A.m_filter_depth == B.m_filter_depth) &&
         ( A.m_num_filters == B.m_num_filters) &&
         ( A.m_stride_x == B.m_stride_x) &&
         ( A.m_stride_y == B.m_stride_y) &&
         ( A.m_activation_function == B.m_activation_function ) &&
         ( A.m_pool_window_size == B.m_pool_window_size ) &&
         ( A.m_pool_stride < B.m_pool_stride )
        )
    {
        return true;
    }

    if ( ( A.m_input_width == B.m_input_width) &&
         ( A.m_input_height == B.m_input_height) &&
         ( A.m_input_depth == B.m_input_depth) &&
         ( A.m_filter_width == B.m_filter_width) &&
         ( A.m_filter_height == B.m_filter_height) &&
         ( A.m_filter_depth == B.m_filter_depth) &&
         ( A.m_num_filters == B.m_num_filters) &&
         ( A.m_stride_x == B.m_stride_x) &&
         ( A.m_stride_y == B.m_stride_y) &&
         ( A.m_activation_function == B.m_activation_function ) &&
         ( A.m_pool_window_size == B.m_pool_window_size ) &&
         ( A.m_pool_stride == B.m_pool_stride )  &&
         ( A.m_output_width < B.m_output_width )
        )
    {
        return true;
    }

    if ( ( A.m_input_width == B.m_input_width) &&
         ( A.m_input_height == B.m_input_height) &&
         ( A.m_input_depth == B.m_input_depth) &&
         ( A.m_filter_width == B.m_filter_width) &&
         ( A.m_filter_height == B.m_filter_height) &&
         ( A.m_filter_depth == B.m_filter_depth) &&
         ( A.m_num_filters == B.m_num_filters) &&
         ( A.m_stride_x == B.m_stride_x) &&
         ( A.m_stride_y == B.m_stride_y) &&
         ( A.m_activation_function == B.m_activation_function ) &&
         ( A.m_pool_window_size == B.m_pool_window_size ) &&
         ( A.m_pool_stride == B.m_pool_stride )  &&
         ( A.m_output_width == B.m_output_width )  &&
         ( A.m_output_height < B.m_output_height )
        )
    {
        return true;
    }


    return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: consider loading kernels from Binary
conv_maxpool_kernel_key ocl_toolkit::prepare_conv_maxpool_kernel(
    uint_least32_t         output_width,
    uint_least32_t         output_height,
    uint_least32_t         input_width,
    uint_least32_t         input_height,
    uint_least32_t         input_depth,
    uint_least32_t         filter_width,
    uint_least32_t         filter_height,
    uint_least32_t         filter_depth,
    uint_least32_t         num_filters,
    uint_least32_t         stride_x,
    uint_least32_t         stride_y,
    NN_ACTIVATION_FUNCTION activation_function,
    uint_least32_t         pool_stride_x,
    uint_least32_t         pool_window_size_x,
    NN_POOLING_MODE        pool_mode,
    uint_least32_t         output_buffer_offset,
    uint_least32_t         output_w_pad_for_next_layer,
    uint_least32_t         output_h_pad_for_next_layer
    )
{
    // Check if we have desired kernel (corresponding to requested dimensions)
    // Search for kernel in container

// TODO: add pads and output_buffer_offset to the key
    conv_maxpool_kernel_key conv_maxpool_kernel_key_to_use( input_width, input_height, input_depth,
                                            filter_width, filter_height, filter_depth, num_filters,
                                            stride_x, stride_y, activation_function , pool_stride_x, pool_window_size_x, output_width, output_height );
    auto g_kit = m_conv_maxpool_kernels.find( conv_maxpool_kernel_key_to_use );

    // If we do not have such a kernel...
    if( g_kit == m_conv_maxpool_kernels.end() )
    {
        // ...then make it
        DBG_PRINTF( "compiling merged convolving+maxpool kernel\n" );

        // Prepare additional compilation args for building program
        std::string extra_compile_args = " -DINPUT_WIDTH=" + std::to_string( input_width );
        extra_compile_args += " -DINPUT_HEIGHT=" + std::to_string( input_height );
        extra_compile_args += " -DINPUT_DEPTH=" + std::to_string( input_depth );

        extra_compile_args += " -DOUTPUT_WIDTH=" + std::to_string( output_width );
        extra_compile_args += " -DOUTPUT_HEIGHT=" + std::to_string( output_height );

        extra_compile_args += " -DFILTER_WIDTH=" + std::to_string( filter_width );
        extra_compile_args += " -DFILTER_HEIGHT=" + std::to_string( filter_height );
        extra_compile_args += " -DFILTER_DEPTH=" + std::to_string( filter_depth );

        extra_compile_args += " -DNUM_FILTERS=" + std::to_string( num_filters );

        extra_compile_args += " -DSTRIDEX=" + std::to_string( stride_x );
        extra_compile_args += " -DSTRIDEY=" + std::to_string( stride_y );

        extra_compile_args += " -DPOOL_WINDOW_SIZE=" + std::to_string( pool_window_size_x );
        extra_compile_args += " -DPOOL_WINDOW_STRIDE=" + std::to_string( pool_stride_x );

        extra_compile_args += " -DOUT_BUFF_OFFSET=" + std::to_string(output_buffer_offset);

        extra_compile_args += " -DOWPAD=" + std::to_string(output_w_pad_for_next_layer);
        extra_compile_args += " -DOHPAD=" + std::to_string(output_h_pad_for_next_layer);


        // Activation function
        switch( activation_function )
        {
        case NN_ACTIVATION_FUNCTION_NONE:
            extra_compile_args += " -Dactivation_function(x)=(x)";
            break;
        case NN_ACTIVATION_FUNCTION_TANH:
            extra_compile_args += " -Dactivation_function(x)=tanh(x)";
            break;
        case NN_ACTIVATION_FUNCTION_RELU:
            extra_compile_args += " -Dactivation_function(x)=fmax(0.0f,x)";
            break;
        case NN_ACTIVATION_FUNCTION_SOFTPLUS:
            extra_compile_args += " -Dactivation_function(x)=log(1.0f+exp(x))";
            break;
        default:
            printf( "Error: Not supported activation function chosen: %d\n", activation_function );
            assert( 0 );
            break;
        }

#ifndef DONT_USE_FAST_RELAXED_MATH
        extra_compile_args += " -cl-fast-relaxed-math";
#endif

        extra_compile_args += " -Dfloatx=float";

        // Check if filter will fit into constant memory area
        unsigned int filter_size = filter_width * filter_height * filter_depth *
                                   num_filters * sizeof( float );

        if( filter_size <= m_constant_mem_size )
        {
            extra_compile_args += " -Dfilter_qualifier=__constant";
        }
        else
        {
            extra_compile_args += " -Dfilter_qualifier=__global";
        }

        if( filter_size <= m_local_mem_size )
        {
            extra_compile_args += " -DUSE_LOCAL_MEMORY";
            DBG_PRINTF( "=== using local memory ===\n" );
        }
        else
        {
            DBG_PRINTF( "=== using constant/global memory ===\n" );
        }

        // For Intel's GPU NUM_ACC 8 is quite efficient
        // TODO: Make proper analysis if it is better to spawn 2 num_acc8 kernels or one num_acc16

        std::vector<std::string> kernels(1, kernelSource);
        conv_maxpool_kernel_variants kernel (
            make_kernels( kernels,
                          "merged_convolve_maxpoolND",
                          extra_compile_args) );

        auto ret = m_conv_maxpool_kernels.insert( std::make_pair( conv_maxpool_kernel_key_to_use, std::move( kernel ) ) );
        // ret.second == false means we are inserting element with key that already exists
        assert( ret.second == true );

    }
    else
    {
        DBG_PRINTF( "reusing existing merged convolving+maxpool kernel\n" );
    }
    return conv_maxpool_kernel_key_to_use;
}
////////////////////////////////////////////////////////////////////////////////////////////
void ocl_toolkit::convolve_maxpool(
    nn_cl_data            *output,
    nn_cl_data            *input,
    nn_cl_data            *filter,
    nn_cl_data            *biases,
    uint_least32_t         output_width,
    uint_least32_t         output_height,
    uint_least32_t         output_depth,
    uint_least32_t         output_buffer_size,
    uint_least32_t         input_width,
    uint_least32_t         input_height,
    uint_least32_t         input_depth,
    uint_least32_t         filter_width,
    uint_least32_t         filter_height,
    uint_least32_t         filter_depth,
    uint_least32_t         num_batches,
    unsigned int           stride_x,
    unsigned int           stride_y,
    NN_ACTIVATION_FUNCTION activation_function,
    uint_least32_t         pool_stride_x,
    uint_least32_t         pool_stride_y,
    uint_least32_t         pool_window_size_x,
    uint_least32_t         pool_window_size_y,
    NN_POOLING_MODE        pool_mode
    )
{
    assert(pool_window_size_x == pool_window_size_y);
    assert(pool_stride_x == pool_stride_y);

    // Get Kernel for a job
    auto g_kit =
        m_conv_maxpool_kernels.find( prepare_conv_maxpool_kernel(
                                                  output_width,                  // output_width,
                                                  output_height,                 // output_height,
                                                  input_width,                   // input_width,
                                                  input_height,                  // input_height,
                                                  input_depth,                   // input_depth,
                                                  filter_width,                  // filter_width,
                                                  filter_height,                 // filter_height,
                                                  filter_depth,                  // filter_depth,
                                                  output_depth,                  // num_filters,
                                                  stride_x,                      // stride_x,
                                                  stride_y,                      // stride_y,
                                                  activation_function,           // activation_function,
                                                  pool_stride_x,                 // pool_stride_x,
                                                  pool_window_size_x,            // pool_window_size_x,
                                                  pool_mode,                      // pool_mode
                                                  0,0,0
                                                  ) );

    // If needed kernel was not there then its creation failed for some reason
    assert(g_kit != m_conv_maxpool_kernels.end());

    //TODO: Bunch of outputs, consider how to implement mapping of those buffers
    for( unsigned int batch = 0; batch < num_batches; ++batch )
    {
        // Set input, output and filter as args of OpenCL convolve kernel
        int retVal = 0;
        retVal = ( g_kit->second ).m_kernel->setArg( 0, *output->parent->cl_buffer[0] );
        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 0 failed with error " );
        }
        retVal = ( g_kit->second ).m_kernel->setArg( 1, *input->parent->cl_subbuffer[0].at(batch) );
        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 1 failed with error: " );
        }
        retVal = ( g_kit->second ).m_kernel->setArg( 2, *filter->parent->cl_buffer[0] );
        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 2 failed with error: " );
        }

        retVal = (g_kit->second).m_kernel->setArg( 3, *biases->parent->cl_buffer[0] );
        if (retVal != CL_SUCCESS)
        {
            THROW_ERROR(retVal, " Error setting OpenCL kernel argument idx: 3 failed with error: ");
        }

        uint32_t ouput_buffer_batch_offset = batch * ( ( output_buffer_size / sizeof( float ) ) / num_batches );
        retVal = ( g_kit->second ).m_kernel->setArg( 4, ouput_buffer_batch_offset );
        if (retVal != CL_SUCCESS)
        {
            THROW_ERROR(retVal, " Error setting OpenCL kernel argument idx: 4 failed with error: ");
        }

        const auto conv_outmap_width = (output_width - 1)*pool_stride_x + pool_window_size_x;
        const auto conv_outmap_height = (output_height - 1)*pool_stride_y + pool_window_size_y;

        cl::NDRange offset( 0, 0, 0 );
        cl::NDRange global_size(output_width, output_height, output_depth);

#if defined( DEBUG )
        // data is  dynamically allocated
        // and pointer to it is passed to as data to callback mechanism
        // after using callback function will free dynamic allocation
        exec_struct *psc = new exec_struct;
        psc->name      = "merged_convmaxpool-pref";
        // TODO: change when the whole batch is processed by one kernel execution
        psc->num_fmads = /*num_batches * */ conv_outmap_width * conv_outmap_height * output_depth * filter_width * filter_height *
                          filter_depth;
        psc->time_event = new cl::Event;
        retVal = m_queue->enqueueNDRangeKernel( *( g_kit->second ).m_kernel,
                                                offset,
                                                global_size,
                                                cl::NullRange,
                                                0,
                                                psc->time_event );
        // Number of MADs to be done to perform this operation
        psc->time_event->setCallback( CL_COMPLETE, &exec_completed, ( void * )psc );
#else
        retVal = m_queue->enqueueNDRangeKernel( *( g_kit->second ).m_kernel, offset, global_size, cl::NullRange );
#endif
        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error executing OpenCL enqueueNDRange. Call failed with error: " );
        }

    }
}
} //namespace device_gpu
