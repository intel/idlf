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
//TODO: Make padding for pooling. currently input view need to contaain whole pooling area
static std::string kernelSource = R"(
// Input dimensions, pooling window size passed via compiler definition
// build options: -DINPUT_WIDTH, -DINPUT_HEIGHT=, -DINPUT_HEIGHT=, -DINPUT_WINDOW_SIZE -DINPUT_WINDOW_STRIDE
//               -DINPUT_START_X, -DINPUT_START_Y
//               -DINPUT_END_X, -DINPUT_END_Y
//               -DOUTPUT_START_X, -DOUTPUT_START_Y
__kernel void max_pool_non_overlapping(__global float* output,  __global float* input)
{
    const unsigned input_stride  = INPUT_WIDTH  * INPUT_HEIGHT;
    const unsigned output_stride = ((INPUT_WIDTH - INPUT_WINDOW_SIZE)/ INPUT_WINDOW_STRIDE + 1) * ((INPUT_HEIGHT - INPUT_WINDOW_SIZE)/INPUT_WINDOW_STRIDE + 1);
    const unsigned output_width_stride = ((INPUT_WIDTH - INPUT_WINDOW_SIZE)/ INPUT_WINDOW_STRIDE + 1);

    const unsigned fm = get_global_id(0);

    __global float* my_input  = input  + fm * input_stride;
    __global float* my_output = output + fm * output_stride;
    unsigned int output_offset_x = OUTPUT_START_X;
    unsigned int output_offset_y = OUTPUT_START_Y;

    for (unsigned row = INPUT_START_Y; row <= INPUT_END_Y - INPUT_WINDOW_SIZE + 1; row += INPUT_WINDOW_STRIDE)
    {
        for (unsigned col = INPUT_START_X; col <= INPUT_END_X - INPUT_WINDOW_SIZE + 1; col += INPUT_WINDOW_STRIDE)
        {
            __global float* cur_input  = my_input  + row * INPUT_WIDTH + col ;
            float cur_max = *cur_input;
            for (unsigned cur_row = 0; cur_row < INPUT_WINDOW_SIZE; ++cur_row)
            {
                for (unsigned cur_col = 0; cur_col < INPUT_WINDOW_SIZE; ++cur_col)
                {
                    cur_max = max(cur_max, cur_input[cur_row * INPUT_WIDTH + cur_col]);
                }
            }
            *(my_output + output_offset_x + output_offset_y*output_width_stride) = cur_max;
            ++output_offset_x;
        }
        output_offset_x = OUTPUT_START_X;
        ++output_offset_y;
    }
}

__kernel void max_pool_overlapping(
#ifdef IMAGE_AS_OUTPUT
    __write_only image2d_t output,
#else
    __global float* output,
#endif
 __global float* input)
{
    const unsigned input_stride  = INPUT_WIDTH  * INPUT_HEIGHT;
    const unsigned output_stride = ((INPUT_WIDTH - INPUT_WINDOW_SIZE)/ INPUT_WINDOW_STRIDE + 1) * ((INPUT_HEIGHT - INPUT_WINDOW_SIZE)/INPUT_WINDOW_STRIDE + 1);
    const unsigned output_width_stride = ((INPUT_WIDTH - INPUT_WINDOW_SIZE)/ INPUT_WINDOW_STRIDE + 1);

    //batches*depth of inputs are stored in Z dimension
    const unsigned fm = get_global_id(2);

    const unsigned X = get_global_id( 0 );
    const unsigned Y = get_global_id( 1 );

    __global float* my_input  = input  + fm * input_stride + Y * INPUT_WIDTH * INPUT_WINDOW_STRIDE + INPUT_START_Y * INPUT_WIDTH + X * INPUT_WINDOW_STRIDE + INPUT_START_X;
#ifdef IMAGE_AS_OUTPUT
    const unsigned output_write_x_coord = (fm %INPUT_DEPTH) * output_stride + ( OUTPUT_START_Y + Y ) * output_width_stride + X + OUTPUT_START_X;
    const unsigned output_write_y_coord = fm/INPUT_DEPTH;

#else
    __global float* my_output = output + fm * output_stride + ( OUTPUT_START_Y + Y ) * output_width_stride + X + OUTPUT_START_X;
#endif

    //uncomment this code after INPUT_START_Z is properly passed
    //__global float* my_input = input  + INPUT_START_Z * input_stride + Y * INPUT_WIDTH * INPUT_WINDOW_STRIDE + INPUT_START_Y * INPUT_WIDTH + X * INPUT_WINDOW_STRIDE + INPUT_START_X;


    float max_value   = 0.0f;
    uint row          = 0;
    uint column       = 0;

    //we need to deal with boundary use cases, therefore split logic into 2 sections
    // this code may be used for zero padding
    // uncomment this code when enabling zero padding, it may need addtional effort
    /*
    if( INPUT_WIDTH % INPUT_WINDOW_SIZE != 0 || INPUT_HEIGHT % INPUT_WINDOW_SIZE != 0 )
    {
        //check if this is boundary work item
        if( get_global_id( 0 ) == get_global_size( 0 ) - 1 ||
            get_global_id( 1 ) == get_global_size( 1 ) - 1 )
        {
            uint maxX = INPUT_WINDOW_SIZE;
            uint maxY = INPUT_WINDOW_SIZE;

            if( get_global_id( 0 ) == get_global_size( 0 ) - 1 )
            {
                 maxX = INPUT_WIDTH  % INPUT_WINDOW_SIZE;
            }
            if( get_global_id( 1 ) == get_global_size( 1 ) - 1 )
            {
                 maxY = INPUT_HEIGHT % INPUT_WINDOW_SIZE;
            }

            for( uint i = 0 ; i < INPUT_WINDOW_SIZE * INPUT_WINDOW_SIZE;i++ )
            {
                if( column < maxX && row < maxY )
                {
                    max_value = max( max_value, *( my_input + column + row * INPUT_WIDTH ) );
                }

                column++;
                if( column == INPUT_WINDOW_SIZE )
                {
                    row++;
                    column = 0;
                }
            }
            *my_output = max_value;
            return;
        }
    }
    */
    //we need to check for Size x Size values to get a max out of them
    for( uint i = 0 ; i < INPUT_WINDOW_SIZE * INPUT_WINDOW_SIZE;i++ )
    {
        max_value = max( max_value, *( my_input + column + row * INPUT_WIDTH ) );
        column++;
        if( column == INPUT_WINDOW_SIZE )
        {
            row++;
            column = 0;
        }
    }

#ifdef IMAGE_AS_OUTPUT
    write_imagef(output,(int2)(output_write_x_coord, output_write_y_coord),max_value);
#else
    *my_output = max_value;
#endif



}

/* OLD kernel left for refference
__kernel void max_pool_overlapping(__global float* output,  __global float* input)
{
    const unsigned input_stride  = INPUT_WIDTH  * INPUT_HEIGHT;
    const unsigned output_stride = ((INPUT_WIDTH - INPUT_WINDOW_SIZE)/ INPUT_WINDOW_STRIDE + 1) * ((INPUT_HEIGHT - INPUT_WINDOW_SIZE)/INPUT_WINDOW_STRIDE + 1);
    const unsigned output_width_stride = ((INPUT_WIDTH - INPUT_WINDOW_SIZE)/ INPUT_WINDOW_STRIDE + 1);

    const unsigned fm = get_global_id(0);

    __global float* my_input  = input  + fm * input_stride;
    __global float* my_output = output + fm * output_stride;


    unsigned int output_offset_x = OUTPUT_START_X;
    unsigned int output_offset_y = OUTPUT_START_Y;

    for (unsigned row = INPUT_START_Y; row <= INPUT_END_Y - INPUT_WINDOW_SIZE + 1; row += INPUT_WINDOW_STRIDE)
    {
#if INPUT_WINDOW_SIZE != INPUT_WINDOW_STRIDE
        float old_share_max;
#endif
        for (unsigned col = INPUT_START_X; col <= INPUT_END_X - INPUT_WINDOW_SIZE + 1; col += INPUT_WINDOW_STRIDE)
        {
            __global float* cur_input  = my_input  + row * INPUT_WIDTH + col ;
            // If some of inputs are shared (overlapping) than last one on right is the one to be shared
            float cur_max = *(cur_input+ INPUT_WINDOW_SIZE - 1);
#if INPUT_WINDOW_SIZE != INPUT_WINDOW_STRIDE
            float val;
            float share_max = cur_max;
#endif
            for (unsigned cur_row = 0; cur_row < INPUT_WINDOW_SIZE; ++cur_row)
            {
#if INPUT_WINDOW_SIZE != INPUT_WINDOW_STRIDE
                if(col == 0) {
#endif
                    for (unsigned cur_col = 0; cur_col < INPUT_WINDOW_STRIDE; ++cur_col)
                    {
                        cur_max = max(cur_max, cur_input[cur_row * INPUT_WIDTH + cur_col]);
                    }
#if INPUT_WINDOW_SIZE != INPUT_WINDOW_STRIDE
                } else {
                    cur_max = max(cur_max,old_share_max);
                    for (unsigned cur_col = 1; cur_col < INPUT_WINDOW_STRIDE; ++cur_col)
                    {
                        cur_max = max(cur_max, cur_input[cur_row * INPUT_WIDTH + cur_col]);
                    }
                }
                for (unsigned cur_col = INPUT_WINDOW_STRIDE; cur_col < INPUT_WINDOW_SIZE; ++cur_col)
                {
                    val = cur_input[cur_row * INPUT_WIDTH + cur_col];
                    cur_max = max(cur_max, val );
                    share_max = max(share_max,val);
                }
#endif
            }
            *(my_output + output_offset_x + output_offset_y*output_width_stride) = cur_max;
            ++output_offset_x;
#if INPUT_WINDOW_SIZE != INPUT_WINDOW_STRIDE
            old_share_max = share_max;
            cur_max = share_max;
#endif
        }
        output_offset_x = OUTPUT_START_X;
        ++output_offset_y;
    }
}*/
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
bool operator < ( const pooling_kernel_key &A, const pooling_kernel_key &B )
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
        ( A.m_input_depth < B.m_input_depth))
    {
        return true;
    }


    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x < B.m_input_start_x))
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x) &&
        ( A.m_input_start_y < B.m_input_start_y))
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x) &&
        ( A.m_input_start_y == B.m_input_start_y) &&
        ( A.m_input_end_x < B.m_input_end_x))
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x) &&
        ( A.m_input_start_y == B.m_input_start_y) &&
        ( A.m_input_end_x == B.m_input_end_x) &&
        ( A.m_input_end_y < B.m_input_end_y))
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x) &&
        ( A.m_input_start_y == B.m_input_start_y) &&
        ( A.m_input_end_x == B.m_input_end_x) &&
        ( A.m_input_end_y == B.m_input_end_y) &&
        ( A.m_output_start_x < B.m_output_start_x))
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x) &&
        ( A.m_input_start_y == B.m_input_start_y) &&
        ( A.m_input_end_x == B.m_input_end_x) &&
        ( A.m_input_end_y == B.m_input_end_y) &&
        ( A.m_output_start_x == B.m_output_start_x)  &&
        ( A.m_output_start_y < B.m_output_start_y))
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x) &&
        ( A.m_input_start_y == B.m_input_start_y) &&
        ( A.m_input_end_x == B.m_input_end_x) &&
        ( A.m_input_end_y == B.m_input_end_y) &&
        ( A.m_output_start_x == B.m_output_start_x)  &&
        ( A.m_output_start_y == B.m_output_start_y)  &&
        ( A.m_input_window_size < B.m_input_window_size ) )
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x) &&
        ( A.m_input_start_y == B.m_input_start_y) &&
        ( A.m_input_end_x == B.m_input_end_x) &&
        ( A.m_input_end_y == B.m_input_end_y) &&
        ( A.m_output_start_x == B.m_output_start_x)  &&
        ( A.m_output_start_y == B.m_output_start_y)  &&
        ( A.m_input_window_size == B.m_input_window_size ) &&
        ( A.m_input_window_stride < B.m_input_window_stride) )
    {
        return true;
    }

    if( ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x) &&
        ( A.m_input_start_y == B.m_input_start_y) &&
        ( A.m_input_end_x == B.m_input_end_x) &&
        ( A.m_input_end_y == B.m_input_end_y) &&
        ( A.m_output_start_x == B.m_output_start_x)  &&
        ( A.m_output_start_y == B.m_output_start_y)  &&
        ( A.m_input_window_size == B.m_input_window_size ) &&
        ( A.m_input_window_stride == B.m_input_window_stride) &&
        ( A.m_image_as_output < B.m_image_as_output) )
    {
        return true;
    }


    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: consider loading kernels from Binary
pooling_kernel_key ocl_toolkit::prepare_pooling_kernel(
    bool           image_as_output,
    uint_least32_t input_width,
    uint_least32_t input_height,
    uint_least32_t input_depth,
    uint_least32_t input_start_x,
    uint_least32_t input_start_y,
    uint_least32_t input_end_x,
    uint_least32_t input_end_y,
    uint_least32_t output_start_x,
    uint_least32_t output_start_y,
    uint_least32_t input_window_size,
    uint_least32_t input_window_stride )
{
    std::map< pooling_kernel_key, std::unique_ptr< cl::Kernel > >::iterator kit;

    // Check if we have desired kernel (corresponding to requested dimensions)
    // Search for kernel in container

    pooling_kernel_key pooling_kernel_to_use( input_width,
                                              input_height,
                                              input_depth,
                                              input_start_x,
                                              input_start_y,
                                              input_end_x,
                                              input_end_y,
                                              output_start_x,
                                              output_start_y,
                                              input_window_size,
                                              input_window_stride,
                                              image_as_output );

    kit = m_pooling_kernels.find( pooling_kernel_to_use );

    // If we do not have such a kernel...
    if( kit == m_pooling_kernels.end() )
    {
        DBG_PRINTF( "compiling pooling kernel\n" );

        // ...then make it
        // Prepare additional compilation args for building program
        std::string extra_compile_args = " -DINPUT_WIDTH=" + std::to_string( input_width );
        extra_compile_args += " -DINPUT_HEIGHT=" + std::to_string( input_height );
        extra_compile_args += " -DINPUT_DEPTH=" + std::to_string( input_depth );
        extra_compile_args += " -DINPUT_WINDOW_SIZE=" + std::to_string( input_window_size );
        extra_compile_args += " -DINPUT_WINDOW_STRIDE=" + std::to_string( input_window_stride );

        extra_compile_args += " -DINPUT_START_X=" + std::to_string( input_start_x );
        extra_compile_args += " -DINPUT_START_Y=" + std::to_string( input_start_y );
        extra_compile_args += " -DINPUT_END_X=" + std::to_string( input_end_x );
        extra_compile_args += " -DINPUT_END_Y=" + std::to_string( input_end_y );

        extra_compile_args += " -DOUTPUT_START_X=" + std::to_string( output_start_x );
        extra_compile_args += " -DOUTPUT_START_Y=" + std::to_string( output_start_y );

        // Should the output area be a buffer or image
        if(image_as_output == true) {
            extra_compile_args += " -DIMAGE_AS_OUTPUT";
        }

#ifndef DONT_USE_FAST_RELAXED_MATH
        extra_compile_args += " -cl-fast-relaxed-math";
#endif

        std::pair< std::map< pooling_kernel_key, std::unique_ptr< cl::Kernel > >::iterator, bool > ret;

        std::vector<std::string> kernels(1, kernelSource);

        //ret = m_pooling_kernels.insert( std::pair< pooling_kernel_key, std::unique_ptr< cl::Kernel> >(
        // pooling_kernel_to_use, make_kernels( kernelSource, "max_pool_non_overlapping", extra_compile_args ) ) );
        ret =
            m_pooling_kernels.insert( std::pair< pooling_kernel_key,
                                                 std::unique_ptr< cl::Kernel > >( pooling_kernel_to_use,
                                                                                  make_kernels(
                                                                                      kernels,
                                                                                      "max_pool_overlapping",
                                                                                      extra_compile_args ) ) );

        // ret.second == false means we are inserting element with key that already exists
        assert( ret.second == true );
    }
    else
    {
        DBG_PRINTF( "reusing existing pooling kernel\n" );
    }

    return pooling_kernel_to_use;
}
////////////////////////////////////////////////////////////////////////////////////////////
void ocl_toolkit::max_pool( nn_cl_data          *output,
                            nn_cl_data          *input,
                            cl::NDRange&         output_start_offset,
                            cl::NDRange&         output_end_offset,
                            cl::NDRange&         input_start_offset,
                            cl::NDRange&         input_end_offset,
                            const uint_least32_t output_batch_start,
                            const uint_least32_t output_batch_end,
                            const uint_least32_t input_batch_start,
                            const uint_least32_t input_batch_end,
                            const uint_least32_t output_width,
                            const uint_least32_t output_height,
                            const uint_least32_t output_depth,
                            const uint_least32_t input_width,
                            const uint_least32_t input_height,
                            const uint_least32_t input_depth,
                            const uint_least32_t num_batches,
                            const uint_least32_t stride_x,
                            const uint_least32_t stride_y,
                            const uint_least32_t size_x,
                            const uint_least32_t size_y,
                            NN_POOLING_MODE      mode)
{
    //TODO: support for other pooling layers
    assert( mode == NN_POOLING_MODE_MAX );

    //TODO: support for non-square pooling areas
    assert( size_x == size_y );

    // Get Kernel for a job
    std::map< pooling_kernel_key, std::unique_ptr< cl::Kernel > >::iterator kit =
        m_pooling_kernels.find( prepare_pooling_kernel(
                                output->parent->cl_buffer[0] == nullptr,  // If no buffer exist than use image
                                input_width, input_height, input_depth,
                                static_cast<const ::size_t*>(input_start_offset)[0],
                                static_cast<const ::size_t*>(input_start_offset)[1],
                                static_cast<const ::size_t*>(input_end_offset)[0],
                                static_cast<const ::size_t*>(input_end_offset)[1],
                                static_cast<const ::size_t*>(output_start_offset)[0],
                                static_cast<const ::size_t*>(output_start_offset)[1],
                                size_x, stride_x ) );
    // If needed kernel was not there then its creation failed for some reason
    assert( kit != m_pooling_kernels.end() );

    // Set input and output as args of OpenCL pooling kernel
    int retVal = 0;
    // Output of pooling may be a buffer as well as an image
    if( output->parent->cl_buffer[0] == nullptr ) {
        retVal = ( kit->second )->setArg( 0, *output->parent->cl_image[0] );
    } else {
        retVal = ( kit->second )->setArg( 0, *output->parent->cl_buffer[0] );
    }
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal,  " Error setting OpenCL pooling kernel argument idx: 0 failed with error: " );
    }

    retVal = ( kit->second )->setArg( 1, *input->parent->cl_buffer[0] );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL pooling kernel argument idx: 1 failed with error: ");
    }
    // Length of View scope for Z dimension (depth) for output and input is the same eg.
    // output_view_end_z - output_view_start_z = input_view_end_z - input_view_start_z
    // It just coords can be shifted

    //Z dimension hold view scope
    //X and Y dimensions hold input width divided by stride
    //basically each work item is responsible for computing one window of data.

    //todo, when zero padding will be enabled, make sure this code will be changed to:
    //cl::NDRange global_size( input_width / stride_x, input_height / stride_x , ( static_cast<const ::size_t*>(output_end_offset)[2] - static_cast<const ::size_t*>(output_start_offset)[2] + 1) * (output_batch_end - output_batch_start + 1) );

    //compute NDRange size, dimension is end - start , then we reduce by windows size , divide by stride and add 1
    size_t dim_size_x = input_end_offset[0] - input_start_offset[0] + 1;
    size_t dim_size_y = input_end_offset[1] - input_start_offset[1] + 1;

    cl::NDRange global_size(( (dim_size_x - size_x) / stride_x + 1 ), ( ( dim_size_y - size_x )/ stride_x + 1 ), ( static_cast<const ::size_t*>(output_end_offset)[2] - static_cast<const ::size_t*>(output_start_offset)[2] + 1) * (output_batch_end - output_batch_start + 1) );


    cl::NDRange offset(0,0,output_start_offset[2]);
#if defined(DEBUG)
    // data is  dynamically allocated
    // and pointer to it is passed to as data to callback mechanism
    // after using callback function will free dynamic allocation
    exec_struct *psc = new exec_struct;
    psc->name = "pooling";
    psc->num_fmads = 0; //No theretical value yet
    psc->time_event = new cl::Event;
    retVal = m_queue->enqueueNDRangeKernel( *( kit->second ), offset, global_size, cl::NullRange, 0, psc->time_event );
    psc->time_event->setCallback( CL_COMPLETE, &exec_completed, ( void * )psc );
#else
    retVal = m_queue->enqueueNDRangeKernel( *( kit->second ), offset, global_size, cl::NullRange );
#endif
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR(retVal,
                         " Error executing OpenCL enqueueNDRange for pooling kernel. Call failed with error: ");
    }
}

} //namespace device_gpu
