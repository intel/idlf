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
#include <stdexcept>
#include <random>
#include <string>
#include <memory>
#include <malloc.h>
#include "device/gpu/api_internal/nn_device_interface_0_internal.h"
#include "layers_opencl.h"

#define USE_RESIDUAL_CONV_KERNELS

extern std::string conv_kernel1;
extern std::string conv_kernel2a;
extern std::string conv_kernel2b;
extern std::string conv_kernel3a;
extern std::string conv_kernel3b;
extern std::string conv_kernel4;
extern std::string conv_kernel5;
extern std::string conv_kernel6;
extern std::string conv_kernel7;
extern std::string conv_kernel8;
extern std::string conv_kernel9;
extern std::string conv_kernel10;


namespace device_gpu
{

bool operator < ( const conv_kernel_key &A, const conv_kernel_key &B )
{

    if( A.m_total_output_depth < B.m_total_output_depth )
    {
        return true;
    }

    if( (A.m_total_output_depth == B.m_total_output_depth) &&
        (A.m_total_input_depth < B.m_total_input_depth) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) && 
        ( A.m_total_input_depth == B.m_total_input_depth ) && 
        ( A.m_input_width < B.m_input_width ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) && 
        ( A.m_input_width == B.m_input_width ) && 
        ( A.m_input_height < B.m_input_height ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth < B.m_input_depth ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth < B.m_input_depth ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_input_start_x < B.m_input_start_x ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y < B.m_input_start_y ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z < B.m_input_start_z ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z == B.m_input_start_z ) &&
        ( A.m_filter_width < B.m_filter_width ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z == B.m_input_start_z ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height < B.m_filter_height ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z == B.m_input_start_z ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth < B.m_filter_depth ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z == B.m_input_start_z ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth == B.m_filter_depth ) &&
        ( A.m_num_filters < B.m_num_filters ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z == B.m_input_start_z ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth == B.m_filter_depth ) &&
        ( A.m_num_filters == B.m_num_filters ) &&
        ( A.m_stride_x < B.m_stride_x ) )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z == B.m_input_start_z ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth == B.m_filter_depth ) &&
        ( A.m_num_filters == B.m_num_filters ) &&
        ( A.m_stride_x == B.m_stride_x ) &&
        ( A.m_stride_y < B.m_stride_y ) )
    {
        return true;
    }


    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z == B.m_input_start_z ) &&
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


    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        (A.m_input_width == B.m_input_width) &&
        (A.m_input_height == B.m_input_height) &&
        (A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z == B.m_input_start_z ) &&
        (A.m_filter_width == B.m_filter_width) &&
        (A.m_filter_height == B.m_filter_height) &&
        (A.m_filter_depth == B.m_filter_depth) &&
        (A.m_num_filters == B.m_num_filters) &&
        (A.m_stride_x == B.m_stride_x) &&
        (A.m_stride_y == B.m_stride_y) &&
        (A.m_activation_function == B.m_activation_function) &&
        (A.m_output_depth < B.m_output_depth)
        )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        (A.m_input_width == B.m_input_width) &&
        (A.m_input_height == B.m_input_height) &&
        (A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z == B.m_input_start_z ) &&
        (A.m_filter_width == B.m_filter_width) &&
        (A.m_filter_height == B.m_filter_height) &&
        (A.m_filter_depth == B.m_filter_depth) &&
        (A.m_num_filters == B.m_num_filters) &&
        (A.m_stride_x == B.m_stride_x) &&
        (A.m_stride_y == B.m_stride_y) &&
        (A.m_activation_function == B.m_activation_function) &&
        (A.m_output_depth == B.m_output_depth) &&
        (A.m_output_width < B.m_output_width)
        )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        (A.m_input_width == B.m_input_width) &&
        (A.m_input_height == B.m_input_height) &&
        (A.m_input_depth == B.m_input_depth) &&
        ( A.m_input_start_x == B.m_input_start_x ) &&
        ( A.m_input_start_y == B.m_input_start_y ) &&
        ( A.m_input_start_z == B.m_input_start_z ) &&
        (A.m_filter_width == B.m_filter_width) &&
        (A.m_filter_height == B.m_filter_height) &&
        (A.m_filter_depth == B.m_filter_depth) &&
        (A.m_num_filters == B.m_num_filters) &&
        (A.m_stride_x == B.m_stride_x) &&
        (A.m_stride_y == B.m_stride_y) &&
        (A.m_activation_function == B.m_activation_function) &&
        (A.m_output_depth == B.m_output_depth) &&
        (A.m_output_width == B.m_output_width) &&
        (A.m_output_height < B.m_output_height)
        )
    {
        return true;
    }

    if ((A.m_total_output_depth == B.m_total_output_depth) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        (A.m_input_width == B.m_input_width) &&
        (A.m_input_height == B.m_input_height) &&
        (A.m_input_depth == B.m_input_depth) &&
        (A.m_input_start_x == B.m_input_start_x ) &&
        (A.m_input_start_y == B.m_input_start_y ) &&
        (A.m_input_start_z == B.m_input_start_z ) &&
        (A.m_filter_width == B.m_filter_width) &&
        (A.m_filter_height == B.m_filter_height) &&
        (A.m_filter_depth == B.m_filter_depth) &&
        (A.m_num_filters == B.m_num_filters) &&
        (A.m_stride_x == B.m_stride_x) &&
        (A.m_stride_y == B.m_stride_y) &&
        (A.m_activation_function == B.m_activation_function) &&
        (A.m_output_depth == B.m_output_depth) &&
        (A.m_output_width == B.m_output_width) &&
        (A.m_output_height == B.m_output_height) &&
        (A.m_output_start_z < B.m_output_start_z) 
        )
    {
        return true;
    }

    if ((A.m_total_output_depth == B.m_total_output_depth) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        (A.m_input_width == B.m_input_width) &&
        (A.m_input_height == B.m_input_height) &&
        (A.m_input_depth == B.m_input_depth) &&
        (A.m_input_start_x == B.m_input_start_x ) &&
        (A.m_input_start_y == B.m_input_start_y ) &&
        (A.m_input_start_z == B.m_input_start_z ) &&
        (A.m_filter_width == B.m_filter_width) &&
        (A.m_filter_height == B.m_filter_height) &&
        (A.m_filter_depth == B.m_filter_depth) &&
        (A.m_num_filters == B.m_num_filters) &&
        (A.m_stride_x == B.m_stride_x) &&
        (A.m_stride_y == B.m_stride_y) &&
        (A.m_activation_function == B.m_activation_function) &&
        (A.m_output_depth == B.m_output_depth) &&
        (A.m_output_width == B.m_output_width) &&
        (A.m_output_height == B.m_output_height) &&
        (A.m_output_start_z == B.m_output_start_z) &&
        (A.m_output_width_pad < B.m_output_width_pad)
        )
    {
        return true;
    }

    if( (A.m_total_output_depth == B.m_total_output_depth) &&
        (A.m_total_input_depth == B.m_total_input_depth) &&
        (A.m_input_width == B.m_input_width) &&
        (A.m_input_height == B.m_input_height) &&
        (A.m_input_depth == B.m_input_depth) &&
        (A.m_input_start_x == B.m_input_start_x ) &&
        (A.m_input_start_y == B.m_input_start_y ) &&
        (A.m_input_start_z == B.m_input_start_z) &&
        (A.m_filter_width == B.m_filter_width) &&
        (A.m_filter_height == B.m_filter_height) &&
        (A.m_filter_depth == B.m_filter_depth) &&
        (A.m_num_filters == B.m_num_filters) &&
        (A.m_stride_x == B.m_stride_x) &&
        (A.m_stride_y == B.m_stride_y) &&
        (A.m_activation_function == B.m_activation_function) &&
        (A.m_output_depth == B.m_output_depth) &&
        (A.m_output_width == B.m_output_width) &&
        (A.m_output_height == B.m_output_height) &&
        (A.m_output_start_z == B.m_output_start_z) &&
        (A.m_output_width_pad == B.m_output_width_pad) &&
        (A.m_output_height_pad < B.m_output_height_pad)
        )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        ( A.m_total_input_depth == B.m_total_input_depth ) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth == B.m_filter_depth ) &&
        ( A.m_num_filters == B.m_num_filters ) &&
        ( A.m_stride_x == B.m_stride_x ) &&
        ( A.m_stride_y == B.m_stride_y ) &&
        ( A.m_activation_function == B.m_activation_function ) &&
        ( A.m_output_depth == B.m_output_depth ) &&
        ( A.m_output_width == B.m_output_width ) &&
        ( A.m_output_height == B.m_output_height ) &&
        ( A.m_output_width_pad == B.m_output_width_pad ) &&
        ( A.m_output_height_pad == B.m_output_height_pad ) &&
        ( A.m_output_buffer_offset < B.m_output_buffer_offset )
        )
    {
        return true;
    }

    if( ( A.m_total_output_depth == B.m_total_output_depth ) &&
        ( A.m_total_input_depth == B.m_total_input_depth ) &&
        ( A.m_input_width == B.m_input_width ) &&
        ( A.m_input_height == B.m_input_height ) &&
        ( A.m_input_depth == B.m_input_depth ) &&
        ( A.m_filter_width == B.m_filter_width ) &&
        ( A.m_filter_height == B.m_filter_height ) &&
        ( A.m_filter_depth == B.m_filter_depth ) &&
        ( A.m_num_filters == B.m_num_filters ) &&
        ( A.m_stride_x == B.m_stride_x ) &&
        ( A.m_stride_y == B.m_stride_y ) &&
        ( A.m_activation_function == B.m_activation_function ) &&
        ( A.m_output_depth == B.m_output_depth ) &&
        ( A.m_output_width == B.m_output_width ) &&
        ( A.m_output_height == B.m_output_height ) &&
        ( A.m_output_width_pad == B.m_output_width_pad ) &&
        ( A.m_output_height_pad == B.m_output_height_pad ) &&
        ( A.m_output_buffer_offset == B.m_output_buffer_offset  )&&
        ( A.m_batch < B.m_batch ))
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
        ( A.m_activation_function == B.m_activation_function ) &&
        ( A.m_output_depth == B.m_output_depth ) &&
        ( A.m_output_width == B.m_output_width ) &&
        ( A.m_output_height == B.m_output_height ) &&
        ( A.m_output_width_pad == B.m_output_width_pad ) &&
        ( A.m_output_height_pad == B.m_output_height_pad ) &&
        ( A.m_output_buffer_offset == B.m_output_buffer_offset  )&&
        ( A.m_batch == B.m_batch ) &&
        ( A.m_image_as_output < B.m_image_as_output ) )       // I'm comparing bool values here. Should all of this work?
    {
        return true;
    }

    return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: consider loading kernels from Binary
uint32_t ocl_toolkit::get_batch(
    bool                   image_as_output,
    uint_least32_t         output_width,
    uint_least32_t         output_height,
    uint_least32_t         output_start_z,
    uint_least32_t         total_output_depth,
    uint_least32_t         total_input_width,
    uint_least32_t         total_input_height,
    uint_least32_t         total_input_depth,
    uint_least32_t         input_width,
    uint_least32_t         input_height,
    uint_least32_t         input_depth,
    uint_least32_t         input_start_x,
    uint_least32_t         input_start_y,
    uint_least32_t         input_start_z,
    uint_least32_t         filter_width,
    uint_least32_t         filter_height,
    uint_least32_t         filter_depth,
    uint_least32_t         num_filters,
    uint_least32_t         stride_x,
    uint_least32_t         stride_y,
    NN_ACTIVATION_FUNCTION activation_function,
    uint_least32_t         num_batches,
    uint_least32_t         output_buffer_offset,
    uint_least32_t         output_w_pad_for_next_layer,
    uint_least32_t         output_h_pad_for_next_layer
    )
{
    // Check if we have desired kernel (corresponding to requested dimensions)
    // Search for kernel in container

    const auto num_output_maps = num_filters;


    conv_kernel_key conv_kernel_key_to_use( total_output_depth, total_input_depth, input_width, input_height, input_depth, 
                                            input_start_x, input_start_y, input_start_z,
                                            filter_width, filter_height, filter_depth, num_filters,
                                            stride_x, stride_y, activation_function,
                                            output_width, output_height, num_output_maps,
                                            output_start_z, output_w_pad_for_next_layer, output_h_pad_for_next_layer, 
                                            output_buffer_offset, num_batches, image_as_output );

    auto g_kit = m_conv_kernels.find(conv_kernel_key_to_use);

    if( g_kit == m_conv_kernels.end( ) )
    {
        DBG_PRINTF( "find error\n" );
        assert( 0 );
        throw NN_API_STATUS_ERROR_OTHER;
    }

    return g_kit->second.m_batch;
}


conv_kernel_key ocl_toolkit::prepare_conv_kernel(
    bool                   image_as_output,
    uint_least32_t         output_width,
    uint_least32_t         output_height,
    uint_least32_t         output_start_z,
    uint_least32_t         total_output_depth,
    uint_least32_t         total_input_width,
    uint_least32_t         total_input_height,
    uint_least32_t         total_input_depth,
    uint_least32_t         input_width,
    uint_least32_t         input_height,
    uint_least32_t         input_depth,
    uint_least32_t         input_start_x,
    uint_least32_t         input_start_y,
    uint_least32_t         input_start_z,
    uint_least32_t         filter_width,
    uint_least32_t         filter_height,
    uint_least32_t         filter_depth,
    uint_least32_t         num_filters,
    uint_least32_t         stride_x,
    uint_least32_t         stride_y,
    NN_ACTIVATION_FUNCTION activation_function,
    uint_least32_t         num_batches,
    uint_least32_t         output_buffer_offset,
    uint_least32_t         output_w_pad_for_next_layer,
    uint_least32_t         output_h_pad_for_next_layer
    )
{
    // Check if we have desired kernel (corresponding to requested dimensions)
    // Search for kernel in container

    const auto num_output_maps = num_filters;

    conv_kernel_key conv_kernel_key_to_use( total_output_depth, total_input_depth, input_width, input_height, input_depth,
                                            input_start_x, input_start_y, input_start_z,
                                            filter_width, filter_height, filter_depth, num_filters,
                                            stride_x, stride_y, activation_function,
                                            output_width, output_height, num_output_maps,
                                            output_start_z, output_w_pad_for_next_layer, output_h_pad_for_next_layer, 
                                            output_buffer_offset, num_batches, image_as_output );

    auto g_kit = m_conv_kernels.find( conv_kernel_key_to_use );

    // If we do not have such a kernel...
    if( g_kit == m_conv_kernels.end() )
    {
        // ...then make it
        // Prepare additional compilation args for building program
        std::string extra_compile_args =
                              " -DINPUT_WIDTH=" + std::to_string( input_width );
        extra_compile_args += " -DINPUT_HEIGHT=" + std::to_string( input_height );
        extra_compile_args += " -DINPUT_DEPTH=" + std::to_string( input_depth );

        extra_compile_args += " -DTOTAL_INPUT_DEPTH_SIZE=" + std::to_string( total_input_depth );
        extra_compile_args += " -DTOTAL_OUTPUT_DEPTH=" + std::to_string( total_output_depth );
        extra_compile_args += " -DINPUT_START_X=" + std::to_string( input_start_x );
        extra_compile_args += " -DINPUT_START_Y=" + std::to_string( input_start_y );
        extra_compile_args += " -DINPUT_START_Z=" + std::to_string( input_start_z );

        extra_compile_args += " -DOUTPUT_WIDTH=" + std::to_string( output_width );
        extra_compile_args += " -DOUTPUT_HEIGHT=" + std::to_string( output_height );

        extra_compile_args += " -DFILTER_WIDTH=" + std::to_string( filter_width );
        extra_compile_args += " -DFILTER_HEIGHT=" + std::to_string( filter_height );

        extra_compile_args += " -DNUM_FILTERS=" + std::to_string( num_filters );

        extra_compile_args += " -DSTRIDEX=" + std::to_string( stride_x );
        extra_compile_args += " -DSTRIDEY=" + std::to_string( stride_y );

        extra_compile_args += " -DOWPAD=" + std::to_string( output_w_pad_for_next_layer );
        extra_compile_args += " -DOHPAD=" + std::to_string( output_h_pad_for_next_layer );
        extra_compile_args += " -DOUT_BUFF_OFFSET=" + std::to_string(output_buffer_offset);
       
        // Should the output area be a buffer or image 
        if(image_as_output == true) {
            extra_compile_args += " -DIMAGE_AS_OUTPUT";
            // Padding for output is needed only when next layer need it 
            // which is valid when next one is convolution
            // but when we need image as output then next one if FF
            // so having image as ouput and non zero output_buffer_offset 
            // is unsupported case.
            if(output_buffer_offset >= 0 ) {
                THROW_ERROR(1, "Error: output padding in convolution when images are requested is not supported scenario! ");
            }
        }

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
                
        auto local_size = cl::NullRange;
        auto global_size = cl::NullRange;
        cl::NDRange offset = {0,0,output_start_z};
        std::string kernel_name;
        auto batched = 1;
        auto req_simd_size = 0;

#ifdef USE_RESIDUAL_CONV_KERNELS
        auto global_size_resx = cl::NullRange;
        cl::NDRange offset_resx = cl::NullRange;

        auto global_size_resy = cl::NullRange;
        cl::NDRange offset_resy = cl::NullRange;

        uint64_t num_fmads = 0;
        uint64_t num_fmads_resx = 0;
        uint64_t num_fmads_resy = 0;
        std::string extra_compile_args_resx, extra_compile_args_resy;
#endif

        if ((11 == filter_width) && (11== filter_height) && (4 == stride_x) && (4 == stride_y)
             && (56 == output_width) && (56 == output_height) && (num_output_maps % 4 == 0))
        {
            // specific for OverFeat C1
            kernel_name = "convolve_11x11x4_v4x2x4_i";

            global_size = { (num_output_maps / 4) * (output_width / 4),
                            output_height / 2,
                            1 };

            local_size = { 14, 28, 1 };
        }
        else if ((11 == filter_width) && (11 == filter_height) && (4 == stride_x) && (4 == stride_y)
            && (num_output_maps % 16 == 0))
        {
            // specific for AlexNet C1
            kernel_name = "convolve_AlexNet_C1";

            const auto output_block_width = 4;
            const auto output_block_height = 3;
            const auto last_block_width = ( output_width % output_block_width == 0 ) ? output_block_width : output_width % output_block_width;
            const auto last_block_height = ( output_height % output_block_height == 0 ) ? output_block_height : output_height % output_block_height;

            // AlexNet kernels are modified to process entire batch
            batched = num_batches;

            global_size = { (output_width + output_block_width - 1) / output_block_width,
                            (output_height + output_block_height - 1) / output_block_height,
                            num_batches * num_output_maps };

            local_size = { 1, 1, 16 };

            // To know full row and column size (parent->length) we pass this info via padding
            // as total_input_width = IWPAD + input_width(width of view)
            // and total_input_height = IHPAD + input_height(height of view)
            auto input_width_pad = total_input_width - input_width;
            auto input_height_pad = total_input_height - input_height;
            extra_compile_args += " -DIWPAD=" + std::to_string(input_width_pad);
            extra_compile_args += " -DIHPAD=" + std::to_string(input_height_pad);
            extra_compile_args += " -DLAST_BLOCK_WIDTH=" + std::to_string( last_block_width );
            extra_compile_args += " -DLAST_BLOCK_HEIGHT=" + std::to_string( last_block_height );

            // Currently this kernel is designed to run only on SIMD16 version
            req_simd_size = 16;
        }
        else if ((5 == filter_width) && (5 == filter_height) && (1 == stride_x) && (1 == stride_y)
            && (24 == output_width) && (24 == output_height) && (num_output_maps % 4 == 0))
        {
            // specific for OverFeat C2
            kernel_name = "convolve_5x5x1_v4x4x2_i_readInColumns";

            global_size = { (num_output_maps / 4) * (output_width / 2),
                            (output_height / 4),
                            1 };

            local_size = { 12, 6, 1 };
        }
        else if ((5 == filter_width) && (5 == filter_height) && (1 == stride_x) && (1 == stride_y)
                 && (num_output_maps % 16 == 0))
        {
            // specific for AlexNet C2
            kernel_name = "convolve_simd16";

            const auto output_block_width = 6;
            const auto output_block_height = 4;
            const auto simd_size = 16;

            extra_compile_args += " -DSIMD_SIZE=" + std::to_string(simd_size);

            // To know full row and column size (parent->length) we pas this info via padding
            // as total_input_width = IWPAD + input_width(width of view)
            // and total_input_height = IHPAD + input_height(height of view)
            auto input_width_pad = total_input_width - input_width;
            auto input_height_pad = total_input_height - input_height;
            extra_compile_args += " -DIWPAD=" + std::to_string(input_width_pad);
            extra_compile_args += " -DIHPAD=" + std::to_string(input_height_pad);
            // AlexNet kernels are modified to process entire batch
            batched = num_batches;

#ifdef USE_RESIDUAL_CONV_KERNELS
            if( num_batches == 1)
#endif
            {
                const auto last_block_width = ( output_width % output_block_width == 0 ) ? output_block_width : output_width % output_block_width;
                const auto last_block_height = ( output_height % output_block_height == 0 ) ? output_block_height : output_height % output_block_height;
                
                global_size = { (output_width  + output_block_width - 1) / output_block_width,
                                (output_height + output_block_height - 1) / output_block_height,
                                num_batches * num_output_maps };

                local_size = { 1, 1, static_cast<size_t>(simd_size) };

                extra_compile_args += " -DOUT_BLOCK_WIDTH=" + std::to_string(output_block_width);
                extra_compile_args += " -DOUT_BLOCK_HEIGHT=" + std::to_string(output_block_height);

                extra_compile_args += " -DIN_BUFFER_SIZE=" + std::to_string(8);

                extra_compile_args += " -DLAST_BLOCK_WIDTH=" + std::to_string( last_block_width );
                extra_compile_args += " -DLAST_BLOCK_HEIGHT=" + std::to_string( last_block_height );
            }
#ifdef USE_RESIDUAL_CONV_KERNELS
            else
            {
                extra_compile_args_resx = extra_compile_args;
                extra_compile_args_resy = extra_compile_args;

                // *** kernel #1 - main part
                auto in_buffer_size = output_block_height + 4;

                auto last_block_width = output_block_width;
                auto last_block_height = output_block_height;

                global_size = { output_width / output_block_width,
                                output_height / output_block_height,
                                num_batches * num_output_maps };

                local_size = { 1, 1, static_cast<size_t>( simd_size ) };

                extra_compile_args += " -DOUT_BLOCK_WIDTH=" + std::to_string( output_block_width );
                extra_compile_args += " -DOUT_BLOCK_HEIGHT=" + std::to_string( output_block_height );

                extra_compile_args += " -DIN_BUFFER_SIZE=" + std::to_string( in_buffer_size );

                extra_compile_args += " -DLAST_BLOCK_WIDTH=" + std::to_string( last_block_width );
                extra_compile_args += " -DLAST_BLOCK_HEIGHT=" + std::to_string( last_block_height );

                extra_compile_args += " -DWRITE_PADDED_VALUES";

                num_fmads = ( uint64_t ) output_block_width * output_block_height * global_size[0] * global_size[1] * global_size[2] *
                            filter_width * filter_height * filter_depth;

                if( (output_width % output_block_width) != 0 )
                {
                    // *** kernel #2 - residue X
                    auto output_block_width_resx = output_width % output_block_width;

                    // Attention: we make block height larger because block width is quite small
                    // TODO: 8 is hardcoded here, will not work for other output sizes or if output_block_height is not 4
                    auto output_block_height_resx = 8;

                    in_buffer_size = output_block_height_resx + 4;

                    last_block_width = output_block_width_resx;
                    last_block_height = output_block_height_resx;

                    global_size_resx = { 1,
                                         output_height / output_block_height_resx,
                                         num_batches * num_output_maps };
                    offset_resx = { global_size[0], 0, offset[2] };

                    extra_compile_args_resx += " -DOUT_BLOCK_WIDTH=" + std::to_string( output_block_width_resx );
                    extra_compile_args_resx += " -DOUT_BLOCK_HEIGHT=" + std::to_string( output_block_height_resx );
                    extra_compile_args_resx += " -DMASTER_OUT_BLOCK_WIDTH=" + std::to_string( output_block_width );


                    extra_compile_args_resx += " -DIN_BUFFER_SIZE=" + std::to_string( in_buffer_size );

                    extra_compile_args_resx += " -DLAST_BLOCK_WIDTH=" + std::to_string( last_block_width );
                    extra_compile_args_resx += " -DLAST_BLOCK_HEIGHT=" + std::to_string( last_block_height );

                    extra_compile_args_resx += " -DWRITE_PADDED_VALUES";

                    num_fmads_resx = ( uint64_t ) output_block_width_resx * output_block_height_resx * global_size_resx[0] * global_size_resx[1] * global_size_resx[2] *
                        filter_width * filter_height * filter_depth;
                }

                if( ( output_height % output_block_height ) != 0 )
                {
                    // *** kernel #3 - residue Y
                    auto output_block_width_resy = output_block_width;
                    auto output_block_height_resy = output_height % output_block_height;

                    in_buffer_size = output_block_height_resy + 4;

                    last_block_width = ( output_width % output_block_width == 0 ) ? output_block_width : output_width % output_block_width;
                    last_block_height = output_block_height_resy;

                    global_size_resy = { ( output_width + output_block_width - 1 ) / output_block_width,
                                         1,
                                         num_batches * num_output_maps };
                    offset_resy = { 0, global_size[1], offset[2] };

                    extra_compile_args_resy += " -DOUT_BLOCK_WIDTH=" + std::to_string( output_block_width_resy );
                    extra_compile_args_resy += " -DOUT_BLOCK_HEIGHT=" + std::to_string( output_block_height_resy );
                    extra_compile_args_resy += " -DMASTER_OUT_BLOCK_HEIGHT=" + std::to_string( output_block_height );

                    extra_compile_args_resy += " -DIN_BUFFER_SIZE=" + std::to_string( in_buffer_size );

                    extra_compile_args_resy += " -DLAST_BLOCK_WIDTH=" + std::to_string( last_block_width );
                    extra_compile_args_resy += " -DLAST_BLOCK_HEIGHT=" + std::to_string( last_block_height );

                    num_fmads_resy = ( uint64_t ) output_block_width_resy * output_block_height_resy * global_size_resy[0] * global_size_resy[1] * global_size_resy[2] *
                        filter_width * filter_height * filter_depth;
                }
            }
#endif
            // Currently this kernel is designed to run only on SIMD16 version
            req_simd_size = 16;
        }
        else if ((3 == filter_width) && (3 == filter_height) && (1 == stride_x) && (1 == stride_y)
                 && (12 == output_height) && (12 == output_height) && (num_output_maps % 8 == 0))
        {
            // specific for OverFeat C3 C4 C5
            kernel_name = "convolve_3x3x1_v8x3x3_i_readInColumns";

            auto grouping = 8;
            if (num_output_maps <= 512)
                grouping = 4; // performs better

            global_size = { (num_output_maps / grouping) * (output_width / 3),
                            output_height / 3,
                            1 };

            local_size = { 4, 4, 1 };

            extra_compile_args += " -DGROUPING=" + std::to_string(grouping);
            extra_compile_args += " -DLWS_X=" + std::to_string(local_size[0]);
            extra_compile_args += " -DLWS_Y=" + std::to_string(local_size[1]);
        }
        else if ((3 == filter_width) && (3 == filter_height) && (1 == stride_x) && (1 == stride_y)
            && (num_output_maps % 16 == 0))
        {
            // specific for AlexNet C3 C4 C5

            auto output_block_width = 7;
            auto output_block_height = 7;
            kernel_name = "convolve_simd8";
            auto simd_size = req_simd_size =  8;
            
            // AlexNet kernels are modified to process entire batch
            batched = num_batches;

            if( ( num_batches * num_output_maps ) <= 256 && ( num_batches * num_output_maps ) > 128 )
            {
                //experimentally this had better performance
                kernel_name = "convolve_simd16";
                output_block_width  = 5;
                output_block_height = 5;
                req_simd_size = simd_size = 16;
            }
            else if( ( num_batches * num_output_maps ) <= 128 )
            {
                //experimentally this had better performance
                kernel_name = "convolve_simd16";
                output_block_width  = 3;
                output_block_height = 3;
                req_simd_size = simd_size = 16;
            }

            extra_compile_args += " -DSIMD_SIZE=" + std::to_string(simd_size);

#ifdef USE_RESIDUAL_CONV_KERNELS
            if( num_batches == 1)
#endif
            {
                const auto in_buffer_size = output_block_height + 2;

                const auto last_block_width = ( output_width % output_block_width == 0 ) ? output_block_width : output_width % output_block_width;
                const auto last_block_height = ( output_height % output_block_height == 0 ) ? output_block_height : output_height % output_block_height;

                global_size = { ( output_width + output_block_width - 1 ) / output_block_width,
                                ( output_height + output_block_height - 1 ) / output_block_height,
                                num_batches * num_output_maps };

                local_size = { 1, 1, static_cast< size_t >( simd_size ) };

                extra_compile_args += " -DSIMD_SIZE=" + std::to_string( simd_size );

                extra_compile_args += " -DOUT_BLOCK_WIDTH=" + std::to_string( output_block_width );
                extra_compile_args += " -DOUT_BLOCK_HEIGHT=" + std::to_string( output_block_height );

                extra_compile_args += " -DIN_BUFFER_SIZE=" + std::to_string( in_buffer_size );

                extra_compile_args += " -DLAST_BLOCK_WIDTH=" + std::to_string( last_block_width );
                extra_compile_args += " -DLAST_BLOCK_HEIGHT=" + std::to_string( last_block_height );
            }
#ifdef USE_RESIDUAL_CONV_KERNELS
            else
            {
                extra_compile_args_resx = extra_compile_args;
                extra_compile_args_resy = extra_compile_args;

                // *** kernel #1 - main part
                auto in_buffer_size = output_block_height + 2;

                auto last_block_width = output_block_width;
                auto last_block_height = output_block_height;

                global_size = { output_width  / output_block_width,
                                output_height / output_block_height,
                                num_batches * num_output_maps };

                local_size = { 1, 1, static_cast<size_t>(simd_size) };

                extra_compile_args += " -DOUT_BLOCK_WIDTH=" + std::to_string(output_block_width);
                extra_compile_args += " -DOUT_BLOCK_HEIGHT=" + std::to_string(output_block_height);

                extra_compile_args += " -DIN_BUFFER_SIZE=" + std::to_string( in_buffer_size );

                extra_compile_args += " -DLAST_BLOCK_WIDTH=" + std::to_string( last_block_width );
                extra_compile_args += " -DLAST_BLOCK_HEIGHT=" + std::to_string( last_block_height );

                extra_compile_args += " -DWRITE_PADDED_VALUES";

                num_fmads = ( uint64_t ) output_block_width * output_block_height * global_size[0] * global_size[1] * global_size[2] *
                    filter_width * filter_height * filter_depth;

                if( ( output_width % output_block_width ) != 0 )
                {
                    // *** kernel #2 - residue X
                    auto output_block_width_resx = output_width % output_block_width;
                    auto output_block_height_resx = output_block_height;

                    in_buffer_size = output_block_height_resx + 2;

                    last_block_width = output_block_width_resx;
                    last_block_height = output_block_height_resx;

                    global_size_resx = { 1,
                                         output_height / output_block_height_resx,
                                         num_batches * num_output_maps };
                    offset_resx = { global_size[0], 0, offset[2]};

                    extra_compile_args_resx += " -DOUT_BLOCK_WIDTH=" + std::to_string(output_block_width_resx);
                    extra_compile_args_resx += " -DOUT_BLOCK_HEIGHT=" + std::to_string(output_block_height_resx);
                    extra_compile_args_resx += " -DMASTER_OUT_BLOCK_WIDTH=" + std::to_string( output_block_width );

                    extra_compile_args_resx += " -DIN_BUFFER_SIZE=" + std::to_string(in_buffer_size);

                    extra_compile_args_resx += " -DLAST_BLOCK_WIDTH=" + std::to_string(last_block_width);
                    extra_compile_args_resx += " -DLAST_BLOCK_HEIGHT=" + std::to_string(last_block_height);

                    extra_compile_args_resx += " -DWRITE_PADDED_VALUES";

                    num_fmads_resx = ( uint64_t ) output_block_width_resx * output_block_height_resx * global_size_resx[0] * global_size_resx[1] * global_size_resx[2] *
                        filter_width * filter_height * filter_depth;
                }

                if( ( output_height % output_block_height ) != 0 )
                {
                    // *** kernel #3 - residue Y
                    auto output_block_width_resy = output_block_width;
                    auto output_block_height_resy = output_height % output_block_height;

                    in_buffer_size = output_block_height_resy + 2;

                    last_block_width = (output_width % output_block_width == 0) ? output_block_width : output_width % output_block_width;
                    last_block_height = output_block_height_resy;

                    global_size_resy = { (output_width + output_block_width - 1) / output_block_width,
                                         1,
                                         num_batches * num_output_maps };
                    offset_resy = { 0, global_size[1], offset[2] };

                    extra_compile_args_resy += " -DOUT_BLOCK_WIDTH=" + std::to_string(output_block_width_resy);
                    extra_compile_args_resy += " -DOUT_BLOCK_HEIGHT=" + std::to_string(output_block_height_resy);
                    extra_compile_args_resy += " -DMASTER_OUT_BLOCK_HEIGHT=" + std::to_string( output_block_height );

                    extra_compile_args_resy += " -DIN_BUFFER_SIZE=" + std::to_string(in_buffer_size);

                    extra_compile_args_resy += " -DLAST_BLOCK_WIDTH=" + std::to_string(last_block_width);
                    extra_compile_args_resy += " -DLAST_BLOCK_HEIGHT=" + std::to_string(last_block_height);

                    num_fmads_resy = ( uint64_t ) output_block_width_resy * output_block_height_resy * global_size_resy[0] * global_size_resy[1] * global_size_resy[2] *
                        filter_width * filter_height * filter_depth;
                }
            }
#endif

        }
        else if ( (6 == filter_width) && (6 == filter_height) && (1 == stride_x) && (1 == stride_y)
            && ((num_output_maps*output_width / 4) % 256 == 0))
        {
            // specific for OverFeat C6
            kernel_name = "convolve_6x6x1_v4x1_i_readInColumns_batch_loop";

            if (num_batches % 8 == 0)
                batched = 8;
            else if (num_batches % 4 == 0)
                batched = 4;
            else
                batched = 1;
            
            auto grouping2 = 4;
            global_size = { num_output_maps * output_width / grouping2,
                            output_height,
                            1 };

            local_size = { 256, 1, 1 };

            extra_compile_args += " -DBATCH_NUM=" + std::to_string(batched);
            extra_compile_args += " -DGROUP2=" + std::to_string(grouping2);

            extra_compile_args += " -DLWS_X=" + std::to_string(256);
            extra_compile_args += " -DLWS_Y=" + std::to_string(1);
            extra_compile_args += " -DLWS_Z=" + std::to_string(1);
            extra_compile_args += " -DREQD_WORK_GROUP_SIZE";
        }
        else
        {
            global_size = { output_width, output_height, num_output_maps };
            kernel_name = "generic_convolve";
        }


        extra_compile_args += " -DINCLUDE_" + kernel_name;
#ifdef USE_RESIDUAL_CONV_KERNELS
        extra_compile_args_resx += " -DINCLUDE_" + kernel_name;
        extra_compile_args_resy += " -DINCLUDE_" + kernel_name;
#endif
        DBG_PRINTF("compiling convolving kernel: %s\n", kernel_name.c_str());
#if 0
        DBG_PRINTF("compile args: %s\n", extra_compile_args.c_str());
        DBG_PRINTF("GWS: %u %u %u \n", global_size[0], global_size[1], global_size[2]);
        DBG_PRINTF("LWS: %u %u %u \n", local_size[0], local_size[1], local_size[2]);
        DBG_PRINTF( "OFF: %u %u %u \n", offset[0], offset[1], offset[2] );
        DBG_PRINTF("output w h d: %u %u %u \n", output_width, output_height, num_output_maps);

#ifdef USE_RESIDUAL_CONV_KERNELS
        DBG_PRINTF( "compile args resx: %s\n", extra_compile_args_resx.c_str( ) );
        DBG_PRINTF( "GWS resx: %u %u %u \n", global_size_resx[0], global_size_resx[1], global_size_resx[2] );
        DBG_PRINTF( "OFF resx: %u %u %u \n", offset_resx[0], offset_resx[1], offset_resx[2] );

        DBG_PRINTF( "compile args resy: %s\n", extra_compile_args_resy.c_str( ) );
        DBG_PRINTF( "GWS resy: %u %u %u \n", global_size_resy[0], global_size_resy[1], global_size_resy[2] );
        DBG_PRINTF( "OFF resy: %u %u %u \n", offset_resy[0], offset_resy[1], offset_resy[2] );
#endif
#endif
        std::vector<std::string> kernels;
        kernels.push_back(conv_kernel1);
        kernels.push_back(conv_kernel2a);
        kernels.push_back(conv_kernel2b);
        kernels.push_back(conv_kernel3a);
        kernels.push_back(conv_kernel3b);
        kernels.push_back(conv_kernel4);
        kernels.push_back(conv_kernel5);
        kernels.push_back(conv_kernel6);
        kernels.push_back(conv_kernel7);
        kernels.push_back(conv_kernel8);
        kernels.push_back(conv_kernel9);
        kernels.push_back(conv_kernel10);

        std::unique_ptr <conv_kernel_variants> kernel;

#ifdef USE_RESIDUAL_CONV_KERNELS
        if (global_size_resx[0] == 0)
        {
            kernel.reset(new conv_kernel_variants(make_kernels(kernels,
                                                               kernel_name,
                                                               extra_compile_args),
                                                  global_size, local_size, offset, batched,
                                                  kernel_name));
        }
        else
        {
            kernel.reset(new conv_kernel_variants(make_kernels(kernels,
                                                               kernel_name,
                                                               extra_compile_args),
                                                  global_size, local_size, offset, 
                                                  make_kernels(kernels,
                                                               kernel_name,
                                                               extra_compile_args_resx),
                                                  global_size_resx, offset_resx, 
                                                  make_kernels(kernels,
                                                               kernel_name,
                                                               extra_compile_args_resy),
                                                  global_size_resy, offset_resy,
                                                  num_fmads, num_fmads_resx, num_fmads_resy,
                                                  batched, kernel_name));
        }
#else
           kernel.reset(new conv_kernel_variants(make_kernels(kernels,
                                                      kernel_name,
                                                      extra_compile_args),
                                                 global_size, local_size, offset, batched,
                                                 kernel_name));
#endif

        // Check if compiled kernel was made using SIMD size it was designed to be done
        // If that is not the case then start another compilation (diffrent definitions, diffrent kernel)
        // TODO: restart the compilation 
        size_t simd_size = 0;
        auto err = kernel->m_kernel->getWorkGroupInfo(m_device,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,&simd_size);
        if(err != CL_SUCCESS) {
            THROW_ERROR(err, "Unable to get Kernel's CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE!");
        }
        if(((req_simd_size != 0) && (req_simd_size != simd_size)) ) {
            assert( 0 );
            THROW_ERROR(1, "Wrong SIMD size selected for given kernel. Check your driver!");
        }
#ifdef USE_RESIDUAL_CONV_KERNELS
        if( kernel->m_kernel_resx != nullptr )
        {
            err = kernel->m_kernel_resx->getWorkGroupInfo( m_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &simd_size );
            if( err != CL_SUCCESS ) {
                THROW_ERROR( err, "Unable to get Kernel's CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE!" );
            }
            if( ( ( req_simd_size != 0 ) && ( req_simd_size != simd_size ) ) ) {
                assert( 0 );
                THROW_ERROR( 1, "Wrong SIMD size selected for given kernel. Check your driver!" );
            }
        }
        if( kernel->m_kernel_resy != nullptr )
        {
            err = kernel->m_kernel_resy->getWorkGroupInfo(m_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &simd_size);
            if (err != CL_SUCCESS) {
                THROW_ERROR(err, "Unable to get Kernel's CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE!");
            }
            if (((req_simd_size != 0) && (req_simd_size != simd_size))) {
                assert(0);
                THROW_ERROR(1, "Wrong SIMD size selected for given kernel. Check your driver!");
            }
        }
#endif
        auto ret = m_conv_kernels.insert(std::make_pair( conv_kernel_key_to_use, std::move(*kernel)));
        // ret.second == false means we are inserting element with key that already exists
        assert(ret.second == true);

    }
    else
    {
        DBG_PRINTF( "reusing existing %s kernel\n", ( g_kit->second ).m_kernel_name.c_str() );

    }

    return conv_kernel_key_to_use;
}
////////////////////////////////////////////////////////////////////////////////////////////
void ocl_toolkit::convolve( nn_cl_data            *output,
                            nn_cl_data            *input,
                            nn_cl_data            *filter,
                            nn_cl_data            *bias,
                            uint_least32_t         total_output_depth,
                            uint_least32_t         output_width,
                            uint_least32_t         output_height,
                            uint_least32_t         output_depth,
                            uint_least32_t         output_buffer_size,
                            uint_least32_t         output_start_z,
                            uint_least32_t         total_input_width,
                            uint_least32_t         total_input_height,
                            uint_least32_t         total_input_depth,
                            uint_least32_t         input_width,
                            uint_least32_t         input_height,
                            uint_least32_t         input_depth,
                            uint_least32_t         input_start_x,
                            uint_least32_t         input_start_y,
                            uint_least32_t         input_start_z,
                            uint_least32_t         filter_width,
                            uint_least32_t         filter_height,
                            uint_least32_t         filter_depth,
                            uint_least32_t         num_batches,
                            unsigned int           stride_x,
                            unsigned int           stride_y,
                            NN_ACTIVATION_FUNCTION activation_function,
                            uint_least32_t         output_buffer_offset,
                            uint_least32_t         output_w_pad_for_next_layer,
                            uint_least32_t         output_h_pad_for_next_layer )
{
    // Get Kernel for a job
    auto g_kit =
        m_conv_kernels.find( prepare_conv_kernel( output->parent->cl_buffer[0] == nullptr,  // If no buffer exist than use image
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
                                                  filter_width,
                                                  filter_height,
                                                  filter_depth,
                                                  output_depth,
                                                  stride_x,
                                                  stride_y,
                                                  activation_function,
                                                  num_batches,
                                                  output_buffer_offset,
                                                  output_w_pad_for_next_layer,
                                                  output_h_pad_for_next_layer ) );

    // If needed kernel was not there then its creation failed for some reason
    assert( g_kit != m_conv_kernels.end() );

    //TODO: Bunch of outputs, consider how to implment mapping of those buffers
    for (unsigned int batch = 0; batch < num_batches; batch += (g_kit->second).m_batch)
    {
        // Set input, output and filter as args of OpenCL convolve kernel
        int retVal = 0;
        // Output of Convolution may be a buffer as well as an image
        if( output->parent->cl_buffer[0] == nullptr ) {
            retVal = ( g_kit->second ).m_kernel->setArg( 0, *output->parent->cl_image[0] );
        } else {
            retVal = ( g_kit->second ).m_kernel->setArg( 0, *output->parent->cl_buffer[0] );
        }

        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 0 failed with error " );
        }

        retVal = ( g_kit->second ).m_kernel->setArg( 1, *input->parent->cl_subbuffer[0].at(batch) );
        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 1 failed with error: " );
        }

        // TODO: support for multiple weights buffers.
        retVal = ( g_kit->second ).m_kernel->setArg( 2, *filter->parent->cl_buffer[0]);
        if (retVal != CL_SUCCESS)
        {
            THROW_ERROR(retVal, " Error setting OpenCL kernel argument idx: 2 failed with error: ");
        }

        retVal = ( g_kit->second ).m_kernel->setArg( 3, *bias->parent->cl_buffer[0] );

        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 3 failed with error: " );
        } 

        // For buffer as output scenario we need offset in output buffer corresponding 
        // to selected batch we would like to process...
        uint32_t output_buffer_batch_offset = 0;
        if( output->parent->cl_buffer[0] != nullptr ) {
            output_buffer_batch_offset = batch * ( ( output_buffer_size / sizeof( float ) ) / num_batches );
        } else {
            //...for images just batch number is enough as it 
            //will be used as second coord (y) to adress output image 
            output_buffer_batch_offset = batch;
        }
        retVal = ( g_kit->second ).m_kernel->setArg( 4, output_buffer_batch_offset );

        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 4 failed with error: " );
        }

#ifdef USE_RESIDUAL_CONV_KERNELS
        if( ( g_kit->second ).m_kernel_resx != nullptr )
        {
            if( output->parent->cl_buffer[0] == nullptr ) {
                retVal = ( g_kit->second ).m_kernel_resx->setArg( 0, *output->parent->cl_image[0] );
            }
            else {
                retVal = ( g_kit->second ).m_kernel_resx->setArg( 0, *output->parent->cl_buffer[0] );
            }

            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 0 failed with error " );
            }

            retVal = ( g_kit->second ).m_kernel_resx->setArg( 1, *input->parent->cl_subbuffer[0].at( batch ) );
            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 1 failed with error: " );
            }

            // TODO: support for multiple weights buffers.
            retVal = ( g_kit->second ).m_kernel_resx->setArg( 2, *filter->parent->cl_buffer[0] );
            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 2 failed with error: " );
            }

            retVal = ( g_kit->second ).m_kernel_resx->setArg( 3, *bias->parent->cl_buffer[0] );

            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 3 failed with error: " );
            }

            // For buffer as output scenario we need offset in output buffer corresponding 
            // to selected batch we would like to process...
            uint32_t output_buffer_batch_offset = 0;
            if( output->parent->cl_buffer[0] != nullptr ) {
                output_buffer_batch_offset = batch * ( ( output_buffer_size / sizeof( float ) ) / num_batches );
            }
            else {
                //...for images just batch number is enough as it 
                //will be used as second coord (y) to adress output image 
                output_buffer_batch_offset = batch;
            }
            retVal = ( g_kit->second ).m_kernel_resx->setArg( 4, output_buffer_batch_offset );

            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 4 failed with error: " );
            }
        }

        if( ( g_kit->second ).m_kernel_resy != nullptr )
        {
            if( output->parent->cl_buffer[0] == nullptr ) {
                retVal = ( g_kit->second ).m_kernel_resy->setArg( 0, *output->parent->cl_image[0] );
            }
            else {
                retVal = ( g_kit->second ).m_kernel_resy->setArg( 0, *output->parent->cl_buffer[0] );
            }

            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 0 failed with error " );
            }

            retVal = ( g_kit->second ).m_kernel_resy->setArg( 1, *input->parent->cl_subbuffer[0].at( batch ) );
            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 1 failed with error: " );
            }

            // TODO: support for multiple weights buffers.
            retVal = ( g_kit->second ).m_kernel_resy->setArg( 2, *filter->parent->cl_buffer[0] );
            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 2 failed with error: " );
            }

            retVal = ( g_kit->second ).m_kernel_resy->setArg( 3, *bias->parent->cl_buffer[0] );

            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 3 failed with error: " );
            }

            // For buffer as output scenario we need offset in output buffer corresponding 
            // to selected batch we would like to process...
            uint32_t output_buffer_batch_offset = 0;
            if( output->parent->cl_buffer[0] != nullptr ) {
                output_buffer_batch_offset = batch * ( ( output_buffer_size / sizeof( float ) ) / num_batches );
            }
            else {
                //...for images just batch number is enough as it 
                //will be used as second coord (y) to adress output image 
                output_buffer_batch_offset = batch;
            }
            retVal = ( g_kit->second ).m_kernel_resy->setArg( 4, output_buffer_batch_offset );

            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL kernel argument idx: 4 failed with error: " );
            }
        }
#endif

#if defined( DEBUG )
        // data is  dynamically allocated
        // and pointer to it is passed to as data to callback mechanism
        // after using callback function will free dynamic allocation
        
        exec_struct *psc = new exec_struct;
        static auto conv_layer_num = 1;
        psc->name = ( g_kit->second ).m_kernel_name + "(" + std::to_string( conv_layer_num ) + ")";
    
        if( ( g_kit->second ).m_num_fmads != 0 )
        {
            psc->num_fmads = ( g_kit->second ).m_num_fmads;
        }
        else
        {
            psc->num_fmads = ( uint64_t ) ( g_kit->second ).m_batch * output_width * output_height * output_depth
                * filter_width * filter_height * filter_depth;
        }

        psc->time_event = new cl::Event;
        retVal = m_queue->enqueueNDRangeKernel( *( g_kit->second ).m_kernel,
                                                g_kit->second.m_offset,
                                                g_kit->second.m_gws,
                                                g_kit->second.m_lws,
                                                0,
                                                psc->time_event );
        // Number of MADs to be done to perform this operation
        psc->time_event->setCallback(CL_COMPLETE, &exec_completed, psc);

        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error executing OpenCL enqueueNDRange. Call failed with error: " );
        }

#ifdef USE_RESIDUAL_CONV_KERNELS
        if ( (g_kit->second).m_kernel_resx != nullptr)
        {
            exec_struct *psc1 = new exec_struct;
            psc1->name = ( g_kit->second ).m_kernel_name + "(" + std::to_string( conv_layer_num ) + "')";
            
            psc1->num_fmads = ( g_kit->second ).m_num_fmads_resx;

            psc1->time_event = new cl::Event;

            retVal = m_queue->enqueueNDRangeKernel( *( g_kit->second ).m_kernel_resx,
                                                    g_kit->second.m_offset_resx,
                                                    g_kit->second.m_gws_resx,
                                                    g_kit->second.m_lws,
                                                    0,
                                                    psc1->time_event );
            // Number of MADs to be done to perform this operation
            psc1->time_event->setCallback(CL_COMPLETE, &exec_completed, psc1);

            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error executing OpenCL enqueueNDRange. Call failed with error: " );
            }
        }

        if ((g_kit->second).m_kernel_resy != nullptr)
        {
            exec_struct *psc1 = new exec_struct;
            psc1->name = ( g_kit->second ).m_kernel_name + "(" + std::to_string( conv_layer_num ) + "\")";
            
            psc1->num_fmads = ( g_kit->second ).m_num_fmads_resy;

            psc1->time_event = new cl::Event;
    
            retVal = m_queue->enqueueNDRangeKernel( *( g_kit->second ).m_kernel_resy,
                                                    g_kit->second.m_offset_resy,
                                                    g_kit->second.m_gws_resy,
                                                    g_kit->second.m_lws,
                                                    0,
                                                    psc1->time_event );
            // Number of MADs to be done to perform this operation
            psc1->time_event->setCallback(CL_COMPLETE, &exec_completed, psc1);

            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error executing OpenCL enqueueNDRange. Call failed with error: " );
            }
        }
#endif
        conv_layer_num++;

#else
        retVal = m_queue->enqueueNDRangeKernel( *( g_kit->second ).m_kernel, g_kit->second.m_offset, g_kit->second.m_gws, g_kit->second.m_lws );

        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error executing OpenCL enqueueNDRange. Call failed with error: " );
        }

        if( ( g_kit->second ).m_kernel_resx != nullptr )
        {
            retVal = m_queue->enqueueNDRangeKernel( *( g_kit->second ).m_kernel_resx, g_kit->second.m_offset_resx, g_kit->second.m_gws_resx, g_kit->second.m_lws );

            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error executing OpenCL enqueueNDRange. Call failed with error: " );
            }
        }

        if( ( g_kit->second ).m_kernel_resy != nullptr )
        {
            retVal = m_queue->enqueueNDRangeKernel( *( g_kit->second ).m_kernel_resy, g_kit->second.m_offset_resy, g_kit->second.m_gws_resy, g_kit->second.m_lws );

            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error executing OpenCL enqueueNDRange. Call failed with error: " );
            }
        }

#endif

    }
}

} //namespace device_gpu
