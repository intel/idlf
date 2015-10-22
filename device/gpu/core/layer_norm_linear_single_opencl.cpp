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
//TODO:  consider vectorization of this code
// First each local work_item is reading some of values
// TODO: Change a , b to be compile time defines
static std::string kernelSource = R"(
//build options: -DCOEFF_A, -DCOEFF_B
__kernel void norm_linear_single(__global float* output,  __global float* input)
{
    // Calculate offset of data (input and corressponding output) to be processed
    // forth dimension is batch in any LAYOUT so no need to worry about that one
#ifdef CONVERSION_ZXY_TO_3D
    unsigned int z = get_global_id(0) % TOTAL_INPUT_DEPTH;
    unsigned int x = (get_global_id(0) / TOTAL_INPUT_DEPTH) % TOTAL_INPUT_WIDTH;
    unsigned int y =  (get_global_id(0) / (TOTAL_INPUT_DEPTH*TOTAL_INPUT_WIDTH)) % TOTAL_INPUT_HEIGHT;
    unsigned int n =  (get_global_id(0) / (TOTAL_INPUT_WIDTH*TOTAL_INPUT_HEIGHT*TOTAL_INPUT_DEPTH));
    unsigned output_offset =  n*TOTAL_INPUT_WIDTH*TOTAL_INPUT_HEIGHT*TOTAL_INPUT_DEPTH + z*TOTAL_INPUT_WIDTH*TOTAL_INPUT_HEIGHT + y*TOTAL_INPUT_WIDTH + x;
#else
    // No conversion scenario can use same value for input and output offset
    unsigned output_offset = get_global_id(0);
#endif
    output[output_offset] = COEFF_A*input[get_global_id(0)] + COEFF_B;
}
)";
////////////////////////////////////////////////////////////////////////////////////////////////////
bool operator < ( const norm_linear_single_kernel_key &A, const norm_linear_single_kernel_key &B )
{

    if( A.m_conv_to_perform < B.m_conv_to_perform )
    {
        return true;
    }

    if( ( A.m_conv_to_perform == B.m_conv_to_perform ) &&
        ( A.m_total_input_width < B.m_total_input_width ) )
    {
        return true;
    }

    if( ( A.m_conv_to_perform == B.m_conv_to_perform ) &&
        ( A.m_total_input_width == B.m_total_input_width ) &&
        ( A.m_total_input_height < B.m_total_input_height )  )
    {
        return true;
    }

    if( ( A.m_conv_to_perform == B.m_conv_to_perform ) &&
        ( A.m_total_input_width == B.m_total_input_width ) &&
        ( A.m_total_input_height == B.m_total_input_height ) &&
        ( A.m_total_input_depth < B.m_total_input_depth )  )
    {
        return true;
    }


    if( ( A.m_conv_to_perform == B.m_conv_to_perform ) &&
        ( A.m_total_input_width == B.m_total_input_width ) &&
        ( A.m_total_input_height == B.m_total_input_height ) &&
        ( A.m_total_input_depth == B.m_total_input_depth ) &&
        ( A.m_coeff_a < B.m_coeff_a ) )
    {
        return true;
    }

    if(( A.m_conv_to_perform == B.m_conv_to_perform ) &&
       ( A.m_total_input_width == B.m_total_input_width ) &&
       ( A.m_total_input_height == B.m_total_input_height ) &&
       ( A.m_total_input_depth == B.m_total_input_depth ) &&
       ( A.m_coeff_a == B.m_coeff_a ) &&
       ( A.m_coeff_b < B.m_coeff_b ) )
    {
        return true;
    }

    return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: consider loading kernels from Binary
norm_linear_single_kernel_key ocl_toolkit::prepare_norm_linear_single_kernel(
    NN_WORKLOAD_DATA_TYPE output_layout,
    NN_WORKLOAD_DATA_TYPE input_layout,
    const uint_least32_t  total_input_width,
    const uint_least32_t  total_input_height,
    const uint_least32_t  total_input_depth,
    float                 coeff_a,
    float                 coeff_b )
{
    std::map< norm_linear_single_kernel_key, std::unique_ptr< cl::Kernel > >::iterator kit;

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
    norm_linear_single_kernel_key norm_linear_single_kernel_key_to_use(
                conv_to_perform,
                total_input_width,
                total_input_height,
                total_input_depth,
                coeff_a,
                coeff_b );
    kit = m_norm_linear_single_kernels.find( norm_linear_single_kernel_key_to_use );

    // If we do not have such a kernel...
    if( kit == m_norm_linear_single_kernels.end() )
    {
        DBG_PRINTF( "compiling norm_linear_single kernel\n" );

        // Prepare additional compilation args for building program
        std::string extra_compile_args = "";

        switch( conv_to_perform )
        {
        case Conversion::CONVERSION_ZXY_TO_3D:
            extra_compile_args += " -DCONVERSION_ZXY_TO_3D=1";
            break;
        default:
            break;
        }

        extra_compile_args += " -DCOEFF_A=" + std::to_string( coeff_a );
        extra_compile_args += " -DCOEFF_B=" + std::to_string( coeff_b );

        extra_compile_args += " -DTOTAL_INPUT_WIDTH=" + std::to_string( total_input_width );
        extra_compile_args += " -DTOTAL_INPUT_HEIGHT=" + std::to_string( total_input_height );
        extra_compile_args += " -DTOTAL_INPUT_DEPTH=" + std::to_string( total_input_depth );


#ifndef DONT_USE_FAST_RELAXED_MATH
        extra_compile_args += " -cl-fast-relaxed-math";
#endif
        std::string kernel_name = "norm_linear_single";

        std::vector<std::string> kernels(1, kernelSource);


        std::pair< std::map< norm_linear_single_kernel_key, std::unique_ptr< cl::Kernel > >::iterator, bool > ret;
        ret =
            m_norm_linear_single_kernels.insert( std::pair< norm_linear_single_kernel_key,
                                                    std::unique_ptr< cl::Kernel > >( norm_linear_single_kernel_key_to_use,
                                                                                     make_kernels( kernels,
                                                                                                   kernel_name,
                                                                                                   extra_compile_args ) ) );
    }
    else
    {
        DBG_PRINTF( "reusing existing norm_linear_single kernel\n" );
    }

    return norm_linear_single_kernel_key_to_use;
}
////////////////////////////////////////////////////////////////////////////////////////////
void ocl_toolkit::norm_linear_single(
    nn_cl_data            *output,
    nn_cl_data            *input,
    NN_WORKLOAD_DATA_TYPE output_layout,
    NN_WORKLOAD_DATA_TYPE input_layout,
    const uint_least32_t  num_input_feature_maps,
    const uint_least32_t  input_feature_map_width,
    const uint_least32_t  input_feature_map_height,
    float                 coeff_a,
    float                 coeff_b,
    const uint_least32_t  num_batches )
{
    // Get Kernel for a job
    std::map< norm_linear_single_kernel_key, std::unique_ptr< cl::Kernel > >::iterator kit =
        m_norm_linear_single_kernels.find( prepare_norm_linear_single_kernel(
                                               output_layout,
                                               input_layout,
                                               input_feature_map_width,
                                               input_feature_map_height,
                                               num_input_feature_maps,
                                               coeff_a, coeff_b ) );

    // If needed kernel was not there then its creation failed for some reason
    assert( kit != m_norm_linear_single_kernels.end() );

    // Set input and output as args of OpenCL norm_linear_single kernel
    int retVal = 0;
    retVal = (kit->second)->setArg(0, *output->parent->cl_buffer[0]);
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL norm_linear_single kernel argument idx: 0 failed with error: " );
    }

    retVal = (kit->second)->setArg(1, *input->parent->cl_buffer[0]);
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR(retVal, " Error setting OpenCL norm_linear_single kernel argument idx: 1 failed with error: ");
    }

    cl::NDRange offset( 0, 0, 0 );

#if defined(DEBUG)
    // data is  dynamically allocated
    // and pointer to it is passed to as data to callback mechanism
    // after using callback function will free dynamic allocation
    exec_struct *psc = new exec_struct;
    psc->name ="norm_linear_single" ;
    psc->num_fmads  = num_input_feature_maps * input_feature_map_width * input_feature_map_height * num_batches;
    psc->time_event = new cl::Event;
    retVal = m_queue->enqueueNDRangeKernel(*( kit->second ),
                                            offset,
                                            num_input_feature_maps*input_feature_map_width*input_feature_map_height*num_batches,
                                            cl::NullRange, 0, psc->time_event );// PROFILING
    psc->time_event->setCallback( CL_COMPLETE, &exec_completed, ( void * )psc );
#else
    retVal = m_queue->enqueueNDRangeKernel(*( kit->second ),
                                            offset,
                                            num_input_feature_maps*input_feature_map_width*input_feature_map_height*num_batches
                                            );
#endif

    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal,
                     " Error executing OpenCL enqueueNDRange for norm_linear_single kernel. Call failed with error: " );
    }

}

} //namespace device_gpu
