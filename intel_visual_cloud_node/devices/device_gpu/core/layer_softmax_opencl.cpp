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
#include "../../../common/common.h"
#include "../api_internal/nn_device_interface_0_internal.h"
#include "layers_opencl.h"

#define PREFERRED_WORK_GROUP_SIZE 16 // This value was read from softmax kernels build for ULTs

namespace device_gpu
{
//TODO: Make softmax kernel more optimal
// First each local work_item is reading some of values
static std::string kernelSource = R"( 
//Input dimensions, softmax window size passed via compiler definition
//build options: -DLWS -DNUM_SAMPLES -DNUM_BATCHES
__kernel __attribute__((reqd_work_group_size(LWS, 1, 1))) void softmax(__global float* output,  __global float* input)
{
    unsigned int batch_offset = (get_global_id(0) / NUM_SAMPLES)* NUM_SAMPLES;
    float sum = 0.0f;
    for(unsigned int i = 0; i< NUM_SAMPLES; ++i) {
       sum += exp(input[i + batch_offset]); 
    }
    output[get_global_id(0)] = exp(input[get_global_id(0)])/sum;
}

__kernel __attribute__((reqd_work_group_size(LWS, 1, 1))) void softmax_opt(__global float* output,  __global float* input)
{
    float sum = 0.0f;
    __local float partial_sums[LWS]; 
    unsigned long batch_offset = (get_global_id(0) / (NUM_SAMPLES - REM_WORK))*NUM_SAMPLES;
    unsigned long gid = get_global_id(0) % (NUM_SAMPLES - REM_WORK);
    unsigned long base_idx =  batch_offset + get_local_id(0)*WORK_PER_KERNEL;
    for(unsigned long e = base_idx ; e < base_idx + WORK_PER_KERNEL; ++e) {
        sum += exp(input[e]); 
    }
#if REM_WORK != 0
    if(get_local_id(0) < REM_WORK) {
        unsigned int rem_idx = batch_offset + NUM_SAMPLES - REM_WORK + get_local_id(0);
        sum += exp(input[rem_idx]);
    }
#endif
    partial_sums[get_local_id(0)] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = LWS/2; i > 0; i >>= 1) {
        if(get_local_id(0) < i) {
            partial_sums[get_local_id(0)] += partial_sums[i + get_local_id(0)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    unsigned long gid_batch = gid + batch_offset;
    output[gid_batch] = exp(input[gid_batch])/partial_sums[0];
#if REM_WORK != 0
    if( gid < REM_WORK) {
        unsigned long gidx = gid_batch + NUM_SAMPLES - REM_WORK;
        output[gidx] = exp(input[gidx])/partial_sums[0];
    }
#endif
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
bool operator < ( const softmax_kernel_key &A, const softmax_kernel_key &B )
{
    if( A.m_num_samples < B.m_num_samples )
    {
        return true;
    }

    if( (A.m_num_samples == B.m_num_samples) && (A.m_num_batches < B.m_num_batches))
    {
        return true;
    }

    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: consider loading kernels from Binary
softmax_kernel_key ocl_toolkit::prepare_softmax_kernel(
    uint_least32_t         num_samples,
    uint_least32_t         num_batches )
{
    std::map< softmax_kernel_key, softmax_kernel >::iterator kit;

    // Check if we have desired kernel (corresponding to requested dimensions)
    // Search for kernel in container

    softmax_kernel_key softmax_kernel_key_to_use( num_samples, num_batches );
    kit = m_softmax_kernels.find( softmax_kernel_key_to_use );

    // If we do not have such a kernel...
    if( kit == m_softmax_kernels.end() )
    {
        // ...then make it
        DBG_PRINTF( "compiling softmax kernel\n" );

        // Prepare additional compilation args for building program
        std::string extra_compile_args = " -DNUM_SAMPLES=" + std::to_string( num_samples );

        extra_compile_args += " -DNUM_BATCHES=" + std::to_string( num_batches );
#ifndef DONT_USE_FAST_RELAXED_MATH
        extra_compile_args += " -cl-fast-relaxed-math";
#endif

        // If number of samples is less than SIMD width then run naive version
        std::pair< std::map< softmax_kernel_key, softmax_kernel >::iterator, bool > ret;
        unsigned int                                                                lws         = 0;
        unsigned int                                                                gws         = 0;
        std::string                                                                 kernel_name = "";
        if( num_samples < PREFERRED_WORK_GROUP_SIZE )
        {
            lws         = num_samples;
            gws         = num_samples * num_batches;
            kernel_name = "softmax";
        }
        else
        {
            // other wise max work group size is to be used
            lws = m_max_work_group_size;
            while( ( num_samples / lws == 0 ) && ( lws > PREFERRED_WORK_GROUP_SIZE ) )
            {
                lws >>= 1;
            }
            lws = lws <= ( m_local_mem_size ) / 4 ? lws : ( m_local_mem_size ) / 4;

            // Find Global work size that is multiple of LWS
            // and is no bigger than number of samples
            gws         = num_batches * ( ( num_samples / lws ) * lws );
            kernel_name = "softmax_opt";
        }
        extra_compile_args += " -DLWS=" + std::to_string( lws );
        // Calculate number of work to be done by single kernel in work group
        extra_compile_args += " -DWORK_PER_KERNEL=" + std::to_string( num_samples / lws );

        // Remaining work to be done
        extra_compile_args += " -DREM_WORK=" + std::to_string( num_samples % lws );

        std::vector<std::string> kernels(1, kernelSource);
        softmax_kernel smk( make_kernels( kernels, kernel_name, extra_compile_args ), gws, lws );
        ret = m_softmax_kernels.insert( std::make_pair( softmax_kernel_key_to_use, std::move( smk ) ) );

        // ret.second == false means we are inserting element with key that already exists
        assert( ret.second == true );
    }
    else
    {
        DBG_PRINTF( "reusing existing softmax kernel\n" );
    }

    return softmax_kernel_key_to_use;
}
////////////////////////////////////////////////////////////////////////////////////////////
void ocl_toolkit::softmax( nn_cl_data    *output,
                           nn_cl_data    *input,
                           uint_least32_t num_samples,
                           uint_least32_t num_batches )
{
    // Get Kernel for a job
    std::map< softmax_kernel_key, softmax_kernel >::iterator kit =
        m_softmax_kernels.find( prepare_softmax_kernel( num_samples, num_batches ) );
    // If needed kernel was not there then its creation failed for some reason
    assert( kit != m_softmax_kernels.end() );

    // Set input and output as args of OpenCL softmax kernel
    int retVal = 0;
    retVal = ( kit->second ).m_kernel->setArg( 0, *output->parent->cl_buffer[0] );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL softmax kernel argument idx: 0 failed with error: " );
    }

    retVal = ( kit->second ).m_kernel->setArg( 1, *input->parent->cl_buffer[0] );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL softmax kernel argument idx: 1 failed with error: " );
    }

    // Code for asking of SIMD kernel was compiled for
    //size_t simd_size = 0;
    //( kit->second ).m_kernel->getWorkGroupInfo(m_device,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &simd_size );
    //printf("SIMD Size= %ld\n",simd_size);


    cl::NDRange offset( 0, 0, 0 );

#if defined(DEBUG)
    // data is  dynamically allocated
    // and pointer to it is passed to as data to callback mechanism
    // after using callback function will free dynamic allocation
    exec_struct *psc = new exec_struct;
    psc->name = "softmax"; 
    psc->num_fmads = 0; //No theretical value yet
    psc->time_event = new cl::Event;
    retVal = m_queue->enqueueNDRangeKernel( *( kit->second ).m_kernel, offset,
                                            ( kit->second ).m_gws, ( kit->second ).m_lws,
                                            0, psc->time_event );      //PROFILING
    psc->time_event->setCallback( CL_COMPLETE, &exec_completed, ( void * )psc );
#else
    retVal = m_queue->enqueueNDRangeKernel( *( ( kit->second ).m_kernel ),
                                            offset,
                                            ( kit->second ).m_gws,
                                            ( kit->second ).m_lws );
#endif
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal,
                     " Error executing OpenCL enqueueNDRange for softmax kernel. Call failed with error: " );
    }
}

} //namespace device_gpu
