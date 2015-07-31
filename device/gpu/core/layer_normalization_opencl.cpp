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

#define STRINGIFY(...) #__VA_ARGS__

namespace device_gpu
{
//TODO: Change layout of data so feature maps corressponding pixels are located subsequenetialy
//TODO: make normalization_opt kernel more optimal. Currently it is slower than naive one. nvestigate that
    
const char* normalization = STRINGIFY(
//Input dimensions, normalization area size passed via compiler definition
//build options: -DLWS -DNUM_INPUT_FEATURE_MAPS -DNORMALIZATION_SIZE -DALPHA -DBETA -DK
__kernel void normalization(__global float* output,  __global const float* input, unsigned out_buff_offset, unsigned owpad, unsigned ohpad)
{

    unsigned int delta_offset  = get_global_size(1)*get_global_size(0);
    unsigned int i_read_index  = get_global_id(2) * (NUM_INPUT_FEATURE_MAPS*delta_offset) + get_global_id(1)*get_global_size(0) + get_global_id(0);    
    unsigned int j_offset      = get_global_id(2) * (NUM_INPUT_FEATURE_MAPS*delta_offset) + get_global_id(1)*get_global_size(0) + get_global_id(0);    

    unsigned int delta_write_offset = (get_global_size(1) + ohpad) * (get_global_size(0) + owpad);
    unsigned int i_write_index = out_buff_offset + get_global_id( 2 ) * NUM_INPUT_FEATURE_MAPS* delta_write_offset + get_global_id( 1 )*( get_global_size( 0 ) + owpad ) + get_global_id( 0 );


    //in case we know only 1 batch will be used , use below functions
    //unsigned int i_write_index =  get_global_id(1)*get_global_size(0) + get_global_id(0);    
    //unsigned int j_offset      =  get_global_id(1)*get_global_size(0) + get_global_id(0);    

    float sum = 0.0f;    
    float powsum = 0.0f;
    float val = 0.0f;    
    uint j;
    //gather Sum of first elements, that is needed to compute element 0
    for( j = 0 ; j <= NORMALIZATION_SIZE/2 ; j++ )
    {   
        val = input[j_offset];    
        sum += val*val;     
        j_offset += delta_offset;    
    }
    //element 0
    powsum = pow((float)K + ALPHA*sum,BETA);
    output[i_write_index] = ( input[i_read_index] ) / powsum;

    for( ; j < NORMALIZATION_SIZE; j++ )
    {
        val            = input[j_offset];    
        sum           += val*val;     
        i_write_index += delta_write_offset;
        i_read_index  += delta_offset;
        j_offset      += delta_offset;    
        //using this sum, compute value for those elements
        powsum = pow((float)K + ALPHA*sum,BETA);
        output[i_write_index] = ( input[i_read_index] ) / powsum;
    }

    //we now have NORMALIZATION_SIZE elemenets in sum, we have also programmed NORMALIZATION_SIZE - NORMALIZATION_SIZE/2 elements, do the middle part
    //in which we add new element at the end and we remove first element from the beginning which is NORMALIZATION_SIZE away.
    for( j = NORMALIZATION_SIZE - NORMALIZATION_SIZE/2 ; j < NUM_INPUT_FEATURE_MAPS - NORMALIZATION_SIZE/2 ; j++ )
    {
        val            = input[j_offset];    
        sum           += val*val;     
        val            = input[ j_offset - NORMALIZATION_SIZE  * delta_offset];
        sum           -= val*val;  
        i_write_index += delta_write_offset;
        i_read_index  += delta_offset;
        j_offset      += delta_offset;    
        //now update the value using current sum value
        powsum = pow((float)K + ALPHA*sum,BETA);
        output[i_write_index] = ( input[i_read_index] ) / powsum;
    }

    //now do the last part in which we just remove elements from the sum and there are no new elements
    for( ; j < NUM_INPUT_FEATURE_MAPS ; j++ )
    {
        val            = input[ j_offset - NORMALIZATION_SIZE  * delta_offset];
        sum           -= val*val;  
        i_write_index += delta_write_offset;
        i_read_index  += delta_offset;
        j_offset      += delta_offset;    
        //compute values for last elements
        powsum = pow((float)K + ALPHA*sum,BETA);
        output[i_write_index] = ( input[i_read_index] ) / powsum;
    }
}
);

static std::string kernelSource( normalization );
/*
//leaving old kernel for refference
static std::string kernelSource =
    "\
//Input dimensions, normalization area size passed via compiler definition\n\
//build options: -DLWS -DNUM_INPUT_FEATURE_MAPS -DNORMALIZATION_SIZE -DALPHA -DBETA -DK\n\
#ifndef LWS\n\
__kernel void normalization(__global float* output,  __global float* input)\n\
{\n\
    unsigned int batch_offset_z = (get_global_id(2) / NUM_INPUT_FEATURE_MAPS)* NUM_INPUT_FEATURE_MAPS;\n\
    unsigned int i_write_index = get_global_id(2)*get_global_size(0)*get_global_size(1) + \n\
                                 get_global_id(1)*get_global_size(0) + get_global_id(0);\n\
    unsigned int commencing_bound = max((int)batch_offset_z,(int)((get_global_id(2) ) - NORMALIZATION_SIZE/2));\n\
    unsigned int finishing_bound = min((int)batch_offset_z + NUM_INPUT_FEATURE_MAPS -1, (int)get_global_id(2) + NORMALIZATION_SIZE/2);\n\
    unsigned int j_offset = commencing_bound*get_global_size(0)*get_global_size(1) + \n\
                            get_global_id(1)*get_global_size(0) + get_global_id(0);\n\
    float sum = 0.0f;\n\
    float val = 0.0f;\n\
    unsigned int delta_offset = get_global_size(1)*get_global_size(0);\n\
    for(unsigned int j = commencing_bound; j<= finishing_bound; ++j) {\n\
        val = input[j_offset];\n\
        sum += val*val; \n\
        j_offset += delta_offset;\n\
    }\n\
    sum = pow((float)K + ALPHA*sum,BETA);\n\
    output[i_write_index] = (input[i_write_index])/sum;\n\
}\n\
\n\
#else\n\
__kernel __attribute__((reqd_work_group_size(1, 1, LWS))) void normalization_opt(__global float* output,  __global float* input)\n\
{\n\
    __local float squares[LWS]; \n\
    unsigned int batch_offset_z = (get_global_id(2) / NUM_INPUT_FEATURE_MAPS)* NUM_INPUT_FEATURE_MAPS;\n\
    unsigned int l_offset = (batch_offset_z + get_local_id(2))*get_global_size(0)*get_global_size(1) + get_global_id(1)*get_global_size(0) + get_global_id(0);\n\
    float val = input[l_offset];\n\
    squares[get_local_id(2)] = val*val;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    \n\
    unsigned int i_write_index = get_global_id(2)*get_global_size(0)*get_global_size(1) + \n\
                                 get_global_id(1)*get_global_size(0) + get_global_id(0);\n\
    unsigned int commencing_bound = max(batch_offset_z,(unsigned int)(get_global_id(2) ) - NORMALIZATION_SIZE/2);\n\
    unsigned int finishing_bound = min(batch_offset_z + NUM_INPUT_FEATURE_MAPS -1, (unsigned int)get_global_id(2) + NORMALIZATION_SIZE/2);\n\
    unsigned int j_offset = commencing_bound*get_global_size(0)*get_global_size(1) + \n\
                            get_global_id(1)*get_global_size(0) + get_global_id(0);\n\
    \n\
    float sum = 0.0f;\n\
    unsigned int delta_offset = get_global_size(1)*get_global_size(0);\n\
    for(unsigned int j = commencing_bound; j<= finishing_bound; ++j) {\n\
        sum += squares[j_offset]; \n\
        j_offset += delta_offset;\n\
    }\n\
    sum = pow((float)K + ALPHA*sum,BETA);\n\
    output[i_write_index] = (input[i_write_index])/sum;\n\
}\n\
#endif\n\
";
*/
////////////////////////////////////////////////////////////////////////////////////////////////////
bool operator < ( const normalization_kernel_key &A, const normalization_kernel_key &B )
{
    if( A.m_num_input_feature_maps < B.m_num_input_feature_maps )
    {
        return true;
    }

    if( ( A.m_num_input_feature_maps == B.m_num_input_feature_maps ) && ( A.m_k < B.m_k ) )
    {
        return true;
    }

    if( ( A.m_num_input_feature_maps == B.m_num_input_feature_maps ) && ( A.m_k == B.m_k ) &&
        ( A.m_alpha < B.m_alpha ) )
    {
        return true;
    }

    if( ( A.m_num_input_feature_maps == B.m_num_input_feature_maps ) && ( A.m_k == B.m_k ) &&
        ( A.m_alpha == B.m_alpha ) && ( A.m_beta < B.m_beta ) )
    {
        return true;
    }

    if( ( A.m_num_input_feature_maps == B.m_num_input_feature_maps ) && ( A.m_k == B.m_k ) &&
        ( A.m_alpha == B.m_alpha ) && ( A.m_beta == B.m_beta ) && ( A.m_size < B.m_size ) )
    {
        return true;
    }

    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: consider loading kernels from Binary
normalization_kernel_key ocl_toolkit::prepare_normalization_kernel(
                             const uint_least32_t        num_input_feature_maps,
                             const uint_least32_t        k,
                             const float                 alpha,
                             const float                 beta,
                             const uint_least32_t        size )
{
    std::map< normalization_kernel_key, std::unique_ptr< cl::Kernel>>::iterator kit;

    // Check if we have desired kernel (corresponding to requested dimensions)
    // Search for kernel in container

    normalization_kernel_key normalization_kernel_key_to_use(num_input_feature_maps, k, alpha, beta, size);
    kit = m_normalization_kernels.find( normalization_kernel_key_to_use );

    // If we do not have such a kernel...
    if( kit == m_normalization_kernels.end() )
    {
        // ...then make it
        DBG_PRINTF( "compiling normalization kernel\n" );

        // Prepare additional compilation args for building program
        std::string extra_compile_args = " -DNUM_INPUT_FEATURE_MAPS=" + std::to_string( num_input_feature_maps );

        //extra_compile_args += " -DLWS=" + std::to_string( num_input_feature_maps );

        extra_compile_args += " -DK=" + std::to_string( k );
        extra_compile_args += " -DALPHA=" + std::to_string( alpha );
        extra_compile_args += " -DBETA=" + std::to_string( beta );

        extra_compile_args += " -DNORMALIZATION_SIZE=" + std::to_string( size );
#ifndef DONT_USE_FAST_RELAXED_MATH
        extra_compile_args += " -cl-fast-relaxed-math";
#endif

        // If number of samples is less than SIMD width then run naive version
        std::string kernel_name = "normalization";
        std::vector<std::string> kernels(1, kernelSource);
        std::pair< std::map< normalization_kernel_key, std::unique_ptr< cl::Kernel> >::iterator, bool > ret;
        ret = m_normalization_kernels.insert( std::pair< normalization_kernel_key, std::unique_ptr< cl::Kernel> >( normalization_kernel_key_to_use, make_kernels( kernels, kernel_name, extra_compile_args ) ) );

        // ret.second == false means we are inserting element with key that already exists
        assert( ret.second == true );
    }
    else
    {
        DBG_PRINTF( "reusing existing normalization kernel\n" );
    }

    return normalization_kernel_key_to_use;
}
////////////////////////////////////////////////////////////////////////////////////////////
void ocl_toolkit::normalize( nn_cl_data                 *output,
                             nn_cl_data                 *input,
                             const uint_least32_t        num_batches,
                             const uint_least32_t        num_input_feature_maps,
                             const uint_least32_t        input_feature_map_width,
                             const uint_least32_t        input_feature_map_height,
                             const uint_least32_t        k,
                             const float                 alpha,
                             const float                 beta,
                             const uint_least32_t        size,
                             const NN_NORMALIZATION_MODE mode,
                             const uint_least32_t        output_buffer_size,
                             const uint_least32_t        output_buffer_offset,
                             const uint_least32_t        owpad_for_next_layer,
                             const uint_least32_t        ohpad_for_next_layer
                             )
{
    // Get Kernel for a job
    std::map< normalization_kernel_key, std::unique_ptr< cl::Kernel>>::iterator kit =
        m_normalization_kernels.find( prepare_normalization_kernel( num_input_feature_maps, k, alpha, beta, size) );
    // If needed kernel was not there then its creation failed for some reason
    assert( kit != m_normalization_kernels.end() );

    // Set input and output as args of OpenCL normalization kernel
    int retVal = 0;
    retVal = ( kit->second )->setArg( 0, *output->parent->cl_buffer[0] );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL normalization kernel argument idx: 0 failed with error: " );
    }

    retVal = ( kit->second )->setArg( 1, *input->parent->cl_buffer[0] );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL normalization kernel argument idx: 1 failed with error: " );
    }

    retVal = ( kit->second )->setArg( 2, output_buffer_offset );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL normalization kernel argument idx: 2 failed with error: " );
    }

    retVal = ( kit->second )->setArg( 3, owpad_for_next_layer );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL normalization kernel argument idx: 3 failed with error: " );
    }

    retVal = ( kit->second )->setArg( 4, ohpad_for_next_layer );
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL normalization kernel argument idx: 4 failed with error: " );
    }
    cl::NDRange offset( 0, 0, 0 );
#if defined(DEBUG)
    // data is  dynamically allocated
    // and pointer to it is passed to as data to callback mechanism
    // after using callback function will free dynamic allocation
    exec_struct *psc = new exec_struct;
    psc->name ="normalization" ; 
    psc->num_fmads = 0; //No theretical value yet
    psc->time_event = new cl::Event;

    retVal = m_queue->enqueueNDRangeKernel( *(kit->second),
                                            offset,
                                            cl::NDRange(input_feature_map_width,
                                                        input_feature_map_height,
                                                        num_batches ),
                                            cl::NullRange, 0, psc->time_event );// PROFILING
    psc->time_event->setCallback( CL_COMPLETE, &exec_completed, ( void * )psc );
#else
    retVal = m_queue->enqueueNDRangeKernel( *(kit->second),
                                            offset,
                                            cl::NDRange(input_feature_map_width,
                                                        input_feature_map_height,
                                                        num_batches),
                                            cl::NullRange );
#endif

    //TODO: Enable more optimal normalization kernel
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal,
                     " Error executing OpenCL enqueueNDRange for normalization kernel. Call failed with error: " );
    }

}

} //namespace device_gpu
