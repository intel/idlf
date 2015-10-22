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
#include <iostream>
#include <fstream>
#include <malloc.h>
#include "device/api/nn_device_interface_0.h"
#include "layers_opencl.h"

namespace device_gpu
{

cl_uint   ocl_toolkit::s_cagf;
cl_uint   ocl_toolkit::s_max_compute_units;
const int ocl_toolkit::s_fmad_eu_issue_rate = 8;   // Taken from Bspec , true for IVB,HSW,BDW

//TODO: Turn exec_completed into functor and following params into ocl_toolkit members
////////////////////////////////////////////////////////////////////////////////////////////////////
void CL_CALLBACK ocl_toolkit::exec_completed( cl_event e, cl_int status, void *data )
{
    int flops = (ocl_toolkit::s_max_compute_units) * (ocl_toolkit::s_fmad_eu_issue_rate); //GT2 Floating point operations per second

    exec_struct *exec_data = static_cast<exec_struct *>(data);

    cl_int   err;
    cl_ulong timeStart = 0;
    cl_ulong timeEnd   = 0;
    err = clGetEventProfilingInfo( e, CL_PROFILING_COMMAND_START, sizeof( timeStart ), &timeStart, 0 );
    err = clGetEventProfilingInfo( e, CL_PROFILING_COMMAND_END, sizeof( timeEnd ), &timeEnd, 0 );
    cl_ulong timeDelta = timeEnd - timeStart;

    static cl_ulong totalTime = 0u;
    totalTime += timeDelta;

    float empirical_efficiency = double(exec_data->num_fmads)/timeDelta;
    auto gpu_freq = ocl_toolkit::s_cagf;
    float theretical_efficiency = gpu_freq * flops / 1000.0f;

    printf( "GPU Efficiency for %25s ( assuming GPU clock fixed to %u MHz ):  %.3f %%,  %.3f[ms]  sum: %.2f[ms]\n",
                exec_data->name.c_str( ), gpu_freq, ( empirical_efficiency / theretical_efficiency ) * 100.0f,
                ( timeDelta / 1000000.0f ), ( totalTime / 1000000.0f ) );

    delete exec_data->time_event;
    delete exec_data;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
ocl_toolkit::ocl_toolkit( void ) : m_constant_mem_size(0), m_local_mem_size(0), m_global_mem_size(0), m_max_work_group_size(0), m_preferred_num_acc(8), m_max_buffer_size(0)
{
    std::vector< cl::Platform > platforms;
    std::vector< cl::Device >   devices;

    cl_int err = 0;
    err =  cl::Platform::get( &platforms );
    if( err != CL_SUCCESS )
    {
        THROW_ERROR(err,"No computing platforms found!\n");
    }

    auto has_requested_extensions = [this](cl::Device& device)
    {
        std::string           device_param_string_value;
        device.getInfo( CL_DEVICE_EXTENSIONS, &device_param_string_value );

        if(device_param_string_value.find("cl_intel_subgroups") == std::string::npos ) {
            device.getInfo( CL_DEVICE_NAME, &device_param_string_value );
            DBG_PRINTF( "   Device: %s does not contain required cl_intel_subgroups extension!\n", device_param_string_value.c_str() );
            return false;
        }
        return true;
    };

    // Get first found GPU Device
    for( std::vector< cl::Platform >::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it )
    {
        if( (plat_it->getDevices( CL_DEVICE_TYPE_GPU, &devices ) == CL_SUCCESS) && has_requested_extensions(devices[0]) )
        {
            // Store chosen device for further usage
            m_device = devices[0];
            print_cl_caps();
            break;
        }
    }

    if( m_device() == nullptr )
    {
        THROW_ERROR( CL_DEVICE_NOT_FOUND, "Error: No suitable GPU OpenCL devices found!\n" );
    }

    // create CL context
    m_context = std::unique_ptr< cl::Context >( new cl::Context( devices[0], NULL, NULL, NULL, &err ) );

    if( err != CL_SUCCESS )
    {
        THROW_ERROR(err, "Error creating OpenCL context ");
    }

    //TODO: create another queue  with profiling property enabled
    // when we have support for it in work_item interface

    // create CL Command Queue
#if defined(DEBUG)
    m_queue = std::unique_ptr< cl::CommandQueue >( new cl::CommandQueue( *m_context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err ) ); // PROFILING
#else
    m_queue = std::unique_ptr< cl::CommandQueue >( new cl::CommandQueue( *m_context, devices[0], 0, &err ) );
#endif

    if( err != CL_SUCCESS )
    {
        THROW_ERROR(err, " Error creating OpenCL command queue " );
    }

}
////////////////////////////////////////////////////////////////////////////////////////////////////
void ocl_toolkit::print_cl_caps(void)
{
    std::string           platform_param_value;
    std::string           device_param_string_value;
    std::vector< size_t > device_param_numeric_values;
    cl_uint               device_param_numeric_value;
    size_t                device_param_size_t_value;
    cl_device_type        device_type;
    cl_bool               device_param_bool_value;

    // Print Device Info
    m_device.getInfo( CL_DEVICE_NAME, &device_param_string_value );
    DBG_PRINTF( "   Device: %s\n", device_param_string_value.c_str() );

    m_device.getInfo( CL_DEVICE_VENDOR, &device_param_string_value );
    DBG_PRINTF( "       vendor: %s\n", device_param_string_value.c_str() );

    m_device.getInfo( CL_DEVICE_TYPE, &device_type );
    DBG_PRINTF( "       type: %d\n", device_type );

    m_device.getInfo( CL_DEVICE_EXTENSIONS, &device_param_string_value );
    DBG_PRINTF( "       extensions: %s\n", device_param_string_value.c_str() );

    m_device.getInfo( CL_DEVICE_VERSION, &device_param_string_value );
    DBG_PRINTF( "       device version: %s\n", device_param_string_value.c_str() );

    m_device.getInfo( CL_DRIVER_VERSION, &device_param_string_value );
    DBG_PRINTF( "       driver version: %s\n", device_param_string_value.c_str() );

    m_device.getInfo( CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &device_param_numeric_value );
    DBG_PRINTF( "       preferred vector width float: %d\n", device_param_numeric_value );

    m_device.getInfo( CL_DEVICE_GLOBAL_MEM_SIZE, &m_global_mem_size );
    DBG_PRINTF( "       global memory size: %lu\n", m_global_mem_size );

    m_device.getInfo( CL_DEVICE_MAX_CLOCK_FREQUENCY, &s_cagf );
    DBG_PRINTF( "       clock: %u\n", s_cagf );

    m_device.getInfo( CL_DEVICE_MAX_COMPUTE_UNITS, &s_max_compute_units );
    DBG_PRINTF( "       compute_units: %u\n", s_max_compute_units );

    m_device.getInfo( CL_DEVICE_LOCAL_MEM_SIZE, &m_local_mem_size );
    DBG_PRINTF( "       local memory size: %lu\n", m_local_mem_size );

    m_device.getInfo( CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &m_constant_mem_size );
    DBG_PRINTF( "       constant memory size: %lu\n", m_constant_mem_size );

    m_device.getInfo( CL_DEVICE_MAX_WORK_ITEM_SIZES, &device_param_numeric_values );
    DBG_PRINTF( "       max_work_item_sizes: %d x %d x %d\n",
            device_param_numeric_values[0],
            device_param_numeric_values[1],
            device_param_numeric_values[2] );

    m_device.getInfo( CL_DEVICE_MAX_WORK_GROUP_SIZE, &m_max_work_group_size );
    DBG_PRINTF( "       max_work_group_size: %d\n", m_max_work_group_size);

    m_device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &m_max_buffer_size);
    DBG_PRINTF( "       max_buffer_size: %d\n", m_max_buffer_size);

    m_device.getInfo( CL_DEVICE_AVAILABLE, &device_param_bool_value );
    DBG_PRINTF( "       availability: %d\n", device_param_bool_value );

    m_device.getInfo( CL_DEVICE_IMAGE2D_MAX_WIDTH, &device_param_size_t_value );
    DBG_PRINTF( "       max image2D width:  %d\n", device_param_size_t_value );

    m_device.getInfo( CL_DEVICE_IMAGE2D_MAX_HEIGHT, &device_param_size_t_value );
    DBG_PRINTF( "       max image2D height: %d\n", device_param_size_t_value );


}
////////////////////////////////////////////////////////////////////////////////////////////////////
std::unique_ptr< cl::Kernel > ocl_toolkit::make_kernels_from_file(const std::string & fileName,
                                                                  const std::string kernelName,
                                                                  const std::string extra_compile_args)
{
    std::ifstream file(fileName.c_str(), std::ios::binary);

    if (!file.good())
    {
        THROW_ERROR(-1, " Error opening file ");
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::end);
    file_size = (size_t)file.tellg() - file_size;

    char* source = new char[file_size+1];

    file.seekg(0, std::ios::beg);
    file.read(source, file_size);
    file.close();

    source[file_size] = '\0';
    std::string kernel_source(source);
    std::vector<std::string> kernels(1, kernel_source);
    auto kernel = make_kernels(kernels,
                               kernelName,
                               extra_compile_args);

    delete[] source;
    return kernel;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
std::unique_ptr< cl::Kernel > ocl_toolkit::make_kernels( std::vector<std::string> &kernels,
                                                         const std::string kernelName,
                                                         const std::string extra_compile_args )
{
    cl_int err = 0;

    //cl::Program::Sources kern_sources( 1, std::make_pair( kernelSource.c_str(), kernelSource.length() + 1 ) );
    cl::Program::Sources kern_sources;
    for(auto& kernel_source : kernels) {
        kern_sources.push_back( std::make_pair( kernel_source.c_str(), kernel_source.length() + 1 ) );
    }

    cl::Program program( *m_context, kern_sources, &err );
    if( err != CL_SUCCESS )
    {
        printf( " Error creating OpenCL program from source: %d\n", err );
        return std::unique_ptr< cl::Kernel >( nullptr );
    }

    std::vector< cl::Device > targetDevices( 1, m_device );
    std::string dev_version = m_device.getInfo< CL_DEVICE_VERSION >();

    std::string buildOptions( "-cl-std=CL" );
    // Parse Device Version string to see what CL is supported.
    // eg. format of string is :"OpenCL major_number.minor_number"
    // get 3 characters eg. "major.minor" and append them to options
    buildOptions.append( dev_version.substr( 7, 3 ) );

    // add extra arguments as compilation options
    buildOptions += extra_compile_args;

    err = program.build( targetDevices, buildOptions.c_str() );
    if( err != CL_SUCCESS )
    {
        printf( " Error Building OpenCL program failed: %d\n", err );
        std::string log;
        err = program.getBuildInfo( m_device, CL_PROGRAM_BUILD_LOG, &log );
        printf( " OpenCL program build log: %s\n", log.c_str() );
        return std::unique_ptr< cl::Kernel >( nullptr );
    }

    std::unique_ptr< cl::Kernel > kernel( new cl::Kernel( program, kernelName.c_str(), &err ) );
    if( err != CL_SUCCESS )
    {
        printf( " Error creating OpenCL kernel : %d\n", err );
        return std::unique_ptr< cl::Kernel >( nullptr );
    }

    return kernel;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
bool ocl_toolkit::prepare_buffer( cl::Buffer &clbuff, float *buffer, unsigned int bufSize, cl_mem_flags flags )
{
    cl_int err = 0;
    clbuff = cl::Buffer( *m_context, flags, bufSize, &buffer[0], &err );

    if( err != CL_SUCCESS )
    {
        printf( "Error creating OpenCL buffer : %d\n", err );
        return false;
    }

    return true;
}

cl::CommandQueue& ocl_toolkit::get_command_queue()
{
    return *m_queue.get();
}

cl::Context& ocl_toolkit::get_context()
{
    return *m_context.get();
}

} //namespace device_gpu
