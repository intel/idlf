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
#ifndef __LAYER_CONVOLUTION_OPENCL__
#define __LAYER_CONVOLUTION_OPENCL__

// Diagnostic print (empty in release, and printing in Debug)
#if defined(DEBUG)
#   include<cstdio>
#   define DBG_PRINTF(...) printf(__VA_ARGS__)
#else
#   define DBG_PRINTF(...)
#endif

#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
//silence the warnings in newer OCL SDK
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>

#define THROW_ERROR(err_code,...) throw device_gpu::runtime_error(__VA_ARGS__,err_code,__LINE__,__FILE__)

#define NN_ALIGN( a, b )                 ( ( a + b - 1 ) / b * b )
#define BUFFER_SIZE_ALIGNEMENT 4096

struct nn_cl_data;

namespace device_gpu
{
void CL_CALLBACK exec_completed( cl_event e, cl_int status, void *data );

enum class Conversion: unsigned short {NO_CONVERSION = 0, CONVERSION_ZXY_TO_3D = 1 };

//Device GPU exception class
class runtime_error : public std::runtime_error
{
private:
    const std::string  m_err_msg;
    const unsigned int m_except_line;
    const std::string  m_except_file;
public:
    const int32_t m_err_code;

public:
    explicit runtime_error( const char* err_msg, int err_code, unsigned int except_line, const char *except_file )
        : std::runtime_error(""),  m_err_msg( err_msg ), m_err_code( err_code ), m_except_line( except_line ), m_except_file( except_file )
    {}
    // Returning error message (along with line, file and error code)
    std::string what( void )
    {
        return std::string( " NN_Device_GPU ERROR[" ) + m_except_file + std::string( ":" ) + std::to_string(
            m_except_line) + std::string( "]: " ) + m_err_msg + std::to_string(m_err_code);
    }
};


// This is a key for map of convolving kernels
class conv_kernel_key
{
public:
    unsigned int           m_total_output_depth;
    unsigned int           m_total_input_depth;
    unsigned int           m_input_width;
    unsigned int           m_input_height;
    unsigned int           m_input_depth;
    unsigned int           m_input_start_x;
    unsigned int           m_input_start_y;
    unsigned int           m_input_start_z;
    unsigned int           m_output_width;
    unsigned int           m_output_height;
    unsigned int           m_output_depth;
    unsigned int           m_output_start_z;
    unsigned int           m_filter_width;
    unsigned int           m_filter_height;
    unsigned int           m_filter_depth;
    unsigned int           m_num_filters;
    unsigned int           m_stride_x;
    unsigned int           m_stride_y;
    unsigned int           m_output_width_pad;
    unsigned int           m_output_height_pad;
    unsigned int           m_output_buffer_offset;
    unsigned int           m_batch;
    bool                   m_image_as_output;
    NN_ACTIVATION_FUNCTION m_activation_function;
public:
    conv_kernel_key( unsigned int total_out_depth,
             unsigned int total_in_depth,//taby?
                     unsigned int in_width,
                     unsigned int in_height,
                     unsigned int in_depth,
                     unsigned int in_start_x,
                     unsigned int in_start_y,
                     unsigned int in_start_z,
                     unsigned int f_width,
                     unsigned int f_height,
                     unsigned int f_depth,
                     unsigned int nm_filters,
                     unsigned int stride_x,
                     unsigned int stride_y,
                     NN_ACTIVATION_FUNCTION activation_function,
                     unsigned int out_width,
                     unsigned int out_height,
                     unsigned int out_depth,
                     unsigned int out_start_z,
                     unsigned int output_width_pad,
                     unsigned int output_height_pad,
                     unsigned int output_buffer_offset,
                     unsigned int batch,
                     bool image_as_output ) :
        m_total_output_depth( total_out_depth ), m_total_input_depth( total_in_depth ), m_input_width( in_width ), m_input_height( in_height ), m_input_depth( in_depth ),
        m_input_start_x( in_start_x ), m_input_start_y( in_start_y ), m_input_start_z( in_start_z ),
        m_filter_width( f_width ), m_filter_height( f_height ), m_filter_depth( f_depth ),
        m_num_filters( nm_filters ), m_stride_x( stride_x ), m_stride_y( stride_y ),
        m_activation_function( activation_function ),
        m_output_width( out_width ), m_output_height( out_height ), m_output_depth( out_depth ),
        m_output_start_z( out_start_z ), m_output_width_pad( output_width_pad ),
        m_output_height_pad( output_height_pad ),
        m_output_buffer_offset( output_buffer_offset ), m_batch( batch ), m_image_as_output( image_as_output )
    {}
};

struct conv_kernel_variants
{
    std::unique_ptr< cl::Kernel > m_kernel;        // main kernel
    std::unique_ptr< cl::Kernel > m_kernel_resx;   // kernel for residual outputs in x dimension
    std::unique_ptr< cl::Kernel > m_kernel_resy;   // kernel for residual outputs in y dimension

    cl::NDRange                   m_gws;      //< GWS to be used
    cl::NDRange                   m_lws;      //< LWS to be used
    cl::NDRange                   m_offset;   //< EnqueueNDRange offset

    cl::NDRange                   m_gws_resx;      //< GWS to be used by m_kernel_resx
    cl::NDRange                   m_offset_resx;   //< EnqueueNDRange offset for m_kernel_resx

    cl::NDRange                   m_gws_resy;      //< GWS to be used by m_kernel_resy
    cl::NDRange                   m_offset_resy;   //< EnqueueNDRange offset for m_kernel_resy

    std::string                   m_kernel_name;    //< Name of kernel
    uint32_t                      m_batch;

    uint64_t                      m_num_fmads;
    uint64_t                      m_num_fmads_resx;
    uint64_t                      m_num_fmads_resy;

    conv_kernel_variants( conv_kernel_variants && arg )
        : m_kernel(std::move(arg.m_kernel)), m_kernel_resx(std::move(arg.m_kernel_resx)), m_kernel_resy(std::move(arg.m_kernel_resy)),
        m_gws(std::move(arg.m_gws)), m_lws(std::move(arg.m_lws)), m_offset(arg.m_offset),
        m_gws_resx(std::move(arg.m_gws_resx)), m_offset_resx(arg.m_offset_resx),
        m_gws_resy(std::move(arg.m_gws_resy)), m_offset_resy(arg.m_offset_resy),
        m_batch(arg.m_batch), m_kernel_name(std::move(arg.m_kernel_name)),
        m_num_fmads( arg.m_num_fmads ), m_num_fmads_resx( arg.m_num_fmads_resx ), m_num_fmads_resy( arg.m_num_fmads_resy )
    {}

    conv_kernel_variants( std::unique_ptr< cl::Kernel >&& kernel, cl::NDRange gws, cl::NDRange lws, cl::NDRange offset,
                          uint32_t batch, std::string kernel_name) :
        m_kernel(std::move(kernel)), m_gws(std::move(gws)), m_lws(std::move(lws)), m_offset(std::move(offset)),
        m_batch(batch), m_kernel_name(std::move(kernel_name)),
        m_kernel_resx(nullptr), m_gws_resx(cl::NullRange), m_offset_resx(cl::NullRange),
        m_kernel_resy(nullptr), m_gws_resy(cl::NullRange), m_offset_resy(cl::NullRange),
        m_num_fmads( 0 ), m_num_fmads_resx( 0 ), m_num_fmads_resy( 0 )
    {}

    conv_kernel_variants( std::unique_ptr< cl::Kernel >&& kernel, cl::NDRange gws, cl::NDRange lws, cl::NDRange offset,
                          std::unique_ptr< cl::Kernel >&& kernel_resx, cl::NDRange gws_resx, cl::NDRange offset_resx,
                          std::unique_ptr< cl::Kernel >&& kernel_resy, cl::NDRange gws_resy, cl::NDRange offset_resy,
                          uint64_t num_fmads, uint64_t num_fmads_resx, uint64_t num_fmads_resy,
                          uint32_t batch, std::string kernel_name) :
        m_kernel(std::move(kernel)), m_gws(std::move(gws)), m_lws(std::move(lws)), m_offset(std::move(offset)),
        m_kernel_resx(std::move(kernel_resx)), m_gws_resx(std::move(gws_resx)), m_offset_resx(std::move(offset_resx)),
        m_kernel_resy(std::move(kernel_resy)), m_gws_resy(std::move(gws_resy)), m_offset_resy(std::move(offset_resy)),
        m_num_fmads( num_fmads ), m_num_fmads_resx( num_fmads_resx ), m_num_fmads_resy( num_fmads_resy ),
        m_batch(batch), m_kernel_name(std::move(kernel_name))
    {}
};

class arithmetic_kernel_key
{
public:
    Conversion             m_conv_to_perform;
    NN_ARITHMETIC_FUNCTION m_arithmetic_function;
    unsigned int           m_total_input_width;
    unsigned int           m_total_input_height;
    unsigned int           m_total_input_depth;
    unsigned int           m_num_batches;

public:
    arithmetic_kernel_key(Conversion conv_to_perform, NN_ARITHMETIC_FUNCTION arithmetic_function,
                          unsigned int total_input_width, unsigned int total_input_height, unsigned int total_input_depth, unsigned int num_batches) :
        m_conv_to_perform(conv_to_perform), m_arithmetic_function(arithmetic_function),
        m_total_input_width(total_input_width), m_total_input_height(total_input_height), m_total_input_depth(total_input_depth),
        m_num_batches(num_batches)
    {}
};


// This is a key for map of mergeed convolution+max_pooling kernels
class conv_maxpool_kernel_key
{
public:
    unsigned int           m_output_width;
    unsigned int           m_output_height;
    unsigned int           m_input_width;
    unsigned int           m_input_height;
    unsigned int           m_input_depth;
    unsigned int           m_filter_width;
    unsigned int           m_filter_height;
    unsigned int           m_filter_depth;
    unsigned int           m_num_filters;
    unsigned int           m_stride_x;
    unsigned int           m_stride_y;
    unsigned int           m_pool_stride;
    unsigned int           m_pool_window_size;
    NN_ACTIVATION_FUNCTION m_activation_function;

public:
    conv_maxpool_kernel_key(unsigned int in_width, unsigned int in_height, unsigned int in_depth,
        unsigned int f_width, unsigned int f_height, unsigned int f_depth, unsigned int nm_filters,
        unsigned int stride_x, unsigned int stride_y, NN_ACTIVATION_FUNCTION activation_function,
        unsigned int pool_window_size, unsigned int pool_stride, unsigned int output_width, unsigned int output_height ) :
        m_input_width(in_width), m_input_height(in_height), m_input_depth(in_depth),
        m_filter_width(f_width), m_filter_height(f_height), m_filter_depth(f_depth),
        m_num_filters(nm_filters), m_stride_x(stride_x), m_stride_y(stride_y),
        m_activation_function(activation_function), m_pool_window_size(pool_window_size), m_pool_stride(pool_stride), m_output_width(output_width), m_output_height(output_height)

    {}
};

struct conv_maxpool_kernel_variants
{
    std::unique_ptr< cl::Kernel > m_kernel;

    conv_maxpool_kernel_variants(conv_maxpool_kernel_variants && arg)
        : m_kernel(std::move(arg.m_kernel))
    {}

    conv_maxpool_kernel_variants(std::unique_ptr< cl::Kernel >&& kernel) :
        m_kernel(std::move(kernel))
    {}
};

// This is a key for map of pooling kernels
class pooling_kernel_key
{
public:
    unsigned int   m_input_width;
    unsigned int   m_input_height;
    unsigned int   m_input_depth;
    uint_least32_t m_input_start_x;
    uint_least32_t m_input_start_y;
    uint_least32_t m_input_end_x;
    uint_least32_t m_input_end_y;
    uint_least32_t m_output_start_x;
    uint_least32_t m_output_start_y;
    unsigned int   m_input_window_size;
    unsigned int   m_input_window_stride;
    bool           m_image_as_output;

public:
    pooling_kernel_key( unsigned int in_width,
                        unsigned int in_height,
                        unsigned int in_depth,
                        uint_least32_t in_start_x,
                        uint_least32_t in_start_y,
                        uint_least32_t in_end_x,
                        uint_least32_t in_end_y,
                        uint_least32_t out_start_x,
                        uint_least32_t out_start_y,
                        unsigned int in_window_size,
                        unsigned int in_window_stride,
                        bool image_as_output )    :
        m_input_width( in_width ), m_input_height( in_height ),m_input_depth( in_depth ),
        m_input_start_x( in_start_x ),
        m_input_start_y( in_start_y ),
        m_input_end_x( in_end_x ),
        m_input_end_y( in_end_y ),
        m_output_start_x( out_start_x ),
        m_output_start_y( out_start_y ),
        m_input_window_size( in_window_size ),
        m_input_window_stride( in_window_stride ),
        m_image_as_output( image_as_output )
    {}
};

// This is a key for map of pooling kernels
class fully_connected_kernel_key
{
public:
    uint32_t               m_num_inputs; /// number of inputs of given
    uint32_t               m_num_outputs; /// number of outputs given
    uint32_t               m_batch_tile; /// batch tile used by the precompiled kernel
    NN_ACTIVATION_FUNCTION m_activation_function;
    bool                   m_image_as_output;
public:
    fully_connected_kernel_key( uint32_t num_inputs, uint32_t num_outputs, uint32_t batch_tile, NN_ACTIVATION_FUNCTION activation_function, bool image_as_output) :
        m_num_inputs( num_inputs ), m_num_outputs( num_outputs ), m_batch_tile( batch_tile ), m_activation_function( activation_function ), m_image_as_output( image_as_output )  {}
};


struct fully_connected_kernel_variants
{
    std::unique_ptr< cl::Kernel > m_kernel;         // preferred kernel
    std::string                   m_kernel_name;    // Name of kernel
    uint32_t                      m_batch_tile;

    fully_connected_kernel_variants( fully_connected_kernel_variants && arg )
        : m_kernel( std::move( arg.m_kernel ) ), m_batch_tile( arg.m_batch_tile ),
        m_kernel_name( std::move( arg.m_kernel_name ) )
    {}

    fully_connected_kernel_variants( std::unique_ptr< cl::Kernel >&& kernel,
                                      uint32_t batch_tile, std::string kernel_name ) :
        m_kernel( std::move( kernel ) ), m_batch_tile( batch_tile ),
        m_kernel_name( std::move( kernel_name ) )
    {}

};

// This is a key for map of softmax kernels
class softmax_kernel_key
{
public:
    unsigned int m_num_samples;
    unsigned int m_num_batches;
public:
    softmax_kernel_key( unsigned int num_samples, unsigned int num_batches ) :
        m_num_samples( num_samples ), m_num_batches( num_batches )
    {}
};

struct softmax_kernel
{
    std::unique_ptr< cl::Kernel > m_kernel;   // kernel to be used
    cl::NDRange                   m_gws; //< GWS(0) to be used
    cl::NDRange                   m_lws; //< LWS(0) to be used

    softmax_kernel( softmax_kernel && arg )
        : m_kernel( std::move( arg.m_kernel ) ), m_gws( std::move( arg.m_gws )), m_lws( std::move( arg.m_lws ) )
    {}

    softmax_kernel( std::unique_ptr< cl::Kernel >&& kernel, unsigned int gws, unsigned int lws ) :
        m_kernel( std::move( kernel ) ), m_gws( gws ), m_lws( lws )
    {}
};

// This is a key for map of normalization kernels
class normalization_kernel_key
{
public:
    uint_least32_t m_num_input_feature_maps;
    uint_least32_t m_k;
    float          m_alpha;
    float          m_beta;
    uint_least32_t m_size;
public:
    normalization_kernel_key( uint_least32_t num_input_feature_maps, uint_least32_t k,
                        float alpha, float beta, uint_least32_t size ) :
        m_num_input_feature_maps( num_input_feature_maps ), m_k( k ),
        m_alpha( alpha), m_beta( beta), m_size( size )
    {}
};

class norm_linear_single_kernel_key
{
public:
    Conversion   m_conv_to_perform;
    unsigned int m_total_input_width;
    unsigned int m_total_input_height;
    unsigned int m_total_input_depth;
    float        m_coeff_a;
    float        m_coeff_b;
    norm_linear_single_kernel_key( Conversion conv_to_perform,
                                   unsigned int total_input_width,
                                   unsigned int total_input_height,
                                   unsigned int total_input_depth,
                                   float coeff_a,
                                   float coeff_b ) :
        m_conv_to_perform( conv_to_perform ), m_total_input_width( total_input_width ),
        m_total_input_height( total_input_height ), m_total_input_depth( total_input_depth ),
        m_coeff_a( coeff_a ), m_coeff_b( coeff_b ) {}
};

class ocl_toolkit
{
typedef struct exec_struct
{
    std::string name;
    uint64_t num_fmads;
    cl::Event *time_event;
}exect_struct_t;
private:
    std::map< arithmetic_kernel_key, std::unique_ptr< cl::Kernel > >         m_arithmetic_kernels;
    std::map< conv_kernel_key, conv_kernel_variants >                        m_conv_kernels;
    std::map< pooling_kernel_key, std::unique_ptr< cl::Kernel > >            m_pooling_kernels;
    std::map< softmax_kernel_key, softmax_kernel >                           m_softmax_kernels;
    std::map< fully_connected_kernel_key, fully_connected_kernel_variants >  m_fully_connected_kernels;
    std::map< normalization_kernel_key, std::unique_ptr< cl::Kernel > >      m_normalization_kernels;
    std::map< conv_maxpool_kernel_key, conv_maxpool_kernel_variants >        m_conv_maxpool_kernels;
    std::map< norm_linear_single_kernel_key, std::unique_ptr< cl::Kernel > > m_norm_linear_single_kernels;

    std::unique_ptr< cl::Context >      m_context;
    std::unique_ptr< cl::CommandQueue > m_queue;

    cl::Device                          m_device;
    cl_ulong                            m_constant_mem_size;
    cl_ulong                            m_local_mem_size;
    cl_ulong                            m_global_mem_size;
    size_t                              m_max_work_group_size;
    const int                           m_preferred_num_acc = 8;
public:
    cl_ulong         m_max_buffer_size;
    static cl_uint   s_cagf;
    static cl_uint   s_max_compute_units;
    static const int s_fmad_eu_issue_rate;// = 8;   // Taken from Bspec , true for IVB,HSW,BDW
public:

    ocl_toolkit( void );

    bool prepare_buffer( cl::Buffer &clbuff, float *buffer, unsigned int bufSize, cl_mem_flags flags );

    cl::CommandQueue& get_command_queue();
    cl::Context& get_context();

    static void CL_CALLBACK exec_completed( cl_event e, cl_int status, void *data );

    arithmetic_kernel_key prepare_arithmetic_kernel(
        NN_WORKLOAD_DATA_TYPE  output_layout,
        NN_WORKLOAD_DATA_TYPE  input_layout,
        const uint_least32_t   total_input_width,
        const uint_least32_t   total_input_height,
        const uint_least32_t   total_input_depth,
        NN_ARITHMETIC_FUNCTION arithmetic_function,
        unsigned int           num_batches );


    conv_kernel_key prepare_conv_kernel(
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
        uint_least32_t         output_h_pad_for_next_layer);


    uint32_t get_batch(
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
        uint_least32_t         output_h_pad_for_next_layer);

    conv_maxpool_kernel_key prepare_conv_maxpool_kernel(
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
        uint_least32_t         output_h_pad_for_next_layer);

    pooling_kernel_key prepare_pooling_kernel(
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
        uint_least32_t input_window_stride );

    fully_connected_kernel_key prepare_fully_connected_kernel(
        bool                   image_as_output,
        uint_least32_t         num_inputs,
        uint_least32_t         num_outputs,    // number of neurons(neuron outputs)
        uint_least32_t         num_batch,
        NN_ACTIVATION_FUNCTION activation_function );

    softmax_kernel_key prepare_softmax_kernel(
        uint_least32_t num_samples,
        uint_least32_t num_batches );

    norm_linear_single_kernel_key prepare_norm_linear_single_kernel(
        NN_WORKLOAD_DATA_TYPE output_layout,
        NN_WORKLOAD_DATA_TYPE input_layout,
        const uint_least32_t  total_input_width,
        const uint_least32_t  total_input_height,
        const uint_least32_t  total_input_depth,
        float                 coeff_a,
        float                 coeff_b );

    normalization_kernel_key prepare_normalization_kernel(
        const uint_least32_t num_input_feature_maps,
        const uint_least32_t k,
        const float          alpha,
        const float          beta,
        const uint_least32_t size );

    void print_cl_caps( void );

    void finish( void ) { m_queue->finish( ); };
    cl_int flush( void ) { return m_queue->flush( ); };

    void arithmetize( nn_cl_data                 *output,
                      nn_cl_data                 *input,
                      nn_cl_data                 *factor,
                      NN_WORKLOAD_DATA_TYPE      output_layout,
                      NN_WORKLOAD_DATA_TYPE      input_layout,
                      const uint_least32_t       num_input_feature_maps,
                      const uint_least32_t       input_feature_map_width,
                      const uint_least32_t       input_feature_map_height,
                      NN_ARITHMETIC_FUNCTION     arithmetic_function,
                      const uint_least32_t       num_batches);

    void convolve( nn_cl_data            *output,
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
                   uint_least32_t         output_h_pad_for_next_layer );

    void max_pool( nn_cl_data          *output,
                   nn_cl_data          *input,
                   cl::NDRange         &output_start_offset,
                   cl::NDRange         &output_end_offset,
                   cl::NDRange         &input_start_offset,
                   cl::NDRange         &input_end_offset,
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
                   NN_POOLING_MODE      mode );

    void convolve_maxpool( nn_cl_data            *output,
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
                           );

    void softmax( nn_cl_data    *output,
                  nn_cl_data    *input,
                  uint_least32_t num_samples,
                  uint_least32_t num_batches );

    void norm_linear_single(
        nn_cl_data            *output,
        nn_cl_data            *input,
        NN_WORKLOAD_DATA_TYPE output_layout,
        NN_WORKLOAD_DATA_TYPE input_layout,
        const uint_least32_t  num_input_feature_maps,
        const uint_least32_t  input_feature_map_width,
        const uint_least32_t  input_feature_map_height,
        float                 coeff_a,
        float                 coeff_b,
        const uint_least32_t  num_batches );

    void fully_connect( nn_cl_data            *output,
                        nn_cl_data            *input,
                        nn_cl_data            *filter,
                        nn_cl_data            *biases,
                        uint_least32_t         num_outputs,
                        uint_least32_t         input_width,
                        uint_least32_t         input_height,
                        uint_least32_t         input_depth,
                        uint_least32_t         total_num_weights,
                        uint_least32_t         num_batches,
                        NN_ACTIVATION_FUNCTION activation_function );

    void normalize( nn_cl_data                 *output,
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
                    const uint_least32_t         output_buffer_size,
                    const uint_least32_t         output_buffer_offset,
                    const uint_least32_t         output_w_pad_for_next_layer,
                    const uint_least32_t         output_h_pad_for_next_layer
                    );

private:

std::unique_ptr< cl::Kernel > make_kernels( std::vector<std::string> &kernels,
                                                         const std::string kernelName,
                                                         const std::string extra_compile_args );

    std::unique_ptr< cl::Kernel > make_kernels_from_file(const std::string & fileName,
                                                         const std::string kernelName,
                                                         const std::string extra_compile_args);
};

//void *MapBufferContent( unsigned int output_number,
//unsigned int bufSize );

//void UnMapBufferContent( unsigned int output_number,
//void         *mappedPtr );

}
#endif //__LAYER_CONVOLUTION_OPENCL__
