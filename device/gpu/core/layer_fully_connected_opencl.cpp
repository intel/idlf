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

//TODO: Make fully connected OCL kernel more optimal (try sending buffers as float16 and use casting rather than vload)
//TODO: Make weights using constant memory if they can fit into it

static std::string kernel_source = R"(
//Input dimensions, fully_connected window size passed via compiler definition\
//build options: -DNUM_INPUTS, -DNUM_VEC16_CHUNKS=, -Dactivation_function

#ifdef INCLUDE_fully_connected_v2
__kernel void fully_connected_v2(
#ifdef IMAGE_AS_OUTPUT
    __write_only image2d_t output,
#else
    __global float* output,
#endif
  __global float* input, __global float* weights,  __global float* weights2, __global float* biases)
{
    float dotProd0 = 0.0f;
    
    __global float* processed_neuron_weights;

    if(get_global_id(0) >= WEIGHTS2_NEURON_INDEX ) {
        processed_neuron_weights = weights2 + (get_global_id(0) - WEIGHTS2_NEURON_INDEX) * NUM_INPUTS; 
    } else {
        processed_neuron_weights = weights + get_global_id(0) * NUM_INPUTS; 
    }

    __global float* processed_input_batch =  input + get_global_id(1) * NUM_INPUTS; 

    for(unsigned int weight_idx = 0; weight_idx < NUM_INPUTS; ++weight_idx ) 
    {
        dotProd0 += processed_input_batch[weight_idx]*processed_neuron_weights[weight_idx];
    }
#ifdef IMAGE_AS_OUTPUT 
    write_imagef(output,(int2)(get_global_id(0), get_global_id(1)),activation_function(dotProd0 + biases[get_global_id(0)]));
#else
    unsigned int output_idx = get_global_id(1)*get_global_size(0) + get_global_id(0);
    output[output_idx] = activation_function(dotProd0 + biases[get_global_id(0)]);
#endif

}
#endif //#ifdef INCLUDE_fully_connected_v2

#ifdef INCLUDE_fully_connected_generic
__kernel void fully_connected_generic(__global float* output,  __global float* input, __global float* weights,  __global float* biases)
{
    float16 dotProd16 = (float16)(0.0f);

    __global float* processed_neuron_weights = weights + get_global_id(0) * NUM_INPUTS; 
    __global float* processed_input_batch =  input + get_global_id(1) * NUM_INPUTS; 

    for(unsigned int weight_idx = 0; weight_idx < NUM_VEC16_CHUNKS; ++weight_idx ) 
    {
        dotProd16 += vload16(weight_idx,processed_input_batch)*vload16(weight_idx,processed_neuron_weights);
    }
    float dotProd = dotProd16.s0 + dotProd16.s1 + dotProd16.s2 + dotProd16.s3 + dotProd16.s4 + dotProd16.s5 + dotProd16.s6 + dotProd16.s7 + dotProd16.s8 + dotProd16.s9 + dotProd16.sA + dotProd16.sB + dotProd16.sC + dotProd16.sD + dotProd16.sE + dotProd16.sF;
    unsigned int start_rems = NUM_VEC16_CHUNKS*16;
    for(unsigned int weight_idx = start_rems; weight_idx < NUM_INPUTS; ++weight_idx ) 
    {
        dotProd += processed_input_batch[weight_idx]*processed_neuron_weights[weight_idx];
    }
    unsigned int output_idx = get_global_id(1)*get_global_size(0) + get_global_id(0);
    output[output_idx] = activation_function(dotProd + biases[get_global_id(0)]);
}
#endif // #ifdef INCLUDE_fully_connected_generic

#ifdef INCLUDE_fully_connected_8x8


#ifdef IMAGE_AS_OUTPUT 
#define WRITE_TO_OUTPUT_0(value) write_imagef(output,(int2)(dst_write0_x_coord, dst_write0_y_coord),value); ++dst_write0_y_coord; 
#define WRITE_TO_OUTPUT_1(value) write_imagef(output,(int2)(dst_write1_x_coord, dst_write1_y_coord), value); ++dst_write1_y_coord; 
#else 
#define WRITE_TO_OUTPUT_0(value) dst_write0[ 0 ] = value; dst_write0 += NUM_OUTPUTS;
#define WRITE_TO_OUTPUT_1(value) dst_write1[ 0 ] = value; dst_write1 += NUM_OUTPUTS; 
#endif

// A: batch x inputs
// B: inputs x neurons
// C: batch x neurons
//atile is M rows x K columns.
//btile is K rows x N columns.
//Result ctile is M rows x N columns

//build options: -DNUM_INPUTS, -DNUM_OUTPUTS, -Dactivation_function, -DBATCH_TILE, -DOUTPUT_TILE
// #define BATCH_TILE  32  // provided as a build parameter
// #define OUTPUT_TILE 16  // provided as a build parameter
#define INPUT_TILE 8
__attribute__((reqd_work_group_size(8, 1, 1)))
__kernel void fully_connected_8x8(
#ifdef IMAGE_AS_OUTPUT
    __write_only image2d_t output,
#else
    __global float* output,
#endif
    __read_only image2d_t input,
    __read_only image2d_t weights,
    __global float *biases
    )
{
    const int group_x = get_group_id(0); // 0 .. (num_outputs / 2 -1)
    const int group_y = get_group_id(1); // 0 .. (batch / BATCH_TILE)
    const int local_x = get_local_id(0);

    //Result ctile is M rows x N columns
    //M = 32, we have 1 rows of work-items, so we need 32/1 = 32 results down = 4 x float8
    //N = 16, we have 8 columns of work-items, so we need 16/8 = 2 results across

    float8 blockC00 = 0.0f;
    float8 blockC10 = 0.0f;

#if BATCH_TILE >= 16
    float8 blockC01 = 0.0f;
    float8 blockC11 = 0.0f;
#endif
#if BATCH_TILE >= 24
    float8 blockC02 = 0.0f;
    float8 blockC12 = 0.0f;
#endif
#if BATCH_TILE == 32
    float8 blockC03 = 0.0f;
    float8 blockC13 = 0.0f;
#endif

    //input is directly used as atile.
    //It starts at the left side of input and walks across.
    //atile is M rows x K columns.
    int2    coordA = (int2)( 0, group_y * BATCH_TILE );

    //weights is directly used as btile.
    //It starts at the top of weights and walks down.
    //btile is K rows x N columns.
    int2    coordB = (int2)( ( group_x * OUTPUT_TILE ) * sizeof(uint), 0 );

    //Walk ACROSS input and DOWN weights:
    int w = 0;
    do
    {
#define TRANSPOSE_BLOCK_8( _block, _col )   \
        (float8)( intel_sub_group_shuffle( _block.s0, _col ), \
                  intel_sub_group_shuffle( _block.s1, _col ), \
                  intel_sub_group_shuffle( _block.s2, _col ), \
                  intel_sub_group_shuffle( _block.s3, _col ), \
                  intel_sub_group_shuffle( _block.s4, _col ), \
                  intel_sub_group_shuffle( _block.s5, _col ), \
                  intel_sub_group_shuffle( _block.s6, _col ), \
                  intel_sub_group_shuffle( _block.s7, _col ) );

#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )    \
        {   \
            const float8  acol0 = TRANSPOSE_BLOCK_8( _blockA, 0 );    \
            const float8  acol1 = TRANSPOSE_BLOCK_8( _blockA, 1 );    \
            const float8  acol2 = TRANSPOSE_BLOCK_8( _blockA, 2 );    \
            const float8  acol3 = TRANSPOSE_BLOCK_8( _blockA, 3 );    \
            const float8  acol4 = TRANSPOSE_BLOCK_8( _blockA, 4 );    \
            const float8  acol5 = TRANSPOSE_BLOCK_8( _blockA, 5 );    \
            const float8  acol6 = TRANSPOSE_BLOCK_8( _blockA, 6 );    \
            const float8  acol7 = TRANSPOSE_BLOCK_8( _blockA, 7 );    \
            _result = mad( (float8)(_blockB.s0), acol0, _result );    \
            _result = mad( (float8)(_blockB.s1), acol1, _result );    \
            _result = mad( (float8)(_blockB.s2), acol2, _result );    \
            _result = mad( (float8)(_blockB.s3), acol3, _result );    \
            _result = mad( (float8)(_blockB.s4), acol4, _result );    \
            _result = mad( (float8)(_blockB.s5), acol5, _result );    \
            _result = mad( (float8)(_blockB.s6), acol6, _result );    \
            _result = mad( (float8)(_blockB.s7), acol7, _result );    \
        }

        //Now load btile, which is K rows x N columns
        //K = 8, we have 1 row of work-items, so each work-item must load 8/1 = 8 rows
        //N = 16, we have 8 columns of work-items, so each work-item must load 16/8 = 2 column
        int2    coordBTemp = coordB;

        float8  blockB00 = as_float8( intel_sub_group_block_read8( weights, coordBTemp ) );   

		coordBTemp.x += 8 * sizeof(uint);
		
        //We want to load atile, which is M rows x K columns
        //M = 32, we have 1 row of work-items, so each work-item must load 32/1 = 32 rows
        //K = 8, we have 8 columns of work-items, so each work-item must load 8/8 = 1 column
        int2    coordATemp = coordA;

        float8  blockA00 = as_float8( intel_sub_group_block_read8( input, coordATemp ) );   coordATemp.y += 8;
        MULTIPLY_BLOCKS_8x8( blockC00, blockA00, blockB00 );

#if BATCH_TILE >= 16
        float8  blockA01 = as_float8( intel_sub_group_block_read8( input, coordATemp ) );   coordATemp.y += 8;

        MULTIPLY_BLOCKS_8x8( blockC01, blockA01, blockB00 );
#endif
#if BATCH_TILE >= 24
        float8  blockA02 = as_float8( intel_sub_group_block_read8( input, coordATemp ) );   coordATemp.y += 8;

        MULTIPLY_BLOCKS_8x8( blockC02, blockA02, blockB00 );
#endif

#if BATCH_TILE >= 32
        float8  blockA03 = as_float8( intel_sub_group_block_read8( input, coordATemp ) );   coordATemp.y += 8;

        MULTIPLY_BLOCKS_8x8( blockC03, blockA03, blockB00 );
#endif

#if OUTPUT_TILE == 16
        float8  blockB10 = as_float8( intel_sub_group_block_read8( weights, coordBTemp ) );   

        MULTIPLY_BLOCKS_8x8( blockC10, blockA00, blockB10 );
#if BATCH_TILE >= 16
        MULTIPLY_BLOCKS_8x8( blockC11, blockA01, blockB10 );
#endif
#if BATCH_TILE >= 24
        MULTIPLY_BLOCKS_8x8( blockC12, blockA02, blockB10 );
#endif
#if BATCH_TILE >= 32
        MULTIPLY_BLOCKS_8x8( blockC13, blockA03, blockB10 );
#endif

#endif // #if OUTPUT_TILE == 16

#undef TRANSPOSE_BLOCK_8
#undef MULTIPLY_BLOCKS_8x8

        coordA.x += INPUT_TILE * sizeof(uint);	
        coordB.y += INPUT_TILE;	
        w += INPUT_TILE;
    }
    while( w < NUM_INPUTS );

    
// TODO 1: remember to initialize biases to random numbers in ULT to make sure this is OK
// TODO 2: remember to make sure what is perf impact of reading biases here

    __global float  *bias0 = biases + local_x + ( group_x * ( OUTPUT_TILE ) );

#ifdef IMAGE_AS_OUTPUT
    const unsigned dst_write0_x_coord = local_x + ( group_x * ( OUTPUT_TILE ) );
    unsigned dst_write0_y_coord = group_y * BATCH_TILE;

#if OUTPUT_TILE == 16
    const unsigned dst_write1_x_coord = dst_write0_x_coord  + ( OUTPUT_TILE / 2 );/// Is it 
    unsigned dst_write1_y_coord = dst_write0_y_coord; 
    __global float  *bias1 = bias0 + ( OUTPUT_TILE / 2 );
#endif

#else   // .. writting to buffer as output
    __global float  *dst_write0 = output + local_x + ( group_x * ( OUTPUT_TILE ) ) + ( group_y * BATCH_TILE ) * NUM_OUTPUTS;
#if OUTPUT_TILE == 16
    __global float  *dst_write1 = dst_write0 + ( OUTPUT_TILE / 2 );
    __global float  *bias1 = bias0 + ( OUTPUT_TILE / 2 );
#endif

#endif // #if IMAGE_AS_OUTPUT

    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC00.s0));
#if BATCH_TILE > 1
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC00.s1));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC00.s2));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC00.s3));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC00.s4));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC00.s5));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC00.s6));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC00.s7));
#endif

#if BATCH_TILE >= 16
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC01.s0));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC01.s1));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC01.s2));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC01.s3));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC01.s4));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC01.s5));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC01.s6));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC01.s7));
#endif

#if BATCH_TILE >= 24
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC02.s0));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC02.s1));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC02.s2));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC02.s3));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC02.s4));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC02.s5));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC02.s6));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC02.s7));
#endif

#if BATCH_TILE >= 32
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC03.s0));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC03.s1));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC03.s2));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC03.s3));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC03.s4));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC03.s5));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC03.s6));
    WRITE_TO_OUTPUT_0(activation_function(bias0[ 0 ] + blockC03.s7));
#endif

#if OUTPUT_TILE == 16
    
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC10.s0));
#if BATCH_TILE > 1
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC10.s1));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC10.s2));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC10.s3));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC10.s4));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC10.s5));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC10.s6));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC10.s7));
#endif
#if BATCH_TILE >= 16
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC11.s0));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC11.s1));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC11.s2));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC11.s3));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC11.s4));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC11.s5));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC11.s6));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC11.s7));
#endif
#if BATCH_TILE >= 24
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC12.s0));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC12.s1));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC12.s2));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC12.s3));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC12.s4));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC12.s5));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC12.s6));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC12.s7));
#endif
#if BATCH_TILE >= 32
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC13.s0));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC13.s1));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC13.s2));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC13.s3));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC13.s4));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC13.s5));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC13.s6));
    WRITE_TO_OUTPUT_1(activation_function(bias1[ 0 ] + blockC13.s7));
#endif

#endif // #if OUTPUT_TILE == 16

}
#endif // #ifdef INCLUDE_fully_connected_8x8
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
bool operator < ( const fully_connected_kernel_key &A, const fully_connected_kernel_key &B )
{
    if( A.m_num_inputs < B.m_num_inputs )
    {
        return true;
    }

    if( ( A.m_num_inputs == B.m_num_inputs ) && ( A.m_num_outputs < B.m_num_outputs ) )
    {
        return true;
    }

    if( ( A.m_num_inputs == B.m_num_inputs ) && ( A.m_num_outputs == B.m_num_outputs ) &&
        ( A.m_batch_tile < B.m_batch_tile ) )
    {
        return true;
    }

    if ( ( A.m_num_inputs == B.m_num_inputs ) && ( A.m_num_outputs == B.m_num_outputs ) &&
         ( A.m_batch_tile == B.m_batch_tile ) &&
         ( A.m_activation_function < B.m_activation_function ) )
    {
        return true;
    }

    if ( ( A.m_num_inputs == B.m_num_inputs ) && ( A.m_num_outputs == B.m_num_outputs ) &&
         ( A.m_batch_tile == B.m_batch_tile ) &&
         ( A.m_activation_function == B.m_activation_function ) &&
         ( A.m_image_as_output < B.m_image_as_output ) )
    {
        return true;
    }

    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//TODO: consider loading kernels from Binary
fully_connected_kernel_key ocl_toolkit::prepare_fully_connected_kernel(
    bool                   image_as_output,
    uint_least32_t         num_inputs,
    uint_least32_t         num_outputs,    // number of neurons(neuron outputs)
    uint_least32_t         num_batch,
    NN_ACTIVATION_FUNCTION activation_function)
{
    // Check if we have desired kernel (corresponding to requested dimensions)
    // Search for kernel in container

    auto batch_tile = 0u;
    auto output_tile = 0u;

    auto use_fully_connected_8x8 = [num_inputs, num_outputs, num_batch, this] ( ) {
        return( ( num_inputs % 8 == 0 ) && ( num_outputs % 8 == 0 ) &&
                ( num_batch % 8 == 0 ) &&
                ( num_inputs*num_outputs*sizeof( float ) <= this->m_max_buffer_size ) );
    };
    
    if( use_fully_connected_8x8() )
    {
        if( num_batch % 32 == 0 )
            batch_tile = 32;
        else if( num_batch % 24 == 0 )
            batch_tile = 24;
        else if( num_batch % 16 == 0 )
            batch_tile = 16;
        else if( num_batch % 8 == 0 )
            batch_tile = 8;
        else
            batch_tile = 1;

        if( num_outputs % 16 == 0 )
            output_tile = 16;
        else
            output_tile = 8;
    }

    fully_connected_kernel_key fully_connected_kernel_to_use( num_inputs, num_outputs, 
                                                              batch_tile, 
                                                              activation_function,
                                                              image_as_output);

    auto kit = m_fully_connected_kernels.find( fully_connected_kernel_to_use );

    // If we do not have such a kernel...
    if( kit == m_fully_connected_kernels.end() )
    {
        // ...then make it
        // Prepare additional compilation args for building program
        std::string extra_compile_args = " -DNUM_INPUTS=" + std::to_string(num_inputs);

        extra_compile_args += " -DNUM_OUTPUTS=" + std::to_string(num_outputs);

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

        // Should the output area be a buffer or image 
        if(image_as_output == true) {
            extra_compile_args += " -DIMAGE_AS_OUTPUT";
        }
#ifndef DONT_USE_FAST_RELAXED_MATH
        extra_compile_args += " -cl-fast-relaxed-math";
#endif

        // If total number of weights is bigger than single allocation size
        //  then we need to split weights into two CL buffers
        std::string kernel_name;
        if(num_inputs*num_outputs*sizeof(float) > m_max_buffer_size) {
            // Determine which neuron's weights are to be taken from second buffer with weights
            extra_compile_args += " -DWEIGHTS2_NEURON_INDEX=" + std::to_string(num_outputs/2);
            kernel_name = "fully_connected_v2";
        }
        else if( use_fully_connected_8x8( ) )
        {
            kernel_name = "fully_connected_8x8";

            extra_compile_args += " -DBATCH_TILE=" + std::to_string( batch_tile );
            extra_compile_args += " -DOUTPUT_TILE=" + std::to_string( output_tile );

        }
        else{
            extra_compile_args += " -DNUM_VEC16_CHUNKS=" + std::to_string( num_inputs / 16 );
            kernel_name = "fully_connected_generic";
        }

        extra_compile_args += " -DINCLUDE_" + kernel_name;
        
        DBG_PRINTF( "compiling fully_connected kernel: %s \n", kernel_name.c_str());
        //DBG_PRINTF( "compile args: %s\n", extra_compile_args.c_str( ) );
        
        std::vector<std::string> kernels(1, kernel_source);
        fully_connected_kernel_variants kernel( make_kernels( kernels,
                                                              kernel_name.c_str(),
                                                              extra_compile_args ),
                                                              batch_tile,
                                                              kernel_name );

        auto ret = m_fully_connected_kernels.insert( std::make_pair( fully_connected_kernel_to_use, std::move( kernel ) ) );


        // ret.second == false means we are inserting element with key that already exists
        assert( ret.second == true );
    }
    else
    {
        DBG_PRINTF( "reusing existing %s kernel\n", (kit->second).m_kernel_name.c_str() );
    }

    return fully_connected_kernel_to_use;
}
////////////////////////////////////////////////////////////////////////////////////////////
void ocl_toolkit::fully_connect( nn_cl_data            *output,
                                 nn_cl_data            *input,
                                 nn_cl_data            *filter,
                                 nn_cl_data            *biases,
                                 uint_least32_t         num_outputs,    // number of neurons(neuron outputs)
                                 uint_least32_t         input_width,
                                 uint_least32_t         input_height,
                                 uint_least32_t         input_depth,
                                 uint_least32_t         total_num_weights,      
                                 uint_least32_t         num_batches,
                                 NN_ACTIVATION_FUNCTION activation_function )
{
    auto num_inputs = input_width*input_height*input_depth;
    // Get Kernel for a job
    auto kit =
        m_fully_connected_kernels.find( prepare_fully_connected_kernel( 
                            output->parent->cl_buffer[0] == nullptr,  // If no buffer exist than use image
                            num_inputs, num_outputs, num_batches, activation_function));
    // If needed kernel was not there then its creation failed for some reason
    assert( kit != m_fully_connected_kernels.end() );

    // Set input and output as args of OpenCL fully_connected kernel
    int retVal = 0;
    // Output of Convolution may be a buffer as well as an image
    if( output->parent->cl_buffer[0] == nullptr ) {
        retVal = ( kit->second ).m_kernel->setArg( 0, *output->parent->cl_image[0] );
    } else {
        retVal = ( kit->second ).m_kernel->setArg( 0, *output->parent->cl_buffer[0] );
    }
    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR( retVal, " Error setting OpenCL fully_connected kernel argument idx: 0 failed with error: " );
    }

    cl::NDRange offset( 0, 0, 0 );
    cl::NDRange global_size = cl::NullRange;
    cl::NDRange local_size = cl::NullRange;

    if( ( kit->second ).m_kernel_name == "fully_connected_8x8" )
    {
        retVal = ( kit->second ).m_kernel->setArg( 1, *input->parent->cl_image[0] );
        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error setting OpenCL fully_connected kernel argument idx: 1 failed with error: " );
        }

        retVal = ( kit->second ).m_kernel->setArg( 2, *filter->parent->cl_image[0] );
        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error setting OpenCL fully_connected kernel argument idx: 2 failed with error: " );
        }

        retVal = (kit->second).m_kernel->setArg(3, *biases->parent->cl_buffer[0]);
        if (retVal != CL_SUCCESS)
        {
            THROW_ERROR(retVal, " Error setting OpenCL fully_connected kernel argument idx: 3 failed with error: ");
        }

        global_size = { num_outputs / ( ( num_outputs % 16 == 0 ) ? 2 : 1 ), num_batches / ( kit->second ).m_batch_tile };
        local_size = { 8, 1, 1 };

    }
    else 
    {
        // Set input and output as args of OpenCL fully_connected kernel
        retVal = ( kit->second ).m_kernel->setArg( 1, *input->parent->cl_buffer[0] );
        if( retVal != CL_SUCCESS )
        {
            THROW_ERROR( retVal, " Error setting OpenCL fully_connected kernel argument idx: 1 failed with error: " );
        }

        if( total_num_weights*sizeof( float ) > m_max_buffer_size ) 
        {
            retVal = ( kit->second ).m_kernel->setArg( 2, *filter->parent->cl_buffer[0]);
            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL fully_connected kernel argument idx: 2 failed with error: " );
            }

            retVal = ( kit->second ).m_kernel->setArg( 3, *filter->parent->cl_buffer[1] );
            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL fully_connected kernel argument idx: 3 failed with error: " );
            }

            retVal = (kit->second).m_kernel->setArg(4, *biases->parent->cl_buffer[0]);
            if (retVal != CL_SUCCESS)
            {
                THROW_ERROR(retVal, " Error setting OpenCL fully_connected kernel argument idx: 4 failed with error: ");
            }
        }
        else
        {
            retVal = ( kit->second ).m_kernel->setArg( 2, *filter->parent->cl_buffer[0] );
            if( retVal != CL_SUCCESS )
            {
                THROW_ERROR( retVal, " Error setting OpenCL fully_connected kernel argument idx: 2 failed with error: " );
            }

            retVal = (kit->second).m_kernel->setArg(3, *biases->parent->cl_buffer[0]);
            if (retVal != CL_SUCCESS)
            {
                THROW_ERROR(retVal, " Error setting OpenCL fully_connected kernel argument idx: 3 failed with error: ");
            }
        }
        
        global_size = { num_outputs, num_batches };
    }
    
    //DBG_PRINTF("GWS: %u %u \n", global_size[0], global_size[1], 1);
    
#if defined(DEBUG)
    // data is  dynamically allocated
    // and pointer to it is passed to as data to callback mechanism
    // after using callback function will free dynamic allocation
    exec_struct *psc = new exec_struct;
    psc->name = ( kit->second ).m_kernel_name;
    psc->num_fmads = num_inputs*num_outputs*num_batches;
    psc->time_event = new cl::Event;
    retVal = m_queue->enqueueNDRangeKernel( *( kit->second ).m_kernel, offset, global_size, local_size, 0, psc->time_event ); //PROFILING
    psc->time_event->setCallback( CL_COMPLETE, &exec_completed, ( void * )psc );
#else
    retVal = m_queue->enqueueNDRangeKernel( *( kit->second ).m_kernel, offset, global_size, local_size);
#endif

    if( retVal != CL_SUCCESS )
    {
        THROW_ERROR(retVal, 
                         " Error executing OpenCL enqueueNDRange for fully_connected kernel. Call failed with error: " );
    }
}

} //namespace device_gpu
