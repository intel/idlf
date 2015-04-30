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

#include <random>
#include <cassert>
#include <malloc.h>
#include <algorithm>
#include "common.h"
/////////////////////////////////////////////////////////////////////////////////////////////////////
// Create huge buffer that will contain all input
void generate_input_data( float *        &buffer,
                          uint_least32_t input_width,
                          uint_least32_t input_height,
                          uint_least32_t input_depth,
                          uint_least32_t num_inputs )
{
    std::uniform_real_distribution< float > rand_val( 0.0f, 1.0f );
    std::random_device rd;

    // Sanity check
    assert( buffer == nullptr );

    uint_least32_t total_size = input_width * input_height * input_depth * num_inputs * sizeof( float );
    // Memory need to be aligned to page , so that OCL driver not to
    // duplicate buffer data by creating aligned copy
#ifdef __linux__
    buffer = ( float * )memalign( page_alignment, total_size );
#else
    buffer = ( float * )_aligned_malloc( total_size, page_alignment );
#endif //__linux__

    for( uint_least32_t i = 0; i < total_size / sizeof( float ); ++i )
    {
        buffer[i] = rand_val( rd );
        //buffer[i] = i;
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void generate_filter_data( float *        &filters,
                           uint_least32_t filter_width,
                           uint_least32_t filter_height,
                           uint_least32_t filter_depth,
                           uint_least32_t num_filters )
{
    std::uniform_real_distribution< float > rand_val( 0.0f, 1.0f );
    std::random_device rd;

    unsigned int total_size = filter_width * filter_height * filter_depth * num_filters *
                              sizeof( float );

    // All filters are set in memory in sequence order eg.
    // First replicated filter then second replicated filter etc.
#ifdef __linux__
    filters = ( float * )memalign( page_alignment, total_size );
#else
    filters = ( float * )_aligned_malloc( total_size, page_alignment );
#endif         //__linux__
    // copy and multiply given filter data so that we have
    // pointer to buffer containing multiplied filter eg.
    // w0,w0,w0,w0,w0,w1,w1,w1,w1,w2,w2,w2,w2...
    //float tmp = 1.0;
    for( unsigned int i = 0; i < total_size / sizeof( float ); ++i )
    {
        float tmp = rand_val( rd );
        filters[i] = tmp;
    }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float mytanh(float val)
{
    return tanh(val);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float relu(float val)
{
    return std::max(0.0f,val);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float none(float val)
{
    return val;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float softplus( float val )
{
    return logf( 1.0f + expf( val ) );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float none2f(float val1, float val2)
{
    return val1;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float add2f(float val1, float val2)
{
    return (val1 + val2);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float sub2f(float val1, float val2)
{
    return (val1 - val2);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float mul2f(float val1, float val2)
{
    return (val1 * val2);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
float div2f(float val1, float val2)
{
    return (val1 / val2);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void softmax_ref( float                     *outputs,
                  const float        *const inputs,
                  uint_least32_t            num_samples, // length of single input(batch) to be  processed (softmax normalize)
                  uint_least32_t            num_batches )   
{
    for( uint_least32_t batch = 0; batch < num_batches; ++batch )
    {
        uint_least32_t base_read_write_index = batch*num_samples;
        // Get sum of exponents (normalization factor)
        float sum = 0.0f;
        for( unsigned int s = 0; s < num_samples; ++s )
        {
            sum += expf( inputs[base_read_write_index + s] );
        }

        // normalize given entry by normalization factor computed above
        for( unsigned int s = 0; s < num_samples; ++s )
        {
            outputs[base_read_write_index + s] = expf( inputs[base_read_write_index + s] ) / sum;
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void arithmetic_ref( float                       *outputs,
                     const float          *const inputs,
                     const float          *const factor,
                     const uint_least32_t        num_input_feature_maps,
                     const uint_least32_t        input_feature_map_width,
                     const uint_least32_t        input_feature_map_height,
                     uint_least32_t              num_batches,
                     fp_func_arithmetic          FA )
{
    for( unsigned int n = 0; n < num_batches; ++n )
    {
        // for each batch we do arithmetic operations using factor data 
        for( uint_least32_t z = 0; z < num_input_feature_maps; ++z )
        {
            for( unsigned int y = 0; y < input_feature_map_height; ++y )
            {
                for( unsigned int x = 0; x < input_feature_map_width; ++x )
                {
                    unsigned int input_output_offset = n * input_feature_map_width * input_feature_map_height * num_input_feature_maps +
                                                z * input_feature_map_width * input_feature_map_height +
                                                y * input_feature_map_width + x;

                    unsigned int factor_offset = z * input_feature_map_width * input_feature_map_height +
                                                 y * input_feature_map_width + x;

                     outputs[input_output_offset] = FA( inputs[input_output_offset], factor[factor_offset] );
                }
            }
        }
    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
void normalize_ref( float                     *outputs,
                    const float               *const inputs,
                    uint_least32_t            num_batches,
                    const uint_least32_t      num_input_feature_maps,
                    const uint_least32_t      input_feature_map_width,
                    const uint_least32_t      input_feature_map_height,
                    const uint_least32_t      normalization_size,            // normalization area
                    const uint_least32_t      k,                             //hyper parameter k
                    const float               alpha,                //hyper parameter alpha
                    const float               beta,                 //hyper parameter k
                    NN_NORMALIZATION_MODE     normalization_mode )           // mode of naormalization
{
    if( normalization_mode == NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS )
    {
        for( unsigned int n = 0; n < num_batches; ++n )
        {
            for( unsigned int y = 0; y < input_feature_map_height; ++y )
            {
                for( unsigned int x = 0; x < input_feature_map_width; ++x )
                {
                    for( uint_least32_t i = 0; i < num_input_feature_maps; ++i )
                    {
                        float sum = 0.0f;
                        // For given feature map go from i -n/2 to , i + n/2
                        for( uint_least32_t j = std::max( (int)0, (int)(i - normalization_size / 2) );
                             j <= std::min( num_input_feature_maps - 1, i + normalization_size / 2 ); ++j )
                        {
                            unsigned int input_offset = n * input_feature_map_width * input_feature_map_height *
                                                        num_input_feature_maps +
                                                        j * input_feature_map_width * input_feature_map_height +
                                                        y * input_feature_map_width +
                                                        x;
                            sum += inputs[input_offset] * inputs[input_offset];
                        }
                        sum = pow( ( float )k + sum * alpha, beta );
                        unsigned int input_index = n * input_feature_map_width * input_feature_map_height *
                                                   num_input_feature_maps +
                                                   i * input_feature_map_width * input_feature_map_height +
                                                   y * input_feature_map_width +
                                                   x;
                        outputs[input_index] = inputs[input_index] / sum;
                    }
                }
            }
        }
    }
    else if (normalization_mode == NN_NORMALIZATION_MODE_LINEAR_SINGLE)
    {
        for (unsigned int s = 0; s < (num_batches*num_input_feature_maps*input_feature_map_width*input_feature_map_height); ++s)
        {
            outputs[s] = alpha * inputs[s] + beta;
        }
    }
    else
    {
        //TODO: Implement other normalization modes
        assert( 0 );
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void view_ref( float                     *outputs,
               const float        *const inputs,
               nn_workload_data_coords_t          output_view_begin,
               nn_workload_data_coords_t          output_view_end,
               unsigned int output_width,
               unsigned int output_height,
               unsigned int output_depth)
{
    unsigned int output_view_num_batches = output_view_end.t[NN_DATA_COORD_n]  - output_view_begin.t[NN_DATA_COORD_n] + 1;  
    unsigned int output_view_depth = output_view_end.t[NN_DATA_COORD_z]  - output_view_begin.t[NN_DATA_COORD_z] + 1;  
    unsigned int output_view_height = output_view_end.t[NN_DATA_COORD_y]  - output_view_begin.t[NN_DATA_COORD_y] + 1;  
    unsigned int output_view_width = output_view_end.t[NN_DATA_COORD_x]  - output_view_begin.t[NN_DATA_COORD_x] + 1;  

    for( unsigned int n = 0; n < output_view_num_batches; ++n )
    {
        for( unsigned int z = 0; z < output_view_depth; ++z )
        {
            for( unsigned int y = 0; y < output_view_height; ++y )
            {
                for( unsigned int x = 0; x < output_view_width; ++x )
                {
                    // x,y,z are to iterate through output, so to use them for base
                    // input calculation need to include stride
                    // stride_x and stride_y are used only for setting x,y coords on given feature map
                    unsigned int output_offset = (n + output_view_begin.t[NN_DATA_COORD_n]) * output_width * output_height * output_depth +
                                                (z + output_view_begin.t[NN_DATA_COORD_z]) * output_width * output_height +
                                                (y + output_view_begin.t[NN_DATA_COORD_y])* output_width +
                                                (x + output_view_begin.t[NN_DATA_COORD_x]);
                    // Layout is first_output_map, second_output_map,...
                    outputs[output_offset] = inputs[output_offset];
                }
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void pool_ref( float                     *outputs,
               const float        *const inputs,
               nn_workload_data_coords_t output_view_begin,
               nn_workload_data_coords_t output_view_end,
               nn_workload_data_coords_t input_view_begin,
               nn_workload_data_coords_t input_view_end,
               const unsigned int        outputs_width,
               const unsigned int        outputs_height,
               const unsigned int        outputs_depth,
               const unsigned int        inputs_width,
               const unsigned int        inputs_height,
               const unsigned int        inputs_depth,
               const unsigned int        window_size,
               const unsigned int        stride_x,
               const unsigned int        stride_y,
               uint_least32_t            num_batches ) 
{

    unsigned int output_view_num_batches = output_view_end.t[NN_DATA_COORD_n]  - output_view_begin.t[NN_DATA_COORD_n] + 1;  
    unsigned int output_view_depth = output_view_end.t[NN_DATA_COORD_z]  - output_view_begin.t[NN_DATA_COORD_z] + 1;  
    unsigned int output_view_height = output_view_end.t[NN_DATA_COORD_y]  - output_view_begin.t[NN_DATA_COORD_y] + 1;  
    unsigned int output_view_width = output_view_end.t[NN_DATA_COORD_x]  - output_view_begin.t[NN_DATA_COORD_x] + 1;  

    for( unsigned int n = 0; n < output_view_num_batches; ++n )
    {
        for( unsigned int z = 0; z < output_view_depth; ++z )
        {
            for( unsigned int y = 0; y < output_view_height; ++y )
            {
                for( unsigned int x = 0; x < output_view_width; ++x )
                {
                    // x,y,z are to iterate through output, so to use them for base
                    // input calculation need to include stride
                    // stride_x and stride_y are used only for setting x,y coords on given feature map
                    unsigned int input_offset = (n + input_view_begin.t[NN_DATA_COORD_n]) * inputs_width * inputs_height * inputs_depth +
                                                (z + input_view_begin.t[NN_DATA_COORD_z]) * inputs_width * inputs_height +
                                                (y + input_view_begin.t[NN_DATA_COORD_y]) * stride_y * inputs_width +
                                                (x + input_view_begin.t[NN_DATA_COORD_x]) * stride_x;
                    // pooling starts here
                    float max_val = inputs[input_offset];
                    for( uint_least32_t j = 0; j < window_size; ++j )
                    {
                        for( uint_least32_t i = 0; i < window_size; ++i )
                        {
                            max_val = std::max( max_val, inputs[input_offset + i] );
                        }
                        input_offset += inputs_width; // New row of input so we need to add input_width
                    }
                    // Layout is first_output_map, second_output_map,...
                    outputs[(n + output_view_begin.t[NN_DATA_COORD_n] ) * outputs_width * outputs_height * outputs_depth +
                            (z + output_view_begin.t[NN_DATA_COORD_z] ) * outputs_width * outputs_height +
                            (y + output_view_begin.t[NN_DATA_COORD_y] ) * outputs_width +
                            (x + output_view_begin.t[NN_DATA_COORD_x] )] = max_val;
                }
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void fully_connect_ref( fp_func_activ             FA,
                        float                     *outputs,
                        const float        *const inputs,
                        const float        *const filters,
                        const float        *const biases,
                        const unsigned int        num_outputs,
                        const unsigned int        inputs_width,
                        const unsigned int        inputs_height,
                        const unsigned int        inputs_depth,
                        const unsigned int        num_batches )
{
    for( unsigned int batch = 0; batch < num_batches; ++batch )
    {
        // Iterate through input according to width and height of output buffer params
        for( unsigned int n = 0; n < num_outputs; ++n )
        {
            float dotProd = 0.0f;
            for( unsigned int z = 0; z < inputs_depth; ++z )
            {
                for( unsigned int y = 0; y < inputs_height; ++y )
                {
                    for( unsigned int x = 0; x < inputs_width; ++x )
                    {
                        // Batch is a a chunk of inputs
                        unsigned int element_index = batch * inputs_depth * inputs_width * inputs_height +
                                                     z * inputs_width * inputs_height +
                                                     y * inputs_width + x;
                        // Weights are nto related to batches they are part of given neuron
                        unsigned int weight_index = n * inputs_depth * inputs_width * inputs_height +
                                                    z * inputs_width * inputs_height +
                                                    y * inputs_width + x;
                        dotProd += inputs[element_index] * filters[weight_index];
                    }
                }

            }
            // Add Result of multiplication to bias
            // (output was initialized with biases when biases == nullptr)
            if( biases == nullptr )
            {
                outputs[batch*num_outputs + n] = FA( outputs[batch*num_outputs + n] + dotProd );
            }
            else
            {
                outputs[batch*num_outputs + n] = FA( dotProd + biases[n] );
            }
        }
    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
void convolve_ref( fp_func_activ             FA,
                   float                     *outputs,
                   const float        *const inputs,
                   const float        *const filters,
                   const float        *const biases,
                   nn_workload_data_coords_t          output_view_begin,
                   nn_workload_data_coords_t          output_view_end,
                   nn_workload_data_coords_t          input_view_begin,
                   nn_workload_data_coords_t          input_view_end,
                   const unsigned int        outputs_width,
                   const unsigned int        outputs_height,
                   const unsigned int        outputs_depth,
                   const unsigned int        inputs_width,
                   const unsigned int        inputs_height,
                   const unsigned int        inputs_depth,
                   const unsigned int        weights_width,
                   const unsigned int        weights_height,
                   const unsigned int        weights_depth,
                   unsigned int              stride_x,
                   unsigned int              stride_y,
                   unsigned int              center_x,
                   unsigned int              center_y,
                   const unsigned int        num_batches )
{
    unsigned int output_view_num_batches = output_view_end.t[NN_DATA_COORD_n]  - output_view_begin.t[NN_DATA_COORD_n] + 1;  
    unsigned int output_view_depth = output_view_end.t[NN_DATA_COORD_z]  - output_view_begin.t[NN_DATA_COORD_z] + 1;  
    unsigned int output_view_height = output_view_end.t[NN_DATA_COORD_y]  - output_view_begin.t[NN_DATA_COORD_y] + 1;  
    unsigned int output_view_width = output_view_end.t[NN_DATA_COORD_x]  - output_view_begin.t[NN_DATA_COORD_x] + 1;  

    for( unsigned int batch = 0; batch < output_view_num_batches; ++batch )
    {
        // Iterate through input according to width and height of output buffer params
        for( unsigned int z = 0; z < output_view_depth; ++z )
        {
            for( unsigned int y = 0; y < output_view_height; ++y )
            {
                for( unsigned int x = 0; x < output_view_width; ++x )
                {
                    unsigned int out_map_size = outputs_width * outputs_height;
                    unsigned int filter_size  = weights_depth * weights_height * weights_width;
                    unsigned int input_size   = inputs_width * inputs_height * inputs_depth;

                    unsigned int batch_out_map_size = out_map_size * outputs_depth;

                    unsigned int filter_offset = z * filter_size;

                    // x,y are to iterate through output, so to use them for base
                    // input calculation need to include stride
                    // stride_x and stride_y are used only for setting x,y coords on given feature map
                    unsigned int input_offset = (batch + input_view_begin.t[NN_DATA_COORD_n]) * input_size +
                                                (input_view_begin.t[NN_DATA_COORD_z])*inputs_width*inputs_height +
                                                ((y - center_y)* stride_y + input_view_begin.t[NN_DATA_COORD_y])  * inputs_width +
                                                ((x - center_x)* stride_x + input_view_begin.t[NN_DATA_COORD_x]) ;

                    // Layout is first_output_map, second_output_map,...
                    unsigned int output_offset = ( batch + output_view_begin.t[NN_DATA_COORD_n] ) * batch_out_map_size +
                                                 ( z + output_view_begin.t[NN_DATA_COORD_z] ) * out_map_size +
                                                 ( y + output_view_begin.t[NN_DATA_COORD_y] ) * outputs_width + 
                                                 ( x + output_view_begin.t[NN_DATA_COORD_x] );

                    //unsigned int output_offset = batch * batch_out_map_size + z * out_map_size + y *
                                                 //outputs_width + x;
                    {
                        float dotProd = 0.0f;

                        for( unsigned int k = 0; k < weights_depth; ++k )
                        {
                            for( unsigned int j = 0; j < weights_height; ++j )
                            {
                                for( unsigned int i = 0; i < weights_width; ++i )
                                {
                                    float signal;
                                    // If value to be read is out of input scope then ignore that one
                                    if( (x + i - center_x + input_view_begin.t[NN_DATA_COORD_x] >= 0) && (x + i - center_x + input_view_begin.t[NN_DATA_COORD_x] < inputs_width) &&
                                        (y + j - center_y + input_view_begin.t[NN_DATA_COORD_y]>= 0) && (y + j - center_y + input_view_begin.t[NN_DATA_COORD_y]< inputs_height) )
                                    {
                                        signal = inputs[input_offset];
                                        dotProd += signal * filters[filter_offset ];
                                    }
                                    ++input_offset;
                                    ++filter_offset;
                                }
                                input_offset += inputs_width - weights_width;
                            }
                            input_offset += (inputs_height - weights_height)*inputs_width;
                        }
                        // Add Result of convolution to bias
                        // (output was initialized with biases when biases == nullptr)
                        outputs[output_offset] = FA( dotProd + biases[z] );

                    }
                }
            }
        }
    }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void init_data( float * &buffer, uint_least32_t bufferCount, float initValue )
{
    uint_least32_t totalSize = bufferCount * sizeof( float );

    // Memory need to be aligned to page , so that OCL driver not to
    // duplicate buffer data by creating aligned copy
#ifdef __linux__
    buffer = ( float * )memalign( page_alignment, totalSize );
#else
    buffer = ( float * )_aligned_malloc( totalSize, page_alignment );
#endif //__linux__

    for( uint_least32_t i = 0; i < bufferCount; ++i )
    {
        buffer[i] = initValue;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
std::unique_ptr< nn::nn_workload_data_t< float > > create_nn_workload_data_using_buffer( const float        *const buffer,
                                                                       nn_workload_data_layout_t          &buffer_layout,
                                                                       nn_workload_data_coords_t          &buffer_coords )
{
    return std::unique_ptr< nn::nn_workload_data_t< float > >( new nn::nn_workload_data_t< float >( ( void * )buffer, buffer_coords,
                                                                                  buffer_layout ) );
}
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Compare content of work_item's output with given reference buffers. count is number of elements
bool verify_output( std::unique_ptr<nn::data<float, 0>> &output_data, float * const ref_buf)
{
    for(auto index=0u; index<output_data->count(); ++index) {
        float candidate = static_cast<float *>(output_data->buffer)[index];
        float reference = ref_buf[index];
        if(fabs(candidate-reference)/reference > 0.01f) {
            return false;
        }
    }

    return true;
}
