/*
Copyright (c) 2014, Intel Corporation

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

#include "device/common/nn_workload_data.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "layer_convolution_avx2.h"
#include "helper_zxyn_f32.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <tuple>
#include "device/cpu/api_internal/data_helper.h"

// Pragmas inside macros.
#if defined _MSC_VER
#   define PRAGMA_MACRO(x) __pragma(x)
#else
#   define PRAGMA_MACRO(x) _Pragma(#x)
#endif

namespace layer {
namespace convolution_f32_impl {
// SIMD width for this implementation
const auto C_simd_width = sizeof(__m256) / sizeof(float);
const uint32_t C_slice_size = 2 * C_simd_width;

///////////////////////////////////////////////////////////////////////////////////////////////////
// forward implementation

///////////////////////////////////////////////////////////////////////////////////////////////////
// defines to handle multiple variants of code loops
// Necessary evil because of compile bug resulting in bad ISA quality when AVX regsiters are in
// C++ array and all AVX registers are to be used in final ISA for intermost loop.
///////////////////////////////////////////////////////////////////////////////////////////////////
#define CREATE_ACC(acc0, acc1, num) \
    __m256 vout_##acc0 = _mm256_setzero_ps(); \
    __m256 vout_##acc1 = _mm256_setzero_ps();

#define MAD_ACC(acc0, acc1, num) \
    bc = _mm256_broadcast_ss(inp_ptr + offs##num); \
    vout_##acc0 = _mm256_fmadd_ps(vwt0, bc, vout_##acc0); \
    vout_##acc1 = _mm256_fmadd_ps(vwt1, bc, vout_##acc1);

#define STORE_ACC(acc0, acc1, num) \
    vout_##acc0 = _mm256_add_ps(vout_##acc0, bias0); \
    vout_##acc1 = _mm256_add_ps(vout_##acc1, bias1); \
    \
    if(T_activation == NN_ACTIVATION_FUNCTION_RELU)\
    { \
        vout_##acc0 = _mm256_max_ps(vout_##acc0, _mm256_setzero_ps()); \
        vout_##acc1 = _mm256_max_ps(vout_##acc1, _mm256_setzero_ps()); \
    } \
    \
    _mm256_store_ps(&output[internal_out_offset0 + (num) * num_output_feature_maps], vout_##acc0); \
    _mm256_store_ps(&output[internal_out_offset1 + (num) * num_output_feature_maps], vout_##acc1);

///////////////////////////////////////////////////////////////////////////////////////////////////
// REPLICATION MACROS
#define SIMPLE_REPLICATION_1(function) function(0,1,0)
#define SIMPLE_REPLICATION_2(function) SIMPLE_REPLICATION_1(function) function(2,3,1)
#define SIMPLE_REPLICATION_3(function) SIMPLE_REPLICATION_2(function) function(4,5,2)
#define SIMPLE_REPLICATION_4(function) SIMPLE_REPLICATION_3(function) function(6,7,3)
#define SIMPLE_REPLICATION_5(function) SIMPLE_REPLICATION_4(function) function(8,9,4)
#define SIMPLE_REPLICATION_6(function) SIMPLE_REPLICATION_5(function) function(10,11,5)

///////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN MACRO
#define NN_CONVOLVE_OPTIMIZED_BLOCK( \
    block_size) \
{ \
    const int offs0 = 0; \
    const int offs1 = num_input_feature_maps * kernel_stride_x; \
    const int offs2 = 2 * offs1 \
    , offs3 = 3 * offs1 \
    , offs4 = 4 * offs1 \
    , offs5 = 5 * offs1 \
    , offs6 = 6 * offs1 \
    ; \
    \
    const auto kernel_depth_size = input_fmap_view_length * C_slice_size; \
    const auto kernel_depth_offset = kernel_input_fmap_view_start * C_slice_size; \
    \
    const auto wfm          = (kernel_feature_map / C_slice_size) * weight_offset; \
    const auto internal_out_offset0  = out_offset + out_feature_map; \
    const auto internal_out_offset1  = internal_out_offset0 + C_simd_width; \
    \
    SIMPLE_REPLICATION_##block_size(CREATE_ACC) \
    \
    float *init_init_kernel_offset_base_ptr = kernel + wfm; \
    float *init_init_input_offset_base_ptr = input + inp_offset_base; \
    for (auto kh = 0U; kh < kernel_height; kh++) \
    { \
        float *init_kernel_offset_base_ptr = init_init_kernel_offset_base_ptr; \
        float *init_inp_ptr = init_init_input_offset_base_ptr; \
        for (auto kw = 0U; kw < kernel_width; ++kw) \
        { \
            float *kernel_offset_base_ptr = init_kernel_offset_base_ptr + kernel_depth_offset; \
            float *inp_ptr = init_inp_ptr + input_fmap_view_start; \
            \
            float *kernel_offset_end_ptr = kernel_offset_base_ptr + kernel_depth_size; \
            if(T_exact_match) \
            { \
                PRAGMA_MACRO(unroll (T_unroll_times)) \
                for (auto ifm = 0U; ifm < input_fmap_view_length; ++ifm) \
                { \
                    __m256 vwt0 = _mm256_load_ps(kernel_offset_base_ptr); \
                    __m256 vwt1 = _mm256_load_ps(kernel_offset_base_ptr + C_simd_width); \
                    __m256 bc; \
                    \
                    SIMPLE_REPLICATION_##block_size(MAD_ACC) \
                    \
                    ++inp_ptr; \
                    kernel_offset_base_ptr += C_slice_size; \
                } \
            } \
            else \
            { \
                for (; kernel_offset_base_ptr < kernel_offset_end_ptr;) \
                { \
                    __m256 vwt0 = _mm256_load_ps(kernel_offset_base_ptr); \
                    __m256 vwt1 = _mm256_load_ps(kernel_offset_base_ptr + C_simd_width); \
                    __m256 bc; \
                    \
                    SIMPLE_REPLICATION_##block_size(MAD_ACC) \
                    \
                    ++inp_ptr; \
                    kernel_offset_base_ptr += C_slice_size; \
                } \
            } \
            init_kernel_offset_base_ptr += kernel_depth_size; \
            init_inp_ptr += num_input_feature_maps; \
        } \
        init_init_kernel_offset_base_ptr += kernel_depth_size * kernel_width; \
        init_init_input_offset_base_ptr += input_row_size; \
    } \
    \
    __m256 bias0 = _mm256_load_ps((float*)bias->parent->data_buffer + bias_feature_map); \
    __m256 bias1 = _mm256_load_ps((float*)bias->parent->data_buffer + bias_feature_map + C_simd_width); \
    \
    SIMPLE_REPLICATION_##block_size(STORE_ACC) \
}

template<bool                  T_exact_match,
        NN_ACTIVATION_FUNCTION T_activation,
        uint32_t               T_unroll_times               = 0, 
        uint32_t               T_input_width                = 0, 
        uint32_t               T_input_height               = 0, 
        uint32_t               T_input_feature_maps         = 0, 
        uint32_t               T_input_fmap_view_start      = 0, 
        uint32_t               T_input_fmap_view_length     = 0, 
        uint32_t               T_kernel_in_fmap_view_start  = 0, 
        uint32_t               T_kernel_width               = 0,
        uint32_t               T_kernel_height              = 0, 
        uint32_t               T_kernel_stride_x            = 0, 
        uint32_t               T_kernel_stride_y            = 0,
        uint32_t               T_output_width               = 0,
        uint32_t               T_output_height              = 0,
        uint32_t               T_output_feature_maps        = 0>
void convolve_internal(
    const nn::workload_data<float> *input_view,
    const size_t center_offset_x,
    const size_t center_offset_y,
    const int32_t stride_x,
    const int32_t stride_y,
    const nn::workload_data<float> *weights,
    const nn::workload_data<float> *bias,
    nn::workload_data<float> *output_view)
{
    float* input = (float*)input_view->parent->data_buffer;
    float* output = (float*)output_view->parent->data_buffer;
    float* kernel = (float*)weights->parent->data_buffer;

    const auto num_output_feature_maps      = (T_exact_match) ? T_output_feature_maps : output_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto num_input_feature_maps       = (T_exact_match) ? T_input_feature_maps : input_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_feature_map_width     = (T_exact_match) ? T_output_width : output_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_feature_map_height    = (T_exact_match) ? T_output_height : output_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto input_feature_map_width      = (T_exact_match) ? T_input_width : input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto input_feature_map_height     = (T_exact_match) ? T_input_height : input_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto kernel_width                 = (T_exact_match) ? T_kernel_width : weights->parent->lengths.t[NN_DATA_COORD_x];
    const auto kernel_height                = (T_exact_match) ? T_kernel_height : weights->parent->lengths.t[NN_DATA_COORD_y];
    const auto kernel_stride_x              = (T_exact_match) ? T_kernel_stride_x : stride_x;
    const auto kernel_stride_y              = (T_exact_match) ? T_kernel_stride_y : stride_y;
    const auto kernel_input_fmap_view_start = (T_exact_match) ? T_kernel_in_fmap_view_start : weights->view_begin.t[NN_DATA_COORD_z];
    const auto input_fmap_view_start        = (T_exact_match) ? T_input_fmap_view_start : input_view->view_begin.t[NN_DATA_COORD_z];
    const auto input_fmap_view_length       = (T_exact_match) ? T_input_fmap_view_length : input_view->view_end.t[NN_DATA_COORD_z] - input_fmap_view_start + 1;

    const auto bias_view_start = bias->view_begin.t[NN_DATA_COORD_x];

    const auto kernel_center_x = center_offset_x;
    const auto kernel_center_y = center_offset_y;
    
    const auto output_fm_view_start = output_view->view_begin.t[NN_DATA_COORD_z];
    const auto output_fm_view_end = output_view->view_end.t[NN_DATA_COORD_z];
    
    const auto output_row_view_start = output_view->view_begin.t[NN_DATA_COORD_y];
    const auto output_row_view_end = output_view->view_end.t[NN_DATA_COORD_y];
    
    const auto output_column_view_start = output_view->view_begin.t[NN_DATA_COORD_x];
    const auto output_column_view_end = output_view->view_end.t[NN_DATA_COORD_x];
    
    const auto input_column_view_start = input_view->view_begin.t[NN_DATA_COORD_x] - kernel_center_x;
    const auto input_row_view_start = input_view->view_begin.t[NN_DATA_COORD_y] - kernel_center_y;
    
    const auto output_view_width = output_column_view_end - output_column_view_start + 1;
    
    const auto output_row_size      = output_feature_map_width * num_output_feature_maps;
    const auto input_row_size       = input_feature_map_width * num_input_feature_maps;
    const auto weight_offset        = weights->parent->lengths.t[NN_DATA_COORD_z] * kernel_width*kernel_height * C_slice_size;
    
    const auto num_blocks_full      = output_view_width / 6;
    const auto partial_block_size   = output_view_width % 6;
    
    const auto output_image_view_start = output_view->view_begin.t[NN_DATA_COORD_n];
    const auto output_image_view_end = output_view->view_end.t[NN_DATA_COORD_n];
    
    const auto input_image_size = input_row_size * input_feature_map_height;
    const auto output_image_size = output_row_size * output_feature_map_height;

    const auto kernel_out_fmap_view_start = weights->view_begin.t[NN_DATA_COORD_q] * C_slice_size;
    
    for(auto out_image = output_image_view_start; out_image <= output_image_view_end; ++out_image)
    {
        auto input_image_offset = out_image*input_image_size;
        auto output_image_offset = out_image*output_image_size;
        
        for (auto out_feature_map = output_fm_view_start, kernel_feature_map = kernel_out_fmap_view_start, bias_feature_map = bias_view_start; 
            out_feature_map <= output_fm_view_end; 
            out_feature_map += C_slice_size, kernel_feature_map += C_slice_size, bias_feature_map += C_slice_size)
        {
            for (auto output_row = output_row_view_start, input_row = 0U; output_row <= output_row_view_end; output_row++, input_row++)
            {
                const auto inp_h        = input_row * kernel_stride_y + input_row_view_start;
                const auto out_offset1  = output_row * output_row_size;
                const auto inp_offset1  = inp_h * input_row_size;
                
                auto out_offset         = out_offset1 + output_column_view_start * num_output_feature_maps + output_image_offset;
                auto inp_offset_base    = inp_offset1 + input_column_view_start * num_input_feature_maps + input_image_offset;
                
                for (auto block = 0U; block < num_blocks_full; block++) {
                    NN_CONVOLVE_OPTIMIZED_BLOCK(6);
                    inp_offset_base += 6 * num_input_feature_maps * kernel_stride_x;
                    out_offset += 6 * num_output_feature_maps;
                }
                
                switch (partial_block_size)
                {
                case 0: break;
                case 1: NN_CONVOLVE_OPTIMIZED_BLOCK(1); break;
                case 2: NN_CONVOLVE_OPTIMIZED_BLOCK(2); break;
                case 3: NN_CONVOLVE_OPTIMIZED_BLOCK(3); break;
                case 4: NN_CONVOLVE_OPTIMIZED_BLOCK(4); break;
                case 5: NN_CONVOLVE_OPTIMIZED_BLOCK(5); break;
                default:
                    /* Execution can never reach here (see 'partial_block_size') calculation.*/
                    /* Need to inform compiler that it should not generate code for 'default'.*/
                    /* [TODO] heed to handle GCC */
                    NN_UNREACHABLE_CODE;
                }
            }
        }
    }
}

using optimized_layer_map_t = std::map<
    std::tuple<NN_ACTIVATION_FUNCTION, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>, 
    decltype(convolve_internal<false, NN_ACTIVATION_FUNCTION_NONE>)*>;

template<NN_ACTIVATION_FUNCTION T_activation,
         uint32_t T_input_width, uint32_t T_input_height, uint32_t T_input_feature_maps, 
         uint32_t T_input_fmap_view_start, uint32_t T_input_fmap_view_length, uint32_t T_kernel_in_fmap_view_start,
         uint32_t T_kernel_width, uint32_t T_kernel_height, uint32_t T_kernel_stride_x, uint32_t T_kernel_stride_y,
         uint32_t T_output_width, uint32_t T_output_height, uint32_t T_output_feature_maps>
optimized_layer_map_t::value_type prepare_entry()
{
    return { 
        optimized_layer_map_t::key_type{ 
            T_activation,
            T_input_width, T_input_height, T_input_feature_maps, 
            T_input_fmap_view_start, T_input_fmap_view_length, T_kernel_in_fmap_view_start, 
            T_kernel_width, T_kernel_height, T_kernel_stride_x, T_kernel_stride_y,
            T_output_width, T_output_height, T_output_feature_maps }, 
        convolve_internal<
            true, 
            T_activation, 
            (T_input_fmap_view_length % 8 == 0) ? 8 : T_input_fmap_view_length,
            T_input_width, T_input_height, T_input_feature_maps, 
            T_input_fmap_view_start, T_input_fmap_view_length, T_kernel_in_fmap_view_start,
            T_kernel_width, T_kernel_height, T_kernel_stride_x, T_kernel_stride_y, 
            T_output_width, T_output_height, T_output_feature_maps> };
}

optimized_layer_map_t optimized_layer_map =
{
    // OverFeat C3.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 14, 14, 256, 0, 256, 0, 3, 3, 1, 1, 14, 14, 512>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 14, 14, 256, 0, 256, 0 ,3, 3, 1, 1, 14, 14, 512>() },

    // OverFeat C4.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 14, 14, 512, 0, 512, 0, 3, 3, 1, 1, 14, 14, 1024>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 14, 14, 512, 0, 512, 0, 3, 3, 1, 1, 14, 14, 1024>() },

    // CaffeNet C1.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 227, 227, 3, 0, 3, 0, 11, 11, 4, 4, 55, 55, 96>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 227, 227, 3, 0, 3, 0, 11, 11, 4, 4, 55, 55, 96>() },

    // CaffeNet C2A.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 31, 31, 96, 0, 48, 0, 5, 5, 1, 1, 27, 27, 256>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 31, 31, 96, 0, 48, 0, 5, 5, 1, 1, 27, 27, 256>() },

    // CaffeNet C2B.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 31, 31, 96, 48, 48, 0, 5, 5, 1, 1, 27, 27, 256>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 31, 31, 96, 48, 48, 0, 5, 5, 1, 1, 27, 27, 256>() },

    // CaffeNet C3.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 15, 15, 256, 0, 256, 0, 3, 3, 1, 1, 15, 15, 384>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 15, 15, 256, 0, 256, 0, 3, 3, 1, 1, 15, 15, 384>() },

    // CaffeNet C4A.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 15, 15, 384, 0, 192, 0, 3, 3, 1, 1, 15, 15, 192>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 15, 15, 384, 0, 192, 0, 3, 3, 1, 1, 15, 15, 192>() },

    // CaffeNet C4B.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 15, 15, 384, 192, 192, 0, 3, 3, 1, 1, 15, 15, 192>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 15, 15, 384, 192, 192, 0, 3, 3, 1, 1, 15, 15, 192>() },

    // CaffeNet C5AB.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 15, 15, 192, 0, 192, 0, 3, 3, 1, 1, 13, 13, 256>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 15, 15, 192, 0, 192, 0, 3, 3, 1, 1, 13, 13, 256>() },
};

void nn_convolve_naive(
    const nn::workload_data<float> *input_view,
    const nn::workload_data<float> *output_view,
    const nn::workload_data<float> *bias_view,
    const nn::workload_data<float> *kernel_view,
    uint32_t center_offset_x,
    uint32_t center_offset_y,
    uint32_t kernel_stride_x,
    uint32_t kernel_stride_y,
    NN_ACTIVATION_FUNCTION activation)
{
    const auto num_output_feature_maps = output_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto num_input_feature_maps = input_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_feature_map_width = output_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_feature_map_height = output_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto input_feature_map_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto input_feature_map_height = input_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto kernel_width = kernel_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto kernel_height = kernel_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto kernel_depth = kernel_view->parent->lengths.t[NN_DATA_COORD_z];

    const auto kernel_ptr = reinterpret_cast<float*>(kernel_view->parent->data_buffer);
    const auto input_ptr = reinterpret_cast<float*>(input_view->parent->data_buffer);
    const auto output_ptr = reinterpret_cast<float*>(output_view->parent->data_buffer);

    const auto bias_ptr = (bias_view) ? reinterpret_cast<float*>(bias_view->parent->data_buffer) : nullptr;

    for (uint32_t batch = output_view->view_begin.t[NN_DATA_COORD_n]; batch <= output_view->view_end.t[NN_DATA_COORD_n]; batch++)
    {
        for (uint32_t output_feature_map = output_view->view_begin.t[NN_DATA_COORD_z]; output_feature_map <= output_view->view_end.t[NN_DATA_COORD_z]; output_feature_map += C_simd_width)
        {
            for (uint32_t input_row = input_view->view_begin.t[NN_DATA_COORD_y] - center_offset_y, output_row = output_view->view_begin.t[NN_DATA_COORD_y]; output_row <= output_view->view_end.t[NN_DATA_COORD_y]; input_row += kernel_stride_y, output_row++)
            {
                for (uint32_t input_column = input_view->view_begin.t[NN_DATA_COORD_x] - center_offset_x, output_column = output_view->view_begin.t[NN_DATA_COORD_x]; output_column <= output_view->view_end.t[NN_DATA_COORD_x]; input_column += kernel_stride_x, output_column++)
                {
                    const uint32_t out_base = 
                          batch*output_feature_map_width*output_feature_map_height*num_output_feature_maps 
                        + output_row*output_feature_map_width*num_output_feature_maps 
                        + output_column*num_output_feature_maps 
                        + output_feature_map;

                    __m256 accumulator0 = _mm256_setzero_ps();

                    for (uint32_t input_feature_map = input_view->view_begin.t[NN_DATA_COORD_z], kernel_input_feature_map =  kernel_view->view_begin.t[NN_DATA_COORD_z];
                                  input_feature_map <= input_view->view_end.t[NN_DATA_COORD_z]; input_feature_map++, ++kernel_input_feature_map)
                    {
                        for (uint32_t kernel_row = kernel_view->view_begin.t[NN_DATA_COORD_y]; kernel_row <= kernel_view->view_end.t[NN_DATA_COORD_y]; kernel_row++)
                        {
                            for (uint_least32_t kernel_column = kernel_view->view_begin.t[NN_DATA_COORD_x]; kernel_column <= kernel_view->view_end.t[NN_DATA_COORD_x]; kernel_column++)
                            {
                                uint32_t kernel_output_map_div = output_feature_map / C_slice_size;
                                uint32_t kernel_output_map_rem = output_feature_map % C_slice_size;

                                __m256 weight0 = _mm256_load_ps(  kernel_ptr
                                                                + kernel_row*C_slice_size*kernel_width*kernel_depth 
                                                                + kernel_input_feature_map*C_slice_size 
                                                                + kernel_column*C_slice_size*kernel_depth 
                                                                + kernel_output_map_div*kernel_width*kernel_height*kernel_depth*C_slice_size 
                                                                + kernel_output_map_rem);

                                __m256 input = _mm256_broadcast_ss(  input_ptr 
                                                                   + batch*input_feature_map_width*input_feature_map_height*num_input_feature_maps 
                                                                   + (input_row+kernel_row)*input_feature_map_width*num_input_feature_maps 
                                                                   + (input_column+kernel_column)*num_input_feature_maps 
                                                                   + input_feature_map);

                                accumulator0 = _mm256_fmadd_ps(weight0, input, accumulator0);
                            }
                        }
                    }

                    if(bias_ptr)
                        accumulator0 = _mm256_add_ps(accumulator0, _mm256_load_ps(bias_ptr + output_feature_map));

                    if(activation == NN_ACTIVATION_FUNCTION_RELU)
                        accumulator0 = _mm256_max_ps(_mm256_setzero_ps(), accumulator0);

                    _mm256_store_ps(  output_ptr 
                                    + batch*output_feature_map_width*output_feature_map_height*num_output_feature_maps 
                                    + output_row*output_feature_map_width*num_output_feature_maps 
                                    + output_column*num_output_feature_maps 
                                    + output_feature_map 
                                    , accumulator0);
                }
            }
        }
    }
}


template <NN_ACTIVATION_FUNCTION T_activation>
void run_convolution(const nn::workload_data<float> *input_view,
                     const NN_PADDING_MODE padding,
                     const int32_t center_offset_x,
                     const int32_t center_offset_y,
                     const size_t stride_x,
                     const size_t stride_y,
                     const nn::workload_data<float> *weights,
                     const nn::workload_data<float> *bias,
                     nn::workload_data<float> *output_view) {

    const size_t num_output_feature_maps = output_view->parent->lengths.t[NN_DATA_COORD_z];
    const size_t num_input_feature_maps = input_view->parent->lengths.t[NN_DATA_COORD_z];
    const size_t output_feature_map_width = output_view->parent->lengths.t[NN_DATA_COORD_x];
    const size_t output_feature_map_height = output_view->parent->lengths.t[NN_DATA_COORD_y];
    const size_t input_feature_map_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
    const size_t input_feature_map_height = input_view->parent->lengths.t[NN_DATA_COORD_y];
    const size_t kernel_width = weights->parent->lengths.t[NN_DATA_COORD_x];
    const size_t kernel_height = weights->parent->lengths.t[NN_DATA_COORD_y];
    const size_t kernel_input_fmap_view_start = weights->view_begin.t[NN_DATA_COORD_z];
    const size_t input_fmap_view_start = input_view->view_begin.t[NN_DATA_COORD_z];
    const size_t input_fmap_view_length = input_view->view_end.t[NN_DATA_COORD_z] - input_fmap_view_start + 1;

    auto map_element = optimized_layer_map.find(std::make_tuple(
        T_activation,
        input_feature_map_width,
        input_feature_map_height,
        num_input_feature_maps,
        input_fmap_view_start,
        input_fmap_view_length,
        kernel_input_fmap_view_start,
        kernel_width,
        kernel_height,
        stride_x,
        stride_y,
        output_feature_map_width,
        output_feature_map_height,
        num_output_feature_maps));

    if (map_element != std::end(optimized_layer_map))
    {
        // Optimized.
        map_element->second(input_view, center_offset_x, center_offset_y, stride_x, stride_y, weights, bias, output_view);
    }
    else
    {
        if(output_view->get_length(NN_DATA_COORD_z) % C_slice_size == 0) // Generic.
            convolve_internal<false, T_activation>(input_view, center_offset_x, center_offset_y, stride_x, stride_y, weights, bias, output_view);
        else // Special naive version for non16 aligned convolutions' OFM views.
            nn_convolve_naive(
                input_view, 
                output_view, 
                bias, 
                weights,
                center_offset_x,
                center_offset_y,
                stride_x,
                stride_y,
                T_activation);
    }
}

void choose_convolution_padding_mode_and_activation(const nn::workload_data<float> *input,
                                                    const NN_PADDING_MODE padding,
                                                    const int32_t center_offset_x,
                                                    const int32_t center_offset_y,
                                                    const size_t stride_x,
                                                    const size_t stride_y,
                                                    const nn_argument_activation_t &activation,
                                                    const nn::workload_data<float> *weights,
                                                    const nn::workload_data<float> *bias,
                                                    nn::workload_data<float> *output) {

    switch (padding)
    {
    case NN_PADDING_MODE_DATA_OR_ZERO:
        {
            // Get basic data about convolution.

            const int32_t  kernel_width = weights->parent->lengths.t[NN_DATA_COORD_x];
            const int32_t  kernel_height = weights->parent->lengths.t[NN_DATA_COORD_y];
            const uint32_t kernel_depth = weights->parent->lengths.t[NN_DATA_COORD_z];

            const uint32_t num_ofm = output->parent->lengths.t[NN_DATA_COORD_z];
            const uint32_t ofm_width = output->parent->lengths.t[NN_DATA_COORD_x];
            const uint32_t ofm_height = output->parent->lengths.t[NN_DATA_COORD_y];

            const uint32_t num_ifm = input->parent->lengths.t[NN_DATA_COORD_z];
            const int32_t  ifm_width = input->parent->lengths.t[NN_DATA_COORD_x];
            const int32_t  ifm_height = input->parent->lengths.t[NN_DATA_COORD_y];

            const uint32_t output_fm_view_start = output->view_begin.t[NN_DATA_COORD_z];
            const uint32_t output_fm_view_end = output->view_end.t[NN_DATA_COORD_z];

            // Get data about base view.
            const int32_t input_base_view_start_x = input->view_begin.t[NN_DATA_COORD_x];
            const int32_t input_base_view_start_y = input->view_begin.t[NN_DATA_COORD_y];
            const int32_t output_base_view_start_x = output->view_begin.t[NN_DATA_COORD_x];
            const int32_t output_base_view_start_y = output->view_begin.t[NN_DATA_COORD_y];
            const int32_t output_base_view_end_x = output->view_end.t[NN_DATA_COORD_x];
            const int32_t output_base_view_end_y = output->view_end.t[NN_DATA_COORD_y];

            // Get offsets of convolution central point.
            const int32_t required_center_offset_from_left = center_offset_x;
            const int32_t required_center_offset_from_up = center_offset_y;
            const int32_t required_center_offset_from_right = weights->parent->lengths.t[NN_DATA_COORD_x] - (required_center_offset_from_left + 1);
            const int32_t required_center_offset_from_down = weights->parent->lengths.t[NN_DATA_COORD_y] - (required_center_offset_from_up + 1);

            // Get number of input elements required along with their absolute end offsets.
            const int32_t  input_elements_required_x = (output->get_length(NN_DATA_COORD_x) - 1) * stride_x + 1;
            const int32_t  input_elements_required_y = (output->get_length(NN_DATA_COORD_y) - 1) * stride_y + 1;
            const int32_t  input_absolute_end_offset_x = input_base_view_start_x + input_elements_required_x;
            const int32_t  input_absolute_end_offset_y = input_base_view_start_y + input_elements_required_y;

            // Get length of raw buffer data after used view to check if we need out-of-data padding.
            const int32_t valid_data_length_after_view_left_barrier = input_base_view_start_x;
            const int32_t valid_data_length_after_view_up_barrier = input_base_view_start_y;
            const int32_t valid_data_length_after_view_right_barrier = std::max(ifm_width - input_absolute_end_offset_x, 0);
            const int32_t valid_data_length_after_view_down_barrier = std::max(ifm_height - input_absolute_end_offset_y, 0);

            // Crop views for default implementation.
            const uint32_t crop_from_left = std::max(required_center_offset_from_left - valid_data_length_after_view_left_barrier, 0);
            const uint32_t crop_from_up = std::max(required_center_offset_from_up - valid_data_length_after_view_up_barrier, 0);
            const uint32_t crop_from_right = std::max(required_center_offset_from_right - valid_data_length_after_view_right_barrier, 0);
            const uint32_t crop_from_down = std::max(required_center_offset_from_down - valid_data_length_after_view_down_barrier, 0);

            // Adjust crop for filter strides.
            // For input it should be aligned to stride.
            const uint32_t input_crop_from_left = (crop_from_left + stride_x - 1) / stride_x * stride_x;
            const uint32_t input_crop_from_up = (crop_from_up + stride_y - 1) / stride_y * stride_y;
            const uint32_t input_crop_from_right = (crop_from_right + stride_x - 1) / stride_x * stride_x;
            const uint32_t input_crop_from_down = (crop_from_down + stride_y - 1) / stride_y * stride_y;

            // Output crop should be input crop divided by stride.
            const uint32_t output_crop_from_left = input_crop_from_left / stride_x;
            const uint32_t output_crop_from_up = input_crop_from_up / stride_y;
            const uint32_t output_crop_from_right = input_crop_from_right / stride_x;
            const uint32_t output_crop_from_down = input_crop_from_down / stride_y;

            // Prepare cropped views.
            nn_workload_data_coords_t input_view_start(
                0,
                input_crop_from_left,
                input_crop_from_up,
                0,
                0,
                0
            );

            nn_workload_data_coords_t input_view_end(
                input->get_length(NN_DATA_COORD_n) - 1,
                input->get_length(NN_DATA_COORD_x) - 1 - input_crop_from_right,
                input->get_length(NN_DATA_COORD_y) - 1 - input_crop_from_down,
                input->get_length(NN_DATA_COORD_z) - 1,
                input->get_length(NN_DATA_COORD_p) - 1,
                input->get_length(NN_DATA_COORD_q) - 1
            );

            nn_workload_data_coords_t output_view_start(
                0,
                output_crop_from_left,
                output_crop_from_up,
                0,
                0,
                0
            );

            nn_workload_data_coords_t output_view_end(
                output->get_length(NN_DATA_COORD_n) - 1,
                output->get_length(NN_DATA_COORD_x) - 1 - output_crop_from_right,
                output->get_length(NN_DATA_COORD_y) - 1 - output_crop_from_down,
                output->get_length(NN_DATA_COORD_z) - 1,
                output->get_length(NN_DATA_COORD_p) - 1,
                output->get_length(NN_DATA_COORD_q) - 1
            );

            bool valid_optimized_view = false;
            nn::workload_data<float>* input_subview = nullptr;
            nn::workload_data<float>* output_subview = nullptr;

            // Run optimized convolution on subview if there is anything to process after crop.
            if (static_cast<int32_t>(output_view_start.t[NN_DATA_COORD_x]) <= static_cast<int32_t>(output_view_end.t[NN_DATA_COORD_x]) &&
                static_cast<int32_t>(output_view_start.t[NN_DATA_COORD_y]) <= static_cast<int32_t>(output_view_end.t[NN_DATA_COORD_y]))
            {
                valid_optimized_view = true;
                input_subview = new nn::workload_data<float>(
                    *const_cast<nn::workload_data<float> *>(input), input_view_start, input_view_end);
                output_subview = new nn::workload_data<float>(*output, output_view_start, output_view_end);

                switch (activation.function) {
                case NN_ACTIVATION_FUNCTION_NONE: run_convolution<NN_ACTIVATION_FUNCTION_NONE>(input_subview, padding, center_offset_x, center_offset_y, stride_x, stride_y, weights, bias, output_subview); break;
                case NN_ACTIVATION_FUNCTION_RELU: run_convolution<NN_ACTIVATION_FUNCTION_RELU>(input_subview, padding, center_offset_x, center_offset_y, stride_x, stride_y, weights, bias, output_subview); break;
                }
            }

            // Process cropped items - if there are any.
            if (output_crop_from_left > 0 ||
                output_crop_from_up > 0 ||
                output_crop_from_right > 0 ||
                output_crop_from_down > 0 )
            {
                // Naive implementation for the rest of view.
                const auto output_image_view_start = output->view_begin.t[NN_DATA_COORD_n];
                const auto output_image_view_end = output->view_end.t[NN_DATA_COORD_n];
                const auto kernel_out_fmap_view_start = weights->view_begin.t[NN_DATA_COORD_q] * C_slice_size;
                const auto bias_view_start = bias->view_begin.t[NN_DATA_COORD_x];

                for (int32_t output_y = output_base_view_start_y; output_y <= output_base_view_end_y; ++output_y)
                {
                    for (int32_t output_x = output_base_view_start_x; output_x <= output_base_view_end_x; ++output_x)
                    {
                        if (valid_optimized_view &&
                            output_x >= output_subview->view_begin.t[NN_DATA_COORD_x] && output_x <= output_subview->view_end.t[NN_DATA_COORD_x] &&
                            output_y >= output_subview->view_begin.t[NN_DATA_COORD_y] && output_y <= output_subview->view_end.t[NN_DATA_COORD_y])
                        {
                            // Already processed by optimized code..
                            continue;
                        }

                        for (auto out_image = output_image_view_start;
                            out_image <= output_image_view_end;
                            ++out_image)
                        {
                            const auto input_image_offset = num_ifm * ifm_width * ifm_height * out_image;
                            const auto output_image_offset = num_ofm * ofm_width * ofm_height * out_image;
                            for (auto out_feature_map = output_fm_view_start, kernel_feature_map = kernel_out_fmap_view_start, bias_feature_map = bias_view_start; 
                                out_feature_map <= output_fm_view_end; 
                                out_feature_map += C_simd_width, kernel_feature_map += C_simd_width, bias_feature_map += C_simd_width)
                            {

                                // Input reading offset for left-upper corner.
                                int32_t left_up_read_offset_x = (output_x - output_base_view_start_x)*stride_x - required_center_offset_from_left + input_base_view_start_x;
                                int32_t left_up_read_offset_y = (output_y - output_base_view_start_y)*stride_y - required_center_offset_from_up + input_base_view_start_y;

                                // Input reading offset for right-bottom corner.
                                int32_t right_down_read_offset_x = left_up_read_offset_x + kernel_width;
                                int32_t right_down_read_offset_y = left_up_read_offset_y + kernel_height;

                                // Compute which part of kernel should be used from(if input reading offset was negative we must crop filter).
                                uint32_t kernel_start_offset_x = std::max(-left_up_read_offset_x, 0);
                                uint32_t kernel_start_offset_y = std::max(-left_up_read_offset_y, 0);

                                // Compute which part of kernel should be used to(if input reading end offset goes out-of-data we must crop the filter).
                                uint32_t kernel_end_offset_x = kernel_width + std::min(ifm_width - right_down_read_offset_x, 0);
                                uint32_t kernel_end_offset_y = kernel_height + std::min(ifm_height - right_down_read_offset_y, 0);

                                // Get starting point for input.
                                uint32_t input_start_offset_x = std::max(left_up_read_offset_x, 0);
                                uint32_t input_start_offset_y = std::max(left_up_read_offset_y, 0);

                                // Compute data buffer offsets for input, output and weights.
                                uint32_t output_element = num_ofm * output_x + num_ofm * ofm_width * output_y + out_feature_map + output_image_offset;
                                uint32_t input_element = num_ifm * input_start_offset_x + num_ifm * ifm_width * input_start_offset_y + input_image_offset;

                                uint32_t weight_slice_id = kernel_feature_map / C_slice_size;
                                uint32_t weight_slice_element = kernel_feature_map % C_slice_size;

                                uint32_t input_fmap_view_start = input->view_begin.t[NN_DATA_COORD_z];
                                uint32_t input_fmap_view_length = input->view_end.t[NN_DATA_COORD_z] - input_fmap_view_start + 1;
                                uint32_t kernel_input_fmap_view_start = weights->view_begin.t[NN_DATA_COORD_z];

                                uint32_t kernel_depth_size = input_fmap_view_length * C_slice_size;
                                uint32_t kernel_depth_offset = kernel_input_fmap_view_start * C_slice_size;

                                uint32_t weight_element =
                                    weight_slice_element
                                    + kernel_start_offset_y * C_slice_size * kernel_depth * kernel_width
                                    + kernel_width*kernel_height*kernel_depth*weight_slice_id*C_slice_size
                                    + kernel_start_offset_x * C_slice_size * kernel_depth;

                                // Read from bias.
                                __m256 acc0 = _mm256_load_ps(reinterpret_cast<float*>(bias->parent->data_buffer) + bias_feature_map);

                                // Run convolution.
                                for (uint32_t kernel_y = kernel_start_offset_y; kernel_y < kernel_end_offset_y; ++kernel_y)
                                {
                                    uint32_t input_y_offset = input_element + (kernel_y - kernel_start_offset_y)*num_ifm*ifm_width;
                                    for (uint32_t kernel_x = kernel_start_offset_x; kernel_x < kernel_end_offset_x; ++kernel_x)
                                    {
                                        uint32_t weight_ptr_offset = weight_element + kernel_depth_offset;
                                        uint32_t input_ptr_offset = input_y_offset + input_fmap_view_start;
                                        for (uint32_t kernel_z = 0u; kernel_z < input_fmap_view_length; ++kernel_z)
                                        {
                                            __m256 weight0 = _mm256_load_ps(reinterpret_cast<float*>(weights->parent->data_buffer) + weight_ptr_offset);
                                            __m256 inp = _mm256_broadcast_ss(reinterpret_cast<float*>(input->parent->data_buffer) + input_ptr_offset);

                                            acc0 = _mm256_fmadd_ps(weight0, inp, acc0);

                                            weight_ptr_offset += C_slice_size;
                                            ++input_ptr_offset;
                                        }
                                        input_y_offset += num_ifm;
                                        weight_element += kernel_depth*C_slice_size;
                                    }
                                }

                                if (activation.function == NN_ACTIVATION_FUNCTION_RELU)
                                {
                                    acc0 = _mm256_max_ps(_mm256_setzero_ps(), acc0);
                                }

                                _mm256_store_ps(reinterpret_cast<float*>(output->parent->data_buffer) + output_element, acc0);
                            }
                        }
                    }
                }
            }

            // Cleanup dynamic data.
            // TODO FIX THIS, temp workaround for workload_data bug with multi threading - this is a memory leak currently.
            delete reinterpret_cast<nn_workload_data_t*>(output_subview);
            delete reinterpret_cast<nn_workload_data_t*>(input_subview);
        }
        break;
    default:
        break;
    }
}

void sum_over_simd(__m256& acc)
{
    __m256i permuter = _mm256_set_epi32(0, 1, 4, 5, 2, 3, 6, 7);

    acc = _mm256_hadd_ps(acc, acc);
    acc = _mm256_permutevar8x32_ps(acc, permuter);
    acc = _mm256_hadd_ps(acc, acc);
    acc = _mm256_hadd_ps(acc, acc);
}

template <size_t T_num_accumulators>
void run_convolution_backprop_error(
                    const nn::workload_data<float> *forward_weights_view,
                    const nn::workload_data<float> *backward_input_view,
                    nn::workload_data<float> *backward_output_view,
                    uint32_t stride_x,
                    uint32_t stride_y)
{
    const auto& in_begin = backward_output_view->view_begin;
    const auto& in_end = backward_output_view->view_end;
    const auto& out_begin = backward_input_view->view_begin;
    const auto& out_end = backward_input_view->view_end;
    const auto& kernel_begin = forward_weights_view->view_begin;
    const auto& kernel_end = forward_weights_view->view_end;

    const auto output_width = backward_input_view->get_length(NN_DATA_COORD_x);
    const auto output_height = backward_input_view->get_length(NN_DATA_COORD_y);

    const auto kernel_w = forward_weights_view->get_length(NN_DATA_COORD_x);
    const auto kernel_h = forward_weights_view->get_length(NN_DATA_COORD_y);

    const auto slice_size = convolution_f32_impl::C_slice_size;

    auto forward_weights_buffer = static_cast<float*>(forward_weights_view->parent->data_buffer);
    auto backward_input_buffer = static_cast<float*>(backward_input_view->parent->data_buffer);
    auto backward_output_buffer = static_cast<float*>(backward_output_view->parent->data_buffer);

    // For backpropagation in [m] layer, we require s[m] = F'[m](n[m]) * W[m+1]^T * s[m+1] <=> s[m] = F'[m](n[m]) * U[m+1] => U[m+1] = W[m+1]^T * s[m+1].
    // Backpropagation for next layer already has computed U[m+1], now we need to compute F'[m](n[m]) part.
    // If there is no activation, then F'[m](n[m]) = 1. It means s[m] = U[m+1], so we'll use just s[m].
    // We have access to both s[m] and W[m] and so, we can compute U[m] = W[m]^T * s[m] required for previous layer.
    // Also compute weights gradient as s[m] * a[m-1]^T and compute bias gradients as s[m].
    // We already have s[m], and a[m-1] is just raw input to this layer.

    for (uint32_t batch = in_begin.t[NN_DATA_COORD_n];
         batch <= in_end.t[NN_DATA_COORD_n];
         ++batch)
    {
        for (uint32_t input_row = in_begin.t[NN_DATA_COORD_y];
             input_row <= in_end.t[NN_DATA_COORD_y];
             ++input_row)
        {
            for (uint32_t input_column = in_begin.t[NN_DATA_COORD_x];
                 input_column <= in_end.t[NN_DATA_COORD_x];
                 ++input_column)
            {
                // Find out output range that have been computed by this specific input.
                auto output_width_range_start = 0;
                auto output_height_range_start = 0;
                auto output_width_range_end = input_column / stride_x;
                auto output_height_range_end = input_row / stride_y;

                if((int32_t)input_column - ((int32_t)kernel_w - 1) >= 0) output_width_range_start = (input_column - (kernel_w - 1)) / stride_x + (((input_column - (kernel_w - 1)) % stride_x != 0 ) ? 1 : 0);
                if((int32_t)input_row - ((int32_t)kernel_h - 1) >= 0) output_height_range_start = (input_row - (kernel_h - 1)) / stride_y + (((input_row - (kernel_h - 1)) % stride_y != 0 ) ? 1 : 0);

                for (uint32_t input_feature_map = in_begin.t[NN_DATA_COORD_z], kernel_input_feature_map = kernel_begin.t[NN_DATA_COORD_z]; 
                     input_feature_map <= in_end.t[NN_DATA_COORD_z]; 
                     input_feature_map += 4, kernel_input_feature_map += 4)
                {
                    __m256 error_acc_l[T_num_accumulators];
                    __m256 error_acc_h[T_num_accumulators];

#pragma unroll(T_num_accumulators)
                    for (auto acc = 0u; acc < T_num_accumulators; ++acc)
                    {
                        error_acc_l[acc] = _mm256_setzero_ps();
                        error_acc_h[acc] = _mm256_setzero_ps();
                    }

                    uint32_t kernel_input_feature_map_offs = kernel_input_feature_map * convolution_f32_impl::C_slice_size;

                    for (int32_t output_row = output_height_range_start;
                         output_row <= output_height_range_end;
                         ++output_row)
                    {
                        for (int32_t output_column = output_width_range_start;
                            output_column <= output_width_range_end;
                            ++output_column)
                        {
                            // Compute input offset from view beginning (its output base offset as well.)
                            int32_t input_offset_w = input_column - in_begin.t[NN_DATA_COORD_x];
                            int32_t input_offset_h = input_row - in_begin.t[NN_DATA_COORD_y];

                            // Check if this output is in the view.
                            if(output_column >= 0 &&
                               output_row >= 0 &&
                               output_column < output_width &&
                               output_row < output_height)
                            {
                                // Compute weight addresses.
                                uint32_t kernel_column = input_offset_w - output_column * stride_x;
                                uint32_t kernel_row = input_offset_h - output_row * stride_y;

                                for (uint32_t output_feature_map = out_begin.t[NN_DATA_COORD_z], kernel_output_feature_map = kernel_begin.t[NN_DATA_COORD_q] * convolution_f32_impl::C_slice_size;
                                        output_feature_map <= out_end.t[NN_DATA_COORD_z];
                                        output_feature_map += convolution_f32_impl::C_slice_size, kernel_output_feature_map += convolution_f32_impl::C_slice_size)
                                {
                                    auto weight_base_offset = forward_weights_buffer
                                                              + kernel_input_feature_map_offs
                                                              + kernel_column * convolution_f32_impl::C_slice_size * forward_weights_view->parent->lengths.t[NN_DATA_COORD_z]
                                                              + kernel_row * convolution_f32_impl::C_slice_size * forward_weights_view->parent->lengths.t[NN_DATA_COORD_z] * forward_weights_view->parent->lengths.t[NN_DATA_COORD_x]
                                                              + kernel_output_feature_map / convolution_f32_impl::C_slice_size * convolution_f32_impl::C_slice_size * forward_weights_view->parent->lengths.t[NN_DATA_COORD_z] * forward_weights_view->parent->lengths.t[NN_DATA_COORD_x] * forward_weights_view->parent->lengths.t[NN_DATA_COORD_y];

                                    auto input_base_offset = backward_input_buffer
                                                             + output_feature_map 
                                                             + output_column * backward_input_view->parent->lengths.t[NN_DATA_COORD_z]
                                                             + output_row * backward_input_view->parent->lengths.t[NN_DATA_COORD_z] * backward_input_view->parent->lengths.t[NN_DATA_COORD_x]
                                                             + batch * backward_input_view->parent->lengths.t[NN_DATA_COORD_z] * backward_input_view->parent->lengths.t[NN_DATA_COORD_x] * backward_input_view->parent->lengths.t[NN_DATA_COORD_y];

                                    __m256 backward_input_l = _mm256_load_ps(input_base_offset + 0);
                                    __m256 backward_input_h = _mm256_load_ps(input_base_offset + 8);

#pragma unroll(T_num_accumulators)
                                    for (auto acc = 0u; acc < T_num_accumulators; ++acc)
                                    {
                                        error_acc_l[acc] = _mm256_fmadd_ps(backward_input_l, _mm256_load_ps(weight_base_offset + 0 + acc * convolution_f32_impl::C_slice_size), error_acc_l[acc]);
                                        error_acc_h[acc] = _mm256_fmadd_ps(backward_input_h, _mm256_load_ps(weight_base_offset + 8 + acc * convolution_f32_impl::C_slice_size), error_acc_h[acc]);
                                    }
                                }
                            }
                        }
                    }

                    __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF);

                    auto output_base_offset = backward_output_buffer 
                                              + input_feature_map 
                                              + input_column * backward_output_view->parent->lengths.t[NN_DATA_COORD_z]
                                              + input_row * backward_output_view->parent->lengths.t[NN_DATA_COORD_z] * backward_output_view->parent->lengths.t[NN_DATA_COORD_x]
                                              + batch * backward_output_view->parent->lengths.t[NN_DATA_COORD_z] * backward_output_view->parent->lengths.t[NN_DATA_COORD_x] * backward_output_view->parent->lengths.t[NN_DATA_COORD_y];
                                        
                    
                    // Squash 16 values into 1 value and save.
#pragma unroll(T_num_accumulators)
                    for (auto acc = 0u; acc < T_num_accumulators; ++acc)
                    {
                        error_acc_l[acc] = _mm256_add_ps(error_acc_l[acc], error_acc_h[acc]);
                        sum_over_simd(error_acc_l[acc]);
                        _mm256_maskstore_ps(output_base_offset + acc, mask, error_acc_l[acc]);
                    }
                }
            }
        }
    }
}

template <size_t T_num_accumulators>
void run_convolution_backprop_weight(
                    const nn::workload_data<float> *forward_input_view,
                    const nn::workload_data<float> *backward_input_view,
                    nn::workload_data<float> *backward_weights_delta_view,
                    uint32_t stride_x,
                    uint32_t stride_y,
                    uint32_t center_offset_x,
                    uint32_t center_offset_y)
{
    const auto& in_begin = forward_input_view->view_begin;
    const auto& in_end = forward_input_view->view_end;
    const auto& out_begin = backward_input_view->view_begin;
    const auto& out_end = backward_input_view->view_end;
    const auto& kernel_begin = backward_weights_delta_view->view_begin;
    const auto& kernel_end = backward_weights_delta_view->view_end;

    auto forward_input_buffer = static_cast<float*>(forward_input_view->parent->data_buffer);
    auto backward_input_buffer = static_cast<float*>(backward_input_view->parent->data_buffer);
    auto backward_weights_delta_buffer = static_cast<float*>(backward_weights_delta_view->parent->data_buffer);

    // For backpropagation in [m] layer, we require s[m] = F'[m](n[m]) * W[m+1]^T * s[m+1] <=> s[m] = F'[m](n[m]) * U[m+1] => U[m+1] = W[m+1]^T * s[m+1].
    // Backpropagation for next layer already has computed U[m+1], now we need to compute F'[m](n[m]) part.
    // If there is no activation, then F'[m](n[m]) = 1. It means s[m] = U[m+1], so we'll use just s[m].
    // We have access to both s[m] and W[m] and so, we can compute U[m] = W[m]^T * s[m] required for previous layer.
    // Also compute weights gradient as s[m] * a[m-1]^T and compute bias gradients as s[m].
    // We already have s[m], and a[m-1] is just raw input to this layer.

    for (uint32_t kernel_row = kernel_begin.t[NN_DATA_COORD_y]; kernel_row <= kernel_end.t[NN_DATA_COORD_y]; ++kernel_row)
    {
        for (uint32_t kernel_column = kernel_begin.t[NN_DATA_COORD_x]; kernel_column <= kernel_end.t[NN_DATA_COORD_x]; ++kernel_column)
        {
            for (uint32_t output_feature_map = kernel_begin.t[NN_DATA_COORD_q];
                        output_feature_map <= kernel_end.t[NN_DATA_COORD_q];
                        ++output_feature_map)
            {
                for (uint32_t input_feature_map = kernel_begin.t[NN_DATA_COORD_z]; input_feature_map <= kernel_end.t[NN_DATA_COORD_z]; input_feature_map += 4)
                {
                    __m256 weight_acc_l[T_num_accumulators];
                    __m256 weight_acc_h[T_num_accumulators];
                    __m256 forward_input[T_num_accumulators];

#pragma unroll(T_num_accumulators)
                    for (auto acc = 0u; acc < T_num_accumulators; ++acc)
                    {
                        weight_acc_l[acc] = _mm256_setzero_ps();
                        weight_acc_h[acc] = _mm256_setzero_ps();
                    }

                    for (uint32_t input_row = in_begin.t[NN_DATA_COORD_y] - center_offset_y, output_row = out_begin.t[NN_DATA_COORD_y];
                            output_row <= out_end.t[NN_DATA_COORD_y];
                            input_row += stride_y, ++output_row)
                    {
                        for (uint32_t input_column = in_begin.t[NN_DATA_COORD_x] - center_offset_x, output_column = out_begin.t[NN_DATA_COORD_x];
                                output_column <= out_end.t[NN_DATA_COORD_x];
                                input_column += stride_x, ++output_column)
                        {
                            for (int32_t batch = (int32_t)in_begin.t[NN_DATA_COORD_n];
                                    batch <= in_end.t[NN_DATA_COORD_n];
                                    ++batch)
                            {
                                auto back_input_base_offset = backward_input_buffer 
                                                              + output_feature_map * convolution_f32_impl::C_slice_size
                                                              + output_column * backward_input_view->parent->lengths.t[NN_DATA_COORD_z]
                                                              + output_row * backward_input_view->parent->lengths.t[NN_DATA_COORD_z] * backward_input_view->parent->lengths.t[NN_DATA_COORD_x];

                                auto forw_input_base_offset = forward_input_buffer 
                                                              + input_feature_map 
                                                              + (input_column + kernel_column) * forward_input_view->parent->lengths.t[NN_DATA_COORD_z]
                                                              + (input_row + kernel_row) * forward_input_view->parent->lengths.t[NN_DATA_COORD_z] * forward_input_view->parent->lengths.t[NN_DATA_COORD_x];

                                __m256 backward_input_l = _mm256_load_ps(back_input_base_offset + 0);
                                __m256 backward_input_h = _mm256_load_ps(back_input_base_offset + 8);

                                
#pragma unroll(T_num_accumulators)
                                for (auto acc = 0u; acc < T_num_accumulators; ++acc)
                                {
                                    forward_input[acc] = _mm256_broadcast_ss(forw_input_base_offset + acc);
                                }

#pragma unroll(T_num_accumulators)
                                for (auto acc = 0u; acc < T_num_accumulators; ++acc)
                                {
                                    weight_acc_l[acc] = _mm256_fmadd_ps(backward_input_l, forward_input[acc], weight_acc_l[acc]);
                                    weight_acc_h[acc] = _mm256_fmadd_ps(backward_input_h, forward_input[acc], weight_acc_h[acc]);
                                }
                            }
                        }
                    }

                    // Save 16 values.
                    auto output_base_offset = backward_weights_delta_buffer
                                              + input_feature_map * convolution_f32_impl::C_slice_size
                                              + kernel_column * convolution_f32_impl::C_slice_size * backward_weights_delta_view->parent->lengths.t[NN_DATA_COORD_z]
                                              + kernel_row * convolution_f32_impl::C_slice_size * backward_weights_delta_view->parent->lengths.t[NN_DATA_COORD_z] * backward_weights_delta_view->parent->lengths.t[NN_DATA_COORD_x]
                                              + output_feature_map * convolution_f32_impl::C_slice_size * backward_weights_delta_view->parent->lengths.t[NN_DATA_COORD_z] * backward_weights_delta_view->parent->lengths.t[NN_DATA_COORD_x] * backward_weights_delta_view->parent->lengths.t[NN_DATA_COORD_y];
                        
#pragma unroll(T_num_accumulators)
                    for (auto acc = 0u; acc < T_num_accumulators; ++acc)
                    {
                        _mm256_store_ps(output_base_offset + 0 + acc * convolution_f32_impl::C_slice_size, weight_acc_l[acc]);
                        _mm256_store_ps(output_base_offset + 8 + acc * convolution_f32_impl::C_slice_size, weight_acc_h[acc]);
                    }
                }
            }
        }
    }
}

void run_convolution_backprop_bias(
                    const nn::workload_data<float> *backward_input_view,
                    nn::workload_data<float> *backward_bias_delta_view)
{
    const auto& out_begin = backward_input_view->view_begin;
    const auto& out_end = backward_input_view->view_end;
    const auto& bias_begin = backward_bias_delta_view->view_begin;
    const auto& bias_end = backward_bias_delta_view->view_end;

    auto backward_input_buffer = static_cast<float*>(backward_input_view->parent->data_buffer);
    auto backward_bias_delta_buffer = static_cast<float*>(backward_bias_delta_view->parent->data_buffer);

    // For backpropagation in [m] layer, we require s[m] = F'[m](n[m]) * W[m+1]^T * s[m+1] <=> s[m] = F'[m](n[m]) * U[m+1] => U[m+1] = W[m+1]^T * s[m+1].
    // Backpropagation for next layer already has computed U[m+1], now we need to compute F'[m](n[m]) part.
    // If there is no activation, then F'[m](n[m]) = 1. It means s[m] = U[m+1], so we'll use just s[m].
    // We have access to both s[m] and W[m] and so, we can compute U[m] = W[m]^T * s[m] required for previous layer.
    // Also compute weights gradient as s[m] * a[m-1]^T and compute bias gradients as s[m].
    // We already have s[m], and a[m-1] is just raw input to this layer.

    for (uint32_t output_feature_map = bias_begin.t[NN_DATA_COORD_x];
            output_feature_map <= bias_end.t[NN_DATA_COORD_x];
            output_feature_map += convolution_f32_impl::C_slice_size)
    {
        __m256 bias_acc0_l = _mm256_setzero_ps();
        __m256 bias_acc0_h = _mm256_setzero_ps();

        for (uint32_t batch = (int32_t)out_begin.t[NN_DATA_COORD_n];
                batch <= out_end.t[NN_DATA_COORD_n];
                ++batch)
        {
            for (uint32_t output_row = out_begin.t[NN_DATA_COORD_y];
                    output_row <= out_end.t[NN_DATA_COORD_y];
                    ++output_row)
            {
                for (uint32_t output_column = out_begin.t[NN_DATA_COORD_x];
                        output_column <= out_end.t[NN_DATA_COORD_x];
                        ++output_column)
                {
                    __m256 backward_input_l = _mm256_load_ps(backward_input_buffer 
                                                            + 0
                                                            + output_feature_map 
                                                            + output_column * backward_input_view->parent->lengths.t[NN_DATA_COORD_z]
                                                            + output_row * backward_input_view->parent->lengths.t[NN_DATA_COORD_z] * backward_input_view->parent->lengths.t[NN_DATA_COORD_x]
                                                            + batch * backward_input_view->parent->lengths.t[NN_DATA_COORD_z] * backward_input_view->parent->lengths.t[NN_DATA_COORD_x] * backward_input_view->parent->lengths.t[NN_DATA_COORD_y]);

                    __m256 backward_input_h = _mm256_load_ps(backward_input_buffer 
                                                            + 8
                                                            + output_feature_map 
                                                            + output_column * backward_input_view->parent->lengths.t[NN_DATA_COORD_z]
                                                            + output_row * backward_input_view->parent->lengths.t[NN_DATA_COORD_z] * backward_input_view->parent->lengths.t[NN_DATA_COORD_x]
                                                            + batch * backward_input_view->parent->lengths.t[NN_DATA_COORD_z] * backward_input_view->parent->lengths.t[NN_DATA_COORD_x] * backward_input_view->parent->lengths.t[NN_DATA_COORD_y]);

                    bias_acc0_l = _mm256_add_ps(backward_input_l, bias_acc0_l);
                    bias_acc0_h = _mm256_add_ps(backward_input_h, bias_acc0_h);
                }
            }
        }

        // Save 16 values.
        _mm256_store_ps(backward_bias_delta_buffer
                        + 0
                        + output_feature_map
                        , bias_acc0_l);

        _mm256_store_ps(backward_bias_delta_buffer
                        + 8
                        + output_feature_map
                        , bias_acc0_h);
    }
}

void choose_template_convolution_backprop_weight(
                    const nn::workload_data<float> *forward_input_view,
                    const nn::workload_data<float> *backward_input_view,
                    nn::workload_data<float> *backward_weights_delta_view,
                    uint32_t stride_x,
                    uint32_t stride_y,
                    uint32_t center_offset_x,
                    uint32_t center_offset_y,
                    uint32_t num_accumulators)
{
    switch(num_accumulators)
    {
    case 1:  run_convolution_backprop_weight<1>(forward_input_view, backward_input_view, backward_weights_delta_view, stride_x, stride_y, center_offset_x, center_offset_y); break;
    case 2:  run_convolution_backprop_weight<2>(forward_input_view, backward_input_view, backward_weights_delta_view, stride_x, stride_y, center_offset_x, center_offset_y); break;
    case 3:  run_convolution_backprop_weight<3>(forward_input_view, backward_input_view, backward_weights_delta_view, stride_x, stride_y, center_offset_x, center_offset_y); break;
    case 4:  run_convolution_backprop_weight<4>(forward_input_view, backward_input_view, backward_weights_delta_view, stride_x, stride_y, center_offset_x, center_offset_y); break;
    default: throw std::invalid_argument(""); break;
    }
}

void choose_template_convolution_backprop_error(
                    const nn::workload_data<float> *forward_weights_view,
                    const nn::workload_data<float> *backward_input_view,
                    nn::workload_data<float> *backward_output_view,
                    uint32_t stride_x,
                    uint32_t stride_y,
                    uint32_t num_accumulators)
{
    switch(num_accumulators)
    {
    case 1:  run_convolution_backprop_error<1>(forward_weights_view, backward_input_view, backward_output_view, stride_x, stride_y); break;
    case 2:  run_convolution_backprop_error<2>(forward_weights_view, backward_input_view, backward_output_view, stride_x, stride_y); break;
    case 3:  run_convolution_backprop_error<3>(forward_weights_view, backward_input_view, backward_output_view, stride_x, stride_y); break;
    case 4:  run_convolution_backprop_error<4>(forward_weights_view, backward_input_view, backward_output_view, stride_x, stride_y); break;
    default: throw std::invalid_argument(""); break;
    }
}
using parameters_convolution_f32_avx2 = std::tuple<nn::workload_data<float> *,
                                                   NN_PADDING_MODE,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   const nn_argument_activation_t *,
                                                   nn::workload_data<float> *,
                                                   nn::workload_data<float> *,
                                                   nn::workload_data<float> *>;

using parameters_convolution_backprop_error_f32_avx2 = std::tuple<
                                                   const nn::workload_data<float> *,
                                                   const nn::workload_data<float> *, 
                                                   nn::workload_data<float> *,
                                                   size_t,
                                                   size_t,
                                                   size_t>;

using parameters_convolution_backprop_weight_f32_avx2 = std::tuple<
                                                   const nn::workload_data<float> *,
                                                   const nn::workload_data<float> *, 
                                                   nn::workload_data<float> *,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t>;

using parameters_convolution_backprop_bias_f32_avx2 = std::tuple<
                                                   const nn::workload_data<float> *, 
                                                   nn::workload_data<float> *>;

void unpack_convolve_callback_handle(
    void* void_handle)
{
    parameters_convolution_f32_avx2& handle = *reinterpret_cast<parameters_convolution_f32_avx2*>(void_handle);
    choose_convolution_padding_mode_and_activation(std::get<0>(handle),
                                                   std::get<1>(handle),
                                                   std::get<2>(handle),
                                                   std::get<3>(handle),
                                                   std::get<4>(handle),
                                                   std::get<5>(handle),
                                                   *std::get<6>(handle),
                                                   std::get<7>(handle),
                                                   std::get<8>(handle),
                                                   std::get<9>(handle));
}

void unpack_convolve_callback_handle_backward_error(
    void* void_handle)
{
    auto& handle = *reinterpret_cast<parameters_convolution_backprop_error_f32_avx2*>(void_handle);

    choose_template_convolution_backprop_error(
                        std::get<0>(handle),
                        std::get<1>(handle),
                        std::get<2>(handle),
                        std::get<3>(handle),
                        std::get<4>(handle),
                        std::get<5>(handle));
}

void unpack_convolve_callback_handle_backward_weight(
    void* void_handle)
{
    auto& handle = *reinterpret_cast<parameters_convolution_backprop_weight_f32_avx2*>(void_handle);

    choose_template_convolution_backprop_weight(
                                 std::get<0>(handle),
                                 std::get<1>(handle),
                                 std::get<2>(handle),
                                 std::get<3>(handle),
                                 std::get<4>(handle),
                                 std::get<5>(handle),
                                 std::get<6>(handle),
                                 std::get<7>(handle));
}

void unpack_convolve_callback_handle_backward_bias(
    void* void_handle)
{
    auto& handle = *reinterpret_cast<parameters_convolution_backprop_bias_f32_avx2*>(void_handle);
    run_convolution_backprop_bias(
                             std::get<0>(handle),
                             std::get<1>(handle));
}


} // namespace convolution_f32_impl

void convolution_f32::forward(const nn::workload_data<float> *input_buffer,
                              const nn::workload_data<float> *weights_buffer,
                              const nn::workload_data<float> *bias_buffer,
                              nn::workload_data<float> *output_buffer) 
{
    auto num_output_fm_items =
        (output_buffer->view_end.t[NN_DATA_COORD_z] - output_buffer->view_begin.t[NN_DATA_COORD_z] + 1) /
        convolution_f32_impl::C_slice_size;

    const auto num_output_fm_items_remainder =
        (output_buffer->view_end.t[NN_DATA_COORD_z] - output_buffer->view_begin.t[NN_DATA_COORD_z] + 1) %
        convolution_f32_impl::C_slice_size;

    if(num_output_fm_items_remainder) ++num_output_fm_items;

    const auto num_batch_items =
        (output_buffer->view_end.t[NN_DATA_COORD_n] - output_buffer->view_begin.t[NN_DATA_COORD_n] + 1);

    const auto total_workers = num_output_fm_items * num_batch_items;

    if (device->thread_pool.get_num_threads() < 2 || total_workers < 2)
    {
        // Its tiny data or there is only one thread available - just do it singlethreaded way.
        convolution_f32_impl::choose_convolution_padding_mode_and_activation(input_buffer, padding, center_offset_x, center_offset_y, stride_x, stride_y, activation, weights_buffer, bias_buffer, output_buffer);
    }
    else
    {
        // Full cores utilization version.
        std::vector<nn::workload_data<float> *> input_views(total_workers);
        std::vector<nn::workload_data<float> *> weight_views(total_workers);
        std::vector<nn::workload_data<float> *> bias_views(total_workers);
        std::vector<nn::workload_data<float> *> output_views(total_workers);

        const auto cpp_master_input = input_buffer;
        const auto cpp_master_output = output_buffer;
        const auto cpp_master_weights = weights_buffer;

        // Fill slave work items.
        for (auto output_fm_item = 0u; output_fm_item < num_output_fm_items; ++output_fm_item)
        {
            for (auto batch_item = 0u; batch_item < num_batch_items; ++batch_item)
            {
                auto item_in_pool = batch_item + output_fm_item * num_batch_items;

                // Replace nn_workload_datas pointers with views.
                nn_workload_data_coords_t input_view_begin(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                );
                nn_workload_data_coords_t input_view_end(
                    cpp_master_input->get_length(NN_DATA_COORD_n) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_x) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_y) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_z) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_p) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_q) - 1
                );

                nn_workload_data_coords_t output_view_begin(
                    batch_item,
                    0,
                    0,
                    output_fm_item * convolution_f32_impl::C_slice_size,
                    0,
                    0
                );
                nn_workload_data_coords_t output_view_end(
                    batch_item,
                    cpp_master_output->get_length(NN_DATA_COORD_x) - 1,
                    cpp_master_output->get_length(NN_DATA_COORD_y) - 1,
                    (output_fm_item+1) * convolution_f32_impl::C_slice_size - 1,
                    cpp_master_output->get_length(NN_DATA_COORD_p) - 1,
                    cpp_master_output->get_length(NN_DATA_COORD_q) - 1
                );

                nn_workload_data_coords_t weights_view_begin(
                    0,
                    0,
                    0,
                    0,
                    0,
                    output_fm_item
                );
                nn_workload_data_coords_t weights_view_end(
                    cpp_master_weights->get_length(NN_DATA_COORD_n) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_x) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_y) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_z) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_p) - 1,
                    output_fm_item
                );

                if(output_fm_item+1 == num_output_fm_items && num_output_fm_items_remainder)
                {
                    // Case where we need to process only remaining FMaps.
                    output_view_end.t[NN_DATA_COORD_z] = output_view_begin.t[NN_DATA_COORD_z] + num_output_fm_items_remainder - 1;
                    weights_view_end.t[NN_DATA_COORD_p] = num_output_fm_items_remainder - 1;
                }

                input_views[item_in_pool] = 
                    new nn::workload_data<float>(const_cast<nn::workload_data<float>&>(*cpp_master_input), input_view_begin, input_view_end);

                output_views[item_in_pool] =
                    new nn::workload_data<float>(*cpp_master_output, output_view_begin, output_view_end);

                weight_views[item_in_pool] =
                    new nn::workload_data<float>(const_cast<nn::workload_data<float>&>(*cpp_master_weights), weights_view_begin, weights_view_end);

                // Use biases.
                if (bias_buffer != nullptr)
                {
                    const auto cpp_master_biases = bias_buffer;

                    nn_workload_data_coords_t bias_view_begin(
                        0,
                        output_fm_item * convolution_f32_impl::C_slice_size,
                        0,
                        0,
                        0,
                        0
                    );
                    nn_workload_data_coords_t bias_view_end(
                        cpp_master_biases->get_length(NN_DATA_COORD_n) - 1,
                        (output_fm_item+1) * convolution_f32_impl::C_slice_size - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_y) - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_z) - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_p) - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_q) - 1
                    );

                    if(output_fm_item+1 == num_output_fm_items && num_output_fm_items_remainder)
                    {
                        // Case where we need to process only remaining FMaps.
                        bias_view_end.t[NN_DATA_COORD_x] = bias_view_begin.t[NN_DATA_COORD_x] + num_output_fm_items_remainder - 1;
                    }

                    bias_views[item_in_pool] =
                        new nn::workload_data<float>(const_cast<nn::workload_data<float>&>(*cpp_master_biases), bias_view_begin, bias_view_end);
                } else {
                    bias_views[item_in_pool] = nullptr;
                }
            }
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(total_workers);
        std::vector<convolution_f32_impl::parameters_convolution_f32_avx2> request_handles(total_workers);

        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            request_handles[item_in_pool] = std::make_tuple(input_views[item_in_pool],
                                                            padding,
                                                            center_offset_x,
                                                            center_offset_y,
                                                            stride_x,
                                                            stride_y,
                                                            &activation,
                                                            weight_views[item_in_pool],
                                                            bias_views[item_in_pool],
                                                            output_views[item_in_pool]);

            job[item_in_pool].callback = convolution_f32_impl::unpack_convolve_callback_handle;
            job[item_in_pool].request_handle = &request_handles[item_in_pool];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);

        // Cleanup dynamic memory.
        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            delete input_views[item_in_pool];
            delete output_views[item_in_pool];
            delete bias_views[item_in_pool];
            delete weight_views[item_in_pool];
        }
    }
}

void convolution_f32::forward(const std::vector<const nn_workload_data_t *> &inputs,
                              const std::vector<const nn_workload_data_t *> &parameters,
                              const std::vector<nn_workload_data_t *> &outputs) {
    assert(inputs.size() == 1);
    assert(parameters.size() == 2);
    assert(outputs.size() == 1);

    forward(reinterpret_cast<const nn::workload_data<float> *>(inputs[0]),
            reinterpret_cast<const nn::workload_data<float> *>(parameters[0]),
            reinterpret_cast<const nn::workload_data<float> *>(parameters[1]),
            reinterpret_cast<nn::workload_data<float> *>(outputs[0]));
}

void convolution_f32::backward_input_delta(
                               const nn::workload_data<float> *forward_weights_view,
                               const nn::workload_data<float> *backward_input_view,
                               nn::workload_data<float> *backward_output_view)
{
    auto num_input_fm_items = backward_output_view->view_end.t[NN_DATA_COORD_z] - backward_output_view->view_begin.t[NN_DATA_COORD_z] + 1;
    auto num_accumulators = 1;

    // Check how many accumulators we can use.
    if(num_input_fm_items % 4 == 0)
    {
        num_accumulators = 4;
        num_input_fm_items /= 4;
    }
    else if(num_input_fm_items % 3 == 0)
    {
        num_accumulators = 3;
        num_input_fm_items /= 3;
    }
    else if(num_input_fm_items % 2 == 0)
    {
        num_accumulators = 2;
        num_input_fm_items /= 2;
    }

    // Remove paddings from view.
    nn_workload_data_coords_t begin(
        backward_output_view->view_begin.t[NN_DATA_COORD_n],
        backward_output_view->view_begin.t[NN_DATA_COORD_x] - center_offset_x,
        backward_output_view->view_begin.t[NN_DATA_COORD_y] - center_offset_y,
        backward_output_view->view_begin.t[NN_DATA_COORD_z],
        backward_output_view->view_begin.t[NN_DATA_COORD_p],
        backward_output_view->view_begin.t[NN_DATA_COORD_q]
    );

    nn_workload_data_coords_t end(
        backward_output_view->view_end.t[NN_DATA_COORD_n],
        backward_output_view->view_end.t[NN_DATA_COORD_x] + (static_cast<uint32_t>(kernel_w) - 1 - center_offset_x),
        backward_output_view->view_end.t[NN_DATA_COORD_y] + (static_cast<uint32_t>(kernel_h) - 1 - center_offset_y),
        backward_output_view->view_end.t[NN_DATA_COORD_z],
        backward_output_view->view_end.t[NN_DATA_COORD_p],
        backward_output_view->view_end.t[NN_DATA_COORD_q]
    );

    nn::workload_data<float> backward_output_whole_buffer(backward_output_view->parent->data_buffer, backward_output_view->parent->lengths, backward_output_view->parent->layout);
    nn::workload_data<float> backward_output_no_padding_view(backward_output_whole_buffer, begin, end);

    const auto num_batch_items =
        (backward_output_view->view_end.t[NN_DATA_COORD_n] - backward_output_view->view_begin.t[NN_DATA_COORD_n] + 1);

    const auto total_workers = num_input_fm_items * num_batch_items;

    if (device->thread_pool.get_num_threads() < 2 || total_workers < 2)
    {
        // Its tiny data or there is only one thread available - just do it singlethreaded way.
        convolution_f32_impl::choose_template_convolution_backprop_error(
            forward_weights_view,
            backward_input_view,
            &backward_output_no_padding_view,
            stride_x,
            stride_y,
            num_accumulators);
    }
    else
    {
        // Full cores utilization version.
        std::vector<nn::workload_data<float> *> backprop_output_delta_views(total_workers);

        // Fill slave work items.
        for (auto input_fm_item = 0u; input_fm_item < num_input_fm_items; ++input_fm_item)
        {
            for (auto batch_item = 0u; batch_item < num_batch_items; ++batch_item)
            {
                auto item_in_pool = batch_item + input_fm_item * num_batch_items;

                // Replace nn_workload_datas pointers with views.
                nn_workload_data_coords_t output_view_begin(
                    batch_item,
                    0,
                    0,
                    input_fm_item * num_accumulators,
                    0,
                    0
                );
                nn_workload_data_coords_t output_view_end(
                    batch_item,
                    backward_output_no_padding_view.get_length(NN_DATA_COORD_x) - 1,
                    backward_output_no_padding_view.get_length(NN_DATA_COORD_y) - 1,
                    (input_fm_item+1) * num_accumulators - 1,
                    backward_output_no_padding_view.get_length(NN_DATA_COORD_p) - 1,
                    backward_output_no_padding_view.get_length(NN_DATA_COORD_q) - 1
                );

                backprop_output_delta_views[item_in_pool] =
                    new nn::workload_data<float>(backward_output_no_padding_view, output_view_begin, output_view_end);

            }
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(total_workers);
        std::vector<convolution_f32_impl::parameters_convolution_backprop_error_f32_avx2> request_handles(total_workers);

        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            request_handles[item_in_pool] = std::make_tuple(
                                                            forward_weights_view,
                                                            backward_input_view,
                                                            backprop_output_delta_views[item_in_pool],
                                                            stride_x,
                                                            stride_y,
                                                            num_accumulators);

            job[item_in_pool].callback = convolution_f32_impl::unpack_convolve_callback_handle_backward_error;
            job[item_in_pool].request_handle = &request_handles[item_in_pool];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);

        // Cleanup dynamic memory.
        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            delete backprop_output_delta_views[item_in_pool];
        }
    }
}

void convolution_f32::backward_bias_delta(
                               const nn::workload_data<float> *backward_input_view,
                               nn::workload_data<float> *backward_bias_delta_view)
{
    const auto num_output_fm_items =
        (backward_bias_delta_view->view_end.t[NN_DATA_COORD_x] - backward_bias_delta_view->view_begin.t[NN_DATA_COORD_x] + 1) / convolution_f32_impl::C_slice_size;

    const auto total_workers = num_output_fm_items;

    if (device->thread_pool.get_num_threads() < 2 || total_workers < 2)
    {
        // Its tiny data or there is only one thread available - just do it singlethreaded way.
        convolution_f32_impl::run_convolution_backprop_bias(
            backward_input_view,
            backward_bias_delta_view);
    }
    else
    {
        // Full cores utilization version.
        std::vector<nn::workload_data<float> *> backprop_bias_delta_views(total_workers);

        // Fill slave work items.
        for (auto output_fm_item = 0u; output_fm_item < num_output_fm_items; ++output_fm_item)
        {

            auto item_in_pool = output_fm_item;

            nn_workload_data_coords_t bias_view_begin(
                0,
                output_fm_item * convolution_f32_impl::C_slice_size,
                0,
                0,
                0,
                0
            );
            nn_workload_data_coords_t bias_view_end(
                backward_bias_delta_view->get_length(NN_DATA_COORD_n) - 1,
                (output_fm_item+1) * convolution_f32_impl::C_slice_size - 1,
                backward_bias_delta_view->get_length(NN_DATA_COORD_y) - 1,
                backward_bias_delta_view->get_length(NN_DATA_COORD_z) - 1,
                backward_bias_delta_view->get_length(NN_DATA_COORD_p) - 1,
                backward_bias_delta_view->get_length(NN_DATA_COORD_q) - 1
            );

            backprop_bias_delta_views[item_in_pool] =
                new nn::workload_data<float>(*backward_bias_delta_view, bias_view_begin, bias_view_end);
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(total_workers);
        std::vector<convolution_f32_impl::parameters_convolution_backprop_bias_f32_avx2> request_handles(total_workers);

        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            request_handles[item_in_pool] = std::make_tuple(
                                                            backward_input_view,
                                                            backprop_bias_delta_views[item_in_pool]);

            job[item_in_pool].callback = convolution_f32_impl::unpack_convolve_callback_handle_backward_bias;
            job[item_in_pool].request_handle = &request_handles[item_in_pool];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);

        // Cleanup dynamic memory.
        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            delete backprop_bias_delta_views[item_in_pool];
        }
    }
}

void convolution_f32::backward_weights_delta(
                               const nn::workload_data<float> *forward_input_view,
                               const nn::workload_data<float> *backward_input_view,
                               nn::workload_data<float> *backward_weights_delta_view)
{
    const auto num_output_fm_items =
        (backward_input_view->view_end.t[NN_DATA_COORD_z] - backward_input_view->view_begin.t[NN_DATA_COORD_z] + 1) / convolution_f32_impl::C_slice_size;

    auto num_input_fm_items = backward_weights_delta_view->view_end.t[NN_DATA_COORD_z] - backward_weights_delta_view->view_begin.t[NN_DATA_COORD_z] + 1;
    auto num_accumulators = 1;

    // Check how many accumulators we can use.
    if(num_input_fm_items % 4 == 0)
    {
        num_accumulators = 4;
        num_input_fm_items /= 4;
    }
    else if(num_input_fm_items % 3 == 0)
    {
        num_accumulators = 3;
        num_input_fm_items /= 3;
    }
    else if(num_input_fm_items % 2 == 0)
    {
        num_accumulators = 2;
        num_input_fm_items /= 2;
    }

    const auto total_workers = num_output_fm_items * num_input_fm_items;

    if (device->thread_pool.get_num_threads() < 2 || total_workers < 2)
    {
        // Its tiny data or there is only one thread available - just do it singlethreaded way.
        convolution_f32_impl::choose_template_convolution_backprop_weight(
            forward_input_view,
            backward_input_view,
            backward_weights_delta_view,
            stride_x,
            stride_y,
            center_offset_x,
            center_offset_y,
            num_accumulators);
    }
    else
    {
        // Full cores utilization version.
        std::vector<nn::workload_data<float> *> backprop_weight_delta_views(total_workers);

        // Fill slave work items.
        for (auto output_fm_item = 0u; output_fm_item < num_output_fm_items; ++output_fm_item)
        {
            for (auto input_fm_item = 0u; input_fm_item < num_input_fm_items; ++input_fm_item)
            {
                auto item_in_pool = input_fm_item + output_fm_item * num_input_fm_items;

                // Replace nn_workload_datas pointers with views.

                nn_workload_data_coords_t weights_view_begin = {
                    0,
                    0,
                    0,
                    input_fm_item * num_accumulators,
                    0,
                    output_fm_item,
                };
                nn_workload_data_coords_t weights_view_end = {
                    backward_weights_delta_view->get_length(NN_DATA_COORD_n) - 1,
                    backward_weights_delta_view->get_length(NN_DATA_COORD_x) - 1,
                    backward_weights_delta_view->get_length(NN_DATA_COORD_y) - 1,
                    (input_fm_item+1)*num_accumulators-1,
                    backward_weights_delta_view->get_length(NN_DATA_COORD_p) - 1,
                    output_fm_item,
                };

                backprop_weight_delta_views[item_in_pool] =
                    new nn::workload_data<float>(*backward_weights_delta_view, weights_view_begin, weights_view_end);
            }
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(total_workers);
        std::vector<convolution_f32_impl::parameters_convolution_backprop_weight_f32_avx2> request_handles(total_workers);

        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            request_handles[item_in_pool] = std::make_tuple(
                                                            forward_input_view,
                                                            backward_input_view,
                                                            backprop_weight_delta_views[item_in_pool],
                                                            stride_x,
                                                            stride_y,
                                                            center_offset_x,
                                                            center_offset_y,
                                                            num_accumulators);

            job[item_in_pool].callback = convolution_f32_impl::unpack_convolve_callback_handle_backward_weight;
            job[item_in_pool].request_handle = &request_handles[item_in_pool];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);

        // Cleanup dynamic memory.
        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            delete backprop_weight_delta_views[item_in_pool];
        }
    }
}


void convolution_f32::backward(const std::vector<nn_workload_data_t *> &inputs,
                               const std::vector<const nn_workload_data_t *> &parameters,
                               const std::vector<const nn_workload_data_t *> &outputs) {
    assert(inputs.size() == 1);
    assert(parameters.size() >= 1);
    assert(outputs.size() == 1);

    const nn::workload_data<float> backward_input(outputs[0]->parent->delta_buffer, outputs[0]->parent->lengths, outputs[0]->parent->layout);
    nn::workload_data<float> backward_output(inputs[0]->parent->delta_buffer, inputs[0]->parent->lengths, inputs[0]->parent->layout);

    backward_output.view_begin = inputs[0]->view_begin;
    backward_output.view_end= inputs[0]->view_end;

    backward_input_delta(reinterpret_cast<const nn::workload_data<float> *>(parameters[0]),
                         &backward_input,
                         &backward_output);
}

void convolution_f32::backward_parameter(size_t parameter_index,
                                         const std::vector<const nn_workload_data_t *> &inputs,
                                         const std::vector<nn_workload_data_t *> &parameters,
                                         const std::vector<const nn_workload_data_t *> &outputs) {
    assert(inputs.size() == 1);
    assert(parameters.size() >= 1);
    assert(outputs.size() == 1);

    switch (parameter_index)
    {
        case 0:
            {
            const nn::workload_data<float> backward_input(outputs[0]->parent->delta_buffer, outputs[0]->parent->lengths, outputs[0]->parent->layout);
            nn::workload_data<float> backward_weights(parameters[0]->parent->delta_buffer, parameters[0]->parent->lengths, parameters[0]->parent->layout);
            backward_weights_delta(reinterpret_cast<const nn::workload_data<float> *>(inputs[0]),
                                   &backward_input,
                                   &backward_weights);
            break;
            }
        case 1:
            {
            const nn::workload_data<float> backward_input(outputs[0]->parent->delta_buffer, outputs[0]->parent->lengths, outputs[0]->parent->layout);
            nn::workload_data<float> backward_bias(parameters[1]->parent->delta_buffer, parameters[1]->parent->lengths, parameters[1]->parent->layout);
            backward_bias_delta(&backward_input, &backward_bias);
            break;
            }
        default:
            throw std::invalid_argument("index out of range");
    }
}

void convolution_f32::backward(const nn::workload_data<float> *forward_input_view,
                               const nn::workload_data<float> *forward_weights_view,
                               const nn::workload_data<float> *forward_output_view,
                               const nn::workload_data<float> *backward_input_view,
                               nn::workload_data<float> *backward_output_view,
                               nn::workload_data<float> *backward_weights_delta_view,
                               nn::workload_data<float> *backward_bias_delta_view)
{
    backward_input_delta(forward_weights_view,
                         backward_input_view,
                         backward_output_view);

    backward_bias_delta(backward_input_view,
                        backward_bias_delta_view);

    backward_weights_delta(forward_input_view,
                           backward_input_view,
                           backward_weights_delta_view);
}

void run_multithreaded_convolve_work_item_backward(nn_workload_item *const work_item)
{
    auto primitive = static_cast<convolution_f32 *>(work_item->forward_item->primitive);

    primitive->backward(
        reinterpret_cast<nn::workload_data<float> *>(work_item->forward_item->input[0].get_data_view()),
        reinterpret_cast<const nn::workload_data<float> *>(work_item->forward_item->parameters[0]),
        reinterpret_cast<nn::workload_data<float> *>(work_item->forward_item->output[0]),
        reinterpret_cast<nn::workload_data<float> *>(work_item->input[0].get_data_view()),
        reinterpret_cast<nn::workload_data<float> *>(work_item->output[0]),
        reinterpret_cast<nn::workload_data<float> *>(work_item->output[1]),  
        reinterpret_cast<nn::workload_data<float> *>(work_item->output[2])); 
}

convolution_f32::convolution_f32(const size_t kernel_w,
                                 const size_t kernel_h,
                                 const size_t num_input,
                                 const size_t num_output,
                                 const size_t output_w,
                                 const size_t output_h,
                                 const int32_t center_offset_x,
                                 const int32_t center_offset_y,
                                 const size_t stride_x,
                                 const size_t stride_y,
                                 const nn_argument_activation_t &activation,
                                 size_t batch_size,
                                 size_t output_padding_left,
                                 size_t output_padding_right,
                                 size_t output_padding_top,
                                 size_t output_padding_bottom,
                                 nn_device_internal *device)
    : primitive_zxyn_f32_base(batch_size,
                              num_input,
                              output_w,
                              output_h,
                              num_output,
                              output_padding_left,
                              output_padding_right,
                              output_padding_top,
                              output_padding_bottom,
                              device),
      kernel_w(kernel_w),
      kernel_h(kernel_h),
      padding(NN_PADDING_MODE_DATA_OR_ZERO),
      center_offset_x(center_offset_x),
      center_offset_y(center_offset_y),
      stride_x(stride_x),
      stride_y(stride_y),
      activation(activation) {}

bool convolution_f32::validate_input(size_t index, nn_workload_data_t *data)
{
    switch (index) {
    case 0:
        return nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, float>::validate<true>(data,
                                                                                 get_required_input_w() - kernel_w + 1,
                                                                                 get_required_input_h() - kernel_h + 1,
                                                                                 input_size_z,
                                                                                 batch_size,
                                                                                 center_offset_x,
                                                                                 center_offset_y,
                                                                                 kernel_w - center_offset_x - 1,
                                                                                 kernel_h - center_offset_y - 1);
    }

    throw std::invalid_argument("index out of range");
}

size_t convolution_f32::get_required_input_w() { return (output_size_x - 1) * stride_x + kernel_w; }

size_t convolution_f32::get_required_input_h() { return (output_size_y - 1) * stride_y + kernel_h; }

std::vector<nn_workload_data_t *> convolution_f32::create_inputs(bool allocate_delta) {
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, float>::create(device,
                                                                      get_required_input_w() - kernel_w + 1,
                                                                      get_required_input_h() - kernel_h + 1,
                                                                      input_size_z,
                                                                      batch_size,
                                                                      center_offset_x,
                                                                      kernel_w - center_offset_x - 1,
                                                                      center_offset_y,
                                                                      kernel_h - center_offset_y - 1,
                                                                      allocate_delta)};
}

std::vector<nn_workload_data_t *> convolution_f32::create_parameters(bool allocate_delta)
{
    const uint32_t C_simd_width = sizeof(__m256) / sizeof(float);
    const uint32_t C_slice_size = 2 * C_simd_width;

    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIXYO, float>::create(
                device, C_slice_size, kernel_w, kernel_h, input_size_z, output_size_z, allocate_delta),
            nn::data_helper<NN_WORKLOAD_DATA_TAG_O, float>::create(device, output_size_z, allocate_delta)};
}

} // namespace layer

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_convolution_f32_create_0(
    nn_device_t *device,    /* IDLF device handle */
    size_t kernel_w,        /* kernel width */
    size_t kernel_h,        /* kernel height */
    size_t num_input,       /* number of input feature maps */
    size_t num_output,      /* number of output feature maps */
    size_t output_w,        /* output width */
    size_t output_h,        /* output height */
    size_t center_offset_x, /* horizontal offset of kernel's center point w/ relation to top left corner */
    size_t center_offset_y, /* vertical offset of kernel's center point w/ relation to top left corner */
    size_t stride_x,        /* horizontal stride */
    size_t stride_y,        /* vertical stride */
    const nn_argument_activation_t *activation, /* struct parameterizing optional activation function */
    size_t batch_size,                          /* size of input batch */
    const nn_primitives_convolution_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */) {
    SET_STATUS(NN_API_STATUS_OK);

    std::remove_const<std::remove_pointer<decltype(hints)>::type>::type hints_ = {};
    if (hints != nullptr)
        hints_ = *hints;

    return new layer::convolution_f32(kernel_w,
                                      kernel_h,
                                      num_input,
                                      num_output,
                                      output_w,
                                      output_h,
                                      center_offset_x,
                                      center_offset_y,
                                      stride_x,
                                      stride_y,
                                      *activation,
                                      batch_size,
                                      hints_.output_padding.left,
                                      hints_.output_padding.right,
                                      hints_.output_padding.top,
                                      hints_.output_padding.bottom,
                                      reinterpret_cast<nn_device_internal *>(device));
}
