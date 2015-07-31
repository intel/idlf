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
#include "layer_convolution_pooling_avx2.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <map>
#include <tuple>

// NN_CODE_UNREACHABLE signal to supporting compiler that specific location in code cannot be reached
#if defined _MSC_VER 
#   define NN_UNREACHABLE_CODE __assume(0)
#endif

#if defined __GNUC__
#   if (__GNUC__ * 100 + __GNUC_MINOR__) >= 405
#       define NN_UNREACHABLE_CODE __builtin_unreachable()
#   else
#       define NN_UNREACHABLE_CODE
#   endif
#endif

#if defined __clang__
#   if __has_builtin(__builtin_unreachable)
#       define NN_UNREACHABLE_CODE __builtin_unreachable()
#   else
#       define NN_UNREACHABLE_CODE
#   endif
#endif

// Pragmas inside macros.
#if defined _MSC_VER 
#   define PRAGMA_MACRO(x) __pragma(x)
#else
#   define PRAGMA_MACRO(x) _Pragma(#x)
#endif

// SIMD width for this implementation
const auto C_simd_width = sizeof(__m256) / sizeof(float);
const uint32_t C_slice_size = 2 * C_simd_width;

namespace layer
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// forward implementation

///////////////////////////////////////////////////////////////////////////////////////////////////
// defines to handle multiple variants of code loops
// Necessary evil because of compile bug resulting in bad ISA quality when AVX regsiters are in
// C++ array and all AVX registers are to be used in final ISA for intermost loop.

#define CREATE_ACC_UPPER(acc0, acc1, num) \
    __m256 vout_upper_##acc0 = _mm256_setzero_ps(); \
    __m256 vout_upper_##acc1 = _mm256_setzero_ps();

#define CREATE_ACC_LOWER(acc0, acc1, num) \
    __m256 vout_lower_##acc0 = _mm256_setzero_ps(); \
    __m256 vout_lower_##acc1 = _mm256_setzero_ps();

#define MAD_ACC_UPPER(acc0, acc1, num) \
    bc = _mm256_broadcast_ss(inp_ptr + offs##num); \
    vout_upper_##acc0 = _mm256_fmadd_ps(vwt0, bc, vout_upper_##acc0); \
    vout_upper_##acc1 = _mm256_fmadd_ps(vwt1, bc, vout_upper_##acc1);

#define MAD_ACC_LOWER(acc0, acc1, num) \
    bc = _mm256_broadcast_ss(inp_ptr + offs##num); \
    vout_lower_##acc0 = _mm256_fmadd_ps(vwt0, bc, vout_lower_##acc0); \
    vout_lower_##acc1 = _mm256_fmadd_ps(vwt1, bc, vout_lower_##acc1);


#define STORE_ACC(acc_left0,acc_left1,acc_right0,acc_right1,num) \
    vout_upper_##acc_left0 = _mm256_max_ps(vout_upper_##acc_left0, vout_upper_##acc_right0); \
    vout_lower_##acc_left0 = _mm256_max_ps(vout_lower_##acc_left0, vout_lower_##acc_right0); \
    vout_upper_##acc_left0 = _mm256_max_ps(vout_upper_##acc_left0, vout_lower_##acc_left0); \
    vout_upper_##acc_left0 = _mm256_add_ps(vout_upper_##acc_left0, bias0); \
    \
    vout_upper_##acc_left1 = _mm256_max_ps(vout_upper_##acc_left1, vout_upper_##acc_right1); \
    vout_lower_##acc_left1 = _mm256_max_ps(vout_lower_##acc_left1, vout_lower_##acc_right1); \
    vout_upper_##acc_left1 = _mm256_max_ps(vout_upper_##acc_left1, vout_lower_##acc_left1); \
    vout_upper_##acc_left1 = _mm256_add_ps(vout_upper_##acc_left1, bias1); \
    \
    if(T_activation == NN_ACTIVATION_FUNCTION_RELU)\
    { \
        vout_upper_##acc_left0 = _mm256_max_ps(vout_upper_##acc_left0, _mm256_setzero_ps()); \
        vout_upper_##acc_left1 = _mm256_max_ps(vout_upper_##acc_left1, _mm256_setzero_ps()); \
    } \
    \
    _mm256_store_ps(&output[internal_out_offset0 + (num) * num_output_feature_maps], vout_upper_##acc_left0); \
    _mm256_store_ps(&output[internal_out_offset1 + (num) * num_output_feature_maps], vout_upper_##acc_left1);

///////////////////////////////////////////////////////////////////////////////////////////////////
// REPLICATION MACROS

#define SIMPLE_REPLICATION_1(function) function(0,1,0)
#define SIMPLE_REPLICATION_2(function) SIMPLE_REPLICATION_1(function) function(2,3,1)
#define SIMPLE_REPLICATION_3(function) SIMPLE_REPLICATION_2(function) function(4,5,2)
#define SIMPLE_REPLICATION_4(function) SIMPLE_REPLICATION_3(function) function(6,7,3)
#define SIMPLE_REPLICATION_5(function) SIMPLE_REPLICATION_4(function) function(8,9,4)
#define SIMPLE_REPLICATION_6(function) SIMPLE_REPLICATION_5(function) function(10,11,5)

#define MERGE_REPLICATION_2(function) function(0,1,2,3,0)
#define MERGE_REPLICATION_4(function) MERGE_REPLICATION_2(function) function(4,5,6,7,1)
#define MERGE_REPLICATION_6(function) MERGE_REPLICATION_4(function) function(8,9,10,11,2)

///////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN MACRO
#define NN_CONVOLVE_MAXPOOL2x2_OPTIMIZED_BLOCK( \
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
    SIMPLE_REPLICATION_##block_size(CREATE_ACC_UPPER) \
    \
    float *init_init_kernel_offset_base_ptr = kernel + wfm; \
    float *init_init_input_offset_base_ptr = input + inp_offset_base; \
    for (auto kh = 0U; kh < kernel_height; ++kh) \
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
                    SIMPLE_REPLICATION_##block_size(MAD_ACC_UPPER) \
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
                    SIMPLE_REPLICATION_##block_size(MAD_ACC_UPPER) \
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
    SIMPLE_REPLICATION_##block_size(CREATE_ACC_LOWER) \
    \
    init_init_kernel_offset_base_ptr = kernel + wfm; \
    init_init_input_offset_base_ptr = input + inp_offset_base2; \
    for (auto kh = 0U; kh < kernel_height; ++kh) \
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
                    SIMPLE_REPLICATION_##block_size(MAD_ACC_LOWER) \
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
                    SIMPLE_REPLICATION_##block_size(MAD_ACC_LOWER) \
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
    __m256 bias0 = _mm256_load_ps((float*)bias_view->parent->data_buffer + bias_feature_map); \
    __m256 bias1 = _mm256_load_ps((float*)bias_view->parent->data_buffer + bias_feature_map + C_simd_width); \
    \
    MERGE_REPLICATION_##block_size(STORE_ACC) \
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
void convolve_maxpool_internal(const nn::workload_data<float> *input_view,
                               const nn::workload_data<float> *weights_view,
                               const nn::workload_data<float> *bias_view,
                               nn::workload_data<float> *output_view,
                               size_t _kernel_stride_x,
                               size_t _kernel_stride_y,
                               int32_t kernel_center_x,
                               int32_t kernel_center_y) {
    float* input = (float*)input_view->parent->data_buffer;
    float* output = (float*)output_view->parent->data_buffer;
    float* kernel = (float*)weights_view->parent->data_buffer;

    const auto num_output_feature_maps      = (T_exact_match) ? T_output_feature_maps : output_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto num_input_feature_maps       = (T_exact_match) ? T_input_feature_maps : input_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_feature_map_width     = (T_exact_match) ? T_output_width : output_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_feature_map_height    = (T_exact_match) ? T_output_height : output_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto input_feature_map_width      = (T_exact_match) ? T_input_width : input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto input_feature_map_height     = (T_exact_match) ? T_input_height : input_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto kernel_width                 = (T_exact_match) ? T_kernel_width : weights_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto kernel_height                = (T_exact_match) ? T_kernel_height : weights_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto kernel_stride_x              = (T_exact_match) ? T_kernel_stride_x : _kernel_stride_x;
    const auto kernel_stride_y              = (T_exact_match) ? T_kernel_stride_y : _kernel_stride_y;
    const auto kernel_input_fmap_view_start = (T_exact_match) ? T_kernel_in_fmap_view_start : weights_view->view_begin.t[NN_DATA_COORD_z];
    const auto input_fmap_view_start        = (T_exact_match) ? T_input_fmap_view_start : input_view->view_begin.t[NN_DATA_COORD_z];
    const auto input_fmap_view_length       = (T_exact_match) ? T_input_fmap_view_length : input_view->view_end.t[NN_DATA_COORD_z] - input_fmap_view_start + 1;

    const auto bias_view_start = bias_view->view_begin.t[NN_DATA_COORD_x];

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
    const auto weight_offset        = weights_view->parent->lengths.t[NN_DATA_COORD_z] * kernel_width*kernel_height * C_slice_size;
    
    const auto num_blocks_full      = output_view_width / 3;
    const auto partial_block_size   = output_view_width % 3;
    
    const auto output_image_view_start = output_view->view_begin.t[NN_DATA_COORD_n];
    const auto output_image_view_end = output_view->view_end.t[NN_DATA_COORD_n];
    
    const auto input_image_size = input_row_size * input_feature_map_height;
    const auto output_image_size = output_row_size * output_feature_map_height;

    const auto kernel_out_fmap_view_start = weights_view->view_begin.t[NN_DATA_COORD_q] * C_slice_size;
    
    for(auto out_image = output_image_view_start; out_image <= output_image_view_end; ++out_image)
    {
        auto input_image_offset = out_image*input_image_size;
        auto output_image_offset = out_image*output_image_size;
        
        for (auto out_feature_map = output_fm_view_start, kernel_feature_map = kernel_out_fmap_view_start, bias_feature_map = bias_view_start; 
            out_feature_map <= output_fm_view_end; 
            out_feature_map += C_slice_size, kernel_feature_map += C_slice_size, bias_feature_map += C_slice_size)
        {
            for (auto output_row = output_row_view_start, input_row = 0U; output_row <= output_row_view_end; output_row++, input_row+=2)
            {
                const auto inp_h        = input_row * kernel_stride_y + input_row_view_start;
                const auto inp_h2       = (input_row+1) * kernel_stride_y + input_row_view_start;
                const auto out_offset1  = output_row * output_row_size;
                const auto inp_offset1  = inp_h * input_row_size;
                const auto inp_offset2  = inp_h2 * input_row_size;
                
                auto out_offset         = out_offset1 + output_column_view_start * num_output_feature_maps + output_image_offset;
                auto inp_offset_base    = inp_offset1 + input_column_view_start * num_input_feature_maps + input_image_offset;
                auto inp_offset_base2   = inp_offset2 + input_column_view_start * num_input_feature_maps + input_image_offset;
                
                for (auto block = 0U; block < num_blocks_full; block++) {
                NN_CONVOLVE_MAXPOOL2x2_OPTIMIZED_BLOCK(6);
                    inp_offset_base += 6 * num_input_feature_maps * kernel_stride_x;
                    inp_offset_base2 += 6 * num_input_feature_maps * kernel_stride_x;
                    out_offset += 3 * num_output_feature_maps;
                }
                
                switch (partial_block_size)
                {
                case 0: break;
                case 1: NN_CONVOLVE_MAXPOOL2x2_OPTIMIZED_BLOCK(2); break;
                case 2: NN_CONVOLVE_MAXPOOL2x2_OPTIMIZED_BLOCK(4); break;
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

namespace 
{

using optimized_layer_map_t = std::map<
    std::tuple<NN_ACTIVATION_FUNCTION, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>, 
    decltype(convolve_maxpool_internal<false, NN_ACTIVATION_FUNCTION_NONE>)*>;

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
        convolve_maxpool_internal<
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
    // OverFeat C1.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 231, 231, 3, 0, 3, 0, 11, 11, 4, 4, 28, 28, 96>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 231, 231, 3, 0, 3, 0, 11, 11, 4, 4, 28, 28, 96>() },

    // OverFeat C2.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 28, 28, 96, 0, 96, 0, 5, 5, 1, 1, 14, 14, 256>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 28, 28, 96, 0, 96, 0, 5, 5, 1, 1, 14, 14, 256>() },

    // OverFeat C5.
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 14, 14, 1024, 0, 1024, 0, 3, 3, 1, 1, 6, 6, 1024>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 14, 14, 1024, 0, 1024, 0, 3, 3, 1, 1, 6, 6, 1024>() },
};

}

convolution_pooling_f32_2x2stride2::convolution_pooling_f32_2x2stride2(const size_t kernel_w,
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
    : convolution_f32(kernel_w,
                      kernel_h,
                      num_input,
                      num_output,
                      output_w,
                      output_h,
                      center_offset_x,
                      center_offset_y,
                      stride_x,
                      stride_y,
                      activation,
                      batch_size,
                      output_padding_left,
                      output_padding_right,
                      output_padding_top,
                      output_padding_bottom,
                      device) {}

size_t convolution_pooling_f32_2x2stride2::get_required_input_w() {
    return ((output_size_x - 1) * pooling_stride_x + pooling_size_x - 1) * stride_x + kernel_w;
}

size_t convolution_pooling_f32_2x2stride2::get_required_input_h() {
    return ((output_size_y - 1) * pooling_stride_y + pooling_size_y - 1) * stride_y + kernel_h;
}

template <NN_ACTIVATION_FUNCTION T_activation>
void convolution_pooling_f32_2x2stride2::run_convolution_maxpool(const nn::workload_data<float> *input_view,
                                                                 const nn::workload_data<float> *weights_view,
                                                                 const nn::workload_data<float> *bias_view,
                                                                 nn::workload_data<float> *output_view) {
    const auto num_output_feature_maps = output_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto num_input_feature_maps = input_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_feature_map_width = output_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_feature_map_height = output_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto input_feature_map_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto input_feature_map_height = input_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto kernel_width = weights_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto kernel_height = weights_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto kernel_stride_x = this->stride_x;
    const auto kernel_stride_y = this->stride_y;
    const auto kernel_input_fmap_view_start = weights_view->view_begin.t[NN_DATA_COORD_z];
    const auto input_fmap_view_start = input_view->view_begin.t[NN_DATA_COORD_z];
    const auto input_fmap_view_length = input_view->view_end.t[NN_DATA_COORD_z] - input_fmap_view_start + 1;


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
        kernel_stride_x,
        kernel_stride_y,
        output_feature_map_width,
        output_feature_map_height,
        num_output_feature_maps ));

    if (map_element != std::end(optimized_layer_map))
    {
        // Optimized.
        map_element->second(input_view,
                            weights_view,
                            bias_view,
                            output_view,
                            kernel_stride_x,
                            kernel_stride_y,
                            this->center_offset_x,
                            this->center_offset_y);
    }
    else
    {
        // Generic.
        convolve_maxpool_internal<false, T_activation>(input_view,
                                                       weights_view,
                                                       bias_view,
                                                       output_view,
                                                       kernel_stride_x,
                                                       kernel_stride_y,
                                                       this->center_offset_x,
                                                       this->center_offset_y);
    }
}

void convolution_pooling_f32_2x2stride2::choose_convolution_maxpool_padding_mode_and_activation(
    const nn::workload_data<float> *input_view,
    const nn::workload_data<float> *weights_view,
    const nn::workload_data<float> *bias_view,
    nn::workload_data<float> *output_view) {

    switch (padding)
    {
    case NN_PADDING_MODE_DATA_OR_ZERO:
    {
        // Get basic data about convolution.
        const int32_t pool_stride_x = 2;
        const int32_t pool_stride_y = 2;

        const int32_t filter_stride_x = this->stride_x;
        const int32_t filter_stride_y = this->stride_y;

        const int32_t stride_x = filter_stride_x * pool_stride_x;
        const int32_t stride_y = filter_stride_y * pool_stride_y;

        const int32_t  kernel_width = weights_view->parent->lengths.t[NN_DATA_COORD_x];
        const int32_t  kernel_height = weights_view->parent->lengths.t[NN_DATA_COORD_y];
        const uint32_t kernel_depth = weights_view->parent->lengths.t[NN_DATA_COORD_z];

        const uint32_t num_ofm = output_view->parent->lengths.t[NN_DATA_COORD_z];
        const uint32_t ofm_width = output_view->parent->lengths.t[NN_DATA_COORD_x];
        const uint32_t ofm_height = output_view->parent->lengths.t[NN_DATA_COORD_y];

        const uint32_t num_ifm = input_view->parent->lengths.t[NN_DATA_COORD_z];
        const int32_t  ifm_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
        const int32_t  ifm_height = input_view->parent->lengths.t[NN_DATA_COORD_y];

        const uint32_t output_fm_view_start = output_view->view_begin.t[NN_DATA_COORD_z];
        const uint32_t output_fm_view_end = output_view->view_end.t[NN_DATA_COORD_z];

        const uint32_t input_line_size = num_ifm*ifm_width;

        // Get data about base view.
        const int32_t input_base_view_start_x = input_view->view_begin.t[NN_DATA_COORD_x];
        const int32_t input_base_view_start_y = input_view->view_begin.t[NN_DATA_COORD_y];
        const int32_t output_base_view_start_x = output_view->view_begin.t[NN_DATA_COORD_x];
        const int32_t output_base_view_start_y = output_view->view_begin.t[NN_DATA_COORD_y];
        const int32_t output_base_view_end_x = output_view->view_end.t[NN_DATA_COORD_x];
        const int32_t output_base_view_end_y = output_view->view_end.t[NN_DATA_COORD_y];

        // Get offsets of convolution central point (last +1 for pooling adjustment).
        const int32_t required_center_offset_from_left = this->center_offset_x;
        const int32_t required_center_offset_from_up = this->center_offset_y;
        const int32_t required_center_offset_from_right = weights_view->parent->lengths.t[NN_DATA_COORD_x] - (required_center_offset_from_left + 1);
        const int32_t required_center_offset_from_down = weights_view->parent->lengths.t[NN_DATA_COORD_y] - (required_center_offset_from_up + 1);

        // Get number of input elements required along with their absolute end offsets.
        const int32_t  input_elements_required_x = (output_view->get_length(NN_DATA_COORD_x) * pool_stride_x - 1) * filter_stride_x + 1;
        const int32_t  input_elements_required_y = (output_view->get_length(NN_DATA_COORD_y) * pool_stride_y - 1) * filter_stride_x + 1;
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
        nn_workload_data_coords_t input_view_start =
        {
            0,
            input_crop_from_left,
            input_crop_from_up,
            0,
            0,
            0
        };

        nn_workload_data_coords_t input_view_end =
        {
            input_view->get_length(NN_DATA_COORD_n) - 1,
            input_view->get_length(NN_DATA_COORD_x) - 1 - input_crop_from_right,
            input_view->get_length(NN_DATA_COORD_y) - 1 - input_crop_from_down,
            input_view->get_length(NN_DATA_COORD_z) - 1,
            input_view->get_length(NN_DATA_COORD_p) - 1,
            input_view->get_length(NN_DATA_COORD_q) - 1
        };

        nn_workload_data_coords_t output_view_start =
        {
            0,
            output_crop_from_left,
            output_crop_from_up,
            0,
            0,
            0
        };

        nn_workload_data_coords_t output_view_end =
        {
            output_view->get_length(NN_DATA_COORD_n) - 1,
            output_view->get_length(NN_DATA_COORD_x) - 1 - output_crop_from_right,
            output_view->get_length(NN_DATA_COORD_y) - 1 - output_crop_from_down,
            output_view->get_length(NN_DATA_COORD_z) - 1,
            output_view->get_length(NN_DATA_COORD_p) - 1,
            output_view->get_length(NN_DATA_COORD_q) - 1
        };

        bool valid_optimized_view = false;
        nn::workload_data<float>* input_subview = nullptr;
        nn::workload_data<float>* output_subview = nullptr;

        // Run optimized convolution on subview if there is anything to process after crop.
        if (static_cast<int32_t>(output_view_start.t[NN_DATA_COORD_x]) <= static_cast<int32_t>(output_view_end.t[NN_DATA_COORD_x]) &&
            static_cast<int32_t>(output_view_start.t[NN_DATA_COORD_y]) <= static_cast<int32_t>(output_view_end.t[NN_DATA_COORD_y]))
        {
            valid_optimized_view = true;
            input_subview = new nn::workload_data<float>(*input_view, input_view_start, input_view_end);
            output_subview = new nn::workload_data<float>(*output_view, output_view_start, output_view_end);

            switch (this->activation.function)
            {
            case NN_ACTIVATION_FUNCTION_NONE: run_convolution_maxpool<NN_ACTIVATION_FUNCTION_NONE>(input_subview, weights_view, bias_view, output_subview); break;
            case NN_ACTIVATION_FUNCTION_RELU: run_convolution_maxpool<NN_ACTIVATION_FUNCTION_RELU>(input_subview, weights_view, bias_view, output_subview); break;
            }
        }

        // Process cropped items - if there are any.
        if (output_crop_from_left > 0 ||
            output_crop_from_up > 0 ||
            output_crop_from_right > 0 ||
            output_crop_from_down > 0)
        {
            // Naive implementation for the rest of view.
            const auto output_image_view_start = output_view->view_begin.t[NN_DATA_COORD_n];
            const auto output_image_view_end = output_view->view_end.t[NN_DATA_COORD_n];
            const auto kernel_out_fmap_view_start = weights_view->view_begin.t[NN_DATA_COORD_q] * C_slice_size;
            const auto bias_view_start = bias_view->view_begin.t[NN_DATA_COORD_x];

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
                            out_feature_map += C_slice_size, kernel_feature_map += C_slice_size, bias_feature_map += C_slice_size)
                        {

                            // Read from bias.
                            __m256 bias0 = _mm256_load_ps(reinterpret_cast<float*>(bias_view->parent->data_buffer) + bias_feature_map);
                            __m256 bias1 = _mm256_load_ps(reinterpret_cast<float*>(bias_view->parent->data_buffer) + bias_feature_map + C_simd_width);

                            __m256 acc0[2][2] = { { bias0, bias0 }, { bias0, bias0 } };
                            __m256 acc1[2][2] = { { bias1, bias1 }, { bias1, bias1 } };

                            for (uint32_t pool_x = 0; pool_x < 2; ++pool_x)
                            {
                                for (uint32_t pool_y = 0; pool_y < 2; ++pool_y)
                                {
                                    // Input reading offset for left-upper corner.
                                    int32_t left_up_read_offset_x = pool_x*filter_stride_x + (output_x - output_base_view_start_x)*stride_x - required_center_offset_from_left + input_base_view_start_x;
                                    int32_t left_up_read_offset_y = pool_y*filter_stride_y + (output_y - output_base_view_start_y)*stride_y - required_center_offset_from_up + input_base_view_start_y;

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

                                    // Compute data buffer offsets for input and weights.
                                    uint32_t input_element = num_ifm * input_start_offset_x + num_ifm * ifm_width * input_start_offset_y + input_image_offset;

                                    uint32_t weight_slice_id = kernel_feature_map / C_slice_size;
                                    uint32_t weight_slice_element = kernel_feature_map % C_slice_size;

                                    uint32_t input_fmap_view_start = input_view->view_begin.t[NN_DATA_COORD_z];
                                    uint32_t input_fmap_view_length = input_view->view_end.t[NN_DATA_COORD_z] - input_fmap_view_start + 1;
                                    uint32_t kernel_input_fmap_view_start = weights_view->view_begin.t[NN_DATA_COORD_z];

                                    uint32_t kernel_depth_size = input_fmap_view_length * C_slice_size;
                                    uint32_t kernel_depth_offset = kernel_input_fmap_view_start * C_slice_size;

                                    uint32_t weight_element =
                                        weight_slice_element
                                        + kernel_start_offset_y * C_slice_size * kernel_depth * kernel_width
                                        + kernel_width*kernel_height*kernel_depth*weight_slice_id*C_slice_size
                                        + kernel_start_offset_x * C_slice_size * kernel_depth;

                                    // Run convolutions.
                                    for (uint32_t kernel_y = kernel_start_offset_y; kernel_y < kernel_end_offset_y; ++kernel_y)
                                    {
                                        uint32_t input_y_offset = input_element + (kernel_y - kernel_start_offset_y)*input_line_size;
                                        for (uint32_t kernel_x = kernel_start_offset_x; kernel_x < kernel_end_offset_x; ++kernel_x)
                                        {
                                            uint32_t weight_ptr_offset = weight_element + kernel_depth_offset;
                                            uint32_t input_ptr_offset = input_y_offset + input_fmap_view_start;
                                            for (uint32_t kernel_z = 0u; kernel_z < input_fmap_view_length; ++kernel_z)
                                            {
                                                __m256 weight0 = _mm256_load_ps(reinterpret_cast<float*>(weights_view->parent->data_buffer) + weight_ptr_offset);
                                                __m256 weight1 = _mm256_load_ps(reinterpret_cast<float*>(weights_view->parent->data_buffer) + weight_ptr_offset + C_simd_width);

                                                __m256 inp = _mm256_broadcast_ss(reinterpret_cast<float*>(input_view->parent->data_buffer) + input_ptr_offset);

                                                acc0[pool_x][pool_y] = _mm256_fmadd_ps(weight0, inp, acc0[pool_x][pool_y]);
                                                acc1[pool_x][pool_y] = _mm256_fmadd_ps(weight1, inp, acc1[pool_x][pool_y]);

                                                weight_ptr_offset += C_slice_size;
                                                ++input_ptr_offset;
                                            }
                                            input_y_offset += num_ifm;
                                            weight_element += C_slice_size*kernel_depth;
                                        }
                                    }
                                }
                            }


                            acc0[0][0] = _mm256_max_ps(acc0[0][0], acc0[0][1]);
                            acc0[0][0] = _mm256_max_ps(acc0[0][0], acc0[1][0]);
                            acc0[0][0] = _mm256_max_ps(acc0[0][0], acc0[1][1]);

                            acc1[0][0] = _mm256_max_ps(acc1[0][0], acc1[0][1]);
                            acc1[0][0] = _mm256_max_ps(acc1[0][0], acc1[1][0]);
                            acc1[0][0] = _mm256_max_ps(acc1[0][0], acc1[1][1]);

                            if (this->activation.function == NN_ACTIVATION_FUNCTION_RELU)
                            {
                                acc0[0][0] = _mm256_max_ps(_mm256_setzero_ps(), acc0[0][0]);
                                acc1[0][0] = _mm256_max_ps(_mm256_setzero_ps(), acc1[0][0]);
                            }

                            uint32_t output_element = num_ofm * output_x + num_ofm * ofm_width * output_y + out_feature_map + output_image_offset;
                            _mm256_store_ps(reinterpret_cast<float*>(output_view->parent->data_buffer) + output_element, acc0[0][0]);
                            _mm256_store_ps(reinterpret_cast<float*>(output_view->parent->data_buffer) + output_element + C_simd_width, acc1[0][0]);
                        }
                    }
                }
            }
        }

        // Cleanup dynamic data.
        delete output_subview;
        delete input_subview;
    }
    break;
    default:
        break;
    }
}

struct convolution_pooling_f32_2x2stride2_request_handle {
    convolution_pooling_f32_2x2stride2 *primitive;
    const nn::workload_data<float> *input;
    const nn::workload_data<float> *weights;
    const nn::workload_data<float> *bias;
    nn::workload_data<float> *output;
};

void unpack_convolve_maxpooling2x2_stride2x2_callback_handle(
    void* void_handle)
{
    auto handle = reinterpret_cast<convolution_pooling_f32_2x2stride2_request_handle *>(void_handle);
    handle->primitive->choose_convolution_maxpool_padding_mode_and_activation(
        handle->input, handle->weights, handle->bias, handle->output);
}

void convolution_pooling_f32_2x2stride2::forward(const nn::workload_data<float> *input,
                                                 const nn::workload_data<float> *weights,
                                                 const nn::workload_data<float> *bias,
                                                 nn::workload_data<float> *output) {
    const auto num_output_fm_items =
        (output->view_end.t[NN_DATA_COORD_z] - output->view_begin.t[NN_DATA_COORD_z] + 1) / C_slice_size;
    const auto num_batch_items =
        (output->view_end.t[NN_DATA_COORD_n] - output->view_begin.t[NN_DATA_COORD_n] + 1);

    const auto total_workers = num_output_fm_items * num_batch_items;

    if (device->thread_pool.get_num_threads() < 2 || total_workers < 2)
    {
        // Its tiny data or there is only one thread available - just do it singlethreaded way.
        choose_convolution_maxpool_padding_mode_and_activation(input, weights, bias, output);
    }
    else
    {
        // Full cores utilization version.
        std::vector<const nn::workload_data<float> *> input_views(total_workers);
        std::vector<const nn::workload_data<float> *> weights_views(total_workers);
        std::vector<const nn::workload_data<float> *> bias_views(total_workers);
        std::vector<nn::workload_data<float> *> output_views(total_workers);

        // Fill slave work items.
        for (auto output_fm_item = 0u; output_fm_item < num_output_fm_items; ++output_fm_item)
        {
            for (auto batch_item = 0u; batch_item < num_batch_items; ++batch_item)
            {
                auto item_in_pool = batch_item + output_fm_item * num_batch_items;

                // Replace nn_workload_datas pointers with views.
                nn_workload_data_coords_t input_view_begin =
                {
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                };
                nn_workload_data_coords_t input_view_end = {
                    input->get_length(NN_DATA_COORD_n) - 1,
                    input->get_length(NN_DATA_COORD_x) - 1,
                    input->get_length(NN_DATA_COORD_y) - 1,
                    input->get_length(NN_DATA_COORD_z) - 1,
                    input->get_length(NN_DATA_COORD_p) - 1,
                    input->get_length(NN_DATA_COORD_q) - 1
                };

                nn_workload_data_coords_t output_view_begin =
                {
                    batch_item,
                    0,
                    0,
                    output_fm_item * C_slice_size,
                    0,
                    0
                };
                nn_workload_data_coords_t output_view_end =
                {
                    batch_item,
                    output->get_length(NN_DATA_COORD_x) - 1,
                    output->get_length(NN_DATA_COORD_y) - 1,
                    (output_fm_item+1) * C_slice_size - 1,
                    output->get_length(NN_DATA_COORD_p) - 1,
                    output->get_length(NN_DATA_COORD_q) - 1
                };

                nn_workload_data_coords_t weights_view_begin =
                {
                    0,
                    0,
                    0,
                    0,
                    0,
                    output_fm_item
                };
                nn_workload_data_coords_t weights_view_end =
                {
                    weights->get_length(NN_DATA_COORD_n) - 1,
                    weights->get_length(NN_DATA_COORD_x) - 1,
                    weights->get_length(NN_DATA_COORD_y) - 1,
                    weights->get_length(NN_DATA_COORD_z) - 1,
                    weights->get_length(NN_DATA_COORD_p) - 1,
                    output_fm_item
                };

                input_views[item_in_pool] = 
                    new nn::workload_data<float>(*input, input_view_begin, input_view_end);

                output_views[item_in_pool] =
                    new nn::workload_data<float>(*output, output_view_begin, output_view_end);

                weights_views[item_in_pool] =
                    new nn::workload_data<float>(*weights, weights_view_begin, weights_view_end);

                // Use biases.
                if (bias != nullptr)
                {
                    nn_workload_data_coords_t bias_view_begin =
                    {
                        0,
                        output_fm_item * C_slice_size,
                        0,
                        0,
                        0,
                        0
                    };
                    nn_workload_data_coords_t bias_view_end =
                    {
                        bias->get_length(NN_DATA_COORD_n) - 1,
                        (output_fm_item+1) * C_slice_size - 1,
                        bias->get_length(NN_DATA_COORD_y) - 1,
                        bias->get_length(NN_DATA_COORD_z) - 1,
                        bias->get_length(NN_DATA_COORD_p) - 1,
                        bias->get_length(NN_DATA_COORD_q) - 1
                    };

                    bias_views[item_in_pool] = 
                        new nn::workload_data<float>(*bias, bias_view_begin, bias_view_end);
                } else {
                    bias_views[item_in_pool] = nullptr;
                }
            }
        }

        // Run threads.
        std::vector<convolution_pooling_f32_2x2stride2_request_handle> request_handles(total_workers);
        std::vector<nn_multithreaded_request> job(total_workers);
        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            request_handles[item_in_pool].primitive = this;
            request_handles[item_in_pool].input = input_views[item_in_pool];
            request_handles[item_in_pool].weights = weights_views[item_in_pool];
            request_handles[item_in_pool].bias = bias_views[item_in_pool];
            request_handles[item_in_pool].output = output_views[item_in_pool];

            job[item_in_pool].callback = unpack_convolve_maxpooling2x2_stride2x2_callback_handle;
            job[item_in_pool].request_handle = &request_handles[item_in_pool];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);

        // Cleanup dynamic memory.
        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            delete input_views[item_in_pool];
            delete weights_views[item_in_pool];
            delete bias_views[item_in_pool];
            delete output_views[item_in_pool];
        }
    }
}
}

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_convolution_pooling_f32_create_0(
    nn_device_t *device,    /* IDLF device handle */
    size_t kernel_w,        /* convolution kernel width */
    size_t kernel_h,        /* convolution kernel height */
    size_t num_input,       /* number of input feature maps */
    size_t num_output,      /* number of output feature maps */
    size_t output_w,        /* output width */
    size_t output_h,        /* output height */
    size_t center_offset_x, /* horizontal offset of kernel's center point w/ relation to top left corner */
    size_t center_offset_y, /* vertical offset of kernel's center point w/ relation to top left corner */
    size_t stride_x,        /* convolution horizontal stride */
    size_t stride_y,        /* convolution vertical stride */
    const nn_argument_activation_t *activation, /* struct parameterizing optional activation function */
    size_t batch_size,                          /* size of input batch */
    size_t pooling_kernel_w,                    /* width of pooling kernel */
    size_t pooling_kernel_h,                    /* height of pooling kernel */
    size_t pooling_stride_x,                    /* horizontal pooling stride */
    size_t pooling_stride_y,                    /* vertical pooling stride */
    const nn_primitives_convolution_pooling_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */) {

    assert(pooling_kernel_w == 2);
    assert(pooling_kernel_h == 2);
    assert(pooling_stride_x == 2);
    assert(pooling_stride_y == 2);
    SET_STATUS(NN_API_STATUS_OK);

    std::remove_const<std::remove_pointer<decltype(hints)>::type>::type hints_ = {};
    if (hints != nullptr)
        hints_ = *hints;

    return new layer::convolution_pooling_f32_2x2stride2(kernel_w,
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
