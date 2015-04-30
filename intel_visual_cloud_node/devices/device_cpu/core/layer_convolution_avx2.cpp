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

#include "../../common/nn_workload_data.h"
#include "../api_internal/nn_device_interface_0_internal.h"
#include "layer_convolution_avx2.h"
#include "helper_zxyn_f32.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <tuple>

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
    const nn::nn_workload_data_t<float> *input_view,
    const size_t center_offset_x,
    const size_t center_offset_y,
    const int32_t stride_x,
    const int32_t stride_y,
    const nn::nn_workload_data_t<float> *weights,
    const nn::nn_workload_data_t<float> *bias,
    nn::nn_workload_data_t<float> *output_view)
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
    // C3
    { prepare_entry<NN_ACTIVATION_FUNCTION_NONE, 14, 14, 256, 0, 256, 0, 3, 3, 1, 1, 14, 14, 512>() },
    { prepare_entry<NN_ACTIVATION_FUNCTION_RELU, 14, 14, 256, 0, 256, 0 ,3, 3, 1, 1, 14, 14, 512>() },

    //  C4.
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

template <NN_ACTIVATION_FUNCTION T_activation>
void run_convolution(const nn::nn_workload_data_t<float> *input_view,
                     const NN_PADDING_MODE padding,
                     const int32_t center_offset_x,
                     const int32_t center_offset_y,
                     const size_t stride_x,
                     const size_t stride_y,
                     const nn::nn_workload_data_t<float> *weights,
                     const nn::nn_workload_data_t<float> *bias,
                     nn::nn_workload_data_t<float> *output_view) {

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
        // Generic.
        convolve_internal<false, T_activation>(input_view, center_offset_x, center_offset_y, stride_x, stride_y, weights, bias, output_view);
    }
}

void choose_convolution_padding_mode_and_activation(const nn::nn_workload_data_t<float> *input,
                                                    const NN_PADDING_MODE padding,
                                                    const int32_t center_offset_x,
                                                    const int32_t center_offset_y,
                                                    const size_t stride_x,
                                                    const size_t stride_y,
                                                    const nn_argument_activation_t &activation,
                                                    const nn::nn_workload_data_t<float> *weights,
                                                    const nn::nn_workload_data_t<float> *bias,
                                                    nn::nn_workload_data_t<float> *output) {

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
                input->get_length(NN_DATA_COORD_n) - 1,
                input->get_length(NN_DATA_COORD_x) - 1 - input_crop_from_right,
                input->get_length(NN_DATA_COORD_y) - 1 - input_crop_from_down,
                input->get_length(NN_DATA_COORD_z) - 1,
                input->get_length(NN_DATA_COORD_p) - 1,
                input->get_length(NN_DATA_COORD_q) - 1
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
                output->get_length(NN_DATA_COORD_n) - 1,
                output->get_length(NN_DATA_COORD_x) - 1 - output_crop_from_right,
                output->get_length(NN_DATA_COORD_y) - 1 - output_crop_from_down,
                output->get_length(NN_DATA_COORD_z) - 1,
                output->get_length(NN_DATA_COORD_p) - 1,
                output->get_length(NN_DATA_COORD_q) - 1
            };

            bool valid_optimized_view = false;
            nn::nn_workload_data_t<float>* input_subview = nullptr;
            nn::nn_workload_data_t<float>* output_subview = nullptr;

            // Run optimized convolution on subview if there is anything to process after crop.
            if (static_cast<int32_t>(output_view_start.t[NN_DATA_COORD_x]) <= static_cast<int32_t>(output_view_end.t[NN_DATA_COORD_x]) &&
                static_cast<int32_t>(output_view_start.t[NN_DATA_COORD_y]) <= static_cast<int32_t>(output_view_end.t[NN_DATA_COORD_y]))
            {
                valid_optimized_view = true;
                input_subview = new nn::nn_workload_data_t<float>(
                    *const_cast<nn::nn_workload_data_t<float> *>(input), input_view_start, input_view_end);
                output_subview = new nn::nn_workload_data_t<float>(*output, output_view_start, output_view_end);

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
                                out_feature_map += C_slice_size, kernel_feature_map += C_slice_size, bias_feature_map += C_slice_size)
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
                                __m256 acc1 = _mm256_load_ps(reinterpret_cast<float*>(bias->parent->data_buffer) + bias_feature_map + C_simd_width);

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
                                            __m256 weight1 = _mm256_load_ps(reinterpret_cast<float*>(weights->parent->data_buffer) + weight_ptr_offset + C_simd_width);
                                            __m256 inp = _mm256_broadcast_ss(reinterpret_cast<float*>(input->parent->data_buffer) + input_ptr_offset);

                                            acc0 = _mm256_fmadd_ps(weight0, inp, acc0);
                                            acc1 = _mm256_fmadd_ps(weight1, inp, acc1);

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
                                    acc1 = _mm256_max_ps(_mm256_setzero_ps(), acc1);
                                }

                                _mm256_store_ps(reinterpret_cast<float*>(output->parent->data_buffer) + output_element, acc0);
                                _mm256_store_ps(reinterpret_cast<float*>(output->parent->data_buffer) + output_element + C_simd_width, acc1);
                            }
                        }
                    }
                }
            }

            // Cleanup dynamic data.
            // TODO FIX THIS, temp workaround for nn_workload_data_t bug with multi threading - this is a memory leak currently.
            delete reinterpret_cast<nn_workload_data_t*>(output_subview);
            delete reinterpret_cast<nn_workload_data_t*>(input_subview);
        }
        break;
    default:
        break;
    }
}

using parameters_convolution_f32_avx2 = std::tuple<nn::nn_workload_data_t<float> *,
                                                   NN_PADDING_MODE,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   const nn_argument_activation_t *,
                                                   nn::nn_workload_data_t<float> *,
                                                   nn::nn_workload_data_t<float> *,
                                                   nn::nn_workload_data_t<float> *>;

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

nn_opaque_data_t *NN_API_CALL_CONVENTION
create_weights(nn_primitive_handle_t handle, const nn_data_t *weights, NN_API_STATUS *status) {
    auto primitive = static_cast<layer::convolution_f32*>(handle);
    auto result = primitive->create_weights(*nn::data_cast<float, 4>(weights));
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

nn_opaque_data_t *NN_API_CALL_CONVENTION
create_bias(nn_primitive_handle_t handle, const nn_data_t *bias, NN_API_STATUS *status) {
    auto primitive = static_cast<layer::convolution_f32 *>(handle);
    auto result = primitive->create_bias(*nn::data_cast<float, 1>(bias));
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

nn_event_t NN_API_CALL_CONVENTION forward_async(nn_primitive_handle_t handle,
                                                nn_opaque_data_t *input,
                                                nn_opaque_data_t *weights,
                                                nn_opaque_data_t *bias,
                                                nn_opaque_data_t *output,
                                                size_t dependencies_count,
                                                nn_event_t *dependencies,
                                                NN_API_STATUS *status) {
    auto primitive = static_cast<layer::convolution_f32 *>(handle);
    primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(input),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(weights),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(bias),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(output));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_convolution_f32_create(nn_device_t *device,
                                                                       const size_t kernel_w,
                                                                       const size_t kernel_h,
                                                                       const size_t num_input,
                                                                       const size_t num_output,
                                                                       const size_t output_w,
                                                                       const size_t output_h,
                                                                       const int32_t center_offset_x,
                                                                       const int32_t center_offset_y,
                                                                       const size_t stride_x,
                                                                       const size_t stride_y,
                                                                       const nn_argument_activation_t *activation,
                                                                       size_t batch_size,
                                                                       NN_API_STATUS *status) {
    SET_STATUS(NN_API_STATUS_OK);
    return layer::convolution_f32::create(kernel_w,
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
                                             device);
}

} // namespace convolution_f32_impl

void convolution_f32::forward(const nn::nn_workload_data_t<float> *input_buffer,
                              const nn::nn_workload_data_t<float> *weights_buffer,
                              const nn::nn_workload_data_t<float> *bias_buffer,
                              nn::nn_workload_data_t<float> *output_buffer) {
    const auto num_output_fm_items =
        (output_buffer->view_end.t[NN_DATA_COORD_z] - output_buffer->view_begin.t[NN_DATA_COORD_z] + 1) /
        convolution_f32_impl::C_slice_size;
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
        std::vector<nn::nn_workload_data_t<float> *> input_views(total_workers);
        std::vector<nn::nn_workload_data_t<float> *> weight_views(total_workers);
        std::vector<nn::nn_workload_data_t<float> *> bias_views(total_workers);
        std::vector<nn::nn_workload_data_t<float> *> output_views(total_workers);

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
                    cpp_master_input->get_length(NN_DATA_COORD_n) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_x) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_y) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_z) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_p) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_q) - 1
                };

                nn_workload_data_coords_t output_view_begin =
                {
                    batch_item,
                    0,
                    0,
                    output_fm_item * convolution_f32_impl::C_slice_size,
                    0,
                    0
                };
                nn_workload_data_coords_t output_view_end =
                {
                    batch_item,
                    cpp_master_output->get_length(NN_DATA_COORD_x) - 1,
                    cpp_master_output->get_length(NN_DATA_COORD_y) - 1,
                    (output_fm_item+1) * convolution_f32_impl::C_slice_size - 1,
                    cpp_master_output->get_length(NN_DATA_COORD_p) - 1,
                    cpp_master_output->get_length(NN_DATA_COORD_q) - 1
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
                    cpp_master_weights->get_length(NN_DATA_COORD_n) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_x) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_y) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_z) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_p) - 1,
                    output_fm_item
                };

                input_views[item_in_pool] = 
                    new nn::nn_workload_data_t<float>(const_cast<nn::nn_workload_data_t<float>&>(*cpp_master_input), input_view_begin, input_view_end);

                output_views[item_in_pool] =
                    new nn::nn_workload_data_t<float>(*cpp_master_output, output_view_begin, output_view_end);

                weight_views[item_in_pool] =
                    new nn::nn_workload_data_t<float>(const_cast<nn::nn_workload_data_t<float>&>(*cpp_master_weights), weights_view_begin, weights_view_end);

                // Use biases.
                if (bias_buffer != nullptr)
                {
                    const auto cpp_master_biases = bias_buffer;

                    nn_workload_data_coords_t bias_view_begin =
                    {
                        0,
                        output_fm_item * convolution_f32_impl::C_slice_size,
                        0,
                        0,
                        0,
                        0
                    };
                    nn_workload_data_coords_t bias_view_end =
                    {
                        cpp_master_biases->get_length(NN_DATA_COORD_n) - 1,
                        (output_fm_item+1) * convolution_f32_impl::C_slice_size - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_y) - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_z) - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_p) - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_q) - 1
                    };

                    bias_views[item_in_pool] =
                        new nn::nn_workload_data_t<float>(const_cast<nn::nn_workload_data_t<float>&>(*cpp_master_biases), bias_view_begin, bias_view_end);
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

void run_multithreaded_convolve_work_item(nn_workload_item *const work_item, nn_device_internal *device) {
    auto primitive = static_cast<convolution_f32 *>(work_item->primitive);
    primitive->forward(
        reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->input[0]->output),
        reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->arguments.forward_convolution.weights),
        reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->arguments.forward_convolution.biases),
        reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->output));
}

convolution_f32 *convolution_f32::create(size_t kernel_w,
                                         size_t kernel_h,
                                         size_t num_input,
                                         size_t num_output,
                                         size_t output_w,
                                         size_t output_h,
                                         int32_t center_offset_x,
                                         int32_t center_offset_y,
                                         size_t stride_x,
                                         size_t stride_y,
                                         const nn_argument_activation_t &activation,
                                         size_t batch_size,
                                         nn_device_t *device) {
    return new convolution_f32(kernel_w,
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
                               reinterpret_cast<nn_device_internal *>(device));
}

nn::nn_workload_data_t<float> *convolution_f32::create_weights(const nn::data<float, 4> &weights) {
    //TODO: validate weight format
    nn_workload_data_layout_t layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q, NN_DATA_COORD_n }, // ordering
        NN_DATATYPE_FLOAT
    };

    const uint32_t C_simd_width = sizeof(__m256) / sizeof(float);
    const uint32_t C_slice_size = 2 * C_simd_width;
    nn_workload_data_coords_t size = {
        1,
        static_cast<uint32_t>(weights.size[0]),               // kernel width
        static_cast<uint32_t>(weights.size[1]),               // kernel height
        static_cast<uint32_t>(weights.size[2]),               // number of input feature maps
        C_slice_size,                                         // output feature maps slice size
        static_cast<uint32_t>(weights.size[3]) / C_slice_size // number of slices of output feature maps
    };

    nn::nn_workload_data_t<float> *load_weights = new nn::nn_workload_data_t<float>(size, layout);
    /*
    Code below this comment is a performance optimized version of:
    for (size_t q = 0u; q < size.t[5]; ++q)
        for (size_t p = 0u; p < size.t[4]; ++p)
            for (size_t z = 0u; z < size.t[3]; ++z)
                for (size_t y = 0u; y < size.t[2]; ++y)
                    for (size_t x = 0u; x < size.t[1]; ++x)
                        //              n, x, y, z, p, q  =            x, y, i, o
                        (*load_weights)(0, x, y, z, p, q) = weights.at(x, y, z, q * C_slice_size + p);
    ...which is left here for readability.
    */
    auto dst = static_cast<float *>(load_weights->parent->data_buffer);
    auto src = static_cast<float *>(weights.buffer);
    auto src_stride_y = weights.size[0];
    auto src_stride_i = src_stride_y*weights.size[1];
    auto src_stride_o = src_stride_i*weights.size[2];
    for(size_t q = 0u; q < size.t[5]; ++q)
        for(size_t y = 0u; y < size.t[2]; ++y)
            for(size_t x = 0u; x < size.t[1]; ++x)
                for(size_t z = 0u; z < size.t[3]; ++z)
                    for(size_t p = 0u; p < size.t[4]; ++p)
                        *(dst++) = src[x + src_stride_y*y +src_stride_i*z + src_stride_o*(q * C_slice_size + p)];
    return load_weights;
}

nn::nn_workload_data_t<float>* convolution_f32::create_bias(const nn::data<float, 1> &bias)
{
    //TODO: validate bias format
    nn_workload_data_layout_t layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
            NN_DATATYPE_FLOAT
    };

    nn_workload_data_coords_t size = {1, static_cast<uint32_t>(bias.size[0]), 1, 1, 1, 1};
    nn::nn_workload_data_t<float> *load_biases = new nn::nn_workload_data_t<float>(size, layout);
    for (size_t index = 0u; index < load_biases->get_length(1); ++index) {
        (*load_biases)(0, index, 0, 0, 0, 0) = bias.at(index);
    }

    return load_biases;
}

nn::nn_workload_data_t<float>* convolution_f32::create_input(const nn::data<float, 4>& input)
{
    const size_t max_padding_w = kernel_w - 1, max_padding_h = kernel_h - 1;

    // workaround for Caffe ULTs - accept input width that is discarded by incomplete strides
    if ((get_required_input_w() > input.size[1] + max_padding_w) ||
        (get_required_input_w() + stride_x - 1 < input.size[1]))
        throw std::invalid_argument("input width");

    if ((get_required_input_h() > input.size[2] + max_padding_h) ||
        (get_required_input_h() + stride_y - 1 < input.size[2]))
        throw std::invalid_argument("input height");

    if(input_size_z != input.size[0])
        throw std::invalid_argument("input feature maps");

    if(batch_size != input.size[3])
        throw std::invalid_argument("input batch size");

    const size_t padding_w = get_required_input_w() > input.size[1] ? get_required_input_w() - input.size[1] : 0;
    const size_t padding_h = get_required_input_h() > input.size[2] ? get_required_input_h() - input.size[2] : 0;

    if (padding_w % 2 != 0 || padding_h % 2 != 0)
        throw std::invalid_argument("padding width and height must be even");

    const size_t padding_left = padding_w / 2, padding_top = padding_h / 2;
    const size_t padding_right = padding_w / 2, padding_bottom = padding_h / 2;

    const int view_offset_x = center_offset_x - padding_left, view_offset_y = center_offset_y - padding_top;
    return create_input_impl(input,
                             padding_left,
                             padding_right,
                             padding_top,
                             padding_bottom,
                             view_offset_x,
                             view_offset_y,
                             get_required_input_w() - kernel_w + 1,
                             get_required_input_h() - kernel_h + 1);
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
                                 nn_device_internal *device)
    : primitive_zxyn_f32_base(batch_size, num_input, output_w, output_h, num_output, device),
      kernel_w(kernel_w),
      kernel_h(kernel_h),
      padding(NN_PADDING_MODE_DATA_OR_ZERO),
      center_offset_x(center_offset_x),
      center_offset_y(center_offset_y),
      stride_x(stride_x),
      stride_y(stride_y),
      activation(activation) {}

bool convolution_f32::validate_input(const nn::nn_workload_data_t<float> &input) {
    if (!memcmp(&input.parent->layout, &in_out_layout, sizeof(nn_workload_data_layout_t)))
        return false;

    const auto view_size = input.get_length();

    if (view_size.t[NN_DATA_COORD_n] != batch_size)
        return false;

    // TODO get rid of center_offset = 0 special case
    if (input.view_begin.t[NN_DATA_COORD_x] < center_offset_x ||
        (center_offset_x == 0 && view_size.t[NN_DATA_COORD_x] != get_required_input_w()) ||
        (center_offset_x != 0 && (view_size.t[NN_DATA_COORD_x] != get_required_input_w() - kernel_w + 1 ||
        input.view_end.t[NN_DATA_COORD_x] + kernel_w - center_offset_x - 1 >=
        input.parent->lengths.t[NN_DATA_COORD_x])))
        return false;

    if (input.view_begin.t[NN_DATA_COORD_y] < center_offset_y ||
        (center_offset_y == 0 && view_size.t[NN_DATA_COORD_y] != get_required_input_h()) ||
        (center_offset_y != 0 && (view_size.t[NN_DATA_COORD_y] != get_required_input_h() - kernel_h + 1 ||
        input.view_end.t[NN_DATA_COORD_y] + kernel_h - center_offset_y - 1 >=
        input.parent->lengths.t[NN_DATA_COORD_y])))
        return false;

    if (view_size.t[NN_DATA_COORD_z] != input_size_z)
        return false;

    return true;
}

size_t convolution_f32::get_required_input_w() { return (output_size_x - 1) * stride_x + kernel_w; }

size_t convolution_f32::get_required_input_h() { return (output_size_y - 1) * stride_y + kernel_h; }
} // namespace layer

nn_primitives_convolution_f32_0_t nn_primitives_convolution_f32_0 = {
    layer::convolution_f32_impl::nn_convolution_f32_create,
    layer::convolution_f32_impl::create_weights,
    layer::convolution_f32_impl::create_bias,
    nullptr,
    layer::helper_zxyn_f32::create_input,
    layer::helper_zxyn_f32::map_input,
    layer::helper_zxyn_f32::split_input_z,
    nullptr,                                            // validate input
    layer::helper_zxyn_f32::create_output,              // create_output
    layer::helper_zxyn_f32::create_output_with_padding, // create output with padding
    layer::helper_zxyn_f32::create_output_vector_z,
    nullptr,                                            // create view
    layer::convolution_f32_impl::forward_async,
    nullptr,
    layer::helper_zxyn_f32::copy_output_async};
