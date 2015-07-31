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
#include "layer_pooling_avx2.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <map>
#include <tuple>
#include "device/cpu/api_internal/data_helper.h"

// SIMD width for this implementation
const uint32_t C_simd_width = sizeof(__m256) / sizeof(float);
const uint32_t C_max_num_accumulators = 8;
const auto C_max_block_size = 16;

namespace layer
{
template<uint32_t T_num_acc,
         bool     T_first_run>
inline void pooling_macro(
    float* input_ptr,
    float* output_ptr)
{
    __m256 acc0, acc1, acc2, acc3, 
           acc4, acc5, acc6, acc7, 
           acc8, acc9, acc10, acc11,
           acc12, acc13, acc14, acc15;

    if (T_first_run)
    {
        if (T_num_acc >=  1)  acc0 = _mm256_load_ps(input_ptr +  0 * C_simd_width);
        if (T_num_acc >=  2)  acc1 = _mm256_load_ps(input_ptr +  1 * C_simd_width);
        if (T_num_acc >=  3)  acc2 = _mm256_load_ps(input_ptr +  2 * C_simd_width);
        if (T_num_acc >=  4)  acc3 = _mm256_load_ps(input_ptr +  3 * C_simd_width);
        if (T_num_acc >=  5)  acc4 = _mm256_load_ps(input_ptr +  4 * C_simd_width);
        if (T_num_acc >=  6)  acc5 = _mm256_load_ps(input_ptr +  5 * C_simd_width);
        if (T_num_acc >=  7)  acc6 = _mm256_load_ps(input_ptr +  6 * C_simd_width);
        if (T_num_acc >=  8)  acc7 = _mm256_load_ps(input_ptr +  7 * C_simd_width);
        if (T_num_acc >=  9)  acc8 = _mm256_load_ps(input_ptr +  8 * C_simd_width);
        if (T_num_acc >= 10)  acc9 = _mm256_load_ps(input_ptr +  9 * C_simd_width);
        if (T_num_acc >= 11) acc10 = _mm256_load_ps(input_ptr + 10 * C_simd_width);
        if (T_num_acc >= 12) acc11 = _mm256_load_ps(input_ptr + 11 * C_simd_width);
        if (T_num_acc >= 13) acc12 = _mm256_load_ps(input_ptr + 12 * C_simd_width);
        if (T_num_acc >= 14) acc13 = _mm256_load_ps(input_ptr + 13 * C_simd_width);
        if (T_num_acc >= 15) acc14 = _mm256_load_ps(input_ptr + 14 * C_simd_width);
        if (T_num_acc >= 16) acc15 = _mm256_load_ps(input_ptr + 15 * C_simd_width);
    }
    else
    {
        if (T_num_acc >=  1)  acc0 = _mm256_load_ps(output_ptr +  0 * C_simd_width);
        if (T_num_acc >=  2)  acc1 = _mm256_load_ps(output_ptr +  1 * C_simd_width);
        if (T_num_acc >=  3)  acc2 = _mm256_load_ps(output_ptr +  2 * C_simd_width);
        if (T_num_acc >=  4)  acc3 = _mm256_load_ps(output_ptr +  3 * C_simd_width);
        if (T_num_acc >=  5)  acc4 = _mm256_load_ps(output_ptr +  4 * C_simd_width);
        if (T_num_acc >=  6)  acc5 = _mm256_load_ps(output_ptr +  5 * C_simd_width);
        if (T_num_acc >=  7)  acc6 = _mm256_load_ps(output_ptr +  6 * C_simd_width);
        if (T_num_acc >=  8)  acc7 = _mm256_load_ps(output_ptr +  7 * C_simd_width);
        if (T_num_acc >=  9)  acc8 = _mm256_load_ps(output_ptr +  8 * C_simd_width);
        if (T_num_acc >= 10)  acc9 = _mm256_load_ps(output_ptr +  9 * C_simd_width);
        if (T_num_acc >= 11) acc10 = _mm256_load_ps(output_ptr + 10 * C_simd_width);
        if (T_num_acc >= 12) acc11 = _mm256_load_ps(output_ptr + 11 * C_simd_width);
        if (T_num_acc >= 13) acc12 = _mm256_load_ps(output_ptr + 12 * C_simd_width);
        if (T_num_acc >= 14) acc13 = _mm256_load_ps(output_ptr + 13 * C_simd_width);
        if (T_num_acc >= 15) acc14 = _mm256_load_ps(output_ptr + 14 * C_simd_width);
        if (T_num_acc >= 16) acc15 = _mm256_load_ps(output_ptr + 15 * C_simd_width);

        if (T_num_acc >=  1)  acc0 = _mm256_max_ps(acc0,  _mm256_load_ps(input_ptr +  0 * C_simd_width));
        if (T_num_acc >=  2)  acc1 = _mm256_max_ps(acc1,  _mm256_load_ps(input_ptr +  1 * C_simd_width));
        if (T_num_acc >=  3)  acc2 = _mm256_max_ps(acc2,  _mm256_load_ps(input_ptr +  2 * C_simd_width));
        if (T_num_acc >=  4)  acc3 = _mm256_max_ps(acc3,  _mm256_load_ps(input_ptr +  3 * C_simd_width));
        if (T_num_acc >=  5)  acc4 = _mm256_max_ps(acc4,  _mm256_load_ps(input_ptr +  4 * C_simd_width));
        if (T_num_acc >=  6)  acc5 = _mm256_max_ps(acc5,  _mm256_load_ps(input_ptr +  5 * C_simd_width));
        if (T_num_acc >=  7)  acc6 = _mm256_max_ps(acc6,  _mm256_load_ps(input_ptr +  6 * C_simd_width));
        if (T_num_acc >=  8)  acc7 = _mm256_max_ps(acc7,  _mm256_load_ps(input_ptr +  7 * C_simd_width));
        if (T_num_acc >=  9)  acc8 = _mm256_max_ps(acc8,  _mm256_load_ps(input_ptr +  8 * C_simd_width));
        if (T_num_acc >= 10)  acc9 = _mm256_max_ps(acc9,  _mm256_load_ps(input_ptr +  9 * C_simd_width));
        if (T_num_acc >= 11) acc10 = _mm256_max_ps(acc10, _mm256_load_ps(input_ptr + 10 * C_simd_width));
        if (T_num_acc >= 12) acc11 = _mm256_max_ps(acc11, _mm256_load_ps(input_ptr + 11 * C_simd_width));
        if (T_num_acc >= 13) acc12 = _mm256_max_ps(acc12, _mm256_load_ps(input_ptr + 12 * C_simd_width));
        if (T_num_acc >= 14) acc13 = _mm256_max_ps(acc13, _mm256_load_ps(input_ptr + 13 * C_simd_width));
        if (T_num_acc >= 15) acc14 = _mm256_max_ps(acc14, _mm256_load_ps(input_ptr + 14 * C_simd_width));
        if (T_num_acc >= 16) acc15 = _mm256_max_ps(acc15, _mm256_load_ps(input_ptr + 15 * C_simd_width));
    }


    if (T_num_acc >=  1) _mm256_store_ps(output_ptr +  0 * C_simd_width,  acc0);
    if (T_num_acc >=  2) _mm256_store_ps(output_ptr +  1 * C_simd_width,  acc1);
    if (T_num_acc >=  3) _mm256_store_ps(output_ptr +  2 * C_simd_width,  acc2);
    if (T_num_acc >=  4) _mm256_store_ps(output_ptr +  3 * C_simd_width,  acc3);
    if (T_num_acc >=  5) _mm256_store_ps(output_ptr +  4 * C_simd_width,  acc4);
    if (T_num_acc >=  6) _mm256_store_ps(output_ptr +  5 * C_simd_width,  acc5);
    if (T_num_acc >=  7) _mm256_store_ps(output_ptr +  6 * C_simd_width,  acc6);
    if (T_num_acc >=  8) _mm256_store_ps(output_ptr +  7 * C_simd_width,  acc7);
    if (T_num_acc >=  9) _mm256_store_ps(output_ptr +  8 * C_simd_width,  acc8);
    if (T_num_acc >= 10) _mm256_store_ps(output_ptr +  9 * C_simd_width,  acc9);
    if (T_num_acc >= 11) _mm256_store_ps(output_ptr + 10 * C_simd_width, acc10);
    if (T_num_acc >= 12) _mm256_store_ps(output_ptr + 11 * C_simd_width, acc11);
    if (T_num_acc >= 13) _mm256_store_ps(output_ptr + 12 * C_simd_width, acc12);
    if (T_num_acc >= 14) _mm256_store_ps(output_ptr + 13 * C_simd_width, acc13);
    if (T_num_acc >= 15) _mm256_store_ps(output_ptr + 14 * C_simd_width, acc14);
    if (T_num_acc >= 16) _mm256_store_ps(output_ptr + 15 * C_simd_width, acc15);
}

template<bool T_first_run>
inline void pooling_outer_macro(
    float* input_ptr,
    float* output_ptr,
    uint32_t num_blocks_full,
    uint32_t partial_block_size)
{
    for (uint32_t block = 0; block < num_blocks_full; block++)
    {
        pooling_macro<C_max_block_size, T_first_run>(input_ptr, output_ptr);
        input_ptr += C_max_block_size * C_simd_width;
        output_ptr += C_max_block_size * C_simd_width;
    }

    switch (partial_block_size)
    {
    case  0: break;
    case  1: pooling_macro< 1, T_first_run>(input_ptr, output_ptr); break;
    case  2: pooling_macro< 2, T_first_run>(input_ptr, output_ptr); break;
    case  3: pooling_macro< 3, T_first_run>(input_ptr, output_ptr); break;
    case  4: pooling_macro< 4, T_first_run>(input_ptr, output_ptr); break;
    case  5: pooling_macro< 5, T_first_run>(input_ptr, output_ptr); break;
    case  6: pooling_macro< 6, T_first_run>(input_ptr, output_ptr); break;
    case  7: pooling_macro< 7, T_first_run>(input_ptr, output_ptr); break;
    case  8: pooling_macro< 8, T_first_run>(input_ptr, output_ptr); break;
    case  9: pooling_macro< 9, T_first_run>(input_ptr, output_ptr); break;
    case 10: pooling_macro<10, T_first_run>(input_ptr, output_ptr); break;
    case 11: pooling_macro<11, T_first_run>(input_ptr, output_ptr); break;
    case 12: pooling_macro<12, T_first_run>(input_ptr, output_ptr); break;
    case 13: pooling_macro<13, T_first_run>(input_ptr, output_ptr); break;
    case 14: pooling_macro<14, T_first_run>(input_ptr, output_ptr); break;
    case 15: pooling_macro<15, T_first_run>(input_ptr, output_ptr); break;
    default:
        /* Execution can never reach here (see 'partial_block_size') calculation.*/
        /* Need to inform compiler that it should not generate code for 'default'.*/
        /* [TODO] heed to handle GCC */
        NN_UNREACHABLE_CODE;
    }
}

template<bool    T_exact_match,
        uint32_t T_input_feature_map_width    = 0, 
        uint32_t T_input_feature_map_height   = 0, 
        uint32_t T_feature_maps               = 0, 
        uint32_t T_output_feature_map_width   = 0, 
        uint32_t T_output_feature_map_height  = 0, 
        uint32_t T_pool_stride_x              = 0, 
        uint32_t T_pool_stride_y              = 0,
        uint32_t T_pool_size_x                = 0, 
        uint32_t T_pool_size_y                = 0>
void pooling_internal(
    size_t _pool_size_x,
    size_t _pool_size_y,
    size_t _pool_stride_x,
    size_t _pool_stride_y,
    const nn::workload_data<float> *input_view,
    nn::workload_data<float> *output_view)
{
    const uint32_t num_feature_maps             = (T_exact_match) ? T_feature_maps : output_view->parent->lengths.t[NN_DATA_COORD_z];
    const uint32_t output_feature_map_width     = (T_exact_match) ? T_output_feature_map_width : output_view->parent->lengths.t[NN_DATA_COORD_x];
    const uint32_t output_feature_map_height    = (T_exact_match) ? T_output_feature_map_height : output_view->parent->lengths.t[NN_DATA_COORD_y];
    const uint32_t input_feature_map_width      = (T_exact_match) ? T_input_feature_map_width : input_view->parent->lengths.t[NN_DATA_COORD_x];
    const uint32_t input_feature_map_height     = (T_exact_match) ? T_input_feature_map_height : input_view->parent->lengths.t[NN_DATA_COORD_y];
    const uint32_t pool_stride_x                = (T_exact_match) ? T_pool_stride_x : _pool_stride_x;
    const uint32_t pool_stride_y                = (T_exact_match) ? T_pool_stride_y : _pool_stride_y;
    const uint32_t pool_size_x                  = (T_exact_match) ? T_pool_size_x : _pool_size_x;
    const uint32_t pool_size_y                  = (T_exact_match) ? T_pool_size_y : _pool_size_y;

    auto* const input_buffer = reinterpret_cast<float*>(input_view->parent->data_buffer);
    auto* const output_buffer = reinterpret_cast<float*>(output_view->parent->data_buffer);

    const uint32_t output_row_view_start = output_view->view_begin.t[NN_DATA_COORD_y];
    const uint32_t output_row_view_end = output_view->view_end.t[NN_DATA_COORD_y];
    
    const uint32_t output_column_view_start = output_view->view_begin.t[NN_DATA_COORD_x];
    const uint32_t output_column_view_end = output_view->view_end.t[NN_DATA_COORD_x];

    const uint32_t output_depth_view_start = output_view->view_begin.t[NN_DATA_COORD_z];
    const uint32_t output_depth_view_end = output_view->view_end.t[NN_DATA_COORD_z];
    
    const uint32_t input_column_view_start = input_view->view_begin.t[NN_DATA_COORD_x];
    const uint32_t input_row_view_start = input_view->view_begin.t[NN_DATA_COORD_y];
    const uint32_t input_depth_view_start = input_view->view_begin.t[NN_DATA_COORD_z];

    const uint32_t output_view_width = (output_depth_view_end - output_depth_view_start + 1);
    
    const uint32_t input_row_size    = input_feature_map_width * num_feature_maps;
    const uint32_t output_row_size   = output_feature_map_width * num_feature_maps;

    const uint32_t num_blocks_full = output_view_width / (C_max_block_size * C_simd_width);
    const uint32_t partial_block_size = (output_view_width % (C_max_block_size * C_simd_width) / C_simd_width);
    //const uint32_t partial_subblock_size = output_view_width % C_simd_width;
    
    const uint32_t output_image_view_start = output_view->view_begin.t[NN_DATA_COORD_n];
    const uint32_t output_image_view_end = output_view->view_end.t[NN_DATA_COORD_n];
    
    const uint32_t input_image_size = input_row_size * input_feature_map_height;
    const uint32_t output_image_size = output_row_size * output_feature_map_height;

    for(uint32_t out_image = output_image_view_start; out_image <= output_image_view_end; ++out_image)
    {
        const uint32_t input_image_offset = out_image * input_image_size;
        const uint32_t output_image_offset = out_image * output_image_size;

        for (uint32_t output_row = output_row_view_start, input_row = input_row_view_start; output_row <= output_row_view_end; ++output_row, input_row += pool_stride_y)
        {
            const uint32_t output_row_offset = output_row * output_row_size;

            for (uint32_t pool_y = 0; pool_y < pool_size_y; ++pool_y)
            {
                const uint32_t input_row_offset = (input_row + pool_y) * input_row_size;

                for (uint32_t output_column = output_column_view_start, input_column = input_column_view_start; output_column <= output_column_view_end; ++output_column, input_column += pool_stride_x)
                {
                    const uint32_t output_column_offset = output_column * num_feature_maps;

                    bool first_run = (pool_y == 0) ? true : false;

                    for (uint32_t pool_x = 0; pool_x < pool_size_x; ++pool_x)
                    {
                        const uint32_t input_column_offset = (input_column + pool_x) * num_feature_maps;

                        float* output_ptr = output_buffer + output_image_offset + output_row_offset + output_column_offset + output_depth_view_start;
                        float* input_ptr = input_buffer + input_image_offset + input_row_offset + input_column_offset + input_depth_view_start;

                        if (first_run)
                        {
                            pooling_outer_macro<true>(input_ptr, output_ptr, num_blocks_full, partial_block_size);
                            first_run = false;
                        }
                        else
                        {
                            pooling_outer_macro<false>(input_ptr, output_ptr, num_blocks_full, partial_block_size);
                        }
                    }
                }
            }
        }
    }
}

pooling_f32::pooling_f32(NN_POOLING_MODE pooling_mode,
                         size_t pool_size_x,
                         size_t pool_size_y,
                         size_t pool_stride_x,
                         size_t pool_stride_y,
                         size_t num_feature_maps,
                         size_t output_w,
                         size_t output_h,
                         size_t batch_size,
                         size_t output_padding_left,
                         size_t output_padding_right,
                         size_t output_padding_top,
                         size_t output_padding_bottom,
                         nn_device_internal *device)
    : primitive_zxyn_f32_base(batch_size,
                              num_feature_maps,
                              output_w,
                              output_h,
                              num_feature_maps,
                              output_padding_left,
                              output_padding_right,
                              output_padding_top,
                              output_padding_bottom,
                              device),
      pooling_mode(pooling_mode),
      pool_size_x(pool_size_x),
      pool_size_y(pool_size_y),
      pool_stride_x(pool_stride_x),
      pool_stride_y(pool_stride_y) {}

size_t pooling_f32::get_required_input_w() { return (output_size_x - 1) * pool_stride_x + pool_size_x; }

size_t pooling_f32::get_required_input_h() { return (output_size_y - 1) * pool_stride_y + pool_size_y; }

namespace
{

using optimized_layer_map_t = std::map<
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>, 
    decltype(pooling_internal<false>)*>;

template<uint32_t T_input_feature_map_width,
         uint32_t T_input_feature_map_height,
         uint32_t T_feature_maps,
         uint32_t T_output_feature_map_width,
         uint32_t T_output_feature_map_height,
         uint32_t T_pool_stride_x,
         uint32_t T_pool_stride_y,
         uint32_t T_pool_size_x,
         uint32_t T_pool_size_y>
optimized_layer_map_t::value_type prepare_entry()
{
    return { 
        optimized_layer_map_t::key_type{ 
            T_input_feature_map_width,
            T_input_feature_map_height,
            T_feature_maps,
            T_output_feature_map_width,
            T_output_feature_map_height,
            T_pool_stride_x,
            T_pool_stride_y,
            T_pool_size_x,
            T_pool_size_y},
        pooling_internal<
            true, 
            T_input_feature_map_width,
            T_input_feature_map_height,
            T_feature_maps,
            T_output_feature_map_width,
            T_output_feature_map_height,
            T_pool_stride_x,
            T_pool_stride_y,
            T_pool_size_x,
            T_pool_size_y>
        };
}

optimized_layer_map_t optimized_layer_map =
{
    // OverFeat C1 maxpooling.
    { prepare_entry<56, 56, 96, 28, 28, 2, 2, 2, 2>() },

    // OverFeat C2 maxpooling.
    { prepare_entry<24, 24, 256, 12, 12, 2, 2, 2, 2>() },

    // OverFeat C5 maxpooling.
    { prepare_entry<12, 12, 1024, 6, 6, 2, 2, 2, 2>() },

    // CaffeNet C1 maxpooling.
    { prepare_entry<55, 55, 96, 27, 27, 2, 2, 3, 3>() },

    // CaffeNet C2 maxpooling.
    { prepare_entry<27, 27, 256, 13, 13, 2, 2, 3, 3>() },

    // CaffeNet C5 maxpooling.
    { prepare_entry<13, 13, 256, 6, 6, 2, 2, 3, 3>() },
};
}

void pooling_f32::run_pooling(const nn::workload_data<float> *input_view,
                              nn::workload_data<float> *output_view) {
    const auto num_feature_maps = input_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_feature_map_width = output_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_feature_map_height = output_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto input_feature_map_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto input_feature_map_height = input_view->parent->lengths.t[NN_DATA_COORD_y];
   
    auto map_element = optimized_layer_map.find(std::make_tuple(
        input_feature_map_width,
        input_feature_map_height,
        num_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        pool_stride_x,
        pool_stride_y,
        pool_size_x,
        pool_size_y));

    if (map_element != std::end(optimized_layer_map))
    {
        // Optimized.
        map_element->second(pool_size_x, pool_size_y, pool_stride_x, pool_stride_y, input_view, output_view);
    }
    else
    {
        // Generic.
        pooling_internal<false>(pool_size_x, pool_size_y, pool_stride_x, pool_stride_y, input_view, output_view);
    }
}

struct pooling_f32_request_handle {
    pooling_f32 *primitive;
    const nn::workload_data<float> *input;
    nn::workload_data<float> *output;
};

struct pooling_f32_request_handle_backward {
    pooling_f32 *primitive;
    const nn::workload_data<float> *forward_input;
    const nn::workload_data<float> *forward_output;
    const nn::workload_data<float> *backward_input;
    nn::workload_data<float> *backward_output;
};

void unpack_pooling_callback_handle(void *void_handle) {
    auto handle = reinterpret_cast<pooling_f32_request_handle*>(void_handle);
    handle->primitive->run_pooling(handle->input, handle->output);
}

void unpack_pooling_backward_callback_handle(void *void_handle) {
    auto handle = reinterpret_cast<pooling_f32_request_handle_backward*>(void_handle);
    handle->primitive->run_backward_delta(handle->forward_input, handle->forward_output, handle->backward_input, handle->backward_output);
}

void pooling_f32::backward(
    const nn::workload_data<float> *forward_input,
    const nn::workload_data<float> *forward_output,
    const nn::workload_data<float> *backward_input,
    nn::workload_data<float> *backward_output)
{
    const auto num_batch_items =
        (backward_output->view_end.t[NN_DATA_COORD_n] - backward_output->view_begin.t[NN_DATA_COORD_n] + 1);

    const auto total_workers = num_batch_items;

    if (device->thread_pool.get_num_threads() < 2 || total_workers < 2)
    {
        // Its tiny data or there is only one thread available - just do it singlethreaded way.
        run_backward_delta(forward_input, forward_output, backward_input, backward_output);
    }
    else
    {
        // Full cores utilization version.
        std::vector<nn::workload_data<float> *> backprop_output_delta_views(total_workers);

        // Fill slave work items.

        for (auto batch_item = 0u; batch_item < num_batch_items; ++batch_item)
        {
            auto item_in_pool = batch_item;

            // Replace nn_workload_datas pointers with views.
            nn_workload_data_coords_t output_view_begin =
            {
                batch_item,
                0,
                0,
                0,
                0,
                0
            };
            nn_workload_data_coords_t output_view_end =
            {
                batch_item,
                backward_output->get_length(NN_DATA_COORD_x) - 1,
                backward_output->get_length(NN_DATA_COORD_y) - 1,
                backward_output->get_length(NN_DATA_COORD_z) - 1,
                backward_output->get_length(NN_DATA_COORD_p) - 1,
                backward_output->get_length(NN_DATA_COORD_q) - 1
            };

            backprop_output_delta_views[item_in_pool] =
                new nn::workload_data<float>(*backward_output, output_view_begin, output_view_end);
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(total_workers);
        std::vector<pooling_f32_request_handle_backward> request_handles(total_workers);

        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            request_handles[item_in_pool].primitive = this;
            request_handles[item_in_pool].forward_input = forward_input;
            request_handles[item_in_pool].forward_output = forward_output;
            request_handles[item_in_pool].backward_input = backward_input;
            request_handles[item_in_pool].backward_output = backprop_output_delta_views[item_in_pool];

            job[item_in_pool].callback = unpack_pooling_backward_callback_handle;
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

template<uint32_t T_num_accumulators>
void run_backward_delta_inner(
    float* &input_base_offset,
    float* &output_base_offset,
    float* &error_base_offset,
    float* &final_base_offset,
    uint32_t output_l1_step,
    uint32_t error_l1_step,
    uint32_t output_l2_step,
    uint32_t error_l2_step,
    int32_t output_width_range_start,
    int32_t output_height_range_start,
    int32_t output_width_range_end,
    int32_t output_height_range_end)
{
    auto output_l1_offset = output_base_offset;
    auto error_l1_offset = error_base_offset;

    // Prepare accumulators.
    __m256 acc[T_num_accumulators];
    #pragma unroll (T_num_accumulators)
    for(uint32_t acc_id = 0; acc_id < T_num_accumulators; ++acc_id)
        acc[acc_id] = _mm256_setzero_ps();

    // Load input we wanna check.
    __m256 input[T_num_accumulators];
    #pragma unroll (T_num_accumulators)
    for(uint32_t acc_id = 0; acc_id < T_num_accumulators; ++acc_id)
        input[acc_id] = _mm256_load_ps(input_base_offset + acc_id * C_simd_width);

    for (int32_t output_row = output_height_range_start;
            output_row <= output_height_range_end;
            ++output_row)
    {
        auto output_l2_offset = output_l1_offset;
        auto error_l2_offset = error_l1_offset;

        for (int32_t output_column = output_width_range_start;
            output_column <= output_width_range_end;
            ++output_column)
        {
            // Compare output and input and create masks for errors.
            __m256 mask[T_num_accumulators];
            #pragma unroll (T_num_accumulators)
            for(uint32_t acc_id = 0; acc_id < T_num_accumulators; ++acc_id)
                mask[acc_id] = _mm256_cmp_ps(input[acc_id], _mm256_load_ps(output_l2_offset + acc_id * C_simd_width), _CMP_EQ_OQ);

            // Mask out errors that we not require.
            __m256 error[T_num_accumulators];
            #pragma unroll (T_num_accumulators)
            for(uint32_t acc_id = 0; acc_id < T_num_accumulators; ++acc_id)
                error[acc_id] = _mm256_and_ps(_mm256_load_ps(error_l2_offset + acc_id * C_simd_width), mask[acc_id]);

            // Accumulate errors.
            #pragma unroll (T_num_accumulators)
            for(uint32_t acc_id = 0; acc_id < T_num_accumulators; ++acc_id)
                acc[acc_id] =_mm256_add_ps(error[acc_id], acc[acc_id]);

            output_l2_offset += output_l2_step;
            error_l2_offset += error_l2_step;
        }

        output_l1_offset += output_l1_step;
        error_l1_offset += error_l1_step;
    }
     
    // Store propagated error.
    #pragma unroll (T_num_accumulators)
    for(uint32_t acc_id = 0; acc_id < T_num_accumulators; ++acc_id)
        _mm256_store_ps(final_base_offset + acc_id * C_simd_width, acc[acc_id]);

    input_base_offset += T_num_accumulators * C_simd_width;
    output_base_offset += T_num_accumulators * C_simd_width;
    error_base_offset += T_num_accumulators * C_simd_width;
    final_base_offset += T_num_accumulators * C_simd_width;
}

void pooling_f32::run_backward_delta(
    const nn::workload_data<float> *forward_input,
    const nn::workload_data<float> *forward_output,
    const nn::workload_data<float> *backward_input,
    nn::workload_data<float> *backward_output)
{
    const auto& in_begin = backward_output->view_begin;
    const auto& in_end = backward_output->view_end;
    const auto& out_begin = backward_input->view_begin;
    const auto& out_end = backward_input->view_end;

    const auto output_width = backward_input->get_length(NN_DATA_COORD_x);
    const auto output_height = backward_input->get_length(NN_DATA_COORD_y);

    auto forward_input_buffer = static_cast<float*>(forward_input->parent->data_buffer);
    auto backward_input_buffer = static_cast<float*>(backward_input->parent->data_buffer);

    auto forward_output_buffer = static_cast<float*>(forward_output->parent->data_buffer);
    auto backward_output_buffer = static_cast<float*>(backward_output->parent->data_buffer);

    const auto output_l2_step = forward_output->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_l1_step = output_l2_step * forward_output->parent->lengths.t[NN_DATA_COORD_x];

    const auto error_l2_step = backward_input->parent->lengths.t[NN_DATA_COORD_z];
    const auto error_l1_step = error_l2_step * backward_input->parent->lengths.t[NN_DATA_COORD_x];

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
                int32_t output_width_range_start = 0;
                int32_t output_height_range_start = 0;
                int32_t output_width_range_end = input_column / pool_stride_x;
                int32_t output_height_range_end = input_row / pool_stride_y;
                if((int32_t)input_column - ((int32_t)pool_size_x - 1) >= 0) output_width_range_start = (input_column - (pool_size_x - 1)) / pool_stride_x + (((input_column - (pool_size_x - 1)) % pool_stride_x != 0 ) ? 1 : 0);
                if((int32_t)input_row - ((int32_t)pool_size_y - 1) >= 0) output_height_range_start = (input_row - (pool_size_y - 1)) / pool_stride_y + (((input_row - (pool_size_y - 1)) % pool_stride_y != 0 ) ? 1 : 0);

                // Clamp ranges to view which is <0,size).
                output_width_range_start = std::max(output_width_range_start, 0);
                output_height_range_start = std::max(output_height_range_start, 0);
                output_width_range_end = std::min(output_width_range_end, (int32_t)output_width - 1);
                output_height_range_end = std::min(output_height_range_end, (int32_t)output_height - 1);


                auto input_base_offset = forward_input_buffer
                        + in_begin.t[NN_DATA_COORD_z] 
                        + input_column * forward_input->parent->lengths.t[NN_DATA_COORD_z]
                        + input_row * forward_input->parent->lengths.t[NN_DATA_COORD_z] * forward_input->parent->lengths.t[NN_DATA_COORD_x]
                        + batch * forward_input->parent->lengths.t[NN_DATA_COORD_z] * forward_input->parent->lengths.t[NN_DATA_COORD_x] * forward_input->parent->lengths.t[NN_DATA_COORD_y];

                auto output_base_offset = forward_output_buffer
                        + in_begin.t[NN_DATA_COORD_z] 
                        + output_width_range_start * forward_output->parent->lengths.t[NN_DATA_COORD_z]
                        + output_height_range_start * forward_output->parent->lengths.t[NN_DATA_COORD_z] * forward_output->parent->lengths.t[NN_DATA_COORD_x]
                        + batch * forward_output->parent->lengths.t[NN_DATA_COORD_z] * forward_output->parent->lengths.t[NN_DATA_COORD_x] * forward_output->parent->lengths.t[NN_DATA_COORD_y];

                auto error_base_offset = backward_input_buffer
                        + in_begin.t[NN_DATA_COORD_z] 
                        + output_width_range_start * backward_input->parent->lengths.t[NN_DATA_COORD_z]
                        + output_height_range_start * backward_input->parent->lengths.t[NN_DATA_COORD_z] * backward_input->parent->lengths.t[NN_DATA_COORD_x]
                        + batch * backward_input->parent->lengths.t[NN_DATA_COORD_z] * backward_input->parent->lengths.t[NN_DATA_COORD_x] * backward_input->parent->lengths.t[NN_DATA_COORD_y];

                auto final_base_offset = backward_output_buffer
                        + in_begin.t[NN_DATA_COORD_z] 
                        + input_column * backward_output->parent->lengths.t[NN_DATA_COORD_z]
                        + input_row * backward_output->parent->lengths.t[NN_DATA_COORD_z] * backward_output->parent->lengths.t[NN_DATA_COORD_x]
                        + batch * backward_output->parent->lengths.t[NN_DATA_COORD_z] * backward_output->parent->lengths.t[NN_DATA_COORD_x] * backward_output->parent->lengths.t[NN_DATA_COORD_y];


                // Here..
                auto num_full_passes = backward_output->get_length(NN_DATA_COORD_z) / (C_max_num_accumulators * C_simd_width);
                auto partial_pass_size = backward_output->get_length(NN_DATA_COORD_z) % (C_max_num_accumulators * C_simd_width) / C_simd_width;

                for (uint32_t pass = 0; pass < num_full_passes; ++pass)
                {
                    #pragma forceinline recursive
                    run_backward_delta_inner<C_max_num_accumulators>(input_base_offset, output_base_offset, error_base_offset, final_base_offset, output_l1_step, error_l1_step, output_l2_step,error_l2_step, output_width_range_start, output_height_range_start, output_width_range_end, output_height_range_end);
                }

                switch(partial_pass_size)
                {
                case 1: run_backward_delta_inner<1>(input_base_offset, output_base_offset, error_base_offset, final_base_offset, output_l1_step, error_l1_step, output_l2_step,error_l2_step, output_width_range_start, output_height_range_start, output_width_range_end, output_height_range_end); break;
                case 2: run_backward_delta_inner<2>(input_base_offset, output_base_offset, error_base_offset, final_base_offset, output_l1_step, error_l1_step, output_l2_step,error_l2_step, output_width_range_start, output_height_range_start, output_width_range_end, output_height_range_end); break;
                case 3: run_backward_delta_inner<3>(input_base_offset, output_base_offset, error_base_offset, final_base_offset, output_l1_step, error_l1_step, output_l2_step,error_l2_step, output_width_range_start, output_height_range_start, output_width_range_end, output_height_range_end); break;
                case 4: run_backward_delta_inner<4>(input_base_offset, output_base_offset, error_base_offset, final_base_offset, output_l1_step, error_l1_step, output_l2_step,error_l2_step, output_width_range_start, output_height_range_start, output_width_range_end, output_height_range_end); break;
                case 5: run_backward_delta_inner<5>(input_base_offset, output_base_offset, error_base_offset, final_base_offset, output_l1_step, error_l1_step, output_l2_step,error_l2_step, output_width_range_start, output_height_range_start, output_width_range_end, output_height_range_end); break;
                case 6: run_backward_delta_inner<6>(input_base_offset, output_base_offset, error_base_offset, final_base_offset, output_l1_step, error_l1_step, output_l2_step,error_l2_step, output_width_range_start, output_height_range_start, output_width_range_end, output_height_range_end); break;
                case 7: run_backward_delta_inner<7>(input_base_offset, output_base_offset, error_base_offset, final_base_offset, output_l1_step, error_l1_step, output_l2_step,error_l2_step, output_width_range_start, output_height_range_start, output_width_range_end, output_height_range_end); break;
                default: break;
                }
            }
        }
    }
}

void pooling_f32::backward(const std::vector<nn_workload_data_t *> &inputs,
                           const std::vector<const nn_workload_data_t *> &parameters,
                           const std::vector<const nn_workload_data_t *> &outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    const nn::workload_data<float> backward_input(outputs[0]->parent->delta_buffer, outputs[0]->parent->lengths, outputs[0]->parent->layout);
    nn::workload_data<float> backward_output(inputs[0]->parent->delta_buffer, inputs[0]->parent->lengths, inputs[0]->parent->layout);

    backward(reinterpret_cast<const nn::workload_data<float> *>(inputs[0]),
             reinterpret_cast<const nn::workload_data<float> *>(outputs[0]),
             &backward_input,
             &backward_output);
}

void pooling_f32::forward(const nn::workload_data<float> *input, nn::workload_data<float> *output)
{
    const auto batch = output->parent->lengths.t[NN_DATA_COORD_n];

    const auto num_output_row_items =
        (output->view_end.t[NN_DATA_COORD_y] - output->view_begin.t[NN_DATA_COORD_y] + 1);
    const auto num_batch_items =
        (output->view_end.t[NN_DATA_COORD_n] - output->view_begin.t[NN_DATA_COORD_n] + 1);

    const auto total_workers = (batch == 48) ? num_batch_items : num_output_row_items * num_batch_items;

    if (device->thread_pool.get_num_threads() < 2 || total_workers < 2)
    {
        // Its tiny data or there is only one thread available - just do it single threaded way.
        run_pooling(input, output);
    }
    else
    {
        // Full cores utilization version.
        std::vector<nn::workload_data<float>*> input_views(total_workers);
        std::vector<nn::workload_data<float>*> output_views(total_workers);

        // Fill slave work items.
        // Outer loop will have single iteration in batch 48 
        // because such high batch will give enough data for multithreading.
        for (auto output_row_item = 0u; output_row_item < num_output_row_items; ++output_row_item)
        {
            for (auto batch_item = 0u; batch_item < num_batch_items; ++batch_item)
            {
                auto item_in_pool = batch_item;
                if (batch != 48) item_in_pool += output_row_item * num_batch_items;

                // Replace nn_workload_datas pointers with views.
                nn_workload_data_coords_t input_view_begin =
                {
                    batch_item,
                    0,
                    (batch == 48) ? 0 : static_cast<uint32_t>(output_row_item * pool_stride_y),
                    0,
                    0,
                    0
                };
                nn_workload_data_coords_t input_view_end = {
                    batch_item,
                    input->get_length(NN_DATA_COORD_x) - 1,
                    (batch == 48) 
                        ? input->get_length(NN_DATA_COORD_y) - 1
                        : static_cast<uint32_t>(output_row_item * pool_stride_y + pool_size_y - 1),
                    input->get_length(NN_DATA_COORD_z) - 1,
                    input->get_length(NN_DATA_COORD_p) - 1,
                    input->get_length(NN_DATA_COORD_q) - 1
                };

                nn_workload_data_coords_t output_view_begin =
                {
                    batch_item,
                    0,
                    (batch == 48) ? 0 : output_row_item,
                    0,
                    0,
                    0
                };
                nn_workload_data_coords_t output_view_end =
                {
                    batch_item,
                    output->get_length(NN_DATA_COORD_x) - 1,
                    (batch == 48) ? output->get_length(NN_DATA_COORD_y) - 1 : output_row_item,
                    output->get_length(NN_DATA_COORD_z) - 1,
                    output->get_length(NN_DATA_COORD_p) - 1,
                    output->get_length(NN_DATA_COORD_q) - 1
                };

                input_views[item_in_pool] = new nn::workload_data<float>(
                    *const_cast<nn::workload_data<float> *>(input), input_view_begin, input_view_end);

                output_views[item_in_pool] =
                    new nn::workload_data<float>(*output, output_view_begin, output_view_end);
            }

            if (batch == 48) break;
        }

        // Run threads.
        std::vector<pooling_f32_request_handle> request_handles(total_workers);
        std::vector<nn_multithreaded_request> job(total_workers);
        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            request_handles[item_in_pool].primitive = this;
            request_handles[item_in_pool].input = input_views[item_in_pool];
            request_handles[item_in_pool].output = output_views[item_in_pool];

            job[item_in_pool].callback = unpack_pooling_callback_handle;
            job[item_in_pool].request_handle = &request_handles[item_in_pool];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);

        // Cleanup dynamic memory.
        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            delete input_views[item_in_pool];
            delete output_views[item_in_pool];
        }
    }
}

void pooling_f32::forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 1);
    assert(parameters.size() == 0);
    assert(outputs.size() == 1);

    forward(reinterpret_cast<const nn::workload_data<float> *>(inputs[0]),
            reinterpret_cast<nn::workload_data<float> *>(outputs[0]));
}

bool pooling_f32::validate_input(size_t index, nn_workload_data_t *data)
{
    switch (index) {
    case 0:
        return nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, float>::validate<true>(
            data, get_required_input_w(), get_required_input_h(), input_size_z, batch_size, 0, 0, 0, 0);
    }

    throw std::invalid_argument("index out of range");
}

// Interface wrappers.
void wrapper_pooling_work_item(nn_workload_item *const work_item)
{
    auto primitive = static_cast<pooling_f32 *>(work_item->primitive);
    switch (primitive->pooling_mode) {
    case NN_POOLING_MODE_MAX:
        primitive->forward(reinterpret_cast<nn::workload_data<float> *>(work_item->input[0].get_data_view()),
                           reinterpret_cast<nn::workload_data<float> *>(work_item->output[0]));
        break;
    default:
        assert(0);
    }
}

void wrapper_pooling_work_item_backward(nn_workload_item *const work_item)
{
    auto primitive = static_cast<pooling_f32 *>(work_item->forward_item->primitive);
    switch (primitive->pooling_mode)
    {
    case NN_POOLING_MODE_MAX:
        // Initialize buffers - it should be called before multithreaded dispatch.
        memset(work_item->output[0]->parent->data_buffer, 0, work_item->output[0]->parent->buffer_size);

        primitive->backward(reinterpret_cast<nn::workload_data<float> *>(work_item->forward_item->input[0].get_data_view()), 
                            reinterpret_cast<nn::workload_data<float> *>(work_item->forward_item->output[0]),
                            reinterpret_cast<nn::workload_data<float> *>(work_item->input[0].get_data_view()),
                            reinterpret_cast<nn::workload_data<float> *>(work_item->output[0]));
        break;
    default:
        assert(0);
    }
}
} // namespace layer

nn_primitive_handle_t NN_API_CALL_CONVENTION
nn_primitives_pooling_f32_create_0(nn_device_t *device,
                                   NN_POOLING_MODE pooling_mode,
                                   size_t pool_size_x,
                                   size_t pool_size_y,
                                   size_t pool_stride_x,
                                   size_t pool_stride_y,
                                   size_t num_feature_maps,
                                   size_t output_w,
                                   size_t output_h,
                                   size_t batch_size,
                                   const nn_primitives_pooling_hints_t *hints,
                                   NN_API_STATUS *status) {
    SET_STATUS(NN_API_STATUS_OK);

    std::remove_const<std::remove_pointer<decltype(hints)>::type>::type hints_ = {};
    if(hints != nullptr)
        hints_ = *hints;

    return new layer::pooling_f32(pooling_mode,
                                  pool_size_x,
                                  pool_size_y,
                                  pool_stride_x,
                                  pool_stride_y,
                                  num_feature_maps,
                                  output_w,
                                  output_h,
                                  batch_size,
                                  hints_.output_padding.left,
                                  hints_.output_padding.right,
                                  hints_.output_padding.top,
                                  hints_.output_padding.bottom,
                                  reinterpret_cast<nn_device_internal *>(device));
}
