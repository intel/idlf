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
#include "layer_pooling_avx2.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <map>
#include <tuple>

// SIMD width for this implementation
const auto C_simd_width = sizeof(__m256) / sizeof(float);
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
    const nn::nn_workload_data_t<float> *input_view,
    nn::nn_workload_data_t<float> *output_view)
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

pooling_f32 * pooling_f32::create(NN_POOLING_MODE pooling_mode,
                                 size_t pool_size_x,
                                 size_t pool_size_y,
                                 size_t pool_stride_x,
                                 size_t pool_stride_y,
                                 size_t num_feature_maps,
                                 size_t output_w,
                                 size_t output_h,
                                 size_t batch_size,
                                 nn_device_t *device) {
    return new pooling_f32(pooling_mode,
                           pool_size_x,
                           pool_size_y,
                           pool_stride_x,
                           pool_stride_y,
                           num_feature_maps,
                           output_w,
                           output_h,
                           batch_size,
                           reinterpret_cast<nn_device_internal *>(device));
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
                         nn_device_internal *device)
    : primitive_zxyn_f32_base(batch_size, num_feature_maps, output_w, output_h, num_feature_maps, device),
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
    //  C1 maxpooling.
    { prepare_entry<56, 56, 96, 28, 28, 2, 2, 2, 2>() },

    //  C2 maxpooling.
    { prepare_entry<24, 24, 256, 12, 12, 2, 2, 2, 2>() },

    //  C5 maxpooling.
    { prepare_entry<12, 12, 1024, 6, 6, 2, 2, 2, 2>() },

    // CaffeNet C1 maxpooling.
    { prepare_entry<55, 55, 96, 27, 27, 2, 2, 3, 3>() },

    // CaffeNet C2 maxpooling.
    { prepare_entry<27, 27, 256, 13, 13, 2, 2, 3, 3>() },

    // CaffeNet C5 maxpooling.
    { prepare_entry<13, 13, 256, 6, 6, 2, 2, 3, 3>() },
};
}

void pooling_f32::run_pooling(const nn::nn_workload_data_t<float> *input_view,
                              nn::nn_workload_data_t<float> *output_view) {
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
    const nn::nn_workload_data_t<float> *input;
    nn::nn_workload_data_t<float> *output;
};

void unpack_pooling_callback_handle(void *void_handle) {
    auto handle = reinterpret_cast<pooling_f32_request_handle*>(void_handle);
    handle->primitive->run_pooling(handle->input, handle->output);
}

void pooling_f32::forward(const nn::nn_workload_data_t<float> *input, nn::nn_workload_data_t<float> *output)
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
        std::vector<nn::nn_workload_data_t<float>*> input_views(total_workers);
        std::vector<nn::nn_workload_data_t<float>*> output_views(total_workers);

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

                input_views[item_in_pool] = new nn::nn_workload_data_t<float>(
                    *const_cast<nn::nn_workload_data_t<float> *>(input), input_view_begin, input_view_end);

                output_views[item_in_pool] =
                    new nn::nn_workload_data_t<float>(*output, output_view_begin, output_view_end);
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

// Interface wrappers.
void wrapper_pooling_work_item(nn_workload_item *const work_item, nn_device_internal* device)
{
    auto primitive = static_cast<pooling_f32 *>(work_item->primitive);
    switch (primitive->pooling_mode) {
    case NN_POOLING_MODE_MAX:
        primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->input[0]->output),
                           reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->output));
        break;
    default:
        assert(0);
    }
}

namespace pooling_f32_impl {

nn_event_t NN_API_CALL_CONVENTION forward_async(nn_primitive_handle_t handle,
                                                nn_opaque_data_t *input,
                                                nn_opaque_data_t *output,
                                                size_t dependencies_count,
                                                nn_event_t *dependencies,
                                                NN_API_STATUS *status) {
    auto primitive = static_cast<layer::pooling_f32 *>(handle);
    primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(input),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(output));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}

nn_primitive_handle_t NN_API_CALL_CONVENTION create(nn_device_t *device,
                                                    NN_POOLING_MODE pooling_mode,
                                                    size_t pool_size_x,
                                                    size_t pool_size_y,
                                                    size_t pool_stride_x,
                                                    size_t pool_stride_y,
                                                    size_t num_feature_maps,
                                                    size_t output_w,
                                                    size_t output_h,
                                                    size_t batch_size,
                                                    NN_API_STATUS *status) {
    SET_STATUS(NN_API_STATUS_OK);
    return layer::pooling_f32::create(pooling_mode,
                                      pool_size_x,
                                      pool_size_y,
                                      pool_stride_x,
                                      pool_stride_y,
                                      num_feature_maps,
                                      output_w,
                                      output_h,
                                      batch_size,
                                      device);
}
}
} // namespace layer

nn_primitives_pooling_f32_0_t nn_primitives_pooling_f32_0{layer::pooling_f32_impl::create,
                                                          layer::helper_zxyn_f32::create_input,
                                                          layer::helper_zxyn_f32::validate_input,
                                                          layer::helper_zxyn_f32::create_output,
                                                          layer::helper_zxyn_f32::create_output_with_padding,
                                                          nullptr, // create_view
                                                          layer::pooling_f32_impl::forward_async,
                                                          layer::helper_zxyn_f32::copy_output_async};