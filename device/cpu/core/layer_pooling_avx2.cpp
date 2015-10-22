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
const auto C_max_block_size = 5;
const uint32_t offsets[] = {0, 1, 2, 3, 4, 5, 6, 7};


namespace layer
{
template<uint32_t T_num_acc,
         bool     T_first_run,
         bool     T_final_run,
         bool     T_have_intermediate,
         NN_POOLING_MODE T_mode>
struct pooling_macro;

template<uint32_t T_num_acc,
         bool     T_first_run,
         bool     T_final_run,
         bool     T_have_intermediate>
struct pooling_macro<T_num_acc, T_first_run, T_final_run, T_have_intermediate, NN_POOLING_MODE_MAX>
{
    static inline void macro(
        float* input_buffer,
        float* input_ptr,
        uint32_t* intermediate_output_ptr,
        float* output_ptr,
        const __m256 inverse_of_count)
    {
        uint32_t input_offset = static_cast<uint32_t>(input_ptr - input_buffer);

        __m256i offset0, offset1, offset2, offset3, offset4;

        if (T_have_intermediate)
        {
            __m256i input_offset_vec = _mm256_add_epi32(
                _mm256_set1_epi32(input_offset),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets)));

            if (T_num_acc >= 1) offset0 = _mm256_set1_epi32(0 * C_simd_width);
            if (T_num_acc >= 2) offset1 = _mm256_set1_epi32(1 * C_simd_width);
            if (T_num_acc >= 3) offset2 = _mm256_set1_epi32(2 * C_simd_width);
            if (T_num_acc >= 4) offset3 = _mm256_set1_epi32(3 * C_simd_width);
            if (T_num_acc >= 5) offset4 = _mm256_set1_epi32(4 * C_simd_width);

            if (T_num_acc >= 1) offset0 = _mm256_add_epi32(offset0, input_offset_vec);
            if (T_num_acc >= 2) offset1 = _mm256_add_epi32(offset1, input_offset_vec);
            if (T_num_acc >= 3) offset2 = _mm256_add_epi32(offset2, input_offset_vec);
            if (T_num_acc >= 4) offset3 = _mm256_add_epi32(offset3, input_offset_vec);
            if (T_num_acc >= 5) offset4 = _mm256_add_epi32(offset4, input_offset_vec);
        }

        __m256 acc0, acc1, acc2, acc3, acc4;
        __m256i mask0, mask1, mask2, mask3, mask4;

        if (T_num_acc >= 1) acc0 = _mm256_load_ps(input_ptr + 0 * C_simd_width);
        if (T_num_acc >= 2) acc1 = _mm256_load_ps(input_ptr + 1 * C_simd_width);
        if (T_num_acc >= 3) acc2 = _mm256_load_ps(input_ptr + 2 * C_simd_width);
        if (T_num_acc >= 4) acc3 = _mm256_load_ps(input_ptr + 3 * C_simd_width);
        if (T_num_acc >= 5) acc4 = _mm256_load_ps(input_ptr + 4 * C_simd_width);

        if (T_first_run)
        {
            __m256i full_mask = _mm256_set1_epi32(0xFFFFFFFF);

            if (T_num_acc >= 1) mask0 = full_mask;
            if (T_num_acc >= 2) mask1 = full_mask;
            if (T_num_acc >= 3) mask2 = full_mask;
            if (T_num_acc >= 4) mask3 = full_mask;
            if (T_num_acc >= 5) mask4 = full_mask;
        }
        else
        {
            if (T_num_acc >= 1) mask0 = _mm256_castps_si256(_mm256_cmp_ps(_mm256_load_ps(output_ptr + 0 * C_simd_width), acc0, _CMP_LT_OQ));
            if (T_num_acc >= 2) mask1 = _mm256_castps_si256(_mm256_cmp_ps(_mm256_load_ps(output_ptr + 1 * C_simd_width), acc1, _CMP_LT_OQ));
            if (T_num_acc >= 3) mask2 = _mm256_castps_si256(_mm256_cmp_ps(_mm256_load_ps(output_ptr + 2 * C_simd_width), acc2, _CMP_LT_OQ));
            if (T_num_acc >= 4) mask3 = _mm256_castps_si256(_mm256_cmp_ps(_mm256_load_ps(output_ptr + 3 * C_simd_width), acc3, _CMP_LT_OQ));
            if (T_num_acc >= 5) mask4 = _mm256_castps_si256(_mm256_cmp_ps(_mm256_load_ps(output_ptr + 4 * C_simd_width), acc4, _CMP_LT_OQ));
        }

        if (T_have_intermediate)
        {
            if (T_num_acc >= 1) _mm256_maskstore_epi32(reinterpret_cast<int32_t*>(intermediate_output_ptr)+0 * C_simd_width, mask0, offset0);
            if (T_num_acc >= 2) _mm256_maskstore_epi32(reinterpret_cast<int32_t*>(intermediate_output_ptr)+1 * C_simd_width, mask1, offset1);
            if (T_num_acc >= 3) _mm256_maskstore_epi32(reinterpret_cast<int32_t*>(intermediate_output_ptr)+2 * C_simd_width, mask2, offset2);
            if (T_num_acc >= 4) _mm256_maskstore_epi32(reinterpret_cast<int32_t*>(intermediate_output_ptr)+3 * C_simd_width, mask3, offset3);
            if (T_num_acc >= 5) _mm256_maskstore_epi32(reinterpret_cast<int32_t*>(intermediate_output_ptr)+4 * C_simd_width, mask4, offset4);
        }

        if (T_num_acc >= 1) _mm256_maskstore_ps(output_ptr + 0 * C_simd_width, mask0, acc0);
        if (T_num_acc >= 2) _mm256_maskstore_ps(output_ptr + 1 * C_simd_width, mask1, acc1);
        if (T_num_acc >= 3) _mm256_maskstore_ps(output_ptr + 2 * C_simd_width, mask2, acc2);
        if (T_num_acc >= 4) _mm256_maskstore_ps(output_ptr + 3 * C_simd_width, mask3, acc3);
        if (T_num_acc >= 5) _mm256_maskstore_ps(output_ptr + 4 * C_simd_width, mask4, acc4);
    }
};

template<uint32_t T_num_acc,
    bool     T_first_run,
    bool     T_final_run,
    bool     T_have_intermediate>
struct pooling_macro<T_num_acc, T_first_run, T_final_run, T_have_intermediate, NN_POOLING_MODE_AVERAGE>
{
    static inline void macro(
        float* input_buffer,
        float* input_ptr,
        uint32_t* intermediate_output_ptr,
        float* output_ptr,
        const __m256 inverse_of_count)
    {
        __m256 accs[T_num_acc];

#pragma unroll(T_num_acc)
        for (size_t i = 0; i < T_num_acc; ++i) {
            if (T_first_run) {
                accs[i] = _mm256_load_ps(input_ptr + i * C_simd_width);
            }
            else {
                accs[i] = _mm256_load_ps(output_ptr + i * C_simd_width);
                accs[i] = _mm256_add_ps(accs[i], _mm256_load_ps(input_ptr + i * C_simd_width));
            }

            if (T_final_run) {
                _mm256_store_ps(output_ptr + i * C_simd_width, _mm256_mul_ps(accs[i], inverse_of_count));
            }
            else{
                _mm256_store_ps(output_ptr + i * C_simd_width, accs[i]);
            }
        }
    }
};

template<bool T_first_run, bool T_last_run, bool T_have_intermediate, NN_POOLING_MODE T_mode>
inline void pooling_outer_macro(
    float* input_buffer,
    float* input_ptr,
    uint32_t* intermediate_output_ptr,
    float* output_ptr,
    const __m256 inverse_of_count,
    uint32_t num_blocks_full,
    uint32_t partial_block_size)
{
    for (uint32_t block = 0; block < num_blocks_full; block++)
    {
        pooling_macro<C_max_block_size, T_first_run, T_last_run, T_have_intermediate, T_mode>::macro(input_buffer, input_ptr, intermediate_output_ptr, output_ptr, inverse_of_count);
        input_ptr += C_max_block_size * C_simd_width;
        intermediate_output_ptr += C_max_block_size * C_simd_width;
        output_ptr += C_max_block_size * C_simd_width;
    }

    switch (partial_block_size)
    {
    case  0: break;
    case  1: pooling_macro< 1, T_first_run, T_last_run, T_have_intermediate, T_mode>::macro(input_buffer, input_ptr, intermediate_output_ptr, output_ptr, inverse_of_count); break;
    case  2: pooling_macro< 2, T_first_run, T_last_run, T_have_intermediate, T_mode>::macro(input_buffer, input_ptr, intermediate_output_ptr, output_ptr, inverse_of_count); break;
    case  3: pooling_macro< 3, T_first_run, T_last_run, T_have_intermediate, T_mode>::macro(input_buffer, input_ptr, intermediate_output_ptr, output_ptr, inverse_of_count); break;
    case  4: pooling_macro< 4, T_first_run, T_last_run, T_have_intermediate, T_mode>::macro(input_buffer, input_ptr, intermediate_output_ptr, output_ptr, inverse_of_count); break;
    default:
        /* Execution can never reach here (see 'partial_block_size') calculation.*/
        /* Need to inform compiler that it should not generate code for 'default'.*/
        /* [TODO] heed to handle GCC */
        NN_UNREACHABLE_CODE;
    }
}

template<NN_POOLING_MODE T_mode,
        bool T_exact_match,
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
    const int32_t center_offset_x,
    const int32_t center_offset_y,
    const nn::workload_data<> *input_view,
    nn::workload_data<> *intermediate_output_view,
    nn::workload_data<> *output_view)
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
    auto* const intermediate_output_buffer = (intermediate_output_view) ? reinterpret_cast<uint32_t*>(intermediate_output_view->parent->data_buffer) : nullptr;
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

    const __m256 inverse_of_count = _mm256_set1_ps(1.0f / pool_size_x / pool_size_y);

    for(uint32_t out_image = output_image_view_start; out_image <= output_image_view_end; ++out_image)
    {
        const uint32_t input_image_offset = out_image * input_image_size;
        const uint32_t output_image_offset = out_image * output_image_size;

        for (uint32_t output_row = output_row_view_start, input_row = input_row_view_start; output_row <= output_row_view_end; ++output_row, input_row += pool_stride_y)
        {
            const uint32_t output_row_offset = output_row * output_row_size;

            for (uint32_t pool_y = 0; pool_y < pool_size_y; ++pool_y)
            {
                const uint32_t input_row_offset = (input_row - center_offset_y + pool_y) * input_row_size;

                for (uint32_t output_column = output_column_view_start, input_column = input_column_view_start; output_column <= output_column_view_end; ++output_column, input_column += pool_stride_x)
                {
                    const uint32_t output_column_offset = output_column * num_feature_maps;

                    bool first_run = (pool_y == 0) ? true : false;

                    for (uint32_t pool_x = 0; pool_x < pool_size_x; ++pool_x)
                    {
                        bool last_run = (pool_y == pool_size_y - 1 && pool_x == pool_size_x - 1) ? true : false;
                        const uint32_t input_column_offset = (input_column - center_offset_x + pool_x) * num_feature_maps;

                        float* output_ptr = output_buffer + output_image_offset + output_row_offset + output_column_offset + output_depth_view_start;
                        float* input_ptr = input_buffer + input_image_offset + input_row_offset + input_column_offset + input_depth_view_start;

                        if(intermediate_output_buffer)
                        {
                            uint32_t* intermediate_output_ptr = intermediate_output_buffer + output_image_offset + output_row_offset + output_column_offset + output_depth_view_start;
                            if (first_run)
                            {
                                pooling_outer_macro<true, false, true, T_mode>(input_buffer, input_ptr, intermediate_output_ptr, output_ptr, inverse_of_count, num_blocks_full, partial_block_size);
                                first_run = false;
                            }
                            else if (last_run)
                            {
                                pooling_outer_macro<false, true, true, T_mode>(input_buffer, input_ptr, intermediate_output_ptr, output_ptr, inverse_of_count, num_blocks_full, partial_block_size);
                            }
                            else
                            {
                                pooling_outer_macro<false, false, true, T_mode>(input_buffer, input_ptr, intermediate_output_ptr, output_ptr, inverse_of_count, num_blocks_full, partial_block_size);
                            }
                        }
                        else
                        {
                            if (first_run)
                            {
                                pooling_outer_macro<true, false, false, T_mode>(input_buffer, input_ptr, nullptr, output_ptr, inverse_of_count, num_blocks_full, partial_block_size);
                                first_run = false;
                            }
                            else if (last_run)
                            {
                                pooling_outer_macro<false, true, false, T_mode>(input_buffer, input_ptr, nullptr, output_ptr, inverse_of_count, num_blocks_full, partial_block_size);
                            }
                            else
                            {
                                pooling_outer_macro<false, false, false, T_mode>(input_buffer, input_ptr, nullptr, output_ptr, inverse_of_count, num_blocks_full, partial_block_size);
                            }
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
                         const int32_t center_offset_x,
                         const int32_t center_offset_y,
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
      pool_stride_y(pool_stride_y),
      center_offset_x(center_offset_x),
      center_offset_y(center_offset_y){}

size_t pooling_f32::get_required_input_w() { return (output_size_x - 1) * pool_stride_x + pool_size_x; }

size_t pooling_f32::get_required_input_h() { return (output_size_y - 1) * pool_stride_y + pool_size_y; }

namespace
{

using optimized_layer_map_t = std::map<
    std::tuple<NN_POOLING_MODE, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>,
    decltype(pooling_internal<NN_POOLING_MODE_MAX, false>)*>;

template<NN_POOLING_MODE T_mode,
         uint32_t T_input_feature_map_width,
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
            T_mode,
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
            T_mode,
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
    { prepare_entry<NN_POOLING_MODE_MAX, 56, 56, 96, 28, 28, 2, 2, 2, 2>() },

    // OverFeat C2 maxpooling.
    { prepare_entry<NN_POOLING_MODE_MAX, 24, 24, 256, 12, 12, 2, 2, 2, 2>() },

    // OverFeat C5 maxpooling.
    { prepare_entry<NN_POOLING_MODE_MAX, 12, 12, 1024, 6, 6, 2, 2, 2, 2>() },

    // CaffeNet C1 maxpooling.
    { prepare_entry<NN_POOLING_MODE_MAX, 55, 55, 96, 27, 27, 2, 2, 3, 3>() },

    // CaffeNet C2 maxpooling.
    { prepare_entry<NN_POOLING_MODE_MAX, 27, 27, 256, 13, 13, 2, 2, 3, 3>() },

    // CaffeNet C5 maxpooling.
    { prepare_entry<NN_POOLING_MODE_MAX, 13, 13, 256, 6, 6, 2, 2, 3, 3>() },
};
}

void pooling_f32::run_pooling(const nn::workload_data<> *input_view,
                              nn::workload_data<> *intermediate_output_view,
                              nn::workload_data<> *output_view) {
    const auto num_feature_maps = input_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_feature_map_width = output_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_feature_map_height = output_view->parent->lengths.t[NN_DATA_COORD_y];
    const auto input_feature_map_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto input_feature_map_height = input_view->parent->lengths.t[NN_DATA_COORD_y];

    auto map_element = optimized_layer_map.find(std::make_tuple(
        pooling_mode,
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
        map_element->second(pool_size_x, pool_size_y, pool_stride_x, pool_stride_y, center_offset_x, center_offset_y, input_view, intermediate_output_view, output_view);
    }
    else
    {
        // Generic.
        switch (pooling_mode) {
        case NN_POOLING_MODE_MAX:
            pooling_internal<NN_POOLING_MODE_MAX, false>(pool_size_x, pool_size_y, pool_stride_x, pool_stride_y, center_offset_x, center_offset_y, input_view, intermediate_output_view, output_view);
            break;
        case NN_POOLING_MODE_AVERAGE:
            pooling_internal<NN_POOLING_MODE_AVERAGE, false>(pool_size_x, pool_size_y, pool_stride_x, pool_stride_y, center_offset_x, center_offset_y, input_view, intermediate_output_view, output_view);
            break;
        default:
            assert(0); // pooling mode not supported
        }
    }
}

struct pooling_f32_request_handle {
    pooling_f32 *primitive;
    const nn::workload_data<> *input;
    nn::workload_data<> *intermediate_output;
    nn::workload_data<> *output;
};

struct pooling_f32_request_handle_backward {
    pooling_f32 *primitive;
    const nn::workload_data<> *forward_input;
    const nn::workload_data<> *forward_intermediate;
    const nn::workload_data<> *forward_output;
    const nn::workload_data<> *backward_input;
    nn::workload_data<> *backward_output;
};

void unpack_pooling_callback_handle(void *void_handle) {
    auto handle = reinterpret_cast<pooling_f32_request_handle*>(void_handle);
    handle->primitive->run_pooling(handle->input, handle->intermediate_output, handle->output);
}

void unpack_pooling_backward_callback_handle(void *void_handle) {
    auto handle = reinterpret_cast<pooling_f32_request_handle_backward*>(void_handle);
    handle->primitive->run_backward_delta(handle->forward_input, handle->forward_intermediate, handle->forward_output, handle->backward_input, handle->backward_output);
}

void pooling_f32::backward(
    const nn::workload_data<> *forward_input,
    const nn::workload_data<> *forward_intermediate,
    const nn::workload_data<> *forward_output,
    const nn::workload_data<> *backward_input,
    nn::workload_data<> *backward_output)
{
    run_backward_delta(forward_input, forward_intermediate, forward_output, backward_input, backward_output);
}

void pooling_f32::run_backward_delta(
    const nn::workload_data<> *forward_input,
    const nn::workload_data<> *forward_intermediate,
    const nn::workload_data<> *forward_output,
    const nn::workload_data<> *backward_input,
    nn::workload_data<> *backward_output)
{
    const auto& out_begin = backward_output->view_begin;
    const auto& out_end = backward_output->view_end;
    const auto& in_begin = backward_input->view_begin;
    const auto& in_end = backward_input->view_end;

    auto forward_input_buffer = static_cast<float*>(forward_input->parent->data_buffer);
    auto forward_intermediate_buffer = static_cast<uint32_t*>(forward_intermediate->parent->data_buffer);
    auto backward_input_buffer = static_cast<float*>(backward_input->parent->data_buffer);

    auto forward_output_buffer = static_cast<float*>(forward_output->parent->data_buffer);
    auto backward_output_buffer = static_cast<float*>(backward_output->parent->data_buffer);

    uint32_t output_fmaps = forward_output->parent->lengths.t[NN_DATA_COORD_z];
    uint32_t output_width = forward_output->parent->lengths.t[NN_DATA_COORD_x];
    uint32_t output_height = forward_output->parent->lengths.t[NN_DATA_COORD_y];

    if(backward_input->get_length(NN_DATA_COORD_z) % C_simd_width != 0)
        throw std::runtime_error("pool_backward: number of ifms is not multiply of 8");

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
                auto current_offset = in_begin.t[NN_DATA_COORD_z]
                                    + input_column*output_fmaps
                                    + input_row*output_fmaps*output_width
                                    + batch*output_fmaps*output_width*output_height;

                auto forward_intermediate_ptr = forward_intermediate_buffer + current_offset;
                auto backward_input_ptr = backward_input_buffer + current_offset;
                auto end_ptr = forward_intermediate_ptr + backward_input->get_length(NN_DATA_COORD_z);

                while (forward_intermediate_ptr < end_ptr)
                {
                    backward_output_buffer[*(forward_intermediate_ptr + 0)] += *(backward_input_ptr + 0);
                    backward_output_buffer[*(forward_intermediate_ptr + 1)] += *(backward_input_ptr + 1);
                    backward_output_buffer[*(forward_intermediate_ptr + 2)] += *(backward_input_ptr + 2);
                    backward_output_buffer[*(forward_intermediate_ptr + 3)] += *(backward_input_ptr + 3);
                    backward_output_buffer[*(forward_intermediate_ptr + 4)] += *(backward_input_ptr + 4);
                    backward_output_buffer[*(forward_intermediate_ptr + 5)] += *(backward_input_ptr + 5);
                    backward_output_buffer[*(forward_intermediate_ptr + 6)] += *(backward_input_ptr + 6);
                    backward_output_buffer[*(forward_intermediate_ptr + 7)] += *(backward_input_ptr + 7);

                    forward_intermediate_ptr += C_simd_width;
                    backward_input_ptr += C_simd_width;
                }
            }
        }
    }
}

void pooling_f32::backward(const std::vector<nn_workload_data_t *> &inputs,
                           const std::vector<const nn_workload_data_t *> &parameters,
                           const std::vector<const nn_workload_data_t *> &outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 2);

    const nn::workload_data<> backward_input(outputs[0]->parent->delta_buffer, outputs[0]->parent->lengths, outputs[0]->parent->layout);
    nn::workload_data<> backward_output(inputs[0]->parent->delta_buffer, inputs[0]->parent->lengths, inputs[0]->parent->layout);

    memset(backward_output.parent->data_buffer, 0, backward_output.parent->buffer_size);
    backward(reinterpret_cast<const nn::workload_data<> *>(inputs[0]),
             reinterpret_cast<const nn::workload_data<> *>(outputs[1]),
             reinterpret_cast<const nn::workload_data<> *>(outputs[0]),
             &backward_input,
             &backward_output);
}

void pooling_f32::forward(const nn::workload_data<> *input, nn::workload_data<> *intermediate_output, nn::workload_data<> *output)
{
    const auto batch = output->parent->lengths.t[NN_DATA_COORD_n];
    const bool dispatch_batch_only = (batch == 48 || pool_stride_y != pool_size_y);

    const auto num_output_row_items =
        (output->view_end.t[NN_DATA_COORD_y] - output->view_begin.t[NN_DATA_COORD_y] + 1);
    const auto num_batch_items =
        (output->view_end.t[NN_DATA_COORD_n] - output->view_begin.t[NN_DATA_COORD_n] + 1);

    const auto total_workers = (dispatch_batch_only) ? num_batch_items : num_output_row_items * num_batch_items;

    if (device->thread_pool.get_num_threads() < 2 || total_workers < 2)
    {
        // Its tiny data or there is only one thread available - just do it single threaded way.
        run_pooling(input, intermediate_output, output);
    }
    else
    {
        // Full cores utilization version.
        std::vector<nn::workload_data<>*> input_views(total_workers);
        std::vector<nn::workload_data<>*> output_views(total_workers);

        // Fill slave work items.
        // Outer loop will have single iteration in batch 48
        // because such high batch will give enough data for multithreading.
        for (auto output_row_item = 0u; output_row_item < num_output_row_items; ++output_row_item)
        {
            for (auto batch_item = 0u; batch_item < num_batch_items; ++batch_item)
            {
                auto item_in_pool = batch_item;
                if (!dispatch_batch_only) item_in_pool += output_row_item * num_batch_items;

                // Replace nn_workload_datas pointers with views.
                nn_workload_data_coords_t input_view_begin =
                {
                    batch_item,
                    0,
                    (dispatch_batch_only) ? 0 : static_cast<uint32_t>(output_row_item * pool_stride_y),
                    0,
                    0,
                    0
                };
                nn_workload_data_coords_t input_view_end = {
                    batch_item,
                    input->get_length(NN_DATA_COORD_x) - 1,
                    (dispatch_batch_only)
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
                    (dispatch_batch_only) ? 0 : output_row_item,
                    0,
                    0,
                    0
                };
                nn_workload_data_coords_t output_view_end =
                {
                    batch_item,
                    output->get_length(NN_DATA_COORD_x) - 1,
                    (dispatch_batch_only) ? output->get_length(NN_DATA_COORD_y) - 1 : output_row_item,
                    output->get_length(NN_DATA_COORD_z) - 1,
                    output->get_length(NN_DATA_COORD_p) - 1,
                    output->get_length(NN_DATA_COORD_q) - 1
                };

                input_views[item_in_pool] = new nn::workload_data<>(
                    *const_cast<nn::workload_data<> *>(input), input_view_begin, input_view_end);

                output_views[item_in_pool] =
                    new nn::workload_data<>(*output, output_view_begin, output_view_end);
            }

            if (dispatch_batch_only) break;
        }

        // Run threads.
        std::vector<pooling_f32_request_handle> request_handles(total_workers);
        std::vector<nn_multithreaded_request> job(total_workers);
        for (auto item_in_pool = 0u; item_in_pool < total_workers; ++item_in_pool)
        {
            request_handles[item_in_pool].primitive = this;
            request_handles[item_in_pool].input = input_views[item_in_pool];
            request_handles[item_in_pool].intermediate_output = intermediate_output;
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
    assert(outputs.size() == 1 || outputs.size() == 2);

    forward(reinterpret_cast<const nn::workload_data<> *>(inputs[0]),
            (outputs.size() == 2) ? nn::workload_data_cast<>(outputs[1]) : nullptr,
            nn::workload_data_cast<>(outputs[0]));
}

bool pooling_f32::validate_input(size_t index, nn_workload_data_t *data)
{
    switch (index) {
    case 0:
        return nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::validate<true>(
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
        primitive->forward(nn::workload_data_cast<>(work_item->input[0].get_data_view()),
                           (work_item->output.size() == 2) ? nn::workload_data_cast<>(work_item->output[0]) : nullptr,
                           nn::workload_data_cast<>(work_item->output[0]));
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
    case NN_POOLING_MODE_MAX:{
        // Initialize buffers - it should be called before multithreaded dispatch.
        memset(work_item->output[0]->parent->data_buffer, 0, work_item->output[0]->parent->buffer_size);

        primitive->backward(nn::workload_data_cast<>(work_item->forward_item->input[0].get_data_view()),
                            (work_item->input.size() == 2) ? reinterpret_cast<nn::workload_data<> *>(work_item->input[1].get_data_view()) : nullptr,
                            nn::workload_data_cast<>(work_item->forward_item->output[0]),
                            nn::workload_data_cast<>(work_item->input[0].get_data_view()),
                            nn::workload_data_cast<>(work_item->output[0]));
    }
        break;
    default:
        assert(0);
    }
}

std::vector<nn_workload_data_t *> pooling_f32::create_outputs(bool allocate_delta) {
    if (allocate_delta == true)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::create(device,
            static_cast<uint32_t>(output_size_x),
            static_cast<uint32_t>(output_size_y),
            static_cast<uint32_t>(output_size_z),
            static_cast<uint32_t>(batch_size),
            static_cast<uint32_t>(output_padding_left),
            static_cast<uint32_t>(output_padding_right),
            static_cast<uint32_t>(output_padding_top),
            static_cast<uint32_t>(output_padding_bottom),
            allocate_delta),
            nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::create(device,
            static_cast<uint32_t>(output_size_x),
            static_cast<uint32_t>(output_size_y),
            static_cast<uint32_t>(output_size_z),
            static_cast<uint32_t>(batch_size),
            static_cast<uint32_t>(output_padding_left),
            static_cast<uint32_t>(output_padding_right),
            static_cast<uint32_t>(output_padding_top),
            static_cast<uint32_t>(output_padding_bottom),
            allocate_delta) };
    }
    else
        return helper_zxyn_f32::primitive_zxyn_f32_base::create_outputs(allocate_delta);
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
                                   const int32_t center_offset_x,
                                   const int32_t center_offset_y,
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
                                  center_offset_x,
                                  center_offset_y,
                                  batch_size,
                                  hints_.output_padding.left,
                                  hints_.output_padding.right,
                                  hints_.output_padding.top,
                                  hints_.output_padding.bottom,
                                  reinterpret_cast<nn_device_internal *>(device));
}
