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
#include "layer_softmax_avx2.h"
#include "device/cpu/api_internal/data_helper.h"
#include "nn_intrinsic_power.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>

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

// SIMD width for this implementation
const auto C_simd_width = sizeof(__m256) / sizeof(float);

static const auto C_max_acc_batch1 = 12u;
static const auto C_max_acc_batch8 = 12u;
static const auto C_max_acc_batch48 = 12u;

static const auto C_batch8_size = C_simd_width;
static const auto C_batch48_size = 6 * C_simd_width;

static const auto C_data_stride_batch1 = C_simd_width * C_max_acc_batch1;

namespace layer {
///////////////////////////////////////////////////////////////////////////////////////////////////
// forward implementation

template<uint32_t T_SIZE, uint32_t T_batch_width>
void softmax_finalize_block(
    float* &output_ptr,
    __m256 &acc_sum)
{
    // We are not using table of registers and unroll pragmas
    // due to compiler which have issues with register allocation
    // and needs special, obvious treatment. Template immediate
    // arguments matching will remove all conditions in this code.
    __m256  acc0, acc1, acc2, acc3, acc4,
        acc5, acc6, acc7, acc8, acc9,
        acc10, acc11, acc12, acc13, acc14;

    // Load outputs and perform multiplication.
    if (T_SIZE >=  1)  acc0 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  0 * T_batch_width), acc_sum);
    if (T_SIZE >=  2)  acc1 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  1 * T_batch_width), acc_sum);
    if (T_SIZE >=  3)  acc2 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  2 * T_batch_width), acc_sum);
    if (T_SIZE >=  4)  acc3 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  3 * T_batch_width), acc_sum);
    if (T_SIZE >=  5)  acc4 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  4 * T_batch_width), acc_sum);
    if (T_SIZE >=  6)  acc5 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  5 * T_batch_width), acc_sum);
    if (T_SIZE >=  7)  acc6 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  6 * T_batch_width), acc_sum);
    if (T_SIZE >=  8)  acc7 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  7 * T_batch_width), acc_sum);
    if (T_SIZE >=  9)  acc8 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  8 * T_batch_width), acc_sum);
    if (T_SIZE >= 10)  acc9 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr +  9 * T_batch_width), acc_sum);
    if (T_SIZE >= 11) acc10 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 10 * T_batch_width), acc_sum);
    if (T_SIZE >= 12) acc11 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 11 * T_batch_width), acc_sum);
    if (T_SIZE >= 13) acc12 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 12 * T_batch_width), acc_sum);
    if (T_SIZE >= 14) acc13 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 13 * T_batch_width), acc_sum);
    if (T_SIZE >= 15) acc14 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 14 * T_batch_width), acc_sum);

    // Store results.
    if (T_SIZE >=  1) _mm256_storeu_ps(output_ptr +  0 * T_batch_width,  acc0);
    if (T_SIZE >=  2) _mm256_storeu_ps(output_ptr +  1 * T_batch_width,  acc1);
    if (T_SIZE >=  3) _mm256_storeu_ps(output_ptr +  2 * T_batch_width,  acc2);
    if (T_SIZE >=  4) _mm256_storeu_ps(output_ptr +  3 * T_batch_width,  acc3);
    if (T_SIZE >=  5) _mm256_storeu_ps(output_ptr +  4 * T_batch_width,  acc4);
    if (T_SIZE >=  6) _mm256_storeu_ps(output_ptr +  5 * T_batch_width,  acc5);
    if (T_SIZE >=  7) _mm256_storeu_ps(output_ptr +  6 * T_batch_width,  acc6);
    if (T_SIZE >=  8) _mm256_storeu_ps(output_ptr +  7 * T_batch_width,  acc7);
    if (T_SIZE >=  9) _mm256_storeu_ps(output_ptr +  8 * T_batch_width,  acc8);
    if (T_SIZE >= 10) _mm256_storeu_ps(output_ptr +  9 * T_batch_width,  acc9);
    if (T_SIZE >= 11) _mm256_storeu_ps(output_ptr + 10 * T_batch_width, acc10);
    if (T_SIZE >= 12) _mm256_storeu_ps(output_ptr + 11 * T_batch_width, acc11);
    if (T_SIZE >= 13) _mm256_storeu_ps(output_ptr + 12 * T_batch_width, acc12);
    if (T_SIZE >= 14) _mm256_storeu_ps(output_ptr + 13 * T_batch_width, acc13);
    if (T_SIZE >= 15) _mm256_storeu_ps(output_ptr + 14 * T_batch_width, acc14);

    output_ptr += T_batch_width*T_SIZE;
}

template<uint32_t T_SIZE, uint32_t T_batch_width>
void softmax_compute_block(
    float* &output_ptr,
    __m256 &acc_sum)
{
    // We are not using table of registers and unroll pragmas
    // due to compiler which have issues with register allocation
    // and needs special, obvious treatment. Template immediate
    // arguments matching will remove all conditions in this code.
    __m256  acc0, acc1, acc2, acc3, acc4,
            acc5, acc6, acc7, acc8, acc9,
            acc10, acc11, acc12, acc13, acc14;

    using namespace nn::intrinsics;
    // Load inputs and perform e^x
    if (T_SIZE >=  1)  acc0 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  0 * T_batch_width));
    if (T_SIZE >=  2)  acc1 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  1 * T_batch_width));
    if (T_SIZE >=  3)  acc2 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  2 * T_batch_width));
    if (T_SIZE >=  4)  acc3 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  3 * T_batch_width));
    if (T_SIZE >=  5)  acc4 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  4 * T_batch_width));
    if (T_SIZE >=  6)  acc5 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  5 * T_batch_width));
    if (T_SIZE >=  7)  acc6 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  6 * T_batch_width));
    if (T_SIZE >=  8)  acc7 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  7 * T_batch_width));
    if (T_SIZE >=  9)  acc8 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  8 * T_batch_width));
    if (T_SIZE >= 10)  acc9 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr +  9 * T_batch_width));
    if (T_SIZE >= 11) acc10 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr + 10 * T_batch_width));
    if (T_SIZE >= 12) acc11 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr + 11 * T_batch_width));
    if (T_SIZE >= 13) acc12 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr + 12 * T_batch_width));
    if (T_SIZE >= 14) acc13 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr + 13 * T_batch_width));
    if (T_SIZE >= 15) acc14 = _inner_mm256_exp_ps(_mm256_loadu_ps(output_ptr + 14 * T_batch_width));

    // Store results.
    if (T_SIZE >=  1) _mm256_storeu_ps(output_ptr +  0 * T_batch_width,  acc0);
    if (T_SIZE >=  2) _mm256_storeu_ps(output_ptr +  1 * T_batch_width,  acc1);
    if (T_SIZE >=  3) _mm256_storeu_ps(output_ptr +  2 * T_batch_width,  acc2);
    if (T_SIZE >=  4) _mm256_storeu_ps(output_ptr +  3 * T_batch_width,  acc3);
    if (T_SIZE >=  5) _mm256_storeu_ps(output_ptr +  4 * T_batch_width,  acc4);
    if (T_SIZE >=  6) _mm256_storeu_ps(output_ptr +  5 * T_batch_width,  acc5);
    if (T_SIZE >=  7) _mm256_storeu_ps(output_ptr +  6 * T_batch_width,  acc6);
    if (T_SIZE >=  8) _mm256_storeu_ps(output_ptr +  7 * T_batch_width,  acc7);
    if (T_SIZE >=  9) _mm256_storeu_ps(output_ptr +  8 * T_batch_width,  acc8);
    if (T_SIZE >= 10) _mm256_storeu_ps(output_ptr +  9 * T_batch_width,  acc9);
    if (T_SIZE >= 11) _mm256_storeu_ps(output_ptr + 10 * T_batch_width, acc10);
    if (T_SIZE >= 12) _mm256_storeu_ps(output_ptr + 11 * T_batch_width, acc11);
    if (T_SIZE >= 13) _mm256_storeu_ps(output_ptr + 12 * T_batch_width, acc12);
    if (T_SIZE >= 14) _mm256_storeu_ps(output_ptr + 13 * T_batch_width, acc13);
    if (T_SIZE >= 15) _mm256_storeu_ps(output_ptr + 14 * T_batch_width, acc14);

    // Sum up accumulators.
    if (T_SIZE >=  1) acc_sum = _mm256_add_ps(acc0,  acc_sum);
    if (T_SIZE >=  2) acc_sum = _mm256_add_ps(acc1,  acc_sum);
    if (T_SIZE >=  3) acc_sum = _mm256_add_ps(acc2,  acc_sum);
    if (T_SIZE >=  4) acc_sum = _mm256_add_ps(acc3,  acc_sum);
    if (T_SIZE >=  5) acc_sum = _mm256_add_ps(acc4,  acc_sum);
    if (T_SIZE >=  6) acc_sum = _mm256_add_ps(acc5,  acc_sum);
    if (T_SIZE >=  7) acc_sum = _mm256_add_ps(acc6,  acc_sum);
    if (T_SIZE >=  8) acc_sum = _mm256_add_ps(acc7,  acc_sum);
    if (T_SIZE >=  9) acc_sum = _mm256_add_ps(acc8,  acc_sum);
    if (T_SIZE >= 10) acc_sum = _mm256_add_ps(acc9,  acc_sum);
    if (T_SIZE >= 11) acc_sum = _mm256_add_ps(acc10, acc_sum);
    if (T_SIZE >= 12) acc_sum = _mm256_add_ps(acc11, acc_sum);
    if (T_SIZE >= 13) acc_sum = _mm256_add_ps(acc12, acc_sum);
    if (T_SIZE >= 14) acc_sum = _mm256_add_ps(acc13, acc_sum);
    if (T_SIZE >= 15) acc_sum = _mm256_add_ps(acc14, acc_sum);

    output_ptr += T_batch_width*T_SIZE;
}

void softmax_f32::run_softmax_work_item_batch8(const nn::workload_data<nn::layout_f32> *input_view,
    nn::workload_data<nn::layout_f32> *output_view) {
    const auto input_width = input_view->view_end.t[NN_DATA_COORD_x] - input_view->view_begin.t[NN_DATA_COORD_x] + 1;
    const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto num_full_blocks = output_width / C_max_acc_batch8;
    const auto partial_block_size = output_width % C_max_acc_batch8;

    const auto output_view_start = output_view->view_begin.t[NN_DATA_COORD_x] * C_batch8_size;
    const auto input_view_start = input_view->view_begin.t[NN_DATA_COORD_x] * C_batch8_size;

    // Find max value for each image. We will use 1 accumulator (1*8=8) so each image has its own avx component.
    {
        auto input_buffer = &static_cast<float*>(input_view->parent->data_buffer)[input_view_start];
        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

        __m256 max_values;
        uint32_t input = 0;
        max_values = _mm256_load_ps(input_buffer + (input++) * C_batch8_size);

        while(input < input_width)
            max_values = _mm256_max_ps(max_values, _mm256_load_ps(input_buffer + (input++) * C_batch8_size));

        for(uint32_t output = 0; output < output_width; ++output)
        {
                _mm256_store_ps(
                    output_buffer + output * C_batch8_size,
                    _mm256_sub_ps(
                        _mm256_load_ps(input_buffer + output * C_batch8_size),
                        max_values));
        }
    }

    __m256 acc_sum = _mm256_setzero_ps();

    {
        auto input_buffer = &static_cast<float*>(input_view->parent->data_buffer)[input_view_start];
        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            softmax_compute_block<C_max_acc_batch8, C_batch8_size>(output_buffer, acc_sum);
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: softmax_compute_block< 1, C_batch8_size>(output_buffer, acc_sum); break;
        case  2: softmax_compute_block< 2, C_batch8_size>(output_buffer, acc_sum); break;
        case  3: softmax_compute_block< 3, C_batch8_size>(output_buffer, acc_sum); break;
        case  4: softmax_compute_block< 4, C_batch8_size>(output_buffer, acc_sum); break;
        case  5: softmax_compute_block< 5, C_batch8_size>(output_buffer, acc_sum); break;
        case  6: softmax_compute_block< 6, C_batch8_size>(output_buffer, acc_sum); break;
        case  7: softmax_compute_block< 7, C_batch8_size>(output_buffer, acc_sum); break;
        case  8: softmax_compute_block< 8, C_batch8_size>(output_buffer, acc_sum); break;
        case  9: softmax_compute_block< 9, C_batch8_size>(output_buffer, acc_sum); break;
        case 10: softmax_compute_block<10, C_batch8_size>(output_buffer, acc_sum); break;
        case 11: softmax_compute_block<11, C_batch8_size>(output_buffer, acc_sum); break;
        case 12: softmax_compute_block<12, C_batch8_size>(output_buffer, acc_sum); break;
        case 13: softmax_compute_block<13, C_batch8_size>(output_buffer, acc_sum); break;
        case 14: softmax_compute_block<14, C_batch8_size>(output_buffer, acc_sum); break;
        default: NN_UNREACHABLE_CODE;
        }
    }

    acc_sum = _mm256_div_ps(_mm256_set1_ps(1.0f), acc_sum);

    {
        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            softmax_finalize_block<C_max_acc_batch8, C_batch8_size>(output_buffer, acc_sum);
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: softmax_finalize_block< 1, C_batch8_size>(output_buffer, acc_sum); break;
        case  2: softmax_finalize_block< 2, C_batch8_size>(output_buffer, acc_sum); break;
        case  3: softmax_finalize_block< 3, C_batch8_size>(output_buffer, acc_sum); break;
        case  4: softmax_finalize_block< 4, C_batch8_size>(output_buffer, acc_sum); break;
        case  5: softmax_finalize_block< 5, C_batch8_size>(output_buffer, acc_sum); break;
        case  6: softmax_finalize_block< 6, C_batch8_size>(output_buffer, acc_sum); break;
        case  7: softmax_finalize_block< 7, C_batch8_size>(output_buffer, acc_sum); break;
        case  8: softmax_finalize_block< 8, C_batch8_size>(output_buffer, acc_sum); break;
        case  9: softmax_finalize_block< 9, C_batch8_size>(output_buffer, acc_sum); break;
        case 10: softmax_finalize_block<10, C_batch8_size>(output_buffer, acc_sum); break;
        case 11: softmax_finalize_block<11, C_batch8_size>(output_buffer, acc_sum); break;
        case 12: softmax_finalize_block<12, C_batch8_size>(output_buffer, acc_sum); break;
        case 13: softmax_finalize_block<13, C_batch8_size>(output_buffer, acc_sum); break;
        case 14: softmax_finalize_block<14, C_batch8_size>(output_buffer, acc_sum); break;
        default: NN_UNREACHABLE_CODE;
        }
    }
}

template<uint32_t T_NUM_ITERATIONS>
void softmax_compute_subsimd(
    float* &output_ptr,
    float &acc_sum)
{
    for (auto iteration = 0u; iteration < T_NUM_ITERATIONS; ++iteration)
    {
        float acc0 = std::exp(*output_ptr);
        *output_ptr = acc0;
        acc_sum += acc0;

        ++output_ptr;
    }
}

template<uint32_t T_NUM_ITERATIONS>
void softmax_finalize_subsimd(
    float* &output_ptr,
    float &acc_sum)
{
    for (auto iteration = 0u; iteration < T_NUM_ITERATIONS; ++iteration)
    {
        float acc0 = *output_ptr;
        acc0 *= acc_sum;
        *output_ptr = acc0;

        ++output_ptr;
    }
}

void softmax_f32::run_softmax_work_item_latency(const nn::workload_data<nn::layout_f32> *input_view,
                                                nn::workload_data<nn::layout_f32> *output_view) {
    const auto input_width = input_view->view_end.t[NN_DATA_COORD_x] - input_view->view_begin.t[NN_DATA_COORD_x] + 1;
    const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto num_full_blocks = output_width / C_data_stride_batch1;
    const auto partial_block_size = (output_width / C_simd_width) % C_max_acc_batch1;
    const auto subsimd_block_size = output_width % C_simd_width;

    const auto output_view_start = output_view->view_begin.t[NN_DATA_COORD_x];
    const auto input_view_start = input_view->view_begin.t[NN_DATA_COORD_x];

    // Find max value for each image. We will use 1 scalar accumulator (semi-naive way).
    {
        auto input_buffer = &static_cast<float*>(input_view->parent->data_buffer)[input_view_start];
        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

        float max_value;
        uint32_t input = 0;
        max_value = *(input_buffer + (input++));

        while(input < input_width)
            max_value = std::max(max_value, *(input_buffer + (input++)));

        for(uint32_t output = 0; output < output_width; ++output)
        {
            *(output_buffer + output) = *(input_buffer + output) - max_value;
        }
    }

    __m256 acc_sum = _mm256_setzero_ps();
    float subsimd_sum = 0.0f;

    {
        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            softmax_compute_block<C_max_acc_batch1, C_simd_width>(output_buffer, acc_sum);
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: softmax_compute_block< 1, C_simd_width>(output_buffer, acc_sum); break;
        case  2: softmax_compute_block< 2, C_simd_width>(output_buffer, acc_sum); break;
        case  3: softmax_compute_block< 3, C_simd_width>(output_buffer, acc_sum); break;
        case  4: softmax_compute_block< 4, C_simd_width>(output_buffer, acc_sum); break;
        case  5: softmax_compute_block< 5, C_simd_width>(output_buffer, acc_sum); break;
        case  6: softmax_compute_block< 6, C_simd_width>(output_buffer, acc_sum); break;
        case  7: softmax_compute_block< 7, C_simd_width>(output_buffer, acc_sum); break;
        case  8: softmax_compute_block< 8, C_simd_width>(output_buffer, acc_sum); break;
        case  9: softmax_compute_block< 9, C_simd_width>(output_buffer, acc_sum); break;
        case 10: softmax_compute_block<10, C_simd_width>(output_buffer, acc_sum); break;
        case 11: softmax_compute_block<11, C_simd_width>(output_buffer, acc_sum); break;
        case 12: softmax_compute_block<12, C_simd_width>(output_buffer, acc_sum); break;
        case 13: softmax_compute_block<13, C_simd_width>(output_buffer, acc_sum); break;
        case 14: softmax_compute_block<14, C_simd_width>(output_buffer, acc_sum); break;
        default: NN_UNREACHABLE_CODE;
        }

        switch (subsimd_block_size)
        {
        case 0: break;
        case 1: softmax_compute_subsimd<1>(output_buffer, subsimd_sum); break;
        case 2: softmax_compute_subsimd<2>(output_buffer, subsimd_sum); break;
        case 3: softmax_compute_subsimd<3>(output_buffer, subsimd_sum); break;
        case 4: softmax_compute_subsimd<4>(output_buffer, subsimd_sum); break;
        case 5: softmax_compute_subsimd<5>(output_buffer, subsimd_sum); break;
        case 6: softmax_compute_subsimd<6>(output_buffer, subsimd_sum); break;
        case 7: softmax_compute_subsimd<7>(output_buffer, subsimd_sum); break;
        default: NN_UNREACHABLE_CODE;
        }
    }

    {
        __m256 intermediate_sum = _mm256_hadd_ps(acc_sum, acc_sum);
        intermediate_sum = _mm256_permutevar8x32_ps(intermediate_sum, _mm256_set_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        intermediate_sum = _mm256_hadd_ps(intermediate_sum, intermediate_sum);
        intermediate_sum = _mm256_hadd_ps(intermediate_sum, intermediate_sum);

        acc_sum = _mm256_add_ps(intermediate_sum, _mm256_set1_ps(subsimd_sum));
        subsimd_sum = _mm_cvtss_f32(_mm256_extractf128_ps(acc_sum, 0));

        acc_sum = _mm256_div_ps(_mm256_set1_ps(1.0f), acc_sum);
        subsimd_sum = 1.0f / subsimd_sum;
    }

    {
        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            softmax_finalize_block<C_max_acc_batch1, C_simd_width>(output_buffer, acc_sum);
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: softmax_finalize_block< 1, C_simd_width>(output_buffer, acc_sum); break;
        case  2: softmax_finalize_block< 2, C_simd_width>(output_buffer, acc_sum); break;
        case  3: softmax_finalize_block< 3, C_simd_width>(output_buffer, acc_sum); break;
        case  4: softmax_finalize_block< 4, C_simd_width>(output_buffer, acc_sum); break;
        case  5: softmax_finalize_block< 5, C_simd_width>(output_buffer, acc_sum); break;
        case  6: softmax_finalize_block< 6, C_simd_width>(output_buffer, acc_sum); break;
        case  7: softmax_finalize_block< 7, C_simd_width>(output_buffer, acc_sum); break;
        case  8: softmax_finalize_block< 8, C_simd_width>(output_buffer, acc_sum); break;
        case  9: softmax_finalize_block< 9, C_simd_width>(output_buffer, acc_sum); break;
        case 10: softmax_finalize_block<10, C_simd_width>(output_buffer, acc_sum); break;
        case 11: softmax_finalize_block<11, C_simd_width>(output_buffer, acc_sum); break;
        case 12: softmax_finalize_block<12, C_simd_width>(output_buffer, acc_sum); break;
        case 13: softmax_finalize_block<13, C_simd_width>(output_buffer, acc_sum); break;
        case 14: softmax_finalize_block<14, C_simd_width>(output_buffer, acc_sum); break;
        default: NN_UNREACHABLE_CODE;
        }

        switch (subsimd_block_size)
        {
        case 0: break;
        case 1: softmax_finalize_subsimd<1>(output_buffer, subsimd_sum); break;
        case 2: softmax_finalize_subsimd<2>(output_buffer, subsimd_sum); break;
        case 3: softmax_finalize_subsimd<3>(output_buffer, subsimd_sum); break;
        case 4: softmax_finalize_subsimd<4>(output_buffer, subsimd_sum); break;
        case 5: softmax_finalize_subsimd<5>(output_buffer, subsimd_sum); break;
        case 6: softmax_finalize_subsimd<6>(output_buffer, subsimd_sum); break;
        case 7: softmax_finalize_subsimd<7>(output_buffer, subsimd_sum); break;
        default: NN_UNREACHABLE_CODE;
        }
    }
}

void softmax_f32::run_softmax_work_item_batch48(const nn::workload_data<nn::layout_f32> *input_view,
                                                nn::workload_data<nn::layout_f32> *output_view) {
    const auto input_width = input_view->view_end.t[NN_DATA_COORD_x] - input_view->view_begin.t[NN_DATA_COORD_x] + 1;
    const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto num_full_blocks = output_width / C_max_acc_batch48;
    const auto partial_block_size = output_width % C_max_acc_batch48;
    const auto num_batch_packages = 6;

    // Find max value for each image. We will use 6 accumulators (6*8=48) so each image has its own avx component.
    {
        const auto output_view_start = output_view->view_begin.t[NN_DATA_COORD_x] * C_batch48_size;
        const auto input_view_start = input_view->view_begin.t[NN_DATA_COORD_x] * C_batch48_size;
        auto input_buffer = &static_cast<float*>(input_view->parent->data_buffer)[input_view_start];
        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

        __m256 max_values[num_batch_packages];
        uint32_t input = 0;

        #pragma unroll(num_batch_packages)
        for(auto acc = 0u; acc < num_batch_packages; ++acc)
            max_values[acc] = _mm256_load_ps(input_buffer + input * C_batch48_size + acc * C_simd_width);

        ++input;

        for(;input < input_width; ++input)
            #pragma unroll(num_batch_packages)
            for(auto acc = 0u; acc < num_batch_packages; ++acc)
                max_values[acc] = _mm256_max_ps(max_values[acc], _mm256_load_ps(input_buffer + input * C_batch48_size + acc * C_simd_width));

        for(uint32_t output = 0; output < output_width; ++output)
        {
            #pragma unroll(num_batch_packages)
            for(auto acc = 0u; acc < num_batch_packages; ++acc)
                _mm256_store_ps(
                    output_buffer + output * C_batch48_size + acc * C_simd_width,
                    _mm256_sub_ps(
                        _mm256_load_ps(input_buffer + output * C_batch48_size + acc * C_simd_width),
                        max_values[acc]));
        }
    }

    for (uint32_t batch_package = 0; batch_package < num_batch_packages; ++batch_package)
    {
        const auto output_view_start = output_view->view_begin.t[NN_DATA_COORD_x] * C_batch48_size + batch_package * C_batch8_size;

        __m256 acc_sum = _mm256_setzero_ps();

        {
            auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_compute_block<C_max_acc_batch48, C_batch48_size>(output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
            case  0: break;
            case  1: softmax_compute_block< 1, C_batch48_size>(output_buffer, acc_sum); break;
            case  2: softmax_compute_block< 2, C_batch48_size>(output_buffer, acc_sum); break;
            case  3: softmax_compute_block< 3, C_batch48_size>(output_buffer, acc_sum); break;
            case  4: softmax_compute_block< 4, C_batch48_size>(output_buffer, acc_sum); break;
            case  5: softmax_compute_block< 5, C_batch48_size>(output_buffer, acc_sum); break;
            case  6: softmax_compute_block< 6, C_batch48_size>(output_buffer, acc_sum); break;
            case  7: softmax_compute_block< 7, C_batch48_size>(output_buffer, acc_sum); break;
            case  8: softmax_compute_block< 8, C_batch48_size>(output_buffer, acc_sum); break;
            case  9: softmax_compute_block< 9, C_batch48_size>(output_buffer, acc_sum); break;
            case 10: softmax_compute_block<10, C_batch48_size>(output_buffer, acc_sum); break;
            case 11: softmax_compute_block<11, C_batch48_size>(output_buffer, acc_sum); break;
            case 12: softmax_compute_block<12, C_batch48_size>(output_buffer, acc_sum); break;
            case 13: softmax_compute_block<13, C_batch48_size>(output_buffer, acc_sum); break;
            case 14: softmax_compute_block<14, C_batch48_size>(output_buffer, acc_sum); break;
            default: NN_UNREACHABLE_CODE;
            }
        }

        acc_sum = _mm256_div_ps(_mm256_set1_ps(1.0f), acc_sum);

        {
            auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_finalize_block<C_max_acc_batch48, C_batch48_size>(output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
            case  0: break;
            case  1: softmax_finalize_block< 1, C_batch48_size>(output_buffer, acc_sum); break;
            case  2: softmax_finalize_block< 2, C_batch48_size>(output_buffer, acc_sum); break;
            case  3: softmax_finalize_block< 3, C_batch48_size>(output_buffer, acc_sum); break;
            case  4: softmax_finalize_block< 4, C_batch48_size>(output_buffer, acc_sum); break;
            case  5: softmax_finalize_block< 5, C_batch48_size>(output_buffer, acc_sum); break;
            case  6: softmax_finalize_block< 6, C_batch48_size>(output_buffer, acc_sum); break;
            case  7: softmax_finalize_block< 7, C_batch48_size>(output_buffer, acc_sum); break;
            case  8: softmax_finalize_block< 8, C_batch48_size>(output_buffer, acc_sum); break;
            case  9: softmax_finalize_block< 9, C_batch48_size>(output_buffer, acc_sum); break;
            case 10: softmax_finalize_block<10, C_batch48_size>(output_buffer, acc_sum); break;
            case 11: softmax_finalize_block<11, C_batch48_size>(output_buffer, acc_sum); break;
            case 12: softmax_finalize_block<12, C_batch48_size>(output_buffer, acc_sum); break;
            case 13: softmax_finalize_block<13, C_batch48_size>(output_buffer, acc_sum); break;
            case 14: softmax_finalize_block<14, C_batch48_size>(output_buffer, acc_sum); break;
            default: NN_UNREACHABLE_CODE;
            }
        }
    }
}

void softmax_f32::backward(const std::vector<nn_workload_data_t *> &inputs,
                           const std::vector<const nn_workload_data_t *> &parameters,
                           const std::vector<const nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    const nn::workload_data<nn::layout_f32> backward_input(outputs[0]->parent->delta_buffer, outputs[0]->parent->lengths, outputs[0]->parent->layout);
    nn::workload_data<nn::layout_f32> backward_output(inputs[0]->parent->delta_buffer, inputs[0]->parent->lengths, inputs[0]->parent->layout);

    backward(reinterpret_cast<const nn::workload_data<nn::layout_f32> *>(outputs[0]),
             &backward_input,
             &backward_output);
}

void softmax_f32::forward(const nn::workload_data<nn::layout_f32> *input, nn::workload_data<nn::layout_f32> *output) {
    auto batch_size = input->parent->lengths.t[NN_DATA_COORD_n];
    switch (batch_size)
    {
    case 1:
        run_softmax_work_item_latency(input, output);
        break;
    case 8:
        run_softmax_work_item_batch8(input, output);
        break;
    case 48:
        run_softmax_work_item_batch48(input, output);
        break;
    default:
        break;
    }
}

void softmax_f32::forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 1);
    assert(parameters.size() == 0);
    assert(outputs.size() == 1);

    forward(reinterpret_cast<const nn::workload_data<nn::layout_f32> *>(inputs[0]),
        reinterpret_cast<nn::workload_data<nn::layout_f32> *>(outputs[0]));
}

void softmax_f32::backward(
    const nn::workload_data<nn::layout_f32> *forward_output,
    const nn::workload_data<nn::layout_f32> *backward_input,
    nn::workload_data<nn::layout_f32> *backward_output)
{
    if(batch_size != 48)
    {
        // Naive.
        for (uint32_t n = 0; n < forward_output->get_length(NN_DATA_COORD_n); ++n)
        {
            for (uint32_t in_x = 0; in_x < forward_output->get_length(NN_DATA_COORD_x); ++in_x)
            {
                float accumulator = 0.0f;
                for (uint32_t out_x = 0; out_x < forward_output->get_length(NN_DATA_COORD_x); ++out_x)
                {
                    float derivative = -forward_output->at(n, in_x, 0, 0, 0, 0) * forward_output->at(n, out_x, 0, 0, 0, 0);

                    if (in_x == out_x)
                        derivative += forward_output->at(n, in_x, 0, 0, 0, 0);

                    accumulator += backward_input->at(n, out_x, 0, 0, 0, 0) * derivative;
                }
                (*backward_output)(n, in_x, 0, 0, 0, 0) = accumulator;
            }
        }
    }
    else
    {
        // Naive+ for batch48.
        const auto C_accs = 6 * C_simd_width;

        const auto backward_output_buffer = reinterpret_cast<float*>(backward_output->parent->data_buffer);
        const auto backward_input_buffer = reinterpret_cast<float*>(backward_input->parent->data_buffer);
        const auto forward_output_buffer = reinterpret_cast<float*>(forward_output->parent->data_buffer);

        for (uint32_t in_x = 0; in_x < forward_output->get_length(NN_DATA_COORD_x); ++in_x)
        {
            float accumulator[C_accs] = {0};

            for (uint32_t out_x = 0; out_x < forward_output->get_length(NN_DATA_COORD_x); ++out_x)
            {
                float derivative[C_accs];

                #pragma unroll (C_accs)
                for(uint32_t acc_id = 0; acc_id < C_accs; ++acc_id)
                    derivative[acc_id] =
                                        -*(forward_output_buffer + acc_id + in_x * C_accs) *
                                        *(forward_output_buffer + acc_id + out_x * C_accs);

                if (in_x == out_x)
                {
                    #pragma unroll (C_accs)
                    for(uint32_t acc_id = 0; acc_id < C_accs; ++acc_id)
                        derivative[acc_id] += *(forward_output_buffer + acc_id + in_x * C_accs);
                }

                #pragma unroll (C_accs)
                for(uint32_t acc_id = 0; acc_id < C_accs; ++acc_id)
                    accumulator[acc_id] += *(backward_input_buffer + acc_id + out_x * C_accs) * derivative[acc_id];
            }

            #pragma unroll (C_accs)
            for(uint32_t acc_id = 0; acc_id < C_accs; ++acc_id)
                *(backward_output_buffer + acc_id + in_x * C_accs) = accumulator[acc_id];
        }
    }
}

void wrapper_softmax_work_item_backward(nn_workload_item *const work_item)
{
    auto primitive = static_cast<softmax_f32*>(work_item->forward_item->primitive);
    primitive->backward(reinterpret_cast<nn::workload_data<nn::layout_f32> *>(work_item->forward_item->output[0]),
                        reinterpret_cast<nn::workload_data<nn::layout_f32> *>(work_item->input[0].get_data_view()),
                        reinterpret_cast<nn::workload_data<nn::layout_f32> *>(work_item->output[0]));
}

softmax_f32::softmax_f32(size_t num_features, size_t batch_size, nn_device_internal *device)
    : num_features(num_features),
      batch_size(batch_size),
      device(device),
      in_out_layout(nn::layout_t<nn::layout_nxyzpq_f32>::layout) {}

std::vector<nn_workload_data_t *> softmax_f32::create_inputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::create(device, num_features, batch_size, allocate_delta)};
}

std::vector<nn_workload_data_t *> softmax_f32::create_outputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::create(device, num_features, batch_size, allocate_delta)};
}

bool softmax_f32::validate_input(size_t index, nn_workload_data_t *data)
{
    switch(index){
    case 0:
        return nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::validate(data, num_features, batch_size);
    }

    throw std::invalid_argument("index out of range");
}

} // namespace layer

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_softmax_f32_create_0(
    nn_device_t *device,  /* IDLF device handle */
    size_t num_features,  /* number of input feature maps */
    size_t batch_size,    /* size of input batch */
    const nn_primitives_softmax_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);

    std::remove_const<std::remove_pointer<decltype(hints)>::type>::type hints_ = {};
    if (hints != nullptr)
        hints_ = *hints;

    return new layer::softmax_f32(num_features, batch_size, reinterpret_cast<nn_device_internal *>(device));
}
