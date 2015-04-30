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
#include "layer_softmax_avx2.h"

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

__m256 _inner_mm256_exp_ps(__m256 arg)
{
    arg = _mm256_mul_ps(arg, _mm256_set1_ps(1.4426950408889634073599246810018921374266459541529859f));

    __m256i e = _mm256_add_epi32(
        _mm256_castps_si256(_mm256_cmp_ps(arg, _mm256_set1_ps(0.0f), _CMP_LT_OQ)),
        _mm256_cvttps_epi32(arg));

    arg = _mm256_sub_ps(arg, _mm256_cvtepi32_ps(e));

    __m256 intermediate_result;
    intermediate_result = _mm256_fmadd_ps(_mm256_set1_ps(0.0136779459179717f), arg, _mm256_set1_ps(0.0517692205767896f));
    intermediate_result = _mm256_fmadd_ps(intermediate_result, arg, _mm256_set1_ps(0.241554388295527f));
    intermediate_result = _mm256_fmadd_ps(intermediate_result, arg, _mm256_set1_ps(0.692998430056128f));
    intermediate_result = _mm256_fmadd_ps(intermediate_result, arg, _mm256_set1_ps(0.999999804292074f));
    arg = intermediate_result;

    __m256 res = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(127)), 23));

    res = _mm256_mul_ps(res, arg);

    return res;
}

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
        acc10, acc11, acc12, acc13, acc14, acc15;

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
    float* &input_ptr,
    float* &output_ptr,
    __m256 &acc_sum)
{
    // We are not using table of registers and unroll pragmas
    // due to compiler which have issues with register allocation
    // and needs special, obvious treatment. Template immediate
    // arguments matching will remove all conditions in this code.
    __m256  acc0, acc1, acc2, acc3, acc4, 
            acc5, acc6, acc7, acc8, acc9, 
            acc10, acc11, acc12, acc13, acc14, acc15;

    // Load inputs and perform e^x
    if (T_SIZE >=  1)  acc0 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr +  0 * T_batch_width));
    if (T_SIZE >=  2)  acc1 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr +  1 * T_batch_width));
    if (T_SIZE >=  3)  acc2 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr +  2 * T_batch_width));
    if (T_SIZE >=  4)  acc3 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr +  3 * T_batch_width));
    if (T_SIZE >=  5)  acc4 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr +  4 * T_batch_width));
    if (T_SIZE >=  6)  acc5 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr +  5 * T_batch_width));
    if (T_SIZE >=  7)  acc6 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr +  6 * T_batch_width));
    if (T_SIZE >=  8)  acc7 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr +  7 * T_batch_width));
    if (T_SIZE >=  9)  acc8 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr +  8 * T_batch_width));
    if (T_SIZE >= 10)  acc9 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr +  9 * T_batch_width));
    if (T_SIZE >= 11) acc10 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr + 10 * T_batch_width));
    if (T_SIZE >= 12) acc11 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr + 11 * T_batch_width));
    if (T_SIZE >= 13) acc12 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr + 12 * T_batch_width));
    if (T_SIZE >= 14) acc13 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr + 13 * T_batch_width));
    if (T_SIZE >= 15) acc14 = _inner_mm256_exp_ps(_mm256_loadu_ps(input_ptr + 14 * T_batch_width));

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

    input_ptr += T_batch_width*T_SIZE;
    output_ptr += T_batch_width*T_SIZE;
}

void softmax_f32::run_softmax_work_item_batch8(const nn::nn_workload_data_t<float> *input_view,
                                               nn::nn_workload_data_t<float> *output_view) {
    const auto input_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto num_full_blocks = output_width / C_max_acc_batch8;
    const auto partial_block_size = output_width % C_max_acc_batch8;

    const auto output_view_start = output_view->view_begin.t[NN_DATA_COORD_x] * C_batch8_size;

    const auto input_view_start = input_view->view_begin.t[NN_DATA_COORD_x] * C_batch8_size;

    __m256 acc_sum = _mm256_setzero_ps();

    {
        auto input_buffer = &static_cast<float*>(input_view->parent->data_buffer)[input_view_start];
        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            softmax_compute_block<C_max_acc_batch8, C_batch8_size>(input_buffer, output_buffer, acc_sum);
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: softmax_compute_block< 1, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case  2: softmax_compute_block< 2, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case  3: softmax_compute_block< 3, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case  4: softmax_compute_block< 4, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case  5: softmax_compute_block< 5, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case  6: softmax_compute_block< 6, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case  7: softmax_compute_block< 7, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case  8: softmax_compute_block< 8, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case  9: softmax_compute_block< 9, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case 10: softmax_compute_block<10, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case 11: softmax_compute_block<11, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case 12: softmax_compute_block<12, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case 13: softmax_compute_block<13, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
        case 14: softmax_compute_block<14, C_batch8_size>(input_buffer, output_buffer, acc_sum); break;
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
    float* &input_ptr,
    float* &output_ptr,
    float &acc_sum)
{
    for (auto iteration = 0u; iteration < T_NUM_ITERATIONS; ++iteration)
    {
        float acc0 = std::exp(*input_ptr);
        *output_ptr = acc0;
        acc_sum += acc0;

        ++input_ptr;
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

void softmax_f32::run_softmax_work_item_latency(const nn::nn_workload_data_t<float> *input_view,
                                                nn::nn_workload_data_t<float> *output_view) {
    const auto input_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto num_full_blocks = output_width / C_data_stride_batch1;
    const auto partial_block_size = (output_width / C_simd_width) % C_max_acc_batch1;
    const auto subsimd_block_size = output_width % C_simd_width;

    const auto output_view_start = output_view->view_begin.t[NN_DATA_COORD_x];

    const auto input_view_start = input_view->view_begin.t[NN_DATA_COORD_x];

    __m256 acc_sum = _mm256_setzero_ps();
    float subsimd_sum = 0.0f;

    {
        auto input_buffer = &static_cast<float*>(input_view->parent->data_buffer)[input_view_start];
        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            softmax_compute_block<C_max_acc_batch1, C_simd_width>(input_buffer, output_buffer, acc_sum);
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: softmax_compute_block< 1, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case  2: softmax_compute_block< 2, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case  3: softmax_compute_block< 3, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case  4: softmax_compute_block< 4, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case  5: softmax_compute_block< 5, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case  6: softmax_compute_block< 6, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case  7: softmax_compute_block< 7, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case  8: softmax_compute_block< 8, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case  9: softmax_compute_block< 9, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case 10: softmax_compute_block<10, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case 11: softmax_compute_block<11, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case 12: softmax_compute_block<12, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case 13: softmax_compute_block<13, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        case 14: softmax_compute_block<14, C_simd_width>(input_buffer, output_buffer, acc_sum); break;
        default: NN_UNREACHABLE_CODE;
        }

        switch (subsimd_block_size)
        {
        case 0: break;
        case 1: softmax_compute_subsimd<1>(input_buffer, output_buffer, subsimd_sum); break;
        case 2: softmax_compute_subsimd<2>(input_buffer, output_buffer, subsimd_sum); break;
        case 3: softmax_compute_subsimd<3>(input_buffer, output_buffer, subsimd_sum); break;
        case 4: softmax_compute_subsimd<4>(input_buffer, output_buffer, subsimd_sum); break;
        case 5: softmax_compute_subsimd<5>(input_buffer, output_buffer, subsimd_sum); break;
        case 6: softmax_compute_subsimd<6>(input_buffer, output_buffer, subsimd_sum); break;
        case 7: softmax_compute_subsimd<7>(input_buffer, output_buffer, subsimd_sum); break;
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

void softmax_f32::run_softmax_work_item_batch48(const nn::nn_workload_data_t<float> *input_view,
                                                nn::nn_workload_data_t<float> *output_view) {
    const auto input_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto num_full_blocks = output_width / C_max_acc_batch48;
    const auto partial_block_size = output_width % C_max_acc_batch48;

    for (uint32_t batch_package = 0; batch_package < 6; ++batch_package)
    {
        const auto output_view_start = output_view->view_begin.t[NN_DATA_COORD_x] * C_batch48_size + batch_package * C_batch8_size;
        const auto input_view_start = input_view->view_begin.t[NN_DATA_COORD_x] * C_batch48_size + batch_package * C_batch8_size;

        __m256 acc_sum = _mm256_setzero_ps();

        {
            auto input_buffer = &static_cast<float*>(input_view->parent->data_buffer)[input_view_start];
            auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_compute_block<C_max_acc_batch48, C_batch48_size>(input_buffer, output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
            case  0: break;
            case  1: softmax_compute_block< 1, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case  2: softmax_compute_block< 2, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case  3: softmax_compute_block< 3, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case  4: softmax_compute_block< 4, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case  5: softmax_compute_block< 5, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case  6: softmax_compute_block< 6, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case  7: softmax_compute_block< 7, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case  8: softmax_compute_block< 8, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case  9: softmax_compute_block< 9, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case 10: softmax_compute_block<10, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case 11: softmax_compute_block<11, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case 12: softmax_compute_block<12, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case 13: softmax_compute_block<13, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
            case 14: softmax_compute_block<14, C_batch48_size>(input_buffer, output_buffer, acc_sum); break;
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

void softmax_f32::forward(const nn::nn_workload_data_t<float> *input, nn::nn_workload_data_t<float> *output) {
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

void run_singlethreaded_softmax_work_item(
    nn_workload_item *const work_item,
    nn_workload_data_t *input_view,
    nn_workload_data_t *output_view)
{
    auto primitive = static_cast<softmax_f32*>(work_item->primitive);
    primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(input_view),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(output_view));
}

void wrapper_softmax_work_item(nn_workload_item *const work_item, nn_device_internal* device)
{
    nn_workload_data_coords_t in_out_view_coords =
    {
        work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_n],
        work_item->input[0]->output->parent->buffer_size / static_cast<uint32_t>(sizeof(float)) / work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_n],
        1,
        1,
        1,
        1
    };

    nn_workload_data_layout_t in_out_view_layout =
    {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_FLOAT
    };

    nn::nn_workload_data_t<float>* input_view = new nn::nn_workload_data_t<float>(work_item->input[0]->output->parent->data_buffer, in_out_view_coords, in_out_view_layout);
    nn::nn_workload_data_t<float>* output_view = new nn::nn_workload_data_t<float>(work_item->output->parent->data_buffer, in_out_view_coords, in_out_view_layout);

    run_singlethreaded_softmax_work_item(work_item, input_view, output_view);

    delete output_view;
    delete input_view;
}

softmax_f32 *softmax_f32::create(size_t num_features, size_t batch_size, nn_device_t *device) {
    return new softmax_f32(num_features, batch_size, reinterpret_cast<nn_device_internal *>(device));
}

softmax_f32::softmax_f32(size_t num_features, size_t batch_size, nn_device_internal *device)
    : num_features(num_features),
      batch_size(batch_size),
      device(device),
      in_out_layout({{0, 0, 0, 0, 0, 0}, // tile in log2(size)
                    {0, 0, 0, 0, 0, 0}, // alignment
                    {NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q}, // ordering
                    NN_DATATYPE_FLOAT}) {}

nn::nn_workload_data_t<float> *softmax_f32::create_input(const nn::data<float, 2> &input) {
    assert(input.size[1] == batch_size);
    assert(input.size[0] == num_features);

    nn_workload_data_coords_t size = { static_cast<uint32_t>(batch_size), static_cast<uint32_t>(num_features), 1, 1, 1, 1 };
    auto result = new nn::nn_workload_data_t<float>(size, in_out_layout);

    for (size_t it_batch = 0; it_batch < batch_size; ++it_batch)
        for (size_t it_input = 0; it_input < num_features; ++it_input)
            (*result)(it_batch, it_input, 0, 0, 0, 0) = input.at(it_input, it_batch);

    return result;
}

bool softmax_f32::validate_input(const nn::nn_workload_data_t<float> &input) {
    if (0 != memcmp(&in_out_layout, &input.parent->layout, sizeof(in_out_layout)))
        return false;

    const auto view_size = input.get_length();

    if (view_size.t[NN_DATA_COORD_n] != batch_size)
        return false;

    if (view_size.t[NN_DATA_COORD_x] != num_features)
        return false;

    if ((input.parent->buffer_size / input.parent->data_type_size) != num_features * batch_size)
        return false;

    return true;
}

nn::nn_workload_data_t<float> *softmax_f32::create_output() {
    nn_workload_data_coords_t size = {
        static_cast<uint32_t>(batch_size), static_cast<uint32_t>(num_features), 1, 1, 1, 1};
    return new nn::nn_workload_data_t<float>(size, in_out_layout);
}

void softmax_f32::copy_output(nn::data<float, 2> &destination, const nn::nn_workload_data_t<float> &source) {
    assert(destination.size[1] == batch_size);
    assert(destination.size[0] == num_features);

    nn_workload_data_coords_t size = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(num_features), 1, 1, 1, 1};

    if(source.parent->buffer_size / source.parent->data_type_size == batch_size * num_features){
        for (size_t it_batch = 0; it_batch < batch_size; ++it_batch)
            for (size_t x = 0; x < num_features; ++x)
                ((float *)destination.buffer)[x + it_batch * num_features] =
                    ((float *)source.parent->data_buffer)[it_batch + x * batch_size];
    }else{
        for (size_t it_batch = 0; it_batch < batch_size; ++it_batch)
            for (size_t x = 0; x < num_features; ++x)
                destination.at(x, it_batch) =
                const_cast<nn::nn_workload_data_t<float> &>(source)(it_batch, x, 0, 0, 0, 0);
    }
}

namespace softmax_f32_impl {
nn_opaque_data_t *NN_API_CALL_CONVENTION
create_input(nn_primitive_handle_t handle, const nn_data_t *input, NN_API_STATUS *status) {
    auto primitive = static_cast<layer::softmax_f32 *>(handle);
    auto result = primitive->create_input(*nn::data_cast<float, 2>(input));
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

nn_opaque_data_t *NN_API_CALL_CONVENTION create_output(nn_primitive_handle_t handle, NN_API_STATUS *status) {
    auto primitive = static_cast<layer::softmax_f32 *>(handle);
    auto result = primitive->create_output();
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

int NN_API_CALL_CONVENTION
validate_input(nn_primitive_handle_t handle, /* primitive handle */
               nn_opaque_data_t *opaque_data /* internal data storage handle to validate */) {
    auto primitive = static_cast<layer::softmax_f32 *>(handle);
    return primitive->validate_input(*reinterpret_cast<nn::nn_workload_data_t<float> *>(opaque_data));
}

nn_event_t NN_API_CALL_CONVENTION copy_output_async(nn_primitive_handle_t handle,
                                                    nn_data_t *output,
                                                    nn_opaque_data_t *output_buffer,
                                                    size_t dependencies_count,
                                                    nn_event_t *dependencies,
                                                    NN_API_STATUS *status) {
    auto primitive = static_cast<layer::softmax_f32 *>(handle);
    primitive->copy_output(*nn::data_cast<float, 2>(output),
                           *reinterpret_cast<nn::nn_workload_data_t<float> *>(output_buffer));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}

nn_event_t NN_API_CALL_CONVENTION forward(nn_primitive_handle_t handle,
                                          nn_opaque_data_t *input,
                                          nn_opaque_data_t *output,
                                          size_t dependencies_count,
                                          nn_event_t *dependencies,
                                          NN_API_STATUS *status) {
    auto primitive = static_cast<layer::softmax_f32 *>(handle);
    primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(input),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(output));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}

nn_primitive_handle_t NN_API_CALL_CONVENTION create(nn_device_t *device,  /* IDLF device handle */
                                                    size_t num_features,  /* number of input feature maps */
                                                    size_t batch_size,    /* size of input batch */
                                                    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
                                                    ) {
    SET_STATUS(NN_API_STATUS_OK);
    return layer::softmax_f32::create(num_features, batch_size, device);
}
}

} // namespace layer

nn_primitives_softmax_f32_0_t nn_primitives_softmax_f32_0{layer::softmax_f32_impl::create,
                                                          layer::softmax_f32_impl::create_input,
                                                          layer::softmax_f32_impl::validate_input,
                                                          layer::softmax_f32_impl::create_output,
                                                          layer::softmax_f32_impl::forward,
                                                          layer::softmax_f32_impl::copy_output_async};
