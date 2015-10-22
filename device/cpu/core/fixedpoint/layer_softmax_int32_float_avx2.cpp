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
#include "device/cpu/api_internal/data_helper.h"
#include "layer_softmax_int32_float_avx2.h"

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

static const auto C_max_acc = 12u;
static const auto C_batch_size = C_simd_width;
static const auto C_data_stride = C_batch_size * C_max_acc;

namespace int16_fixedpoint {

    __m256 _inner_mm256_exp_ps1(__m256 arg)
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

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // forward implementation

    template<uint32_t T_SIZE>
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
        if (T_SIZE >= 1)  acc0 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 0 * C_batch_size), acc_sum);
        if (T_SIZE >= 2)  acc1 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 1 * C_batch_size), acc_sum);
        if (T_SIZE >= 3)  acc2 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 2 * C_batch_size), acc_sum);
        if (T_SIZE >= 4)  acc3 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 3 * C_batch_size), acc_sum);
        if (T_SIZE >= 5)  acc4 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 4 * C_batch_size), acc_sum);
        if (T_SIZE >= 6)  acc5 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 5 * C_batch_size), acc_sum);
        if (T_SIZE >= 7)  acc6 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 6 * C_batch_size), acc_sum);
        if (T_SIZE >= 8)  acc7 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 7 * C_batch_size), acc_sum);
        if (T_SIZE >= 9)  acc8 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 8 * C_batch_size), acc_sum);
        if (T_SIZE >= 10)  acc9 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 9 * C_batch_size), acc_sum);
        if (T_SIZE >= 11) acc10 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 10 * C_batch_size), acc_sum);
        if (T_SIZE >= 12) acc11 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 11 * C_batch_size), acc_sum);
        if (T_SIZE >= 13) acc12 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 12 * C_batch_size), acc_sum);
        if (T_SIZE >= 14) acc13 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 13 * C_batch_size), acc_sum);
        if (T_SIZE >= 15) acc14 = _mm256_mul_ps(_mm256_loadu_ps(output_ptr + 14 * C_batch_size), acc_sum);

        // Store results.
        if (T_SIZE >= 1) _mm256_storeu_ps(output_ptr + 0 * C_batch_size, acc0);
        if (T_SIZE >= 2) _mm256_storeu_ps(output_ptr + 1 * C_batch_size, acc1);
        if (T_SIZE >= 3) _mm256_storeu_ps(output_ptr + 2 * C_batch_size, acc2);
        if (T_SIZE >= 4) _mm256_storeu_ps(output_ptr + 3 * C_batch_size, acc3);
        if (T_SIZE >= 5) _mm256_storeu_ps(output_ptr + 4 * C_batch_size, acc4);
        if (T_SIZE >= 6) _mm256_storeu_ps(output_ptr + 5 * C_batch_size, acc5);
        if (T_SIZE >= 7) _mm256_storeu_ps(output_ptr + 6 * C_batch_size, acc6);
        if (T_SIZE >= 8) _mm256_storeu_ps(output_ptr + 7 * C_batch_size, acc7);
        if (T_SIZE >= 9) _mm256_storeu_ps(output_ptr + 8 * C_batch_size, acc8);
        if (T_SIZE >= 10) _mm256_storeu_ps(output_ptr + 9 * C_batch_size, acc9);
        if (T_SIZE >= 11) _mm256_storeu_ps(output_ptr + 10 * C_batch_size, acc10);
        if (T_SIZE >= 12) _mm256_storeu_ps(output_ptr + 11 * C_batch_size, acc11);
        if (T_SIZE >= 13) _mm256_storeu_ps(output_ptr + 12 * C_batch_size, acc12);
        if (T_SIZE >= 14) _mm256_storeu_ps(output_ptr + 13 * C_batch_size, acc13);
        if (T_SIZE >= 15) _mm256_storeu_ps(output_ptr + 14 * C_batch_size, acc14);

        output_ptr += C_batch_size*T_SIZE;
    }

    template<uint32_t T_SIZE>
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
            acc10, acc11, acc12, acc13, acc14;

        // Load inputs and perform e^x
        if (T_SIZE >= 1)  acc0 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 0 * C_batch_size));
        if (T_SIZE >= 2)  acc1 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 1 * C_batch_size));
        if (T_SIZE >= 3)  acc2 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 2 * C_batch_size));
        if (T_SIZE >= 4)  acc3 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 3 * C_batch_size));
        if (T_SIZE >= 5)  acc4 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 4 * C_batch_size));
        if (T_SIZE >= 6)  acc5 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 5 * C_batch_size));
        if (T_SIZE >= 7)  acc6 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 6 * C_batch_size));
        if (T_SIZE >= 8)  acc7 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 7 * C_batch_size));
        if (T_SIZE >= 9)  acc8 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 8 * C_batch_size));
        if (T_SIZE >= 10)  acc9 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 9 * C_batch_size));
        if (T_SIZE >= 11) acc10 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 10 * C_batch_size));
        if (T_SIZE >= 12) acc11 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 11 * C_batch_size));
        if (T_SIZE >= 13) acc12 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 12 * C_batch_size));
        if (T_SIZE >= 14) acc13 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 13 * C_batch_size));
        if (T_SIZE >= 15) acc14 = _inner_mm256_exp_ps1(_mm256_loadu_ps(input_ptr + 14 * C_batch_size));

        // Store results.
        if (T_SIZE >= 1) _mm256_storeu_ps(output_ptr + 0 * C_batch_size, acc0);
        if (T_SIZE >= 2) _mm256_storeu_ps(output_ptr + 1 * C_batch_size, acc1);
        if (T_SIZE >= 3) _mm256_storeu_ps(output_ptr + 2 * C_batch_size, acc2);
        if (T_SIZE >= 4) _mm256_storeu_ps(output_ptr + 3 * C_batch_size, acc3);
        if (T_SIZE >= 5) _mm256_storeu_ps(output_ptr + 4 * C_batch_size, acc4);
        if (T_SIZE >= 6) _mm256_storeu_ps(output_ptr + 5 * C_batch_size, acc5);
        if (T_SIZE >= 7) _mm256_storeu_ps(output_ptr + 6 * C_batch_size, acc6);
        if (T_SIZE >= 8) _mm256_storeu_ps(output_ptr + 7 * C_batch_size, acc7);
        if (T_SIZE >= 9) _mm256_storeu_ps(output_ptr + 8 * C_batch_size, acc8);
        if (T_SIZE >= 10) _mm256_storeu_ps(output_ptr + 9 * C_batch_size, acc9);
        if (T_SIZE >= 11) _mm256_storeu_ps(output_ptr + 10 * C_batch_size, acc10);
        if (T_SIZE >= 12) _mm256_storeu_ps(output_ptr + 11 * C_batch_size, acc11);
        if (T_SIZE >= 13) _mm256_storeu_ps(output_ptr + 12 * C_batch_size, acc12);
        if (T_SIZE >= 14) _mm256_storeu_ps(output_ptr + 13 * C_batch_size, acc13);
        if (T_SIZE >= 15) _mm256_storeu_ps(output_ptr + 14 * C_batch_size, acc14);

        // Sum up accumulators.
        if (T_SIZE >= 1) acc_sum = _mm256_add_ps(acc0, acc_sum);
        if (T_SIZE >= 2) acc_sum = _mm256_add_ps(acc1, acc_sum);
        if (T_SIZE >= 3) acc_sum = _mm256_add_ps(acc2, acc_sum);
        if (T_SIZE >= 4) acc_sum = _mm256_add_ps(acc3, acc_sum);
        if (T_SIZE >= 5) acc_sum = _mm256_add_ps(acc4, acc_sum);
        if (T_SIZE >= 6) acc_sum = _mm256_add_ps(acc5, acc_sum);
        if (T_SIZE >= 7) acc_sum = _mm256_add_ps(acc6, acc_sum);
        if (T_SIZE >= 8) acc_sum = _mm256_add_ps(acc7, acc_sum);
        if (T_SIZE >= 9) acc_sum = _mm256_add_ps(acc8, acc_sum);
        if (T_SIZE >= 10) acc_sum = _mm256_add_ps(acc9, acc_sum);
        if (T_SIZE >= 11) acc_sum = _mm256_add_ps(acc10, acc_sum);
        if (T_SIZE >= 12) acc_sum = _mm256_add_ps(acc11, acc_sum);
        if (T_SIZE >= 13) acc_sum = _mm256_add_ps(acc12, acc_sum);
        if (T_SIZE >= 14) acc_sum = _mm256_add_ps(acc13, acc_sum);
        if (T_SIZE >= 15) acc_sum = _mm256_add_ps(acc14, acc_sum);

        input_ptr += C_batch_size*T_SIZE;
        output_ptr += C_batch_size*T_SIZE;
    }

    softmax_i32::softmax_i32(
        size_t num_features,
        size_t batch_size,
        int8_t input_fraction,
        nn_device_internal *device)
        : num_features(num_features),
        batch_size(batch_size),
        device(device),
        input_fraction(input_fraction) {}

    bool softmax_i32::validate_input(size_t index, nn_workload_data_t *data)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    void softmax_i32::run_softmax_int32_float_work_item_batch8(const nn::workload_data<int32_t> *input_view, nn::workload_data<> *output_view)
    {
        const auto input_width = input_view->parent->lengths.t[NN_DATA_COORD_z] * input_view->parent->lengths.t[NN_DATA_COORD_p];
        const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;
        const auto batch_size = input_view->parent->lengths.t[NN_DATA_COORD_n];

        const auto num_full_blocks = output_width / C_max_acc;
        const auto partial_block_size = output_width % C_max_acc;

        const auto output_view_start = output_view->view_begin.t[NN_DATA_COORD_x] * batch_size;
        const auto input_view_start = input_view->view_begin.t[NN_DATA_COORD_z] * input_view->parent->lengths.t[NN_DATA_COORD_p] * batch_size;

        const auto out_fraction = input_fraction;

        float *input_f = (float *)_mm_malloc(input_width * batch_size * sizeof(float), 64);
        if (input_f == nullptr)
            throw std::bad_alloc();

        auto input_buffer = &static_cast<int32_t*>(input_view->parent->data_buffer)[input_view_start];

        auto shift = out_fraction;
        if (shift > 0)
        {
            for (uint32_t i = 0; i < input_width * batch_size; i++)
                input_f[i] = (float)(input_buffer[i]) / (1 << shift);
        }
        else if (shift < 0)
        {
            for (uint32_t i = 0; i < input_width* batch_size; i++)
                input_f[i] = (float)(input_buffer[i]) * (1 << -shift);
        }
        else
        {
            for (uint32_t i = 0; i < input_width* batch_size; i++)
                input_f[i] = (float)(input_buffer[i]);
        }

        __m256 acc_sum = _mm256_setzero_ps();

        {
            auto input_buffer = input_f;
            auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_compute_block<C_max_acc>(input_buffer, output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
            case  0: break;
            case  1: softmax_compute_block< 1>(input_buffer, output_buffer, acc_sum); break;
            case  2: softmax_compute_block< 2>(input_buffer, output_buffer, acc_sum); break;
            case  3: softmax_compute_block< 3>(input_buffer, output_buffer, acc_sum); break;
            case  4: softmax_compute_block< 4>(input_buffer, output_buffer, acc_sum); break;
            case  5: softmax_compute_block< 5>(input_buffer, output_buffer, acc_sum); break;
            case  6: softmax_compute_block< 6>(input_buffer, output_buffer, acc_sum); break;
            case  7: softmax_compute_block< 7>(input_buffer, output_buffer, acc_sum); break;
            case  8: softmax_compute_block< 8>(input_buffer, output_buffer, acc_sum); break;
            case  9: softmax_compute_block< 9>(input_buffer, output_buffer, acc_sum); break;
            case 10: softmax_compute_block<10>(input_buffer, output_buffer, acc_sum); break;
            case 11: softmax_compute_block<11>(input_buffer, output_buffer, acc_sum); break;
            case 12: softmax_compute_block<12>(input_buffer, output_buffer, acc_sum); break;
            case 13: softmax_compute_block<13>(input_buffer, output_buffer, acc_sum); break;
            case 14: softmax_compute_block<14>(input_buffer, output_buffer, acc_sum); break;
            default: NN_UNREACHABLE_CODE;
            }
        }

        acc_sum = _mm256_div_ps(_mm256_set1_ps(1.0f), acc_sum);

        {
            auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_finalize_block<C_max_acc>(output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
            case  0: break;
            case  1: softmax_finalize_block< 1>(output_buffer, acc_sum); break;
            case  2: softmax_finalize_block< 2>(output_buffer, acc_sum); break;
            case  3: softmax_finalize_block< 3>(output_buffer, acc_sum); break;
            case  4: softmax_finalize_block< 4>(output_buffer, acc_sum); break;
            case  5: softmax_finalize_block< 5>(output_buffer, acc_sum); break;
            case  6: softmax_finalize_block< 6>(output_buffer, acc_sum); break;
            case  7: softmax_finalize_block< 7>(output_buffer, acc_sum); break;
            case  8: softmax_finalize_block< 8>(output_buffer, acc_sum); break;
            case  9: softmax_finalize_block< 9>(output_buffer, acc_sum); break;
            case 10: softmax_finalize_block<10>(output_buffer, acc_sum); break;
            case 11: softmax_finalize_block<11>(output_buffer, acc_sum); break;
            case 12: softmax_finalize_block<12>(output_buffer, acc_sum); break;
            case 13: softmax_finalize_block<13>(output_buffer, acc_sum); break;
            case 14: softmax_finalize_block<14>(output_buffer, acc_sum); break;
            default: NN_UNREACHABLE_CODE;
            }
        }

        _mm_free(input_f);
    }


    void softmax_i32::run_softmax_int32_float_work_item_batch8x(const nn::workload_data<int32_t> *input_view, nn::workload_data<> *output_view, uint16_t NoBatch8)
    {
        const auto input_width = input_view->parent->lengths.t[NN_DATA_COORD_z] * input_view->parent->lengths.t[NN_DATA_COORD_p];
        const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;
        const auto batch_size_global = output_view->parent->lengths.t[NN_DATA_COORD_n];
        const auto batch_size = 8;

        const auto num_full_blocks = output_width / C_max_acc;
        const auto partial_block_size = output_width % C_max_acc;

        const auto output_view_start = output_view->view_begin.t[NN_DATA_COORD_x] * batch_size * NoBatch8;
        const auto input_view_start = input_view->view_begin.t[NN_DATA_COORD_z] * input_view->parent->lengths.t[NN_DATA_COORD_p] * batch_size * NoBatch8;

        const auto out_fraction = input_fraction;

        float *input_f = (float *)_mm_malloc(input_width * batch_size * sizeof(float), 64);
        if (input_f == nullptr)
            throw std::bad_alloc();

        float *output_f = (float *)_mm_malloc(output_width * batch_size * sizeof(float), 64);
        if (output_f == nullptr)
            throw std::bad_alloc();

        auto input_buffer = &static_cast<int32_t*>(input_view->parent->data_buffer)[input_view_start + NoBatch8 * 8 * input_width];
        //auto input_buffer = &static_cast<int32_t*>(input_view->parent->data_buffer)[input_view_start];

        auto shift = out_fraction;
        if (shift > 0)
        {
            for (uint32_t i = 0; i < input_width * batch_size; i++)
                input_f[i] = (float)(input_buffer[i]) / (1 << shift);
        }
        else if (shift < 0)
        {
            for (uint32_t i = 0; i < input_width* batch_size; i++)
                input_f[i] = (float)(input_buffer[i]) * (1 << -shift);
        }
        else
        {
            for (uint32_t i = 0; i < input_width* batch_size; i++)
                input_f[i] = (float)(input_buffer[i]);
        }

        __m256 acc_sum = _mm256_setzero_ps();

        {
            auto input_buffer = input_f;
            //auto output_buffer = &static_cast<float*>(work_item->output->parent->data_buffer)[output_view_start + NoBatch8 * 8 * output_width];
            auto output_buffer = output_f;
            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_compute_block<C_max_acc>(input_buffer, output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
            case  0: break;
            case  1: softmax_compute_block< 1>(input_buffer, output_buffer, acc_sum); break;
            case  2: softmax_compute_block< 2>(input_buffer, output_buffer, acc_sum); break;
            case  3: softmax_compute_block< 3>(input_buffer, output_buffer, acc_sum); break;
            case  4: softmax_compute_block< 4>(input_buffer, output_buffer, acc_sum); break;
            case  5: softmax_compute_block< 5>(input_buffer, output_buffer, acc_sum); break;
            case  6: softmax_compute_block< 6>(input_buffer, output_buffer, acc_sum); break;
            case  7: softmax_compute_block< 7>(input_buffer, output_buffer, acc_sum); break;
            case  8: softmax_compute_block< 8>(input_buffer, output_buffer, acc_sum); break;
            case  9: softmax_compute_block< 9>(input_buffer, output_buffer, acc_sum); break;
            case 10: softmax_compute_block<10>(input_buffer, output_buffer, acc_sum); break;
            case 11: softmax_compute_block<11>(input_buffer, output_buffer, acc_sum); break;
            case 12: softmax_compute_block<12>(input_buffer, output_buffer, acc_sum); break;
            case 13: softmax_compute_block<13>(input_buffer, output_buffer, acc_sum); break;
            case 14: softmax_compute_block<14>(input_buffer, output_buffer, acc_sum); break;
            default: NN_UNREACHABLE_CODE;
            }
        }

        acc_sum = _mm256_div_ps(_mm256_set1_ps(1.0f), acc_sum);

        {
            //auto output_buffer = &static_cast<float*>(work_item->output->parent->data_buffer)[output_view_start + NoBatch8 * 8 * output_width];
            auto output_buffer = output_f;
            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_finalize_block<C_max_acc>(output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
            case  0: break;
            case  1: softmax_finalize_block< 1>(output_buffer, acc_sum); break;
            case  2: softmax_finalize_block< 2>(output_buffer, acc_sum); break;
            case  3: softmax_finalize_block< 3>(output_buffer, acc_sum); break;
            case  4: softmax_finalize_block< 4>(output_buffer, acc_sum); break;
            case  5: softmax_finalize_block< 5>(output_buffer, acc_sum); break;
            case  6: softmax_finalize_block< 6>(output_buffer, acc_sum); break;
            case  7: softmax_finalize_block< 7>(output_buffer, acc_sum); break;
            case  8: softmax_finalize_block< 8>(output_buffer, acc_sum); break;
            case  9: softmax_finalize_block< 9>(output_buffer, acc_sum); break;
            case 10: softmax_finalize_block<10>(output_buffer, acc_sum); break;
            case 11: softmax_finalize_block<11>(output_buffer, acc_sum); break;
            case 12: softmax_finalize_block<12>(output_buffer, acc_sum); break;
            case 13: softmax_finalize_block<13>(output_buffer, acc_sum); break;
            case 14: softmax_finalize_block<14>(output_buffer, acc_sum); break;
            default: NN_UNREACHABLE_CODE;
            }
        }

        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

        for (auto itrW = 0; itrW < output_width; itrW++)
        for (auto itr8 = 0; itr8 < C_batch_size; itr8++)
            output_buffer[itr8 + itrW * batch_size_global + NoBatch8 * C_batch_size] = output_f[itr8 + itrW * C_batch_size];

        _mm_free(input_f);
        _mm_free(output_f);
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

    void softmax_i32::run_softmax_int32_float_work_item_latency(const nn::workload_data<int32_t> *input_view, nn::workload_data<> *output_view)
    {
        const auto input_width = input_view->parent->lengths.t[NN_DATA_COORD_z] * input_view->parent->lengths.t[NN_DATA_COORD_p];
        const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;

        const auto num_full_blocks = output_width / C_data_stride;
        const auto partial_block_size = (output_width / C_simd_width) % C_max_acc;
        const auto subsimd_block_size = output_width % C_simd_width;

        const auto output_view_start = output_view->view_begin.t[NN_DATA_COORD_x];

        const auto input_view_start = input_view->view_begin.t[NN_DATA_COORD_z] * input_view->parent->lengths.t[NN_DATA_COORD_p];

        const auto out_fraction = input_fraction;

        float *input_f = (float *)_mm_malloc(input_width * sizeof(float), 64);
        if (input_f == nullptr)
            throw std::bad_alloc();

        auto input_buffer = &static_cast<int32_t*>(input_view->parent->data_buffer)[input_view_start];

        auto shift = out_fraction;
        if (shift > 0)
        {
            for (uint32_t i = 0; i < input_width; i++)
                input_f[i] = (float)(input_buffer[i]) / (1 << shift);
        }
        else if (shift < 0)
        {
            for (uint32_t i = 0; i < input_width; i++)
                input_f[i] = (float)(input_buffer[i]) * (1 << -shift);
        }
        else
        {
            for (uint32_t i = 0; i < input_width; i++)
                input_f[i] = (float)(input_buffer[i]);
        }

        __m256 acc_sum = _mm256_setzero_ps();
        float subsimd_sum = 0.0f;
        {
            auto input_buffer = input_f;
            auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_start];

            for (auto block = 0u; block < num_full_blocks; ++block)
            {
                // Run computation.
                softmax_compute_block<C_max_acc>(input_buffer, output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
            case  0: break;
            case  1: softmax_compute_block< 1>(input_buffer, output_buffer, acc_sum); break;
            case  2: softmax_compute_block< 2>(input_buffer, output_buffer, acc_sum); break;
            case  3: softmax_compute_block< 3>(input_buffer, output_buffer, acc_sum); break;
            case  4: softmax_compute_block< 4>(input_buffer, output_buffer, acc_sum); break;
            case  5: softmax_compute_block< 5>(input_buffer, output_buffer, acc_sum); break;
            case  6: softmax_compute_block< 6>(input_buffer, output_buffer, acc_sum); break;
            case  7: softmax_compute_block< 7>(input_buffer, output_buffer, acc_sum); break;
            case  8: softmax_compute_block< 8>(input_buffer, output_buffer, acc_sum); break;
            case  9: softmax_compute_block< 9>(input_buffer, output_buffer, acc_sum); break;
            case 10: softmax_compute_block<10>(input_buffer, output_buffer, acc_sum); break;
            case 11: softmax_compute_block<11>(input_buffer, output_buffer, acc_sum); break;
            case 12: softmax_compute_block<12>(input_buffer, output_buffer, acc_sum); break;
            case 13: softmax_compute_block<13>(input_buffer, output_buffer, acc_sum); break;
            case 14: softmax_compute_block<14>(input_buffer, output_buffer, acc_sum); break;
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
                softmax_finalize_block<C_max_acc>(output_buffer, acc_sum);
            }

            switch (partial_block_size)
            {
            case  0: break;
            case  1: softmax_finalize_block< 1>(output_buffer, acc_sum); break;
            case  2: softmax_finalize_block< 2>(output_buffer, acc_sum); break;
            case  3: softmax_finalize_block< 3>(output_buffer, acc_sum); break;
            case  4: softmax_finalize_block< 4>(output_buffer, acc_sum); break;
            case  5: softmax_finalize_block< 5>(output_buffer, acc_sum); break;
            case  6: softmax_finalize_block< 6>(output_buffer, acc_sum); break;
            case  7: softmax_finalize_block< 7>(output_buffer, acc_sum); break;
            case  8: softmax_finalize_block< 8>(output_buffer, acc_sum); break;
            case  9: softmax_finalize_block< 9>(output_buffer, acc_sum); break;
            case 10: softmax_finalize_block<10>(output_buffer, acc_sum); break;
            case 11: softmax_finalize_block<11>(output_buffer, acc_sum); break;
            case 12: softmax_finalize_block<12>(output_buffer, acc_sum); break;
            case 13: softmax_finalize_block<13>(output_buffer, acc_sum); break;
            case 14: softmax_finalize_block<14>(output_buffer, acc_sum); break;
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
        _mm_free(input_f);
    }

    std::vector<nn_workload_data_t *> softmax_i32::create_inputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::create(device, num_features, batch_size) };
    }

    std::vector<nn_workload_data_t *> softmax_i32::create_outputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::create(device, num_features, batch_size) };
    }

    void softmax_i32::forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs)
    {
        assert(inputs.size() == 1);
        assert(outputs.size() == 1);

        forward(reinterpret_cast<const nn::workload_data<int32_t> *>(inputs[0]),
                nn::workload_data_cast<>(outputs[0]));
    }

    void softmax_i32::forward(const nn::workload_data<int32_t> *input, nn::workload_data<> *output)
    {
        auto batch_size = input->parent->lengths.t[NN_DATA_COORD_n];
        switch (batch_size)
        {
        case 1:
            run_softmax_int32_float_work_item_latency(input, output);
            break;
        case 8:
            run_softmax_int32_float_work_item_batch8(input, output);
            break;
        case 16:
        case 24:
        case 32:
            for (auto itrB = 0; itrB < batch_size / 8; ++itrB)
                run_softmax_int32_float_work_item_batch8x(input, output, itrB);
            break;

        default:
            break;
        }
    }

    void run_softmax_int32_float_work_item(nn_workload_item *const work_item, nn_device_internal* device)
    {
        nn::workload_data<int32_t>* input_view = reinterpret_cast<nn::workload_data<int32_t> *>(work_item->input[0].get_data_view());
        nn::workload_data<>* output_view = nn::workload_data_cast<>(work_item->output[0]);

        static_cast<softmax_i32 *>(work_item->primitive)->forward(input_view, output_view);
    }
} // namepace

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_softmax_i32_create_0(
    nn_device_t *device,  /* IDLF device handle */
    size_t num_features,  /* number of input feature maps */
    size_t batch_size,    /* size of input batch */
    const nn_primitives_softmax_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);

    nn_primitives_softmax_hints_t hints_ = {};
    hints_.fixed_point_fraction_bits.accumulator = 16; // set input_fraction default value

    if (hints != nullptr)
        hints_ = *hints;

    return new int16_fixedpoint::softmax_i32(num_features, batch_size, hints_.fixed_point_fraction_bits.accumulator, reinterpret_cast<nn_device_internal *>(device));
}