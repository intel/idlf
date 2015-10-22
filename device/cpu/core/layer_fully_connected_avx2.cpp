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
#include "layer_fully_connected_avx2.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include "device/cpu/api_internal/data_helper.h"

// SIMD width for this implementation
const auto C_simd_width = sizeof(__m256) / sizeof(float);

static const auto C_max_acc_batch1 = 13u;
static const auto C_max_acc_batch8 = 13u;
static const auto C_max_acc_batch48 = 2u;

static const auto C_batch8_size = C_simd_width;
static const auto C_batch48_size = 6 * C_simd_width;

static const auto C_data_stride_batch1 = C_simd_width * C_max_acc_batch1;
static const auto C_data_stride_batch8 = C_batch8_size * C_max_acc_batch8;
static const auto C_data_stride_batch48 = C_batch48_size * C_max_acc_batch48;

namespace layer {
///////////////////////////////////////////////////////////////////////////////////////////////////
// forward implementation

template<uint32_t T_SIZE, NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
void fully_connected_compute_block_batch8(
    float* input_buffer,
    float* output_ptr,
    float* bias_ptr,
    float* weights_buffer,
    uint32_t input_width)
{
    // We are not using table of registers and unroll pragmas
    // due to compiler which have issues with register allocation
    // and needs special, obvious treatment. Template immediate
    // arguments matching will remove all conditions in this code.
    __m256  acc0, acc1, acc2, acc3, acc4,
            acc5, acc6, acc7, acc8, acc9,
            acc10, acc11, acc12, acc13, acc14;

    if (T_NEED_BIAS_COPY)
    {
        if (T_SIZE >=  1)  acc0 = _mm256_setzero_ps();
        if (T_SIZE >=  2)  acc1 = _mm256_setzero_ps();
        if (T_SIZE >=  3)  acc2 = _mm256_setzero_ps();
        if (T_SIZE >=  4)  acc3 = _mm256_setzero_ps();
        if (T_SIZE >=  5)  acc4 = _mm256_setzero_ps();
        if (T_SIZE >=  6)  acc5 = _mm256_setzero_ps();
        if (T_SIZE >=  7)  acc6 = _mm256_setzero_ps();
        if (T_SIZE >=  8)  acc7 = _mm256_setzero_ps();
        if (T_SIZE >=  9)  acc8 = _mm256_setzero_ps();
        if (T_SIZE >= 10)  acc9 = _mm256_setzero_ps();
        if (T_SIZE >= 11) acc10 = _mm256_setzero_ps();
        if (T_SIZE >= 12) acc11 = _mm256_setzero_ps();
        if (T_SIZE >= 13) acc12 = _mm256_setzero_ps();
        if (T_SIZE >= 14) acc13 = _mm256_setzero_ps();
        if (T_SIZE >= 15) acc14 = _mm256_setzero_ps();
    }
    else
    {
        if (T_SIZE >=  1)  acc0 = _mm256_load_ps(output_ptr +  0 * C_batch8_size);
        if (T_SIZE >=  2)  acc1 = _mm256_load_ps(output_ptr +  1 * C_batch8_size);
        if (T_SIZE >=  3)  acc2 = _mm256_load_ps(output_ptr +  2 * C_batch8_size);
        if (T_SIZE >=  4)  acc3 = _mm256_load_ps(output_ptr +  3 * C_batch8_size);
        if (T_SIZE >=  5)  acc4 = _mm256_load_ps(output_ptr +  4 * C_batch8_size);
        if (T_SIZE >=  6)  acc5 = _mm256_load_ps(output_ptr +  5 * C_batch8_size);
        if (T_SIZE >=  7)  acc6 = _mm256_load_ps(output_ptr +  6 * C_batch8_size);
        if (T_SIZE >=  8)  acc7 = _mm256_load_ps(output_ptr +  7 * C_batch8_size);
        if (T_SIZE >=  9)  acc8 = _mm256_load_ps(output_ptr +  8 * C_batch8_size);
        if (T_SIZE >= 10)  acc9 = _mm256_load_ps(output_ptr +  9 * C_batch8_size);
        if (T_SIZE >= 11) acc10 = _mm256_load_ps(output_ptr + 10 * C_batch8_size);
        if (T_SIZE >= 12) acc11 = _mm256_load_ps(output_ptr + 11 * C_batch8_size);
        if (T_SIZE >= 13) acc12 = _mm256_load_ps(output_ptr + 12 * C_batch8_size);
        if (T_SIZE >= 14) acc13 = _mm256_load_ps(output_ptr + 13 * C_batch8_size);
        if (T_SIZE >= 15) acc14 = _mm256_load_ps(output_ptr + 14 * C_batch8_size);
    }

    auto input_ptr = &input_buffer[0];

    const auto input_ptr_end = &input_buffer[input_width*C_batch8_size];

    while (input_ptr < input_ptr_end)
    {
        // Do MADs.
        __m256 input = _mm256_load_ps(input_ptr);
        if (T_SIZE >=  1)  acc0 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  0),  acc0);
        if (T_SIZE >=  2)  acc1 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  1),  acc1);
        if (T_SIZE >=  3)  acc2 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  2),  acc2);
        if (T_SIZE >=  4)  acc3 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  3),  acc3);
        if (T_SIZE >=  5)  acc4 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  4),  acc4);
        if (T_SIZE >=  6)  acc5 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  5),  acc5);
        if (T_SIZE >=  7)  acc6 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  6),  acc6);
        if (T_SIZE >=  8)  acc7 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  7),  acc7);
        if (T_SIZE >=  9)  acc8 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  8),  acc8);
        if (T_SIZE >= 10)  acc9 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer +  9),  acc9);
        if (T_SIZE >= 11) acc10 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer + 10), acc10);
        if (T_SIZE >= 12) acc11 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer + 11), acc11);
        if (T_SIZE >= 13) acc12 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer + 12), acc12);
        if (T_SIZE >= 14) acc13 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer + 13), acc13);
        if (T_SIZE >= 15) acc14 = _mm256_fmadd_ps(input, _mm256_broadcast_ss(weights_buffer + 14), acc14);

        // Increment pointers.
        input_ptr += C_batch8_size;
        weights_buffer += C_max_acc_batch8;
    }

    if (T_NEED_BIAS_COPY)
    {
        // Add biases.
        if (T_SIZE >=  1)  acc0 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  0),  acc0);
        if (T_SIZE >=  2)  acc1 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  1),  acc1);
        if (T_SIZE >=  3)  acc2 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  2),  acc2);
        if (T_SIZE >=  4)  acc3 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  3),  acc3);
        if (T_SIZE >=  5)  acc4 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  4),  acc4);
        if (T_SIZE >=  6)  acc5 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  5),  acc5);
        if (T_SIZE >=  7)  acc6 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  6),  acc6);
        if (T_SIZE >=  8)  acc7 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  7),  acc7);
        if (T_SIZE >=  9)  acc8 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  8),  acc8);
        if (T_SIZE >= 10)  acc9 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr +  9),  acc9);
        if (T_SIZE >= 11) acc10 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 10), acc10);
        if (T_SIZE >= 12) acc11 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 11), acc11);
        if (T_SIZE >= 13) acc12 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 12), acc12);
        if (T_SIZE >= 14) acc13 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 13), acc13);
        if (T_SIZE >= 15) acc14 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 14), acc14);
    }

    if (T_FUNCTION == NN_ACTIVATION_FUNCTION_RELU)
    {
        // Perform ReLU.
        if (T_SIZE >=  1)  acc0 = _mm256_max_ps(_mm256_setzero_ps(),  acc0);
        if (T_SIZE >=  2)  acc1 = _mm256_max_ps(_mm256_setzero_ps(),  acc1);
        if (T_SIZE >=  3)  acc2 = _mm256_max_ps(_mm256_setzero_ps(),  acc2);
        if (T_SIZE >=  4)  acc3 = _mm256_max_ps(_mm256_setzero_ps(),  acc3);
        if (T_SIZE >=  5)  acc4 = _mm256_max_ps(_mm256_setzero_ps(),  acc4);
        if (T_SIZE >=  6)  acc5 = _mm256_max_ps(_mm256_setzero_ps(),  acc5);
        if (T_SIZE >=  7)  acc6 = _mm256_max_ps(_mm256_setzero_ps(),  acc6);
        if (T_SIZE >=  8)  acc7 = _mm256_max_ps(_mm256_setzero_ps(),  acc7);
        if (T_SIZE >=  9)  acc8 = _mm256_max_ps(_mm256_setzero_ps(),  acc8);
        if (T_SIZE >= 10)  acc9 = _mm256_max_ps(_mm256_setzero_ps(),  acc9);
        if (T_SIZE >= 11) acc10 = _mm256_max_ps(_mm256_setzero_ps(), acc10);
        if (T_SIZE >= 12) acc11 = _mm256_max_ps(_mm256_setzero_ps(), acc11);
        if (T_SIZE >= 13) acc12 = _mm256_max_ps(_mm256_setzero_ps(), acc12);
        if (T_SIZE >= 14) acc13 = _mm256_max_ps(_mm256_setzero_ps(), acc13);
        if (T_SIZE >= 15) acc14 = _mm256_max_ps(_mm256_setzero_ps(), acc14);
    }

    // Store results.
    if (T_SIZE >=  1) _mm256_store_ps(output_ptr +  0 * C_batch8_size,  acc0);
    if (T_SIZE >=  2) _mm256_store_ps(output_ptr +  1 * C_batch8_size,  acc1);
    if (T_SIZE >=  3) _mm256_store_ps(output_ptr +  2 * C_batch8_size,  acc2);
    if (T_SIZE >=  4) _mm256_store_ps(output_ptr +  3 * C_batch8_size,  acc3);
    if (T_SIZE >=  5) _mm256_store_ps(output_ptr +  4 * C_batch8_size,  acc4);
    if (T_SIZE >=  6) _mm256_store_ps(output_ptr +  5 * C_batch8_size,  acc5);
    if (T_SIZE >=  7) _mm256_store_ps(output_ptr +  6 * C_batch8_size,  acc6);
    if (T_SIZE >=  8) _mm256_store_ps(output_ptr +  7 * C_batch8_size,  acc7);
    if (T_SIZE >=  9) _mm256_store_ps(output_ptr +  8 * C_batch8_size,  acc8);
    if (T_SIZE >= 10) _mm256_store_ps(output_ptr +  9 * C_batch8_size,  acc9);
    if (T_SIZE >= 11) _mm256_store_ps(output_ptr + 10 * C_batch8_size, acc10);
    if (T_SIZE >= 12) _mm256_store_ps(output_ptr + 11 * C_batch8_size, acc11);
    if (T_SIZE >= 13) _mm256_store_ps(output_ptr + 12 * C_batch8_size, acc12);
    if (T_SIZE >= 14) _mm256_store_ps(output_ptr + 13 * C_batch8_size, acc13);
    if (T_SIZE >= 15) _mm256_store_ps(output_ptr + 14 * C_batch8_size, acc14);
}

template <NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
void fully_connected_f32::run_fully_connected_work_item_internal_batch8(const nn::workload_data<> *input,
                                                                        const nn::workload_data<> *weights,
                                                                        const nn::workload_data<> *bias,
                                                                        nn::workload_data<> *output) {
    const auto input_width = input->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_width = output->view_end.t[NN_DATA_COORD_x] - output->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto num_full_blocks = output_width / C_max_acc_batch8;
    const auto partial_block_size = output_width % C_max_acc_batch8;

    auto input_buffer = static_cast<float*>(input->parent->data_buffer);
    auto output_buffer = static_cast<float*>(output->parent->data_buffer);
    auto weights_buffer = static_cast<float*>(weights->parent->data_buffer);

    // Output views.
    const auto output_view_start = output->view_begin.t[NN_DATA_COORD_x];
    const auto output_view_batch_offset = output_view_start * C_batch8_size;

    // Weight views (for output-related weights).
    auto weight_view_start =
        output_view_start / C_max_acc_batch8 * input_width * C_max_acc_batch8;

    auto weights_ptr = &weights_buffer[weight_view_start];
    auto output_ptr = &output_buffer[output_view_batch_offset];

    float* bias_ptr = nullptr;
    if (T_NEED_BIAS_COPY)
    {
        auto biases_buffer = static_cast<float*>(bias->parent->data_buffer);
        bias_ptr = &biases_buffer[output_view_start];
    }

    for (auto block = 0u; block < num_full_blocks; ++block)
    {
        // Run computation.
        fully_connected_compute_block_batch8<C_max_acc_batch8, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width);

        // Increment pointers.
        output_ptr += C_data_stride_batch8;
        weights_ptr += input_width*C_max_acc_batch8;

        if (T_NEED_BIAS_COPY)
        {
            bias_ptr += C_max_acc_batch8;
        }
    }

    switch (partial_block_size)
    {
    case  0: break;
    case  1: fully_connected_compute_block_batch8< 1, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case  2: fully_connected_compute_block_batch8< 2, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case  3: fully_connected_compute_block_batch8< 3, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case  4: fully_connected_compute_block_batch8< 4, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case  5: fully_connected_compute_block_batch8< 5, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case  6: fully_connected_compute_block_batch8< 6, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case  7: fully_connected_compute_block_batch8< 7, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case  8: fully_connected_compute_block_batch8< 8, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case  9: fully_connected_compute_block_batch8< 9, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case 10: fully_connected_compute_block_batch8<10, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case 11: fully_connected_compute_block_batch8<11, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    case 12: fully_connected_compute_block_batch8<12, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width); break;
    default:
        NN_UNREACHABLE_CODE;
    }
}

template<uint32_t T_SIZE, NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
inline void fully_connected_compute_block_batch48(
    float* input_buffer,
    float* output_ptr,
    float* bias_ptr,
    float* weights_buffer,
    uint32_t input_width,
    bool first_run,
    bool last_run)
{
    // We are not using table of registers and unroll pragmas
    // due to compiler which have issues with register allocation
    // and needs special, obvious treatment. Template immediate
    // arguments matching will remove all conditions in this code.
    __m256  acc0, acc1, acc2, acc3, acc4,
            acc5, acc6, acc7, acc8, acc9,
            acc10, acc11;

    if (first_run)
    {
        if (T_NEED_BIAS_COPY)
        {
            if (T_SIZE >= 1)
            {
                acc0 = _mm256_setzero_ps();
                acc1 = _mm256_setzero_ps();
                acc2 = _mm256_setzero_ps();
                acc3 = _mm256_setzero_ps();
                acc4 = _mm256_setzero_ps();
                acc5 = _mm256_setzero_ps();
            }

            if (T_SIZE >= 2)
            {
                acc6 = _mm256_setzero_ps();
                acc7 = _mm256_setzero_ps();
                acc8 = _mm256_setzero_ps();
                acc9 = _mm256_setzero_ps();
                acc10 = _mm256_setzero_ps();
                acc11 = _mm256_setzero_ps();
            }
        }
        else
        {
            if (T_SIZE >= 1)
            {
                acc0 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 0 * C_simd_width);
                acc1 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 1 * C_simd_width);
                acc2 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 2 * C_simd_width);
                acc3 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 3 * C_simd_width);
                acc4 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 4 * C_simd_width);
                acc5 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 5 * C_simd_width);
            }

            if (T_SIZE >= 2)
            {
                acc6 = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 0 * C_simd_width);
                acc7 = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 1 * C_simd_width);
                acc8 = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 2 * C_simd_width);
                acc9 = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 3 * C_simd_width);
                acc10 = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 4 * C_simd_width);
                acc11 = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 5 * C_simd_width);
            }
        }
    }
    else
    {
        if (T_SIZE >= 1)
        {
            acc0 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 0 * C_simd_width);
            acc1 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 1 * C_simd_width);
            acc2 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 2 * C_simd_width);
            acc3 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 3 * C_simd_width);
            acc4 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 4 * C_simd_width);
            acc5 = _mm256_load_ps(output_ptr + 0 * C_batch48_size + 5 * C_simd_width);
        }

        if (T_SIZE >= 2)
        {
            acc6  = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 0 * C_simd_width);
            acc7  = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 1 * C_simd_width);
            acc8  = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 2 * C_simd_width);
            acc9  = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 3 * C_simd_width);
            acc10 = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 4 * C_simd_width);
            acc11 = _mm256_load_ps(output_ptr + 1 * C_batch48_size + 5 * C_simd_width);
        }
    }

    auto input_ptr = &input_buffer[0];

    const auto input_ptr_end = &input_buffer[input_width*C_batch48_size];

    while (input_ptr < input_ptr_end)
    {
        // Do MADs.
        __m256 weights0 = _mm256_broadcast_ss(weights_buffer + 0);
        __m256 weights1 = _mm256_broadcast_ss(weights_buffer + 1);

        __m256 input = _mm256_load_ps(input_ptr + 0 * C_simd_width);
        if (T_SIZE >= 1)  acc0 = _mm256_fmadd_ps(input, weights0,  acc0);
        if (T_SIZE >= 2)  acc6 = _mm256_fmadd_ps(input, weights1,  acc6);

        input = _mm256_load_ps(input_ptr + 1 * C_simd_width);
        if (T_SIZE >= 1)  acc1 = _mm256_fmadd_ps(input, weights0,  acc1);
        if (T_SIZE >= 2)  acc7 = _mm256_fmadd_ps(input, weights1,  acc7);

        input = _mm256_load_ps(input_ptr + 2 * C_simd_width);
        if (T_SIZE >= 1)  acc2 = _mm256_fmadd_ps(input, weights0,  acc2);
        if (T_SIZE >= 2)  acc8 = _mm256_fmadd_ps(input, weights1,  acc8);

        input = _mm256_load_ps(input_ptr + 3 * C_simd_width);
        if (T_SIZE >= 1)  acc3 = _mm256_fmadd_ps(input, weights0,  acc3);
        if (T_SIZE >= 2)  acc9 = _mm256_fmadd_ps(input, weights1,  acc9);

        input = _mm256_load_ps(input_ptr + 4 * C_simd_width);
        if (T_SIZE >= 1)  acc4 = _mm256_fmadd_ps(input, weights0,  acc4);
        if (T_SIZE >= 2) acc10 = _mm256_fmadd_ps(input, weights1, acc10);

        input = _mm256_load_ps(input_ptr + 5 * C_simd_width);
        if (T_SIZE >= 1)  acc5 = _mm256_fmadd_ps(input, weights0,  acc5);
        if (T_SIZE >= 2) acc11 = _mm256_fmadd_ps(input, weights1, acc11);

        // Increment pointers.
        input_ptr += C_batch48_size;
        weights_buffer += C_max_acc_batch48;
    }

    if (first_run)
    {
        if (T_NEED_BIAS_COPY)
        {
            // Add biases.
            if (T_SIZE >= 1)
            {
                acc0 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 0), acc0);
                acc1 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 0), acc1);
                acc2 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 0), acc2);
                acc3 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 0), acc3);
                acc4 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 0), acc4);
                acc5 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 0), acc5);
            }

            if (T_SIZE >= 2)
            {
                acc6 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 1), acc6);
                acc7 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 1), acc7);
                acc8 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 1), acc8);
                acc9 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 1), acc9);
                acc10 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 1), acc10);
                acc11 = _mm256_add_ps(_mm256_broadcast_ss(bias_ptr + 1), acc11);
            }
        }
    }

    if (last_run)
    {
        if (T_FUNCTION == NN_ACTIVATION_FUNCTION_RELU)
        {
            // Perform ReLU.
            if (T_SIZE >= 1)
            {
                acc0 = _mm256_max_ps(_mm256_setzero_ps(), acc0);
                acc1 = _mm256_max_ps(_mm256_setzero_ps(), acc1);
                acc2 = _mm256_max_ps(_mm256_setzero_ps(), acc2);
                acc3 = _mm256_max_ps(_mm256_setzero_ps(), acc3);
                acc4 = _mm256_max_ps(_mm256_setzero_ps(), acc4);
                acc5 = _mm256_max_ps(_mm256_setzero_ps(), acc5);
            }

            if (T_SIZE >= 2)
            {
                acc6 = _mm256_max_ps(_mm256_setzero_ps(), acc6);
                acc7 = _mm256_max_ps(_mm256_setzero_ps(), acc7);
                acc8 = _mm256_max_ps(_mm256_setzero_ps(), acc8);
                acc9 = _mm256_max_ps(_mm256_setzero_ps(), acc9);
                acc10 = _mm256_max_ps(_mm256_setzero_ps(), acc10);
                acc11 = _mm256_max_ps(_mm256_setzero_ps(), acc11);
            }
        }
    }

    // Store results.
    if (T_SIZE >= 1)
    {
        _mm256_store_ps(output_ptr + 0 * C_batch48_size + 0 * C_simd_width, acc0);
        _mm256_store_ps(output_ptr + 0 * C_batch48_size + 1 * C_simd_width, acc1);
        _mm256_store_ps(output_ptr + 0 * C_batch48_size + 2 * C_simd_width, acc2);
        _mm256_store_ps(output_ptr + 0 * C_batch48_size + 3 * C_simd_width, acc3);
        _mm256_store_ps(output_ptr + 0 * C_batch48_size + 4 * C_simd_width, acc4);
        _mm256_store_ps(output_ptr + 0 * C_batch48_size + 5 * C_simd_width, acc5);
    }

    if (T_SIZE >= 2)
    {
        _mm256_store_ps(output_ptr + 1 * C_batch48_size + 0 * C_simd_width,  acc6);
        _mm256_store_ps(output_ptr + 1 * C_batch48_size + 1 * C_simd_width,  acc7);
        _mm256_store_ps(output_ptr + 1 * C_batch48_size + 2 * C_simd_width,  acc8);
        _mm256_store_ps(output_ptr + 1 * C_batch48_size + 3 * C_simd_width,  acc9);
        _mm256_store_ps(output_ptr + 1 * C_batch48_size + 4 * C_simd_width, acc10);
        _mm256_store_ps(output_ptr + 1 * C_batch48_size + 5 * C_simd_width, acc11);
    }
}

template <NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
void fully_connected_f32::run_fully_connected_work_item_internal_batch48(const nn::workload_data<> *input,
                                                                         const nn::workload_data<> *weights,
                                                                         const nn::workload_data<> *bias,
                                                                         nn::workload_data<> *output) {
    const auto input_width = input->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_width = output->view_end.t[NN_DATA_COORD_x] - output->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto num_full_blocks = output_width / C_max_acc_batch48;
    const auto partial_block_size = output_width % C_max_acc_batch48;

    auto input_buffer = static_cast<float*>(input->parent->data_buffer);
    auto output_buffer = static_cast<float*>(output->parent->data_buffer);
    auto weights_buffer = static_cast<float*>(weights->parent->data_buffer);

    // Output views.
    const auto output_view_start = output->view_begin.t[NN_DATA_COORD_x];
    const auto output_view_batch_offset = output_view_start * C_batch48_size;

    // Weight views (for output-related weights).
    auto weight_view_start =
        output_view_start / C_max_acc_batch48 * input_width * C_max_acc_batch48;

    const auto weights_ptr = &weights_buffer[weight_view_start];
    const auto output_ptr = &output_buffer[output_view_batch_offset];

    const auto C_package_size = 128;

    const auto num_input_packages = input_width / C_package_size;
    const auto package_remainder = input_width % C_package_size;

    bool first_run = true;

    float* bias_ptr = nullptr;
    if (T_NEED_BIAS_COPY)
    {
        auto biases_buffer = static_cast<float*>(bias->parent->data_buffer);
        bias_ptr = &biases_buffer[output_view_start];
    }

    // Full packages.
    for (auto input_package = 0u; input_package < num_input_packages; ++input_package)
    {
        const bool last_run =
            (package_remainder > 0 || input_package + 1 < num_input_packages) ? false : true;
        auto package_weights_ptr = weights_ptr;
        auto package_output_ptr = output_ptr;
        auto package_bias_ptr = bias_ptr;
        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            fully_connected_compute_block_batch48<C_max_acc_batch48, T_FUNCTION, T_NEED_BIAS_COPY>(
                        input_buffer + C_package_size * C_batch48_size * input_package,
                        package_output_ptr,
                        package_bias_ptr,
                        package_weights_ptr + C_package_size * C_max_acc_batch48 * input_package,
                        C_package_size,
                        first_run,
                        last_run);

            // Increment pointers.
            package_output_ptr += C_data_stride_batch48;
            package_weights_ptr += input_width*C_max_acc_batch48;

            if (T_NEED_BIAS_COPY)
            {
                package_bias_ptr += C_max_acc_batch48;
            }
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: fully_connected_compute_block_batch48< 1, T_FUNCTION, T_NEED_BIAS_COPY>(
                            input_buffer + C_package_size * C_batch48_size * input_package,
                            package_output_ptr,
                            package_bias_ptr,
                            package_weights_ptr + C_package_size * C_max_acc_batch48 * input_package,
                            C_package_size,
                            first_run,
                            last_run);
            break;
        default:
            NN_UNREACHABLE_CODE;
        }

        first_run = false;
    }

    // Remaining input items.
    if (package_remainder > 0 )
    {
        auto package_weights_ptr = weights_ptr;
        auto package_output_ptr = output_ptr;
        auto package_bias_ptr = bias_ptr;
        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            fully_connected_compute_block_batch48<C_max_acc_batch48, T_FUNCTION, T_NEED_BIAS_COPY>(
                        input_buffer + C_package_size * C_batch48_size * num_input_packages,
                        package_output_ptr,
                        package_bias_ptr,
                        package_weights_ptr + C_package_size * C_max_acc_batch48 * num_input_packages,
                        package_remainder,
                        first_run,
                        true);

            // Increment pointers.
            package_output_ptr += C_data_stride_batch48;
            package_weights_ptr += input_width*C_max_acc_batch48;

            if (T_NEED_BIAS_COPY)
            {
                package_bias_ptr += C_max_acc_batch48;
            }
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: fully_connected_compute_block_batch48< 1, T_FUNCTION, T_NEED_BIAS_COPY>(
                        input_buffer + C_package_size * C_batch48_size * num_input_packages,
                        package_output_ptr,
                        package_bias_ptr,
                        package_weights_ptr + C_package_size * C_max_acc_batch48 * num_input_packages,
                        package_remainder,
                        first_run,
                        true);
            break;
        default:
            NN_UNREACHABLE_CODE;
        }
    }

}

template<uint32_t T_SIZE, NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
void fully_connected_compute_block_latency(
    float* input_buffer,
    float* &output_buffer,
    float* &bias_buffer,
    float* &weights_buffer,
    uint32_t input_width,
    uint32_t output_length)
{
    auto output_ptr = output_buffer;
    auto bias_ptr = bias_buffer;
    auto weights_ptr = weights_buffer;

    // We are not using table of registers and unroll pragmas
    // due to compiler which have issues with register allocation
    // and needs special, obvious treatment. Template immediate
    // arguments matching will remove all conditions in this code.
    __m256  acc0, acc1, acc2, acc3, acc4,
            acc5, acc6, acc7, acc8, acc9,
            acc10, acc11, acc12, acc13, acc14;

    if (T_NEED_BIAS_COPY)
    {
        if (T_SIZE >=  1)  acc0 = _mm256_setzero_ps();
        if (T_SIZE >=  2)  acc1 = _mm256_setzero_ps();
        if (T_SIZE >=  3)  acc2 = _mm256_setzero_ps();
        if (T_SIZE >=  4)  acc3 = _mm256_setzero_ps();
        if (T_SIZE >=  5)  acc4 = _mm256_setzero_ps();
        if (T_SIZE >=  6)  acc5 = _mm256_setzero_ps();
        if (T_SIZE >=  7)  acc6 = _mm256_setzero_ps();
        if (T_SIZE >=  8)  acc7 = _mm256_setzero_ps();
        if (T_SIZE >=  9)  acc8 = _mm256_setzero_ps();
        if (T_SIZE >= 10)  acc9 = _mm256_setzero_ps();
        if (T_SIZE >= 11) acc10 = _mm256_setzero_ps();
        if (T_SIZE >= 12) acc11 = _mm256_setzero_ps();
        if (T_SIZE >= 13) acc12 = _mm256_setzero_ps();
        if (T_SIZE >= 14) acc13 = _mm256_setzero_ps();
        if (T_SIZE >= 15) acc14 = _mm256_setzero_ps();
    }
    else
    {
        if (T_SIZE >=  1)  acc0 = _mm256_loadu_ps(output_ptr +  0 * C_simd_width);
        if (T_SIZE >=  2)  acc1 = _mm256_loadu_ps(output_ptr +  1 * C_simd_width);
        if (T_SIZE >=  3)  acc2 = _mm256_loadu_ps(output_ptr +  2 * C_simd_width);
        if (T_SIZE >=  4)  acc3 = _mm256_loadu_ps(output_ptr +  3 * C_simd_width);
        if (T_SIZE >=  5)  acc4 = _mm256_loadu_ps(output_ptr +  4 * C_simd_width);
        if (T_SIZE >=  6)  acc5 = _mm256_loadu_ps(output_ptr +  5 * C_simd_width);
        if (T_SIZE >=  7)  acc6 = _mm256_loadu_ps(output_ptr +  6 * C_simd_width);
        if (T_SIZE >=  8)  acc7 = _mm256_loadu_ps(output_ptr +  7 * C_simd_width);
        if (T_SIZE >=  9)  acc8 = _mm256_loadu_ps(output_ptr +  8 * C_simd_width);
        if (T_SIZE >= 10)  acc9 = _mm256_loadu_ps(output_ptr +  9 * C_simd_width);
        if (T_SIZE >= 11) acc10 = _mm256_loadu_ps(output_ptr + 10 * C_simd_width);
        if (T_SIZE >= 12) acc11 = _mm256_loadu_ps(output_ptr + 11 * C_simd_width);
        if (T_SIZE >= 13) acc12 = _mm256_loadu_ps(output_ptr + 12 * C_simd_width);
        if (T_SIZE >= 14) acc13 = _mm256_loadu_ps(output_ptr + 13 * C_simd_width);
        if (T_SIZE >= 15) acc14 = _mm256_loadu_ps(output_ptr + 14 * C_simd_width);
    }


    auto input_ptr = &input_buffer[0];

    const auto input_ptr_end = &input_buffer[input_width];

    while (input_ptr < input_ptr_end)
    {
        // Do MADs.
        __m256 input = _mm256_broadcast_ss(input_ptr);
        if (T_SIZE >=  1)  acc0 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr +  0 * C_simd_width),  acc0);
        if (T_SIZE >=  2)  acc1 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr +  1 * C_simd_width),  acc1);
        if (T_SIZE >=  3)  acc2 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr +  2 * C_simd_width),  acc2);
        if (T_SIZE >=  4)  acc3 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr +  3 * C_simd_width),  acc3);
        if (T_SIZE >=  5)  acc4 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr +  4 * C_simd_width),  acc4);
        if (T_SIZE >=  6)  acc5 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr +  5 * C_simd_width),  acc5);
        if (T_SIZE >=  7)  acc6 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr +  6 * C_simd_width),  acc6);
        if (T_SIZE >=  8)  acc7 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr +  7 * C_simd_width),  acc7);
        if (T_SIZE >=  9)  acc8 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr +  8 * C_simd_width),  acc8);
        if (T_SIZE >= 10)  acc9 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr +  9 * C_simd_width),  acc9);
        if (T_SIZE >= 11) acc10 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 10 * C_simd_width), acc10);
        if (T_SIZE >= 12) acc11 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 11 * C_simd_width), acc11);
        if (T_SIZE >= 13) acc12 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 12 * C_simd_width), acc12);
        if (T_SIZE >= 14) acc13 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 13 * C_simd_width), acc13);
        if (T_SIZE >= 15) acc14 = _mm256_fmadd_ps(input, _mm256_loadu_ps(weights_ptr + 14 * C_simd_width), acc14);

        // Increment pointers.
        ++input_ptr;
        weights_ptr += output_length;
    }

    if (T_NEED_BIAS_COPY)
    {
        // Add biases.
        if (T_SIZE >=  1)  acc0 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr +  0 * C_simd_width),  acc0);
        if (T_SIZE >=  2)  acc1 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr +  1 * C_simd_width),  acc1);
        if (T_SIZE >=  3)  acc2 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr +  2 * C_simd_width),  acc2);
        if (T_SIZE >=  4)  acc3 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr +  3 * C_simd_width),  acc3);
        if (T_SIZE >=  5)  acc4 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr +  4 * C_simd_width),  acc4);
        if (T_SIZE >=  6)  acc5 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr +  5 * C_simd_width),  acc5);
        if (T_SIZE >=  7)  acc6 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr +  6 * C_simd_width),  acc6);
        if (T_SIZE >=  8)  acc7 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr +  7 * C_simd_width),  acc7);
        if (T_SIZE >=  9)  acc8 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr +  8 * C_simd_width),  acc8);
        if (T_SIZE >= 10)  acc9 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr +  9 * C_simd_width),  acc9);
        if (T_SIZE >= 11) acc10 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 10 * C_simd_width), acc10);
        if (T_SIZE >= 12) acc11 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 11 * C_simd_width), acc11);
        if (T_SIZE >= 13) acc12 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 12 * C_simd_width), acc12);
        if (T_SIZE >= 14) acc13 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 13 * C_simd_width), acc13);
        if (T_SIZE >= 15) acc14 = _mm256_add_ps(_mm256_loadu_ps(bias_ptr + 14 * C_simd_width), acc14);
    }

    if (T_FUNCTION == NN_ACTIVATION_FUNCTION_RELU)
    {
        // Perform ReLU.
        if (T_SIZE >=  1)  acc0 = _mm256_max_ps(_mm256_setzero_ps(),  acc0);
        if (T_SIZE >=  2)  acc1 = _mm256_max_ps(_mm256_setzero_ps(),  acc1);
        if (T_SIZE >=  3)  acc2 = _mm256_max_ps(_mm256_setzero_ps(),  acc2);
        if (T_SIZE >=  4)  acc3 = _mm256_max_ps(_mm256_setzero_ps(),  acc3);
        if (T_SIZE >=  5)  acc4 = _mm256_max_ps(_mm256_setzero_ps(),  acc4);
        if (T_SIZE >=  6)  acc5 = _mm256_max_ps(_mm256_setzero_ps(),  acc5);
        if (T_SIZE >=  7)  acc6 = _mm256_max_ps(_mm256_setzero_ps(),  acc6);
        if (T_SIZE >=  8)  acc7 = _mm256_max_ps(_mm256_setzero_ps(),  acc7);
        if (T_SIZE >=  9)  acc8 = _mm256_max_ps(_mm256_setzero_ps(),  acc8);
        if (T_SIZE >= 10)  acc9 = _mm256_max_ps(_mm256_setzero_ps(),  acc9);
        if (T_SIZE >= 11) acc10 = _mm256_max_ps(_mm256_setzero_ps(), acc10);
        if (T_SIZE >= 12) acc11 = _mm256_max_ps(_mm256_setzero_ps(), acc11);
        if (T_SIZE >= 13) acc12 = _mm256_max_ps(_mm256_setzero_ps(), acc12);
        if (T_SIZE >= 14) acc13 = _mm256_max_ps(_mm256_setzero_ps(), acc13);
        if (T_SIZE >= 15) acc14 = _mm256_max_ps(_mm256_setzero_ps(), acc14);
    }

    // Store results.
    if (T_SIZE >=  1) _mm256_storeu_ps(output_ptr +  0 * C_simd_width,  acc0);
    if (T_SIZE >=  2) _mm256_storeu_ps(output_ptr +  1 * C_simd_width,  acc1);
    if (T_SIZE >=  3) _mm256_storeu_ps(output_ptr +  2 * C_simd_width,  acc2);
    if (T_SIZE >=  4) _mm256_storeu_ps(output_ptr +  3 * C_simd_width,  acc3);
    if (T_SIZE >=  5) _mm256_storeu_ps(output_ptr +  4 * C_simd_width,  acc4);
    if (T_SIZE >=  6) _mm256_storeu_ps(output_ptr +  5 * C_simd_width,  acc5);
    if (T_SIZE >=  7) _mm256_storeu_ps(output_ptr +  6 * C_simd_width,  acc6);
    if (T_SIZE >=  8) _mm256_storeu_ps(output_ptr +  7 * C_simd_width,  acc7);
    if (T_SIZE >=  9) _mm256_storeu_ps(output_ptr +  8 * C_simd_width,  acc8);
    if (T_SIZE >= 10) _mm256_storeu_ps(output_ptr +  9 * C_simd_width,  acc9);
    if (T_SIZE >= 11) _mm256_storeu_ps(output_ptr + 10 * C_simd_width, acc10);
    if (T_SIZE >= 12) _mm256_storeu_ps(output_ptr + 11 * C_simd_width, acc11);
    if (T_SIZE >= 13) _mm256_storeu_ps(output_ptr + 12 * C_simd_width, acc12);
    if (T_SIZE >= 14) _mm256_storeu_ps(output_ptr + 13 * C_simd_width, acc13);
    if (T_SIZE >= 15) _mm256_storeu_ps(output_ptr + 14 * C_simd_width, acc14);

    output_buffer += C_simd_width*T_SIZE;
    weights_buffer += C_simd_width*T_SIZE;

    if (T_NEED_BIAS_COPY)
    {
        bias_buffer += C_simd_width*T_SIZE;
    }
}

template<uint32_t T_NUM_ITERATIONS, NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
void fully_connected_compute_subsimd_latency(
    float* input_buffer,
    float* &output_buffer,
    float* &bias_buffer,
    float* &weights_buffer,
    uint32_t input_width,
    uint32_t output_length)
{
    for (auto iteration = 0u; iteration < T_NUM_ITERATIONS; ++iteration)
    {
        auto output_ptr = output_buffer;
        auto bias_ptr = bias_buffer;
        auto weights_ptr = weights_buffer;

        float acc0 = 0.0f;
        if (!T_NEED_BIAS_COPY)
        {
            acc0 = *output_ptr;
        }

        auto input_ptr = &input_buffer[0];
        const auto input_ptr_end = &input_buffer[input_width];

        while (input_ptr < input_ptr_end)
        {
            // Do MADs.
            acc0 += (*input_ptr) * (*weights_ptr);

            // Increment pointers.
            ++input_ptr;
            weights_ptr += output_length;
        }

        if (T_NEED_BIAS_COPY)
        {
            // Add biases.
            acc0 += *bias_ptr;
        }

        if (T_FUNCTION == NN_ACTIVATION_FUNCTION_RELU)
        {
            // Perform ReLU.
            acc0 = std::max(0.0f, acc0);
        }

        // Store results.
        *output_ptr = acc0;

        ++output_buffer;
        ++bias_buffer;
        ++weights_buffer;
    }
}

template <NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
void fully_connected_f32::run_fully_connected_work_item_internal_latency(const nn::workload_data<> *input,
                                                                         const nn::workload_data<> *weights,
                                                                         const nn::workload_data<> *bias,
                                                                         nn::workload_data<> *output) {
    const auto input_width = input->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_width = output->view_end.t[NN_DATA_COORD_x] - output->view_begin.t[NN_DATA_COORD_x] + 1;
    const auto output_length = output->parent->lengths.t[NN_DATA_COORD_x];

    const auto num_full_blocks = output_width / C_data_stride_batch1;
    const auto partial_block_size = (output_width / C_simd_width) % C_max_acc_batch1;
    const auto subsimd_block_size = output_width % C_simd_width;

    auto input_buffer = static_cast<float*>(input->parent->data_buffer);
    auto output_buffer = static_cast<float*>(output->parent->data_buffer);
    auto weights_buffer = static_cast<float*>(weights->parent->data_buffer);

    // Output views.
    const auto output_view_start = output->view_begin.t[NN_DATA_COORD_x];

    auto weights_ptr = &weights_buffer[output_view_start];
    auto output_ptr = &output_buffer[output_view_start];

    float* bias_ptr = nullptr;
    if (T_NEED_BIAS_COPY)
    {
        auto biases_buffer = static_cast<float*>(bias->parent->data_buffer);
        bias_ptr = &biases_buffer[output_view_start];
    }

    for (auto block = 0u; block < num_full_blocks; ++block)
    {
        // Run computation.
        fully_connected_compute_block_latency<C_max_acc_batch1, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length);
    }

    switch (partial_block_size)
    {
    case  0: break;
    case  1: fully_connected_compute_block_latency< 1, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case  2: fully_connected_compute_block_latency< 2, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case  3: fully_connected_compute_block_latency< 3, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case  4: fully_connected_compute_block_latency< 4, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case  5: fully_connected_compute_block_latency< 5, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case  6: fully_connected_compute_block_latency< 6, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case  7: fully_connected_compute_block_latency< 7, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case  8: fully_connected_compute_block_latency< 8, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case  9: fully_connected_compute_block_latency< 9, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 10: fully_connected_compute_block_latency<10, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 11: fully_connected_compute_block_latency<11, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 12: fully_connected_compute_block_latency<12, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 13: fully_connected_compute_block_latency<13, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 14: fully_connected_compute_block_latency<14, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    default:
        NN_UNREACHABLE_CODE;
    }

    switch (subsimd_block_size)
    {
    case 0: break;
    case 1: fully_connected_compute_subsimd_latency<1, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 2: fully_connected_compute_subsimd_latency<2, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 3: fully_connected_compute_subsimd_latency<3, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 4: fully_connected_compute_subsimd_latency<4, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 5: fully_connected_compute_subsimd_latency<5, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 6: fully_connected_compute_subsimd_latency<6, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    case 7: fully_connected_compute_subsimd_latency<7, T_FUNCTION, T_NEED_BIAS_COPY>(input_buffer, output_ptr, bias_ptr, weights_ptr, input_width, output_length); break;
    default:
        NN_UNREACHABLE_CODE;
    }
}

template <NN_ACTIVATION_FUNCTION T_FUNCTION, bool T_NEED_BIAS_COPY>
void fully_connected_f32::choose_fully_connected_work_item_batching_mode(const nn::workload_data<> *input,
                                                                         const nn::workload_data<> *weights,
                                                                         const nn::workload_data<> *bias,
                                                                         nn::workload_data<> *output) {
    switch (batch_size)
    {
    case 1:
        run_fully_connected_work_item_internal_latency<T_FUNCTION, T_NEED_BIAS_COPY>(input, weights, bias, output);
        break;
    case 8:
        run_fully_connected_work_item_internal_batch8<T_FUNCTION, T_NEED_BIAS_COPY>(input, weights, bias, output);
        break;
    case 48:
        run_fully_connected_work_item_internal_batch48<T_FUNCTION, T_NEED_BIAS_COPY>(input, weights, bias, output);
        break;
    default:
        break;
    }
}

template <bool T_NEED_BIAS_COPY>
void fully_connected_f32::choose_fully_connected_work_item_activation(const nn::workload_data<> *input,
                                                                      const nn::workload_data<> *weights,
                                                                      const nn::workload_data<> *bias,
                                                                      nn::workload_data<> *output) {
    switch (activation.function)
    {
    case NN_ACTIVATION_FUNCTION_NONE:
        choose_fully_connected_work_item_batching_mode<NN_ACTIVATION_FUNCTION_NONE, T_NEED_BIAS_COPY>(input, weights, bias, output);
        break;
    case NN_ACTIVATION_FUNCTION_RELU:
        choose_fully_connected_work_item_batching_mode<NN_ACTIVATION_FUNCTION_RELU, T_NEED_BIAS_COPY>(input, weights, bias, output);
        break;
    default:
        break;
    }
}

void fully_connected_f32::run_fully_connected_work_item(const nn::workload_data<> *input,
                                                        const nn::workload_data<> *weights,
                                                        const nn::workload_data<> *bias,
                                                        nn::workload_data<> *output) {
    bool need_bias_copy = (bias != nullptr);

    if (need_bias_copy)
    {
        choose_fully_connected_work_item_activation<true>(input, weights, bias, output);
    }
    else
    {
        choose_fully_connected_work_item_activation<false>(input, weights, bias, output);
    }
}

std::vector<nn_workload_data_t *> fully_connected_f32::create_parameters(bool allocate_delta)
{
    if(batch_size == 1){
        return{nn::data_helper<NN_WORKLOAD_DATA_TAG_OI, nn::layout_oi_f32>::create(device, num_input, num_output, allocate_delta),
                nn::data_helper<NN_WORKLOAD_DATA_TAG_O, nn::layout_o_f32>::create(device, num_output, allocate_delta)};
    }
    else{
        const uint32_t C_max_accumulators = (batch_size == 8) ? 13 : 2;
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_OBLOCKIO, nn::layout_oblockio_f32>::create(
                    device, C_max_accumulators, num_input, num_output, allocate_delta),
                    nn::data_helper<NN_WORKLOAD_DATA_TAG_O, nn::layout_o_f32>::create(device, num_output, allocate_delta)};
    }
}

std::vector<nn_workload_data_t *> fully_connected_f32::create_inputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::create(device, num_input, batch_size, allocate_delta)};
}

std::vector<nn_workload_data_t *> fully_connected_f32::create_outputs(bool allocate_delta)
{
    return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::create(device, num_output, batch_size, allocate_delta)};
}

const nn_workload_data_layout_t& fully_connected_f32::in_out_layout = nn::layout_t<nn::layout_nxyzpq_f32>::layout;

struct fully_connected_f32_request_handle {
    fully_connected_f32 *primitive;
    const nn::workload_data<> *input;
    const nn::workload_data<> *weights;
    const nn::workload_data<> *bias;
    nn::workload_data<> *output;
};

struct fully_connected_f32_backward_input_request_handle {
    fully_connected_f32 *primitive;
    const nn::workload_data<> *backward_input;
    const nn::workload_data<> *forward_weights;
    nn::workload_data<> *output;
};

struct fully_connected_f32_backward_weight_request_handle {
    fully_connected_f32 *primitive;
    const nn::workload_data<> *forward_input;
    const nn::workload_data<> *backward_input;
    nn::workload_data<> *output;
};

struct fully_connected_f32_backward_bias_request_handle {
    fully_connected_f32 *primitive;
    const nn::workload_data<> *backward_input;
    const nn::workload_data<> *bias_multiplier;
    nn::workload_data<> *output;
};

void sum_over_simd(__m256& acc)
{
    acc = _mm256_hadd_ps(acc, acc);
    acc = _mm256_permutevar8x32_ps(acc, _mm256_set_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    acc = _mm256_hadd_ps(acc, acc);
    acc = _mm256_hadd_ps(acc, acc);
}

void run_backward_weights_delta(
    const nn::workload_data<> *forward_input_view,
    const nn::workload_data<> *forward_output_view,
    nn::workload_data<> *forward_weights_delta_view)
{
    const auto& in_begin = forward_input_view->view_begin;
    const auto& in_end = forward_input_view->view_end;
    const auto& out_begin = forward_output_view->view_begin;
    const auto& out_end = forward_output_view->view_end;

    const auto& wght_begin = forward_weights_delta_view->view_begin;
    const auto& wght_end = forward_weights_delta_view->view_end;

    nn::workload_data<nn::layout_f32> forward_input_buffer(NN_WORKLOAD_DATA_TAG_UNKNOWN, forward_input_view->parent->data_buffer, forward_input_view->parent->lengths, forward_input_view->parent->layout);
    nn::workload_data<nn::layout_f32> backward_input_buffer(NN_WORKLOAD_DATA_TAG_UNKNOWN, forward_output_view->parent->delta_buffer, forward_output_view->parent->lengths, forward_output_view->parent->layout);
    nn::workload_data<nn::layout_f32> backward_weights_delta_buffer(NN_WORKLOAD_DATA_TAG_UNKNOWN, forward_weights_delta_view->parent->delta_buffer, forward_weights_delta_view->parent->lengths, forward_weights_delta_view->parent->layout);

    auto batch_size = forward_input_buffer.parent->lengths.t[NN_DATA_COORD_n];

    const uint32_t C_max_accumulators = (batch_size == 8) ? 13 : 2;

    auto remainder_of_slice = forward_output_view->parent->lengths.t[NN_DATA_COORD_x] % C_max_accumulators;

    // For backpropagation in [m] layer, we require s[m] = F'[m](n[m]) * W[m+1]^T * s[m+1] <=> s[m] = F'[m](n[m]) * U[m+1] => U[m+1] = W[m+1]^T * s[m+1].
    // Backpropagation for next layer already has computed U[m+1], now we need to compute F'[m](n[m]) part.
    // If there is no activation, then F'[m](n[m]) = 1. It means s[m] = U[m+1], so we'll use just s[m].
    // We have access to both s[m] and W[m] and so, we can compute U[m] = W[m]^T * s[m] required for previous layer.
    // Also compute weights gradient as s[m] * a[m-1]^T and compute bias gradients as s[m].
    // We already have s[m], and a[m-1] is just raw input to this layer.
    if (batch_size == 1)
    {
        for (uint32_t output_element = wght_begin.t[NN_DATA_COORD_y]; output_element <= wght_end.t[NN_DATA_COORD_y]; ++output_element)
        {
            for (uint32_t input_element = wght_begin.t[NN_DATA_COORD_x]; input_element <= wght_end.t[NN_DATA_COORD_x]; ++input_element)
            {
                float acc = 0.0f;

                for (uint32_t batch = in_begin.t[NN_DATA_COORD_n];
                    batch <= in_end.t[NN_DATA_COORD_n];
                    ++batch)
                {
                    acc +=   backward_input_buffer(batch, output_element, 0, 0, 0, 0)
                            * forward_input_buffer(batch, input_element, 0, 0, 0, 0);
                }

                backward_weights_delta_buffer(0, input_element, output_element, 0, 0, 0) = acc;
            }
        }
    }
    else if(batch_size == 8)
    {
        for (uint32_t output_major_element = wght_begin.t[NN_DATA_COORD_q]; output_major_element <= wght_end.t[NN_DATA_COORD_q]; ++output_major_element)
        {
            for (uint32_t output_minor_element = wght_begin.t[NN_DATA_COORD_p]; output_minor_element <= wght_end.t[NN_DATA_COORD_p]; ++output_minor_element)
            {
                // If its last major element, then stop after doing all required subelements.
                if(output_major_element == wght_end.t[NN_DATA_COORD_q] && remainder_of_slice != 0 && output_minor_element >= remainder_of_slice)
                   break;

                auto output_element = output_minor_element + output_major_element * C_max_accumulators;

                for (uint32_t input_element = wght_begin.t[NN_DATA_COORD_x]; input_element <= wght_end.t[NN_DATA_COORD_x]; ++input_element)
                {
                    float acc = 0.0f;

                    for (uint32_t batch = in_begin.t[NN_DATA_COORD_n];
                        batch <= in_end.t[NN_DATA_COORD_n];
                        ++batch)
                    {
                        acc += backward_input_buffer(batch, output_element, 0, 0, 0, 0)
                            * forward_input_buffer(batch, input_element, 0, 0, 0, 0);
                    }

                    backward_weights_delta_buffer(0, input_element, 0, 0, output_minor_element, output_major_element) = acc;
                }
            }
        }
    }
    else if(batch_size == 48)
    {
        // Optimized version.
        auto backward_input_ptr = reinterpret_cast<float*>(forward_output_view->parent->delta_buffer);
        auto forward_input_ptr = reinterpret_cast<float*>(forward_input_view->parent->data_buffer);
        auto backward_weights_delta_ptr = reinterpret_cast<float*>(forward_weights_delta_view->parent->delta_buffer);

        auto input_size = forward_weights_delta_view->parent->lengths.t[NN_DATA_COORD_x];

        for (uint32_t output_major_element = wght_begin.t[NN_DATA_COORD_q]; output_major_element <= wght_end.t[NN_DATA_COORD_q]; ++output_major_element)
        {
            for (uint32_t output_minor_element = wght_begin.t[NN_DATA_COORD_p]; output_minor_element <= wght_end.t[NN_DATA_COORD_p]; ++output_minor_element)
            {
                // If its last major element, then stop after doing all required subelements.
                if(output_major_element == wght_end.t[NN_DATA_COORD_q] && remainder_of_slice != 0 && output_minor_element >= remainder_of_slice)
                   break;

                auto output_element = output_minor_element + output_major_element * C_max_accumulators;

                auto backward_input_ptr_adjusted = backward_input_ptr + output_element * C_batch48_size;
                auto forward_input_ptr_adjusted = forward_input_ptr + wght_begin.t[NN_DATA_COORD_x] * C_batch48_size;
                auto backward_weights_delta_ptr_adjusted = backward_weights_delta_ptr
                                                            + output_minor_element
                                                            + wght_begin.t[NN_DATA_COORD_x] * C_max_accumulators
                                                            + output_major_element * C_max_accumulators * input_size;

                for (uint32_t input_element = wght_begin.t[NN_DATA_COORD_x]; input_element <= wght_end.t[NN_DATA_COORD_x]; ++input_element)
                {
                    __m256 acc0 = _mm256_load_ps(backward_input_ptr_adjusted + 0 * C_simd_width);
                    __m256 acc1 = _mm256_load_ps(backward_input_ptr_adjusted + 1 * C_simd_width);
                    __m256 acc2 = _mm256_load_ps(backward_input_ptr_adjusted + 2 * C_simd_width);
                    __m256 acc3 = _mm256_load_ps(backward_input_ptr_adjusted + 3 * C_simd_width);
                    __m256 acc4 = _mm256_load_ps(backward_input_ptr_adjusted + 4 * C_simd_width);
                    __m256 acc5 = _mm256_load_ps(backward_input_ptr_adjusted + 5 * C_simd_width);

                    acc0 = _mm256_mul_ps(acc0, _mm256_load_ps(forward_input_ptr_adjusted + 0 * C_simd_width));
                    acc1 = _mm256_mul_ps(acc1, _mm256_load_ps(forward_input_ptr_adjusted + 1 * C_simd_width));
                    acc2 = _mm256_mul_ps(acc2, _mm256_load_ps(forward_input_ptr_adjusted + 2 * C_simd_width));
                    acc3 = _mm256_mul_ps(acc3, _mm256_load_ps(forward_input_ptr_adjusted + 3 * C_simd_width));
                    acc4 = _mm256_mul_ps(acc4, _mm256_load_ps(forward_input_ptr_adjusted + 4 * C_simd_width));
                    acc5 = _mm256_mul_ps(acc5, _mm256_load_ps(forward_input_ptr_adjusted + 5 * C_simd_width));

                    acc0 = _mm256_add_ps(acc0, acc1);
                    acc2 = _mm256_add_ps(acc2, acc3);
                    acc4 = _mm256_add_ps(acc4, acc5);

                    acc0 = _mm256_add_ps(acc0, acc2);
                    acc0 = _mm256_add_ps(acc0, acc4);

                    sum_over_simd(acc0);

                    _mm256_maskstore_ps(backward_weights_delta_ptr_adjusted, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF), acc0);

                    backward_weights_delta_ptr_adjusted += C_max_accumulators;
                    forward_input_ptr_adjusted += C_batch48_size;
                }
            }
        }
    }
}

template <uint32_t T_block_size>
void backward_bias_delta_inner_loop_batch48(
    float* backward_input_buffer,
    float* backward_bias_delta_buffer)
{
    const auto C_batch = 48;
    const auto C_load_offset = 8;

    __m256 acc0[T_block_size];
    __m256 acc2[T_block_size];
    __m256 acc4[T_block_size];

#pragma unroll (T_block_size)
    for(uint32_t block = 0; block < T_block_size; ++block)
    {
        acc0[block] = _mm256_load_ps(backward_input_buffer + 0 * C_load_offset + block * C_batch);
        acc2[block] = _mm256_load_ps(backward_input_buffer + 2 * C_load_offset + block * C_batch);
        acc4[block] = _mm256_load_ps(backward_input_buffer + 4 * C_load_offset + block * C_batch);
    }

#pragma unroll (T_block_size)
    for(uint32_t block = 0; block < T_block_size; ++block)
    {
        acc0[block] = _mm256_add_ps(acc0[block], _mm256_load_ps(backward_input_buffer + 1 * C_load_offset + block * C_batch));
        acc2[block] = _mm256_add_ps(acc2[block], _mm256_load_ps(backward_input_buffer + 3 * C_load_offset + block * C_batch));
        acc4[block] = _mm256_add_ps(acc4[block], _mm256_load_ps(backward_input_buffer + 5 * C_load_offset + block * C_batch));
    }

#pragma unroll (T_block_size)
    for(uint32_t block = 0; block < T_block_size; ++block)
    {
        acc0[block] = _mm256_add_ps(acc0[block], acc2[block]);
        acc0[block] = _mm256_add_ps(acc0[block], acc4[block]);
    }

#pragma unroll (T_block_size)
    for(uint32_t block = 0; block < T_block_size; ++block)
    {
        sum_over_simd(acc0[block]);
    }

#pragma unroll (T_block_size)
    for(uint32_t block = 0; block < T_block_size; ++block)
    {
        _mm256_maskstore_ps(backward_bias_delta_buffer + block, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF), acc0[block]);
    }
}

void run_backward_bias_delta(
    const nn::workload_data<> *forward_output_view,
    const nn::workload_data<> *bias_multiplier,
    nn::workload_data<> *forward_bias_delta_view)
{
    const auto& in_begin = forward_output_view->view_begin;
    const auto& in_end = forward_output_view->view_end;
    const auto& out_begin = forward_bias_delta_view->view_begin;
    const auto& out_end = forward_bias_delta_view->view_end;


    // For backpropagation in [m] layer, we require s[m] = F'[m](n[m]) * W[m+1]^T * s[m+1] <=> s[m] = F'[m](n[m]) * U[m+1] => U[m+1] = W[m+1]^T * s[m+1].
    // Backpropagation for next layer already has computed U[m+1], now we need to compute F'[m](n[m]) part.
    // If there is no activation, then F'[m](n[m]) = 1. It means s[m] = U[m+1], so we'll use just s[m].
    // We have access to both s[m] and W[m] and so, we can compute U[m] = W[m]^T * s[m] required for previous layer.
    // Also compute weights gradient as s[m] * a[m-1]^T and compute bias gradients as s[m].
    // We already have s[m], and a[m-1] is just raw input to this layer.
    const auto batch_size = forward_output_view->parent->lengths.t[NN_DATA_COORD_n];

    // Naive default version.
    if(batch_size != 48)
    {
        nn::workload_data<nn::layout_f32> backward_input_buffer(NN_WORKLOAD_DATA_TAG_UNKNOWN, forward_output_view->parent->delta_buffer, forward_output_view->parent->lengths, forward_output_view->parent->layout);
        nn::workload_data<nn::layout_f32> backward_bias_delta_buffer(NN_WORKLOAD_DATA_TAG_UNKNOWN, forward_bias_delta_view->parent->delta_buffer, forward_bias_delta_view->parent->lengths, forward_bias_delta_view->parent->layout);

        for (uint32_t output_element = out_begin.t[NN_DATA_COORD_x]; output_element <= out_end.t[NN_DATA_COORD_x]; ++output_element)
        {
            float acc = 0.0f;

            for (uint32_t batch = in_begin.t[NN_DATA_COORD_n];
                batch <= in_end.t[NN_DATA_COORD_n];
                ++batch)
            {
                acc += backward_input_buffer(batch, output_element, 0, 0, 0, 0);
            }

            backward_bias_delta_buffer(0, output_element, 0, 0, 0, 0) = acc;
        }
    }
    else
    {
        // Naive+ batch 48 version.
        auto backward_input_buffer = reinterpret_cast<float*>(forward_output_view->parent->delta_buffer);
        auto backward_bias_delta_buffer = reinterpret_cast<float*>(forward_bias_delta_view->parent->delta_buffer);

        const auto C_max_acc = 3;
        const auto C_batch = 48;

        const auto full_passes = forward_bias_delta_view->get_length(NN_DATA_COORD_x) / C_max_acc;
        const auto partial_pass_size = forward_bias_delta_view->get_length(NN_DATA_COORD_x) % C_max_acc;

        backward_input_buffer += out_begin.t[NN_DATA_COORD_x] * C_batch;
        backward_bias_delta_buffer += out_begin.t[NN_DATA_COORD_x];

        const auto end_of_full_pass = backward_bias_delta_buffer + full_passes * C_max_acc;

#pragma forceinline recursive
        {
            while (backward_bias_delta_buffer < end_of_full_pass)
            {
                backward_bias_delta_inner_loop_batch48<3>(backward_input_buffer, backward_bias_delta_buffer);

                backward_input_buffer += C_max_acc * C_batch;
                backward_bias_delta_buffer += C_max_acc;
            }

            switch(partial_pass_size)
            {
            case 1: backward_bias_delta_inner_loop_batch48<1>(backward_input_buffer, backward_bias_delta_buffer); break;
            case 2: backward_bias_delta_inner_loop_batch48<2>(backward_input_buffer, backward_bias_delta_buffer); break;
            }
        }
    }
}

void run_backward_input_delta(
    const nn::workload_data<> *forward_output_view,
    const nn::workload_data<> *forward_weights_view,
    nn::workload_data<> *forward_input_view)
{
    const auto& in_begin = forward_output_view->view_begin;
    const auto& in_end = forward_output_view->view_end;
    const auto& out_begin = forward_input_view->view_begin;
    const auto& out_end = forward_input_view->view_end;

    nn::workload_data<nn::layout_f32> forward_weights_buffer(NN_WORKLOAD_DATA_TAG_UNKNOWN, forward_weights_view->parent->data_buffer, forward_weights_view->parent->lengths, forward_weights_view->parent->layout);
    nn::workload_data<nn::layout_f32> backward_input_buffer(NN_WORKLOAD_DATA_TAG_UNKNOWN, forward_output_view->parent->delta_buffer, forward_output_view->parent->lengths, forward_output_view->parent->layout);
    nn::workload_data<nn::layout_f32> backward_output_buffer(NN_WORKLOAD_DATA_TAG_UNKNOWN, forward_input_view->parent->delta_buffer, forward_input_view->parent->lengths, forward_input_view->parent->layout);

    auto batch_size = forward_output_view->parent->lengths.t[NN_DATA_COORD_n];

    // For backpropagation in [m] layer, we require s[m] = F'[m](n[m]) * W[m+1]^T * s[m+1] <=> s[m] = F'[m](n[m]) * U[m+1] => U[m+1] = W[m+1]^T * s[m+1].
    // Backpropagation for next layer already has computed U[m+1], now we need to compute F'[m](n[m]) part.
    // If there is no activation, then F'[m](n[m]) = 1. It means s[m] = U[m+1], so we'll use just s[m].
    // We have access to both s[m] and W[m] and so, we can compute U[m] = W[m]^T * s[m] required for previous layer.
    // Also compute weights gradient as s[m] * a[m-1]^T and compute bias gradients as s[m].
    // We already have s[m], and a[m-1] is just raw input to this layer.
    if (batch_size == 1)
    {
        for (uint32_t batch = out_begin.t[NN_DATA_COORD_n];
                batch <= out_end.t[NN_DATA_COORD_n];
                ++batch)
        {
            for (uint32_t input_element = out_begin.t[NN_DATA_COORD_x]; input_element <= out_end.t[NN_DATA_COORD_x]; ++input_element)
            {
                float acc = 0.0f;

                for (uint32_t output_element = in_begin.t[NN_DATA_COORD_x]; output_element <= in_end.t[NN_DATA_COORD_x]; ++output_element)
                {
                    acc +=   backward_input_buffer(batch, output_element, 0, 0, 0, 0)
                           * forward_weights_buffer(0, input_element, output_element, 0, 0, 0);
                }

                backward_output_buffer(batch, input_element, 0, 0, 0, 0) = acc;
            }
        }
    }
    else if(batch_size == 8)
    {
        const uint32_t C_max_accumulators = 13;

        for (uint32_t batch = out_begin.t[NN_DATA_COORD_n];
                batch <= out_end.t[NN_DATA_COORD_n];
                ++batch)
        {
            for (uint32_t input_element = out_begin.t[NN_DATA_COORD_x]; input_element <= out_end.t[NN_DATA_COORD_x]; ++input_element)
            {
                float acc = 0.0f;

                for (uint32_t output_element = in_begin.t[NN_DATA_COORD_x]; output_element <= in_end.t[NN_DATA_COORD_x]; ++output_element)
                {
                    acc +=   backward_input_buffer(batch, output_element, 0, 0, 0, 0)
                            * forward_weights_buffer(0, input_element, 0, 0, output_element % C_max_accumulators, output_element / C_max_accumulators);
                }

                backward_output_buffer(batch, input_element, 0, 0, 0, 0) = acc;
            }
        }
    }
    else if(batch_size == 48)
    {
        const uint32_t C_max_accumulators = 2;
        const auto input_size = forward_weights_view->parent->lengths.t[NN_DATA_COORD_x];

        auto backward_input_ptr = reinterpret_cast<float*>(forward_output_view->parent->delta_buffer);
        auto forward_weights_ptr = reinterpret_cast<float*>(forward_weights_view->parent->data_buffer);
        auto backward_output_ptr = reinterpret_cast<float*>(forward_input_view->parent->delta_buffer);

        for (uint32_t input_element = out_begin.t[NN_DATA_COORD_x]; input_element <= out_end.t[NN_DATA_COORD_x]; ++input_element)
        {
            auto backward_output_ptr_adjusted = backward_output_ptr + input_element * C_batch48_size;
            auto backward_input_ptr_adjusted = backward_input_ptr + in_begin.t[NN_DATA_COORD_x] * C_batch48_size;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();

            for (uint32_t output_element = in_begin.t[NN_DATA_COORD_x]; output_element <= in_end.t[NN_DATA_COORD_x]; ++output_element)
            {
                __m256 weight = _mm256_broadcast_ss(forward_weights_ptr
                                                    + output_element % C_max_accumulators
                                                    + input_element * C_max_accumulators
                                                    + output_element / C_max_accumulators * C_max_accumulators * input_size);

                acc0 = _mm256_fmadd_ps(_mm256_load_ps(backward_input_ptr_adjusted + 0 * C_simd_width), weight, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_load_ps(backward_input_ptr_adjusted + 1 * C_simd_width), weight, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_load_ps(backward_input_ptr_adjusted + 2 * C_simd_width), weight, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_load_ps(backward_input_ptr_adjusted + 3 * C_simd_width), weight, acc3);
                acc4 = _mm256_fmadd_ps(_mm256_load_ps(backward_input_ptr_adjusted + 4 * C_simd_width), weight, acc4);
                acc5 = _mm256_fmadd_ps(_mm256_load_ps(backward_input_ptr_adjusted + 5 * C_simd_width), weight, acc5);

                backward_input_ptr_adjusted += C_batch48_size;
            }

            _mm256_store_ps(backward_output_ptr_adjusted + 0 * C_simd_width, acc0);
            _mm256_store_ps(backward_output_ptr_adjusted + 1 * C_simd_width, acc1);
            _mm256_store_ps(backward_output_ptr_adjusted + 2 * C_simd_width, acc2);
            _mm256_store_ps(backward_output_ptr_adjusted + 3 * C_simd_width, acc3);
            _mm256_store_ps(backward_output_ptr_adjusted + 4 * C_simd_width, acc4);
            _mm256_store_ps(backward_output_ptr_adjusted + 5 * C_simd_width, acc5);
        }
    }
}


void unpack_fully_connected_callback_handle(void *void_handle) {
    auto handle = reinterpret_cast<fully_connected_f32_request_handle *>(void_handle);
    handle->primitive->run_fully_connected_work_item(handle->input, handle->weights, handle->bias, handle->output);
}

void unpack_fully_connected_callback_handle_backward_bias(void *void_handle) {
    auto handle = reinterpret_cast<fully_connected_f32_backward_bias_request_handle *>(void_handle);
    run_backward_bias_delta(handle->backward_input, handle->bias_multiplier, handle->output);
}

void unpack_fully_connected_callback_handle_backward_weight(void *void_handle) {
    auto handle = reinterpret_cast<fully_connected_f32_backward_weight_request_handle *>(void_handle);
    run_backward_weights_delta(handle->forward_input, handle->backward_input, handle->output);
}

void unpack_fully_connected_callback_handle_backward_input(void *void_handle) {
    auto handle = reinterpret_cast<fully_connected_f32_backward_input_request_handle *>(void_handle);
    run_backward_input_delta(handle->backward_input, handle->forward_weights, handle->output);
}

void fully_connected_f32::forward(const nn::workload_data<> *input,
                                  const nn::workload_data<> *weights,
                                  const nn::workload_data<> *bias,
                                  nn::workload_data<> *output) {
    auto num_hardware_threads = std::min(device->thread_pool.get_num_threads(), max_threads);

    nn_workload_data_coords_t input_view_coords(
        input->parent->lengths.t[NN_DATA_COORD_n],
        input->parent->buffer_size / static_cast<uint32_t>(sizeof(float)) / input->parent->lengths.t[NN_DATA_COORD_n],
        1,
        1,
        1,
        1
    );

    nn_workload_data_coords_t output_view_coords(
        output->parent->lengths.t[NN_DATA_COORD_n],
        output->parent->buffer_size / static_cast<uint32_t>(sizeof(float)) / output->parent->lengths.t[NN_DATA_COORD_n],
        1,
        1,
        1,
        1
    );

    nn_workload_data_layout_t in_out_view_layout = nn::layout_t<nn::layout_nxyzpq_f32>::layout;

    nn::workload_data<>* input_reinterpret = new nn::workload_data<>(NN_WORKLOAD_DATA_TAG_UNKNOWN, input->parent->data_buffer, input_view_coords, in_out_view_layout);
    nn::workload_data<>* output_reinterpret = new nn::workload_data<>(NN_WORKLOAD_DATA_TAG_UNKNOWN, output->parent->data_buffer, output_view_coords, in_out_view_layout);


    auto item_view_length = output_reinterpret->view_end.t[NN_DATA_COORD_x] - output_reinterpret->view_begin.t[NN_DATA_COORD_x] + 1;

    auto rest_of_items_in_batch = 0u;
    if (batch_size == 8)
    {
        rest_of_items_in_batch = item_view_length % C_max_acc_batch8;
        item_view_length /= C_max_acc_batch8;
    }
    else if (batch_size == 48)
    {
        rest_of_items_in_batch = item_view_length % C_max_acc_batch48;
        item_view_length /= C_max_acc_batch48;
    }

    const auto items_per_thread = item_view_length / num_hardware_threads;
    const auto items_modulo = item_view_length % num_hardware_threads;

    // Check if we have enough data to cover all threads.
    if (items_per_thread == 0 && items_modulo < 2)
    {
        // Its tiny data - just do it singlethreaded way.
        run_fully_connected_work_item(input_reinterpret, weights, bias, output_reinterpret);
    }
    else
    {
        // Not all threads will be used.
        if (items_per_thread == 0)
            num_hardware_threads = items_modulo;

        uint32_t* thread_items_sums = static_cast<uint32_t*>(alloca(num_hardware_threads * sizeof(uint32_t)));

        if (thread_items_sums == nullptr) throw std::bad_alloc();

        // Distribute elements more evenly.
        auto elements_left = items_modulo;
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            thread_items_sums[thread_id] = items_per_thread;
            if (elements_left)
            {
                ++thread_items_sums[thread_id];
                --elements_left;
            }
        }

        // Now create table of thread sums.
        auto thread_sum = 0u;
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            thread_sum += thread_items_sums[thread_id];
            thread_items_sums[thread_id] = thread_sum;

            if (batch_size == 8)
            {
                thread_items_sums[thread_id] *= C_max_acc_batch8;

                if (thread_id+1 == num_hardware_threads)
                {
                    thread_items_sums[thread_id] += rest_of_items_in_batch;
                }
            }
            else if (batch_size == 48)
            {
                thread_items_sums[thread_id] *= C_max_acc_batch48;

                if (thread_id+1 == num_hardware_threads)
                {
                    thread_items_sums[thread_id] += rest_of_items_in_batch;
                }
            }
        }

        std::vector<fully_connected_f32_request_handle> request_handles(num_hardware_threads);

        // Fill slave work items.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            auto& request_handle = request_handles[thread_id];
            request_handle.primitive = this;

            auto work_begin = 0u;
            if (thread_id > 0u)
                work_begin = thread_items_sums[thread_id - 1];

            auto work_end = thread_items_sums[thread_id] - 1;

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
                input_reinterpret->get_length(NN_DATA_COORD_n) - 1,
                input_reinterpret->get_length(NN_DATA_COORD_x) - 1,
                input_reinterpret->get_length(NN_DATA_COORD_y) - 1,
                input_reinterpret->get_length(NN_DATA_COORD_z) - 1,
                input_reinterpret->get_length(NN_DATA_COORD_p) - 1,
                input_reinterpret->get_length(NN_DATA_COORD_q) - 1
            );

            nn_workload_data_coords_t weights_view_begin(
                0,
                0,
                0,
                0,
                0,
                0
            );
            nn_workload_data_coords_t weights_view_end(
                weights->get_length(NN_DATA_COORD_n) - 1,
                weights->get_length(NN_DATA_COORD_x) - 1,
                weights->get_length(NN_DATA_COORD_y) - 1,
                weights->get_length(NN_DATA_COORD_z) - 1,
                weights->get_length(NN_DATA_COORD_p) - 1,
                weights->get_length(NN_DATA_COORD_q) - 1
            );

            nn_workload_data_coords_t output_view_begin(
                0,
                work_begin,
                0,
                0,
                0,
                0
            );
            nn_workload_data_coords_t output_view_end(
                output_reinterpret->get_length(NN_DATA_COORD_n) - 1,
                work_end,
                output_reinterpret->get_length(NN_DATA_COORD_y) - 1,
                output_reinterpret->get_length(NN_DATA_COORD_z) - 1,
                output_reinterpret->get_length(NN_DATA_COORD_p) - 1,
                output_reinterpret->get_length(NN_DATA_COORD_q) - 1
            );

            request_handle.input = new nn::workload_data<>(
                *const_cast<nn::workload_data<> *>(input_reinterpret), input_view_begin, input_view_end);

            request_handle.weights = new nn::workload_data<>(
                *const_cast<nn::workload_data<> *>(weights), weights_view_begin, weights_view_end);

            request_handle.output = new nn::workload_data<>(
                *const_cast<nn::workload_data<> *>(output_reinterpret), output_view_begin, output_view_end);

            if (bias != nullptr)
            {
                nn_workload_data_coords_t bias_view_begin(
                    0,
                    work_begin,
                    0,
                    0,
                    0,
                    0
                );
                nn_workload_data_coords_t bias_view_end(
                    bias->get_length(NN_DATA_COORD_n) - 1,
                    work_end,
                    bias->get_length(NN_DATA_COORD_y) - 1,
                    bias->get_length(NN_DATA_COORD_z) - 1,
                    bias->get_length(NN_DATA_COORD_p) - 1,
                    bias->get_length(NN_DATA_COORD_q) - 1
                );

                request_handle.bias = new nn::workload_data<>(
                    *const_cast<nn::workload_data<> *>(bias), bias_view_begin, bias_view_end);
            }else{
                request_handle.bias = nullptr;
            }
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(num_hardware_threads);

        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            job[thread_id].callback = unpack_fully_connected_callback_handle;
            job[thread_id].request_handle = &request_handles[thread_id];
        }

        // Wait for all sub threads.
        // It can be affordable ( in terms of performance)
        device->thread_pool.push_job_on_physical_cores_if_beneficial(job);

        // Cleanup dynamic memory.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            delete request_handles[thread_id].input;
            delete request_handles[thread_id].weights;
            if (request_handles[thread_id].bias != nullptr)
                delete request_handles[thread_id].bias;
            delete request_handles[thread_id].output;
        }
    }

    delete input_reinterpret;
    delete output_reinterpret;
}

void fully_connected_f32::forward(const std::vector<const nn_workload_data_t *> &inputs,
                                  const std::vector<const nn_workload_data_t *> &parameters,
                                  const std::vector<nn_workload_data_t *> &outputs) {
    assert(inputs.size() == 1);
    assert(parameters.size() == 2);
    assert(outputs.size() == 1);

    forward(reinterpret_cast<const nn::workload_data<> *>(inputs[0]),
            reinterpret_cast<const nn::workload_data<> *>(parameters[0]),
            reinterpret_cast<const nn::workload_data<> *>(parameters[1]),
            nn::workload_data_cast<>(outputs[0]));
}

void fully_connected_f32::backward(
    const std::vector<nn_workload_data_t *> &inputs,
    const std::vector<const nn_workload_data_t *> &parameters,
    const std::vector<const nn_workload_data_t *> &outputs)
{
    dispatch_backward_input_delta(reinterpret_cast<const nn::workload_data<> *>(outputs[0]),
                                  reinterpret_cast<const nn::workload_data<> *>(parameters[0]),
                                  nn::workload_data_cast<>(inputs[0]));
}

void fully_connected_f32::backward_parameter(
    size_t parameter_index,
    const std::vector<const nn_workload_data_t *> &inputs,
    const std::vector<nn_workload_data_t *> &parameters,
    const std::vector<const nn_workload_data_t *> &outputs)
{
    assert(inputs.size() == 1);
    //assert(parameters.size() == 1);
    assert(outputs.size() == 1);
    switch (parameter_index)
    {
        case 0:
            dispatch_backward_weights_delta(reinterpret_cast<const nn::workload_data<> *>(inputs[0]),
                                            reinterpret_cast<const nn::workload_data<> *>(outputs[0]),
                                            nn::workload_data_cast<>(parameters[0]));
            break;
        case 1:
            dispatch_backward_bias_delta(reinterpret_cast<const nn::workload_data<> *>(outputs[0]),
                                         nullptr, //TBD: bias multiplier
                                         nn::workload_data_cast<>(parameters[1]));
            break;
        default:
            throw std::invalid_argument("index out of range");
    }
}

void fully_connected_f32::dispatch_backward_weights_delta(
    const nn::workload_data<> *forward_input_view,
    const nn::workload_data<> *forward_output_view,
    nn::workload_data<> *forward_weights_delta_view)
{
    auto num_hardware_threads = std::min(device->thread_pool.get_num_threads(), max_threads);
    auto item_view_length = forward_weights_delta_view->view_end.t[NN_DATA_COORD_x] - forward_weights_delta_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto items_per_thread = item_view_length / num_hardware_threads;
    const auto items_modulo = item_view_length % num_hardware_threads;

    // Check if we have enough data to cover all threads.
    if (items_per_thread == 0 && items_modulo < 2)
    {
        // Its tiny data - just do it singlethreaded way.
            run_backward_weights_delta(forward_input_view, forward_output_view, forward_weights_delta_view);
    }
    else
    {
        // Not all threads will be used.
        if (items_per_thread == 0)
            num_hardware_threads = items_modulo;

        uint32_t* thread_items_sums = static_cast<uint32_t*>(alloca(num_hardware_threads * sizeof(uint32_t)));

        if (thread_items_sums == nullptr) throw std::bad_alloc();

        // Distribute elements more evenly.
        auto elements_left = items_modulo;
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            thread_items_sums[thread_id] = items_per_thread;
            if (elements_left)
            {
                ++thread_items_sums[thread_id];
                --elements_left;
            }
        }

        // Now create table of thread sums.
        auto thread_sum = 0u;
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            thread_sum += thread_items_sums[thread_id];
            thread_items_sums[thread_id] = thread_sum;
        }

        std::vector<fully_connected_f32_backward_weight_request_handle> request_handles(num_hardware_threads);

        // Fill slave work items.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            auto& request_handle = request_handles[thread_id];
            request_handle.primitive = this;

            auto work_begin = 0u;
            if (thread_id > 0u)
                work_begin = thread_items_sums[thread_id - 1];

            auto work_end = thread_items_sums[thread_id] - 1;

            // Replace nn_workload_datas pointers with views.
            nn_workload_data_coords_t output_view_begin(
                0,
                work_begin,
                0,
                0,
                0,
                0
            );
            nn_workload_data_coords_t output_view_end(
                forward_weights_delta_view->get_length(NN_DATA_COORD_n) - 1,
                work_end,
                forward_weights_delta_view->get_length(NN_DATA_COORD_y) - 1,
                forward_weights_delta_view->get_length(NN_DATA_COORD_z) - 1,
                forward_weights_delta_view->get_length(NN_DATA_COORD_p) - 1,
                forward_weights_delta_view->get_length(NN_DATA_COORD_q) - 1
            );

            request_handle.backward_input = forward_output_view;
            request_handle.forward_input = forward_input_view;
            request_handle.output = new nn::workload_data<>(
                *const_cast<nn::workload_data<> *>(forward_weights_delta_view), output_view_begin, output_view_end);
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(num_hardware_threads);

        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            job[thread_id].callback = unpack_fully_connected_callback_handle_backward_weight;
            job[thread_id].request_handle = &request_handles[thread_id];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);

        // Cleanup dynamic memory.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            delete request_handles[thread_id].output;
    }
}

void fully_connected_f32::dispatch_backward_bias_delta(
    const nn::workload_data<> *forward_output_view,
    const nn::workload_data<> *bias_multiplier,
    nn::workload_data<> *forward_bias_delta_view)
{
    auto num_hardware_threads = std::min(device->thread_pool.get_num_threads(), max_threads);
    auto item_view_length = forward_bias_delta_view->view_end.t[NN_DATA_COORD_x] - forward_bias_delta_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto items_per_thread = item_view_length / num_hardware_threads;
    const auto items_modulo = item_view_length % num_hardware_threads;

    // Check if we have enough data to cover all threads.
    if (items_per_thread == 0 && items_modulo < 2)
    {
        // Its tiny data - just do it singlethreaded way.
        run_backward_bias_delta(forward_output_view, bias_multiplier, forward_bias_delta_view);
    }
    else
    {
        // Not all threads will be used.
        if (items_per_thread == 0)
            num_hardware_threads = items_modulo;

        uint32_t* thread_items_sums = static_cast<uint32_t*>(alloca(num_hardware_threads * sizeof(uint32_t)));

        if (thread_items_sums == nullptr) throw std::bad_alloc();

        // Distribute elements more evenly.
        auto elements_left = items_modulo;
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            thread_items_sums[thread_id] = items_per_thread;
            if (elements_left)
            {
                ++thread_items_sums[thread_id];
                --elements_left;
            }
        }

        // Now create table of thread sums.
        auto thread_sum = 0u;
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            thread_sum += thread_items_sums[thread_id];
            thread_items_sums[thread_id] = thread_sum;
        }

        std::vector<fully_connected_f32_backward_bias_request_handle> request_handles(num_hardware_threads);

        // Fill slave work items.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            auto& request_handle = request_handles[thread_id];
            request_handle.primitive = this;

            auto work_begin = 0u;
            if (thread_id > 0u)
                work_begin = thread_items_sums[thread_id - 1];

            auto work_end = thread_items_sums[thread_id] - 1;

            // Replace nn_workload_datas pointers with views.
            nn_workload_data_coords_t output_view_begin(
                0,
                work_begin,
                0,
                0,
                0,
                0
            );
            nn_workload_data_coords_t output_view_end(
                forward_bias_delta_view->get_length(NN_DATA_COORD_n) - 1,
                work_end,
                forward_bias_delta_view->get_length(NN_DATA_COORD_y) - 1,
                forward_bias_delta_view->get_length(NN_DATA_COORD_z) - 1,
                forward_bias_delta_view->get_length(NN_DATA_COORD_p) - 1,
                forward_bias_delta_view->get_length(NN_DATA_COORD_q) - 1
            );

            request_handle.backward_input = forward_output_view;
            request_handle.bias_multiplier = bias_multiplier;
            request_handle.output = new nn::workload_data<>(
                *const_cast<nn::workload_data<> *>(forward_bias_delta_view), output_view_begin, output_view_end);
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(num_hardware_threads);

        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            job[thread_id].callback = unpack_fully_connected_callback_handle_backward_bias;
            job[thread_id].request_handle = &request_handles[thread_id];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);

        // Cleanup dynamic memory.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            delete request_handles[thread_id].output;
    }
}

void fully_connected_f32::dispatch_backward_input_delta(
    const nn::workload_data<> *forward_output_view,
    const nn::workload_data<> *forward_weights_view,
    nn::workload_data<> *forward_input_view)
{
    auto num_hardware_threads = std::min(device->thread_pool.get_num_threads(), max_threads);
    auto item_view_length = forward_input_view->view_end.t[NN_DATA_COORD_x] - forward_input_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto items_per_thread = item_view_length / num_hardware_threads;
    const auto items_modulo = item_view_length % num_hardware_threads;

    // Check if we have enough data to cover all threads.
    if (items_per_thread == 0 && items_modulo < 2)
    {
        // Its tiny data - just do it singlethreaded way.
        run_backward_input_delta(forward_output_view, forward_weights_view, forward_input_view);
    }
    else
    {
        // Not all threads will be used.
        if (items_per_thread == 0)
            num_hardware_threads = items_modulo;

        uint32_t* thread_items_sums = static_cast<uint32_t*>(alloca(num_hardware_threads * sizeof(uint32_t)));

        if (thread_items_sums == nullptr) throw std::bad_alloc();

        // Distribute elements more evenly.
        auto elements_left = items_modulo;
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            thread_items_sums[thread_id] = items_per_thread;
            if (elements_left)
            {
                ++thread_items_sums[thread_id];
                --elements_left;
            }
        }

        // Now create table of thread sums.
        auto thread_sum = 0u;
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            thread_sum += thread_items_sums[thread_id];
            thread_items_sums[thread_id] = thread_sum;
        }

        std::vector<fully_connected_f32_backward_input_request_handle> request_handles(num_hardware_threads);

        // Fill slave work items.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            auto& request_handle = request_handles[thread_id];
            request_handle.primitive = this;

            auto work_begin = 0u;
            if (thread_id > 0u)
                work_begin = thread_items_sums[thread_id - 1];

            auto work_end = thread_items_sums[thread_id] - 1;

            // Replace nn_workload_datas pointers with views.
            nn_workload_data_coords_t output_view_begin(
                0,
                work_begin,
                0,
                0,
                0,
                0
            );
            nn_workload_data_coords_t output_view_end(
                forward_input_view->get_length(NN_DATA_COORD_n) - 1,
                work_end,
                forward_input_view->get_length(NN_DATA_COORD_y) - 1,
                forward_input_view->get_length(NN_DATA_COORD_z) - 1,
                forward_input_view->get_length(NN_DATA_COORD_p) - 1,
                forward_input_view->get_length(NN_DATA_COORD_q) - 1
            );

            request_handle.backward_input = forward_output_view;
            request_handle.forward_weights = forward_weights_view;

            request_handle.output = new nn::workload_data<>(
                *const_cast<nn::workload_data<> *>(forward_input_view), output_view_begin, output_view_end);
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(num_hardware_threads);

        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            job[thread_id].callback = unpack_fully_connected_callback_handle_backward_input;
            job[thread_id].request_handle = &request_handles[thread_id];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);

        // Cleanup dynamic memory.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            delete request_handles[thread_id].output;
    }
}

void run_multithreaded_FC_work_item_backward(nn_workload_item *const work_item)
{
    auto primitive = static_cast<fully_connected_f32 *>(work_item->forward_item->primitive);

    // Squash dimensions.
    auto forward_item_input = work_item->forward_item->input[0].get_data_view();
    auto forward_item_output = work_item->forward_item->output[0];
    auto backward_item_input = work_item->input[0].get_data_view();
    auto backward_item_output = work_item->output[0];
    auto backward_item_output_weight = work_item->output[1];
    auto backward_item_output_bias = work_item->output[2];

    nn_workload_data_coords_t input_view_coords(
        forward_item_input->parent->lengths.t[NN_DATA_COORD_n],
        forward_item_input->parent->buffer_size / static_cast<uint32_t>(sizeof(float)) / forward_item_input->parent->lengths.t[NN_DATA_COORD_n],
        1, 1, 1, 1 );

    nn_workload_data_coords_t output_view_coords(
        forward_item_output->parent->lengths.t[NN_DATA_COORD_n],
        forward_item_output->parent->buffer_size / static_cast<uint32_t>(sizeof(float)) / forward_item_output->parent->lengths.t[NN_DATA_COORD_n],
        1, 1, 1, 1 );

    nn_workload_data_layout_t in_out_view_layout = nn::layout_t<nn::layout_nxyzpq_f32>::layout;

    nn::workload_data<> forward_input_view(NN_WORKLOAD_DATA_TAG_UNKNOWN, forward_item_input->parent->data_buffer, input_view_coords, in_out_view_layout);
    nn::workload_data<> forward_output_view(NN_WORKLOAD_DATA_TAG_UNKNOWN, forward_item_output->parent->data_buffer, output_view_coords, in_out_view_layout);

    // Workaround for lack of compilation support in device API for delta buffers.
    // We are adding delta buffers to containers.
    forward_input_view.parent->delta_buffer = backward_item_output->parent->data_buffer;
    forward_output_view.parent->delta_buffer = backward_item_input->parent->data_buffer;

    nn::workload_data<> forward_weight_view(NN_WORKLOAD_DATA_TAG_UNKNOWN, work_item->forward_item->parameters[0]->parent->data_buffer, work_item->forward_item->parameters[0]->parent->lengths, work_item->forward_item->parameters[0]->parent->layout);
    nn::workload_data<> forward_bias_view(NN_WORKLOAD_DATA_TAG_UNKNOWN, work_item->forward_item->parameters[1]->parent->data_buffer, work_item->forward_item->parameters[1]->parent->lengths, work_item->forward_item->parameters[1]->parent->layout);

    forward_weight_view.parent->delta_buffer = backward_item_output_weight->parent->data_buffer;
    forward_bias_view.parent->delta_buffer = backward_item_output_bias->parent->data_buffer;

    std::vector<const nn_workload_data_t *> const_inputs;
    std::vector<nn_workload_data_t *> inputs;

    std::vector<const nn_workload_data_t *> const_parameters;
    std::vector<nn_workload_data_t *> parameters;

    std::vector<const nn_workload_data_t *> const_outputs;

    const_inputs.push_back(&forward_input_view);
    inputs.push_back(&forward_input_view);

    const_outputs.push_back(&forward_output_view);

    const_parameters.push_back(&forward_weight_view);
    const_parameters.push_back(&forward_bias_view);
    const_parameters.push_back(nullptr); // bias multiplier

    parameters.push_back(&forward_weight_view);
    parameters.push_back(&forward_bias_view);
    parameters.push_back(nullptr); // bias multiplier

    primitive->backward(
        inputs,
        const_parameters,
        const_outputs);

    primitive->backward_parameter(
        0,
        const_inputs,
        parameters,
        const_outputs);

    primitive->backward_parameter(
        1,
        const_inputs,
        parameters,
        const_outputs);
}

fully_connected_f32::fully_connected_f32(size_t num_input,
                                         size_t num_output,
                                         const nn_argument_activation_t &activation,
                                         size_t batch_size,
                                         nn_device_internal *device)
    : has_3d_input(false),
      input_size_x(1),
      input_size_y(1),
      input_size_z(num_input),
      num_input(num_input),
      num_output(num_output),
      activation(activation),
      batch_size(batch_size),
      device(device) {
    assert(batch_size == 1 || batch_size == 8 || batch_size == 48);
}

fully_connected_f32::fully_connected_f32(size_t input_size_x,
                                         size_t input_size_y,
                                         size_t input_size_z,
                                         size_t num_output,
                                         const nn_argument_activation_t &activation,
                                         size_t batch_size,
                                         nn_device_internal *device)
    : has_3d_input(true),
      input_size_x(input_size_x),
      input_size_y(input_size_y),
      input_size_z(input_size_z),
      num_input(input_size_x * input_size_y * input_size_z),
      num_output(num_output),
      activation(activation),
      batch_size(batch_size),
      device(device) {
    assert(batch_size == 1 || batch_size == 8 || batch_size == 48);
}
bool fully_connected_f32::validate_input(size_t index, nn_workload_data_t *data)
{
    switch(index){
    case 0:
        return nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::validate(data, num_input, batch_size);
    }

    throw std::invalid_argument("index out of range");
}

} // namespace layer

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_fully_connected_f32_create_0(
    nn_device_t *device,                        /* IDLF device handle */
    size_t num_input,                           /* number of input feature maps */
    size_t num_output,                          /* number of output feature maps */
    const nn_argument_activation_t *activation, /* struct parameterizing optional activation function */
    size_t batch_size,                          /* size of input batch */
    const nn_primitives_fully_connected_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);

    std::remove_const<std::remove_pointer<decltype(hints)>::type>::type hints_ = {};
    if(hints != nullptr)
        hints_ = *hints;

    return new layer::fully_connected_f32(
        num_input, num_output, *activation, batch_size, reinterpret_cast<nn_device_internal *>(device));
}
