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
#include "layer_fully_connected_avx2.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>

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
void fully_connected_f32::run_fully_connected_work_item_internal_batch8(const nn::nn_workload_data_t<float> *input,
                                                                        const nn::nn_workload_data_t<float> *weights,
                                                                        const nn::nn_workload_data_t<float> *bias,
                                                                        nn::nn_workload_data_t<float> *output) {
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
void fully_connected_f32::run_fully_connected_work_item_internal_batch48(const nn::nn_workload_data_t<float> *input,
                                                                         const nn::nn_workload_data_t<float> *weights,
                                                                         const nn::nn_workload_data_t<float> *bias,
                                                                         nn::nn_workload_data_t<float> *output) {
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
void fully_connected_f32::run_fully_connected_work_item_internal_latency(const nn::nn_workload_data_t<float> *input,
                                                                         const nn::nn_workload_data_t<float> *weights,
                                                                         const nn::nn_workload_data_t<float> *bias,
                                                                         nn::nn_workload_data_t<float> *output) {
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
void fully_connected_f32::choose_fully_connected_work_item_batching_mode(const nn::nn_workload_data_t<float> *input,
                                                                         const nn::nn_workload_data_t<float> *weights,
                                                                         const nn::nn_workload_data_t<float> *bias,
                                                                         nn::nn_workload_data_t<float> *output) {
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
void fully_connected_f32::choose_fully_connected_work_item_activation(const nn::nn_workload_data_t<float> *input,
                                                                      const nn::nn_workload_data_t<float> *weights,
                                                                      const nn::nn_workload_data_t<float> *bias,
                                                                      nn::nn_workload_data_t<float> *output) {
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

void fully_connected_f32::run_fully_connected_work_item(const nn::nn_workload_data_t<float> *input,
                                                        const nn::nn_workload_data_t<float> *weights,
                                                        const nn::nn_workload_data_t<float> *bias,
                                                        nn::nn_workload_data_t<float> *output) {
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

const nn_workload_data_layout_t fully_connected_f32::in_out_layout = {
    {0, 0, 0, 0, 0, 0}, // tile in log2(size)
    {0, 0, 0, 0, 0, 0}, // alignment
    {NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q}, // ordering
    NN_DATATYPE_FLOAT};

struct fully_connected_f32_request_handle {
    fully_connected_f32 *primitive;
    const nn::nn_workload_data_t<float> *input;
    const nn::nn_workload_data_t<float> *weights;
    const nn::nn_workload_data_t<float> *bias;
    nn::nn_workload_data_t<float> *output;
};

void unpack_fully_connected_callback_handle(void *void_handle) {
    auto handle = reinterpret_cast<fully_connected_f32_request_handle *>(void_handle);
    handle->primitive->run_fully_connected_work_item(handle->input, handle->weights, handle->bias, handle->output);
}

void fully_connected_f32::forward(const nn::nn_workload_data_t<float> *input,
                                  const nn::nn_workload_data_t<float> *weights,
                                  const nn::nn_workload_data_t<float> *bias,
                                  nn::nn_workload_data_t<float> *output) {
    auto num_hardware_threads = std::min(device->thread_pool.get_num_threads(), max_threads);

    auto item_view_length = output->view_end.t[NN_DATA_COORD_x] - output->view_begin.t[NN_DATA_COORD_x] + 1;

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
        run_fully_connected_work_item(input, weights, bias, output);
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

            nn_workload_data_coords_t weights_view_begin =
            {
                0,
                0,
                0,
                0,
                0,
                0
            };
            nn_workload_data_coords_t weights_view_end = {
                weights->get_length(NN_DATA_COORD_n) - 1,
                weights->get_length(NN_DATA_COORD_x) - 1,
                weights->get_length(NN_DATA_COORD_y) - 1,
                weights->get_length(NN_DATA_COORD_z) - 1,
                weights->get_length(NN_DATA_COORD_p) - 1,
                weights->get_length(NN_DATA_COORD_q) - 1
            };

            nn_workload_data_coords_t output_view_begin =
            {
                0,
                work_begin,
                0,
                0,
                0,
                0
            };
            nn_workload_data_coords_t output_view_end =
            {
                output->get_length(NN_DATA_COORD_n) - 1,
                work_end,
                output->get_length(NN_DATA_COORD_y) - 1,
                output->get_length(NN_DATA_COORD_z) - 1,
                output->get_length(NN_DATA_COORD_p) - 1,
                output->get_length(NN_DATA_COORD_q) - 1
            };

            request_handle.input = new nn::nn_workload_data_t<float>(
                *const_cast<nn::nn_workload_data_t<float> *>(input), input_view_begin, input_view_end);

            request_handle.weights = new nn::nn_workload_data_t<float>(
                *const_cast<nn::nn_workload_data_t<float> *>(weights), weights_view_begin, weights_view_end);

            request_handle.output = new nn::nn_workload_data_t<float>(
                *const_cast<nn::nn_workload_data_t<float> *>(output), output_view_begin, output_view_end);

            if (bias != nullptr)
            {
                nn_workload_data_coords_t bias_view_begin =
                {
                    0,
                    work_begin,
                    0,
                    0,
                    0,
                    0
                };
                nn_workload_data_coords_t bias_view_end =
                {
                    bias->get_length(NN_DATA_COORD_n) - 1,
                    work_end,
                    bias->get_length(NN_DATA_COORD_y) - 1,
                    bias->get_length(NN_DATA_COORD_z) - 1,
                    bias->get_length(NN_DATA_COORD_p) - 1,
                    bias->get_length(NN_DATA_COORD_q) - 1
                };

                request_handle.bias = new nn::nn_workload_data_t<float>(
                    *const_cast<nn::nn_workload_data_t<float> *>(bias), bias_view_begin, bias_view_end);
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
        device->thread_pool.push_job(job);

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
}

void wrapper_fully_connected_work_item(nn_workload_item *const work_item)
{
    nn_workload_data_coords_t input_view_coords =
    {
        work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_n],
        work_item->input[0]->output->parent->buffer_size / static_cast<uint32_t>(sizeof(float)) / work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_n],
        1,
        1,
        1,
        1
    };

    nn_workload_data_coords_t output_view_coords =
    {
        work_item->output->parent->lengths.t[NN_DATA_COORD_n],
        work_item->output->parent->buffer_size / static_cast<uint32_t>(sizeof(float)) / work_item->output->parent->lengths.t[NN_DATA_COORD_n],
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

    nn::nn_workload_data_t<float>* input_view = new nn::nn_workload_data_t<float>(work_item->input[0]->output->parent->data_buffer, input_view_coords, in_out_view_layout);
    nn::nn_workload_data_t<float>* output_view = new nn::nn_workload_data_t<float>(work_item->output->parent->data_buffer, output_view_coords, in_out_view_layout);

    if (static_cast<fully_connected_f32 *>(work_item->primitive)->device->thread_pool.get_num_threads() > 1)
    {
        static_cast<fully_connected_f32 *>(work_item->primitive)
            ->forward(
                input_view,
                reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->arguments.forward_fully_connected.weights),
                reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->arguments.forward_fully_connected.biases),
                output_view);
    }
    else
    {
        static_cast<fully_connected_f32 *>(work_item->primitive)
            ->run_fully_connected_work_item(
                input_view,
                reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->arguments.forward_fully_connected.weights),
                reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->arguments.forward_fully_connected.biases),
                output_view);
    }

    delete output_view;
    delete input_view;
}

fully_connected_f32 *fully_connected_f32::create(size_t num_input,
                                                 size_t num_output,
                                                 const nn_argument_activation_t &activation,
                                                 size_t batch_size,
                                                 nn_device_t *device) {
    return new fully_connected_f32(
        num_input, num_output, activation, batch_size, static_cast<nn_device_internal *>(device));
}

fully_connected_f32::fully_connected_f32(size_t num_input,
                                         size_t num_output,
                                         const nn_argument_activation_t &activation,
                                         size_t batch_size,
                                         nn_device_internal *device)
    : num_input(num_input),
      num_output(num_output),
      activation(activation),
      batch_size(batch_size),
      device(device) {}

nn::nn_workload_data_t<float> *fully_connected_f32::create_weights(const nn::data<float, 2> &weights) {
    nn::nn_workload_data_t<float> *result = nullptr;

    switch (batch_size) {
    case 1: { // weights
        //TODO: validate weight format
        nn_workload_data_layout_t layout = {
            { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
            { 0, 0, 0, 0, 0, 0 }, // alignment
            { NN_DATA_COORD_y, NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, // ordering
            NN_DATATYPE_FLOAT
        };

        nn_workload_data_coords_t size = {1,
                                          static_cast<uint32_t>(num_input),
                                          static_cast<uint32_t>(num_output),
                                          1,
                                          1,
                                          1};

        result = new nn::nn_workload_data_t<float>(size, layout);

        /*
        Code below this comment is a performance optimized version of:
        for (size_t y = 0u; y < size.t[2]; ++y)
            for (size_t x = 0u; x < size.t[1]; ++x)
                (*result)(0, x, y, 0, 0, 0) = weights.at(x, y);
        ...which is left here for readability.
        */
        auto src = static_cast<float *>(weights.buffer);
        auto dst = static_cast<float *>(result->parent->data_buffer);
        for (size_t y = 0u; y < num_output; ++y)
            for (size_t x = 0u; x < num_input; ++x)
                dst[y + x*num_output] = src[x + y*num_input];
        break;
    }
    case 8: 
    case 48:
    {
        const uint32_t C_max_accumulators = (batch_size == 8) ? 13 : 2;

        //TODO: validate weight format
        nn_workload_data_layout_t layout = {
            { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
            { 0, 0, 0, 0, 0, 0 }, // alignment
            { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_q, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_n }, // ordering
            NN_DATATYPE_FLOAT
        };
        uint32_t last_non_full_slice = weights.size[1] % C_max_accumulators;
        nn_workload_data_coords_t size = {
            1,                                                                                       // n
            static_cast<uint32_t>(weights.size[0]),                                                  // x
            1,                                                                                       // y
            1,                                                                                       // z
            C_max_accumulators,                                                                      // p (slice size)
            static_cast<uint32_t>((weights.size[1] + C_max_accumulators - 1) / C_max_accumulators)   // q (number of slices)
        };
        result = new nn::nn_workload_data_t<float>(size, layout);
        /*
        Code below this comment is a performance optimized version of:
        auto width = weights.size[0];
        auto height = weights.size[1];
        for(auto x=0u; x<width; ++x)
            for(auto y=0u; y<height; ++y)
                (*result)(0, x, 0, 0, y%C_max_accumulators, y/C_max_accumulators) = weights.at(x,y);
        ...which is left here for readability.
        */
        auto width = weights.size[0];
        auto height = weights.size[1];
        auto src = static_cast<float *>(weights.buffer);
        auto dst = static_cast<float *>(result->parent->data_buffer);
        for(auto y=0u; y<height; ++y) {
            auto y_offset = y%C_max_accumulators + (y/C_max_accumulators)*width*C_max_accumulators;
            for(auto x=0u; x<width; ++x)
                dst[y_offset + x*C_max_accumulators] = src[width*y+x];
        }
        break;
    }
    default:
        assert(0); // batch size unsupported
    }

    return result;
}

nn::nn_workload_data_t<float> *fully_connected_f32::create_weights(const nn::data<float, 4> &weights) {
    nn::nn_workload_data_t<float> *result = nullptr;

    assert(weights.size[0] * weights.size[1] * weights.size[2] == num_input);
    assert(weights.size[3] == num_output);

    const size_t z_size = weights.size[2], y_size = weights.size[1], x_size = weights.size[0];

    switch(batch_size){
    case 1:{
        // 4-dimensional weights.
        //TODO: validate weight format
        nn_workload_data_layout_t layout = {
            { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
            { 0, 0, 0, 0, 0, 0 }, // alignment
            { NN_DATA_COORD_y, NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
            NN_DATATYPE_FLOAT
        };
        nn_workload_data_coords_t size = {1,
                                          static_cast<uint32_t>(num_input),
                                          static_cast<uint32_t>(num_output),
                                          1,
                                          1,
                                          1};

        result = new nn::nn_workload_data_t<float>(size, layout);
        for (auto p = 0u; p < num_output; ++p)
            for (auto z = 0u; z < z_size; ++z)
                for (auto y = 0u; y < y_size; ++y)
                    for (auto x = 0u; x < x_size; ++x)
                        (*result)(0, z + z_size * (x + x_size * y), p, 0, 0, 0) = weights.at(x, y, z, p);
        break;
    }
    case 8:
    case 48:{
        const uint32_t C_max_accumulators = (batch_size == 8) ? 13 : 2;

        //THIS code requires verification
        //TODO: validate weight format
        nn_workload_data_layout_t layout = {
            { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
            { 0, 0, 0, 0, 0, 0 }, // alignment
            { NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q, NN_DATA_COORD_n }, // ordering
            NN_DATATYPE_FLOAT
        };

        nn_workload_data_coords_t size = {
            1,
            static_cast<uint32_t>(num_input),
            1,
            1,
            C_max_accumulators,                                                             // p (slice size)
            static_cast<uint32_t>(num_output + C_max_accumulators - 1) / C_max_accumulators // q (number of slices)
        };
        result = new nn::nn_workload_data_t<float>(size, layout);

        uint32_t last_non_full_slice = weights.size[3] % C_max_accumulators;
        /*
        Code below this comment is a performance optimized version of:
        for (auto z = 0u; z < z_size; ++z)
            for (auto y = 0u; y < y_size; ++y)
                for (auto x = 0u; x < x_size; ++x)
                    for(auto o=0u; o<num_output; ++o)
                        (*result)(0, z + z_size * (x + x_size * y), 0, 0, o%C_max_accumulators, o/C_max_accumulators) =
                            weights.at(x, y, z, o);
        ...which is left here for readability.
        */
        auto src = static_cast<float *>(weights.buffer);
        auto src_y_stride = weights.size[0];
        auto src_z_stride = weights.size[1]*src_y_stride;
        auto src_o_stride = weights.size[2]*src_z_stride;
        auto dst = static_cast<float *>(result->parent->data_buffer);
        for (auto z = 0u; z < z_size; ++z)
            for (auto y = 0u; y < y_size; ++y)
                for (auto x = 0u; x < x_size; ++x) {
                    auto dst_xyz_offset = (z + z_size * (x + x_size * y))*C_max_accumulators;
                    auto src_xyz_offset = x+y*src_y_stride+z*src_z_stride;
                    for(auto o=0u; o<num_output; ++o) {
                        auto dp = o%C_max_accumulators;
                        auto dq = o - dp;
                        dst[dp + dst_xyz_offset + num_input*dq] = src[src_xyz_offset + o*src_o_stride];
                    }
                }
        break;
    }
    default:
        assert(0); // batch size unsupported
    }

    return result;
}

nn::nn_workload_data_t<float> * fully_connected_f32::create_bias(const nn::data<float, 1> &bias)
{
    //TODO: validate bias format
    nn_workload_data_layout_t layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_FLOAT
    };
    nn_workload_data_coords_t size = {1, static_cast<uint32_t>(bias.size[0]), 1, 1, 1, 1};
    auto result = new nn::nn_workload_data_t<float>(size, layout);
    for (size_t index = 0u; index < size.t[1]; ++index)
        (*result)(0, index, 0, 0, 0, 0) = bias.at(index);
    return result;
}

nn::nn_workload_data_t<float> *fully_connected_f32::create_input(const nn::data<float, 2> &input) {
    assert(input.size[1] == batch_size);
    assert(input.size[0] == num_input);

    nn_workload_data_coords_t size = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(num_input), 1, 1, 1, 1};
    auto result = new nn::nn_workload_data_t<float>(size, in_out_layout);

    for (size_t it_batch = 0; it_batch < batch_size; ++it_batch)
        for (size_t it_input = 0; it_input < num_input; ++it_input)
            (*result)(it_batch, it_input, 0, 0, 0, 0) = input.at(it_input, it_batch);

    return result;
}

nn::nn_workload_data_t<float> *fully_connected_f32::create_input(const nn::data<float, 4> &input) {
    assert(input.size[3] == batch_size);
    assert(input.size[0] * input.size[1] * input.size[2] == num_input);

    const size_t z_size = input.size[0], y_size = input.size[2], x_size = input.size[1];

    nn_workload_data_coords_t size = {
        static_cast<uint32_t>(batch_size), static_cast<uint32_t>(z_size * x_size * y_size) * 1, 1, 1, 1, 1};

    auto result = new nn::nn_workload_data_t<float>(size, in_out_layout);

    for (size_t it_batch = 0; it_batch < batch_size; ++it_batch)
        for (auto z = 0u; z < z_size; ++z)
            for (auto y = 0u; y < y_size; ++y)
                for (auto x = 0u; x < x_size; ++x)
                    (*result)(it_batch, z + z_size * (x + x_size * y), 0, 0, 0, 0) = input.at(z, x, y, it_batch);

    return result;
}

bool fully_connected_f32::validate_input(const nn::nn_workload_data_t<float> &input) {
    if (0 != memcmp(&in_out_layout, &input.parent->layout, sizeof(in_out_layout)))
        return false;

    const auto view_size = input.get_length();

    if (view_size.t[NN_DATA_COORD_n] != batch_size)
        return false;

    if (view_size.t[NN_DATA_COORD_x] != num_input &&
        (view_size.t[NN_DATA_COORD_x] * view_size.t[NN_DATA_COORD_y] * view_size.t[NN_DATA_COORD_z] != num_input))
        return false;

    if ((input.parent->buffer_size / input.parent->data_type_size) != num_input * batch_size)
        return false;

    return true;
}

nn::nn_workload_data_t<float> *fully_connected_f32::create_output() {
    nn_workload_data_coords_t size = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(num_output), 1, 1, 1, 1};
    return new nn::nn_workload_data_t<float>(size, in_out_layout);
}

void fully_connected_f32::copy_output(nn::data<float, 2> &destination, const nn::nn_workload_data_t<float> &source) {
    assert(destination.size[1] == batch_size);
    assert(destination.size[0] == num_input);

    nn_workload_data_coords_t size = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(num_input), 1, 1, 1, 1};
    auto result = new nn::nn_workload_data_t<float>(size, in_out_layout);

    for (size_t it_batch = 0; it_batch < batch_size; ++it_batch)
        for (size_t it_input = 0; it_input < num_input; ++it_input)
            destination.at(it_input, it_batch) =
                const_cast<nn::nn_workload_data_t<float> &>(source)(it_batch, it_input, 0, 0, 0, 0);
}

namespace fully_connected_f32_impl {

nn_opaque_data_t *NN_API_CALL_CONVENTION
create_weights(nn_primitive_handle_t handle, const nn_data_t *weights, NN_API_STATUS *status) {
    auto primitive = static_cast<layer::fully_connected_f32 *>(handle);
    nn::nn_workload_data_t<float> *result;
    if(weights->dimension == 2)
        {result = primitive->create_weights(*nn::data_cast<float, 2>(weights));
    } else if (weights->dimension == 4) {
        result = primitive->create_weights(*nn::data_cast<float, 4>(weights));
    }else{
        SET_STATUS(NN_API_STATUS_ERROR_INVALID_MEMORY_LAYOUT);
        return nullptr;
    }

    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

nn_opaque_data_t *NN_API_CALL_CONVENTION
create_bias(nn_primitive_handle_t handle, const nn_data_t *bias, NN_API_STATUS *status) {
    auto primitive = static_cast<layer::fully_connected_f32 *>(handle);
    auto result = primitive->create_bias(*nn::data_cast<float, 1>(bias));
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

nn_opaque_data_t *NN_API_CALL_CONVENTION
create_input(nn_primitive_handle_t handle, const nn_data_t *input, NN_API_STATUS *status) {
    auto primitive = static_cast<layer::fully_connected_f32 *>(handle);
    nn::nn_workload_data_t<float> *result;

    if(input->dimension == 2)
        result = primitive->create_input(*nn::data_cast<float, 2>(input));
    else if(input->dimension == 4)
        result = primitive->create_input(*nn::data_cast<float, 4>(input));
    else
    {
        SET_STATUS(NN_API_STATUS_ERROR_INVALID_MEMORY_LAYOUT);
        return nullptr;
    }

    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

nn_opaque_data_t *NN_API_CALL_CONVENTION create_output(nn_primitive_handle_t handle, NN_API_STATUS *status) {
    auto primitive = static_cast<layer::fully_connected_f32 *>(handle);
    auto result = primitive->create_output();
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}

int NN_API_CALL_CONVENTION
validate_input(nn_primitive_handle_t handle, /* primitive handle */
               nn_opaque_data_t *opaque_data /* internal data storage handle to validate */) {
    auto primitive = static_cast<layer::fully_connected_f32 *>(handle);
    return primitive->validate_input(*reinterpret_cast<nn::nn_workload_data_t<float> *>(opaque_data));
}

nn_event_t NN_API_CALL_CONVENTION copy_output_async(nn_primitive_handle_t handle,
                                                    nn_data_t *output,
                                                    nn_opaque_data_t *output_buffer,
                                                    size_t dependencies_count,
                                                    nn_event_t *dependencies,
                                                    NN_API_STATUS *status) {
    auto primitive = static_cast<layer::fully_connected_f32 *>(handle);
    primitive->copy_output(*nn::data_cast<float, 2>(output),
                           *reinterpret_cast<nn::nn_workload_data_t<float> *>(output_buffer));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}

nn_event_t NN_API_CALL_CONVENTION forward_with_weights_and_bias(nn_primitive_handle_t handle,
                                                                nn_opaque_data_t *input,
                                                                nn_opaque_data_t *weights,
                                                                nn_opaque_data_t *bias,
                                                                nn_opaque_data_t *output,
                                                                size_t dependencies_count,
                                                                nn_event_t *dependencies,
                                                                NN_API_STATUS *status) {
    auto primitive = static_cast<layer::fully_connected_f32 *>(handle);
    primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(input),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(weights),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(bias),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(output));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}

nn_primitive_handle_t NN_API_CALL_CONVENTION
create(nn_device_t *device,                        /* IDLF device handle */
       size_t num_input,                           /* number of input feature maps */
       size_t num_output,                          /* number of output feature maps */
       const nn_argument_activation_t *activation, /* struct parameterizing optional activation function */
       size_t batch_size,                          /* size of input batch */
       NN_API_STATUS *status                       /* NN_API_STATUS_OK on success */
       ) {
    SET_STATUS(NN_API_STATUS_OK);
    return layer::fully_connected_f32::create(num_input, num_output, *activation, batch_size, device);
}
}

} // namespace layer

nn_primitives_fully_connected_f32_0_t nn_primitives_fully_connected_f32_0{
    layer::fully_connected_f32_impl::create,
    layer::fully_connected_f32_impl::create_weights,
    layer::fully_connected_f32_impl::create_bias,
    nullptr, // create_parameters
    layer::fully_connected_f32_impl::create_input,
    layer::fully_connected_f32_impl::validate_input,
    layer::fully_connected_f32_impl::create_output,
    layer::fully_connected_f32_impl::forward_with_weights_and_bias,
    nullptr, // forward_with_parameters_async
    layer::fully_connected_f32_impl::copy_output_async};
