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
#include "layer_normalization_avx2.h"

#include <immintrin.h>
#include <string.h>
#include <thread>
#include <vector>

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

namespace layer {
///////////////////////////////////////////////////////////////////////////////////////////////////
// forward implementation

template<uint32_t T_SIZE>
void normalization_compute_block_linear_single(
    float* &input_ptr,
    float* &output_ptr,
    const __m256 coeff_a,
    const __m256 coeff_b)
{
    // We are not using table of registers and unroll pragmas
    // due to compiler which have issues with register allocation
    // and needs special, obvious treatment. Template immediate
    // arguments matching will remove all conditions in this code.
    __m256  acc0, acc1, acc2, acc3, acc4,
            acc5, acc6, acc7, acc8, acc9,
            acc10, acc11, acc12, acc13, acc14, acc15;

    // Load inputs.
    if (T_SIZE >=  1)  acc0 = _mm256_loadu_ps(input_ptr +  0 * C_batch_size);
    if (T_SIZE >=  2)  acc1 = _mm256_loadu_ps(input_ptr +  1 * C_batch_size);
    if (T_SIZE >=  3)  acc2 = _mm256_loadu_ps(input_ptr +  2 * C_batch_size);
    if (T_SIZE >=  4)  acc3 = _mm256_loadu_ps(input_ptr +  3 * C_batch_size);
    if (T_SIZE >=  5)  acc4 = _mm256_loadu_ps(input_ptr +  4 * C_batch_size);
    if (T_SIZE >=  6)  acc5 = _mm256_loadu_ps(input_ptr +  5 * C_batch_size);
    if (T_SIZE >=  7)  acc6 = _mm256_loadu_ps(input_ptr +  6 * C_batch_size);
    if (T_SIZE >=  8)  acc7 = _mm256_loadu_ps(input_ptr +  7 * C_batch_size);
    if (T_SIZE >=  9)  acc8 = _mm256_loadu_ps(input_ptr +  8 * C_batch_size);
    if (T_SIZE >= 10)  acc9 = _mm256_loadu_ps(input_ptr +  9 * C_batch_size);
    if (T_SIZE >= 11) acc10 = _mm256_loadu_ps(input_ptr + 10 * C_batch_size);
    if (T_SIZE >= 12) acc11 = _mm256_loadu_ps(input_ptr + 11 * C_batch_size);
    if (T_SIZE >= 13) acc12 = _mm256_loadu_ps(input_ptr + 12 * C_batch_size);
    if (T_SIZE >= 14) acc13 = _mm256_loadu_ps(input_ptr + 13 * C_batch_size);
    if (T_SIZE >= 15) acc14 = _mm256_loadu_ps(input_ptr + 14 * C_batch_size);

    // Perform a*x + b
    if (T_SIZE >=  1)  acc0 = _mm256_fmadd_ps(coeff_a,  acc0, coeff_b);
    if (T_SIZE >=  2)  acc1 = _mm256_fmadd_ps(coeff_a,  acc1, coeff_b);
    if (T_SIZE >=  3)  acc2 = _mm256_fmadd_ps(coeff_a,  acc2, coeff_b);
    if (T_SIZE >=  4)  acc3 = _mm256_fmadd_ps(coeff_a,  acc3, coeff_b);
    if (T_SIZE >=  5)  acc4 = _mm256_fmadd_ps(coeff_a,  acc4, coeff_b);
    if (T_SIZE >=  6)  acc5 = _mm256_fmadd_ps(coeff_a,  acc5, coeff_b);
    if (T_SIZE >=  7)  acc6 = _mm256_fmadd_ps(coeff_a,  acc6, coeff_b);
    if (T_SIZE >=  8)  acc7 = _mm256_fmadd_ps(coeff_a,  acc7, coeff_b);
    if (T_SIZE >=  9)  acc8 = _mm256_fmadd_ps(coeff_a,  acc8, coeff_b);
    if (T_SIZE >= 10)  acc9 = _mm256_fmadd_ps(coeff_a,  acc9, coeff_b);
    if (T_SIZE >= 11) acc10 = _mm256_fmadd_ps(coeff_a, acc10, coeff_b);
    if (T_SIZE >= 12) acc11 = _mm256_fmadd_ps(coeff_a, acc11, coeff_b);
    if (T_SIZE >= 13) acc12 = _mm256_fmadd_ps(coeff_a, acc12, coeff_b);
    if (T_SIZE >= 14) acc13 = _mm256_fmadd_ps(coeff_a, acc13, coeff_b);
    if (T_SIZE >= 15) acc14 = _mm256_fmadd_ps(coeff_a, acc14, coeff_b);

    // Store results.
    if (T_SIZE >=  1) _mm256_storeu_ps(output_ptr +  0 * C_batch_size,  acc0);
    if (T_SIZE >=  2) _mm256_storeu_ps(output_ptr +  1 * C_batch_size,  acc1);
    if (T_SIZE >=  3) _mm256_storeu_ps(output_ptr +  2 * C_batch_size,  acc2);
    if (T_SIZE >=  4) _mm256_storeu_ps(output_ptr +  3 * C_batch_size,  acc3);
    if (T_SIZE >=  5) _mm256_storeu_ps(output_ptr +  4 * C_batch_size,  acc4);
    if (T_SIZE >=  6) _mm256_storeu_ps(output_ptr +  5 * C_batch_size,  acc5);
    if (T_SIZE >=  7) _mm256_storeu_ps(output_ptr +  6 * C_batch_size,  acc6);
    if (T_SIZE >=  8) _mm256_storeu_ps(output_ptr +  7 * C_batch_size,  acc7);
    if (T_SIZE >=  9) _mm256_storeu_ps(output_ptr +  8 * C_batch_size,  acc8);
    if (T_SIZE >= 10) _mm256_storeu_ps(output_ptr +  9 * C_batch_size,  acc9);
    if (T_SIZE >= 11) _mm256_storeu_ps(output_ptr + 10 * C_batch_size, acc10);
    if (T_SIZE >= 12) _mm256_storeu_ps(output_ptr + 11 * C_batch_size, acc11);
    if (T_SIZE >= 13) _mm256_storeu_ps(output_ptr + 12 * C_batch_size, acc12);
    if (T_SIZE >= 14) _mm256_storeu_ps(output_ptr + 13 * C_batch_size, acc13);
    if (T_SIZE >= 15) _mm256_storeu_ps(output_ptr + 14 * C_batch_size, acc14);

    input_ptr += T_SIZE*C_batch_size;
    output_ptr += T_SIZE*C_batch_size;
}

void normalization_elementwise_linear_f32::run_normalization_work_item_linear_single_batch8(
    const nn::nn_workload_data_t<float> *input_view, nn::nn_workload_data_t<float> *output_view) {
    const auto input_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto num_full_blocks = output_width / C_max_acc;
    const auto partial_block_size = output_width % C_max_acc;

    const auto output_view_offset = output_view->view_begin.t[NN_DATA_COORD_x] * C_batch_size;
    const auto input_view_offset = input_view->view_begin.t[NN_DATA_COORD_x] * C_batch_size;

    auto input_buffer = &static_cast<float*>(input_view->parent->data_buffer)[input_view_offset];
    auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_offset];

    const __m256 coeff_a = _mm256_set1_ps(alpha);
    const __m256 coeff_b = _mm256_set1_ps(beta);

    {
        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            normalization_compute_block_linear_single<C_max_acc>(input_buffer, output_buffer, coeff_a, coeff_b);
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: normalization_compute_block_linear_single< 1>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  2: normalization_compute_block_linear_single< 2>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  3: normalization_compute_block_linear_single< 3>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  4: normalization_compute_block_linear_single< 4>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  5: normalization_compute_block_linear_single< 5>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  6: normalization_compute_block_linear_single< 6>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  7: normalization_compute_block_linear_single< 7>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  8: normalization_compute_block_linear_single< 8>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  9: normalization_compute_block_linear_single< 9>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case 10: normalization_compute_block_linear_single<10>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case 11: normalization_compute_block_linear_single<11>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case 12: normalization_compute_block_linear_single<12>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case 13: normalization_compute_block_linear_single<13>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case 14: normalization_compute_block_linear_single<14>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        default: NN_UNREACHABLE_CODE;
        }
    }
}

void normalization_elementwise_linear_f32::run_normalization_work_item_linear_single_batch8X(
    const nn::nn_workload_data_t<float> *input_view, nn::nn_workload_data_t<float> *output_view) {
    const auto BatchSize = output_view->parent->lengths.t[NN_DATA_COORD_n];
    const auto NoBatches = BatchSize / C_batch_size;

    const auto input_width = input_view->parent->lengths.t[NN_DATA_COORD_x] * BatchSize;
    const auto output_width = (output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1) * BatchSize;

    const auto num_full_blocks = (output_width / C_simd_width) / C_max_acc;
    const auto partial_block_size = (output_width / C_simd_width ) % C_max_acc;

    const auto output_view_offset = output_view->view_begin.t[NN_DATA_COORD_x] * BatchSize;
    const auto input_view_offset = input_view->view_begin.t[NN_DATA_COORD_x] * BatchSize;

    //for (auto itrB = 0; itrB < BatchSize / C_batch_size; ++itrB)
    {
        auto input_buffer = &static_cast<float*>(input_view->parent->data_buffer)[input_view_offset];
        auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_offset];

        const __m256 coeff_a = _mm256_set1_ps(alpha);
        const __m256 coeff_b = _mm256_set1_ps(beta);

        for (auto block = 0u; block < num_full_blocks; ++block)
        {
            // Run computation.
            normalization_compute_block_linear_single<C_max_acc>(input_buffer, output_buffer, coeff_a, coeff_b);
        }

        switch (partial_block_size)
        {
        case  0: break;
        case  1: normalization_compute_block_linear_single< 1>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  2: normalization_compute_block_linear_single< 2>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  3: normalization_compute_block_linear_single< 3>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  4: normalization_compute_block_linear_single< 4>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  5: normalization_compute_block_linear_single< 5>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  6: normalization_compute_block_linear_single< 6>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  7: normalization_compute_block_linear_single< 7>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  8: normalization_compute_block_linear_single< 8>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case  9: normalization_compute_block_linear_single< 9>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case 10: normalization_compute_block_linear_single<10>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case 11: normalization_compute_block_linear_single<11>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case 12: normalization_compute_block_linear_single<12>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case 13: normalization_compute_block_linear_single<13>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        case 14: normalization_compute_block_linear_single<14>(input_buffer, output_buffer, coeff_a, coeff_b); break;
        default: NN_UNREACHABLE_CODE;
        }

    }
}


template<uint32_t T_NUM_ITERATIONS>
void normalization_compute_subsimd_linear_single(
    float* &input_ptr,
    float* &output_ptr,
    const float coeff_a,
    const float coeff_b)
{
    for (auto iteration = 0u; iteration < T_NUM_ITERATIONS; ++iteration)
    {
        *output_ptr = (*input_ptr) * coeff_a + coeff_b;

        ++input_ptr;
        ++output_ptr;
    }
}

void normalization_elementwise_linear_f32::run_normalization_work_item_linear_single_latency(
    const nn::nn_workload_data_t<float> *input_view, nn::nn_workload_data_t<float> *output_view) {

    const auto input_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
    const auto output_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto num_full_blocks = output_width / C_data_stride;
    const auto partial_block_size = (output_width / C_simd_width) % C_max_acc;
    const auto subsimd_block_size = output_width % C_simd_width;

    const auto output_view_offset = output_view->view_begin.t[NN_DATA_COORD_x];
    const auto input_view_offset = input_view->view_begin.t[NN_DATA_COORD_x];

    auto input_buffer = &static_cast<float*>(input_view->parent->data_buffer)[input_view_offset];
    auto output_buffer = &static_cast<float*>(output_view->parent->data_buffer)[output_view_offset];

    const __m256 coeff_a = _mm256_set1_ps(alpha);
    const __m256 coeff_b = _mm256_set1_ps(beta);

    for (auto block = 0u; block < num_full_blocks; ++block)
    {
        // Run computation.
        normalization_compute_block_linear_single<C_max_acc>(input_buffer, output_buffer, coeff_a, coeff_b);
    }

    switch (partial_block_size)
    {
    case  0: break;
    case  1: normalization_compute_block_linear_single< 1>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case  2: normalization_compute_block_linear_single< 2>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case  3: normalization_compute_block_linear_single< 3>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case  4: normalization_compute_block_linear_single< 4>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case  5: normalization_compute_block_linear_single< 5>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case  6: normalization_compute_block_linear_single< 6>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case  7: normalization_compute_block_linear_single< 7>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case  8: normalization_compute_block_linear_single< 8>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case  9: normalization_compute_block_linear_single< 9>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case 10: normalization_compute_block_linear_single<10>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case 11: normalization_compute_block_linear_single<11>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case 12: normalization_compute_block_linear_single<12>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case 13: normalization_compute_block_linear_single<13>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    case 14: normalization_compute_block_linear_single<14>(input_buffer, output_buffer, coeff_a, coeff_b); break;
    default: NN_UNREACHABLE_CODE;
    }

    switch (subsimd_block_size)
    {
    case 0: break;
    case 1: normalization_compute_subsimd_linear_single<1>(input_buffer, output_buffer, alpha, beta); break;
    case 2: normalization_compute_subsimd_linear_single<2>(input_buffer, output_buffer, alpha, beta); break;
    case 3: normalization_compute_subsimd_linear_single<3>(input_buffer, output_buffer, alpha, beta); break;
    case 4: normalization_compute_subsimd_linear_single<4>(input_buffer, output_buffer, alpha, beta); break;
    case 5: normalization_compute_subsimd_linear_single<5>(input_buffer, output_buffer, alpha, beta); break;
    case 6: normalization_compute_subsimd_linear_single<6>(input_buffer, output_buffer, alpha, beta); break;
    case 7: normalization_compute_subsimd_linear_single<7>(input_buffer, output_buffer, alpha, beta); break;
    default: NN_UNREACHABLE_CODE;
    }
}

void normalization_elementwise_linear_f32::choose_normalization_work_item_linear_single_batching_mode(
    const nn::nn_workload_data_t<float> *input_view, nn::nn_workload_data_t<float> *output_view) {
    auto batch_size = input_view->parent->lengths.t[NN_DATA_COORD_n];
    switch (batch_size)
    {
    case 1:
        run_normalization_work_item_linear_single_latency(input_view, output_view);
        break;
    case 8:
        run_normalization_work_item_linear_single_batch8(input_view, output_view);
        break;
    case 16:
    case 24:
    case 32:
    case 48:
        run_normalization_work_item_linear_single_batch8X(input_view, output_view);
        break;

    default:
        break;
    }
}

struct normalization_elementwise_linear_f32_request_handle {
    normalization_elementwise_linear_f32 *primitive;
    const nn::nn_workload_data_t<float> *input_view;
    nn::nn_workload_data_t<float> *output_view;
};

void unpack_1d_normalization_callback_handle(
    void* void_handle)
{
    auto handle = reinterpret_cast<normalization_elementwise_linear_f32_request_handle *>(void_handle);
    handle->primitive->choose_normalization_work_item_linear_single_batching_mode(handle->input_view,
                                                                                  handle->output_view);
}

void normalization_elementwise_linear_f32::run_multithreaded_1d_normalization_work_item(
    const nn::nn_workload_data_t<float> *input, nn::nn_workload_data_t<float> *output) {
    auto num_hardware_threads = std::min(device->thread_pool.get_num_threads(), 18u);

    const auto item_view_length =
        output->view_end.t[NN_DATA_COORD_x] - output->view_begin.t[NN_DATA_COORD_x] + 1;

    const auto items_per_thread = item_view_length / num_hardware_threads;
    const auto items_modulo = item_view_length % num_hardware_threads;

    // Check if we have enough data to cover all threads.
    if (items_per_thread == 0 && items_modulo < 2)
    {
        // Its tiny data - just do it single threaded way.
        choose_normalization_work_item_linear_single_batching_mode(input, output);
    }
    else
    {
        // Not all threads will be used.
        if (items_per_thread == 0)
            num_hardware_threads = items_modulo;

        // Full cores utilization version.
        std::vector<normalization_elementwise_linear_f32_request_handle> request_handles(num_hardware_threads);
        std::vector<const nn::nn_workload_data_t<float> *> input_views(num_hardware_threads);
        std::vector<nn::nn_workload_data_t<float> *> output_views(num_hardware_threads);

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

        // Fill slave work items.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            auto work_begin = 0u;
            if (thread_id > 0u)
                work_begin = thread_items_sums[thread_id - 1];

            auto work_end = thread_items_sums[thread_id] - 1;

            // Replace nn_workload_datas pointers with views.
            nn_workload_data_coords_t nn_view_begin =
            {
                0,
                work_begin,
                0,
                0,
                0,
                0
            };

            nn_workload_data_coords_t nn_view_end =
            {
                input->get_length(NN_DATA_COORD_n) - 1,
                work_end,
                input->get_length(NN_DATA_COORD_y) - 1,
                input->get_length(NN_DATA_COORD_z) - 1,
                input->get_length(NN_DATA_COORD_p) - 1,
                input->get_length(NN_DATA_COORD_q) - 1
            };

            input_views[thread_id] = new nn::nn_workload_data_t<float>(*input, nn_view_begin, nn_view_end);

            output_views[thread_id] = new nn::nn_workload_data_t<float>(*output, nn_view_begin, nn_view_end);
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(num_hardware_threads);
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            request_handles[thread_id].primitive = this;
            request_handles[thread_id].input_view = input_views[thread_id];
            request_handles[thread_id].output_view = output_views[thread_id];

            job[thread_id].callback = unpack_1d_normalization_callback_handle;
            job[thread_id].request_handle = &request_handles[thread_id];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);

        // Cleanup dynamic memory.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            delete input_views[thread_id];
            delete output_views[thread_id];
        }
    }
}

void normalization_elementwise_linear_f32::forward(const nn::nn_workload_data_t<float> *input,
                                                   nn::nn_workload_data_t<float> *output) {
    nn_workload_data_coords_t in_out_view_coords =
    {
        input->parent->lengths.t[NN_DATA_COORD_n],
        input->parent->buffer_size / static_cast<uint32_t>(sizeof(float)) / input->parent->lengths.t[NN_DATA_COORD_n],
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

    nn::nn_workload_data_t<float>* input_view = new nn::nn_workload_data_t<float>(input->parent->data_buffer, in_out_view_coords, in_out_view_layout);
    nn::nn_workload_data_t<float>* output_view = new nn::nn_workload_data_t<float>(output->parent->data_buffer, in_out_view_coords, in_out_view_layout);

    if (device->thread_pool.get_num_threads() > 1)
    {
        run_multithreaded_1d_normalization_work_item(input_view, output_view);
    }
    else
    {
        choose_normalization_work_item_linear_single_batching_mode(input_view, output_view);
    }

    delete output_view;
    delete input_view;
}

__m256 _inner_mm256_invpow075_ps(__m256 arg)
{
    __m256i e = _mm256_slli_epi32(
                    _mm256_sub_epi32(
                        _mm256_and_si256(
                            _mm256_castps_si256(arg), 
                            _mm256_set1_epi32(0x7f800000)),
                        _mm256_set1_epi32(0x3f800000)),
                    1);

    __m256 p0 = _mm256_castsi256_ps(
                        _mm256_srli_epi32(
                            _mm256_add_epi32(
                                _mm256_mullo_epi32(
                                    _mm256_srai_epi32(
                                        _mm256_and_si256(
                                            e,
                                            _mm256_set1_epi32(0xfc000000)),
                                        2),
                                    _mm256_set1_epi32(-3)),
                                _mm256_set1_epi32(0x7f000000)),
                            1));

    __m256 p1 = _mm256_blendv_ps(
                    _mm256_set1_ps(0.59460355750136053335874998528f),
                    _mm256_set1_ps(1.0f), 
                    _mm256_castsi256_ps(
                        _mm256_cmpeq_epi32(
                            _mm256_and_si256(
                                e, 
                                _mm256_set1_epi32(1<<24)),
                            _mm256_set1_epi32(0))));

    __m256 p2 = _mm256_blendv_ps(
                    _mm256_set1_ps(0.35355339059327376220042218105f),
                    _mm256_set1_ps(1.0f), 
                    _mm256_castsi256_ps(
                        _mm256_cmpeq_epi32(
                            _mm256_and_si256(
                                e, 
                                _mm256_set1_epi32(2<<24)),
                            _mm256_set1_epi32(0))));

    arg = _mm256_castsi256_ps(
            _mm256_or_si256(
                _mm256_and_si256(
                    _mm256_castps_si256(arg),
                    _mm256_set1_epi32(0x007fffff)),
                _mm256_set1_epi32(0x3f800000)));

    __m256 intermediate_result;
    intermediate_result = _mm256_fmadd_ps(arg, _mm256_set1_ps(-0.06251362156237f), _mm256_set1_ps(0.56657226995864f));
    intermediate_result = _mm256_fmadd_ps(arg, intermediate_result, _mm256_set1_ps(-2.12314847503624f));
    intermediate_result = _mm256_fmadd_ps(arg, intermediate_result, _mm256_set1_ps(4.22879355263332f));
    intermediate_result = _mm256_fmadd_ps(arg, intermediate_result, _mm256_set1_ps(-4.79039952143706f));
    intermediate_result = _mm256_fmadd_ps(arg, intermediate_result, _mm256_set1_ps(3.18069569544757f));

    intermediate_result = 
        _mm256_mul_ps(
                _mm256_mul_ps(
                    p0, 
                    p1),
                _mm256_mul_ps(
                    p2, 
                    intermediate_result));

    return intermediate_result;
}

void normalization_response_across_maps_f32::run_3d_normalization_work_item(const nn::nn_workload_data_t<float> *input_view,
                                                                            nn::nn_workload_data_t<float> *output_view) {
    const auto input_column_size = input_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto input_row_size = input_view->parent->lengths.t[NN_DATA_COORD_x] * input_column_size;
    const auto input_batch_size = input_view->parent->lengths.t[NN_DATA_COORD_y] * input_row_size;

    const auto output_column_size = output_view->parent->lengths.t[NN_DATA_COORD_z];
    const auto output_row_size = output_view->parent->lengths.t[NN_DATA_COORD_x] * output_column_size;
    const auto output_batch_size = output_view->parent->lengths.t[NN_DATA_COORD_y] * output_row_size;

    auto input_buffer = static_cast<float*>(input_view->parent->data_buffer);
    auto output_buffer = static_cast<float*>(output_view->parent->data_buffer);

    // Const data.
    const uint32_t permutation_mask[8] = { 1, 2, 3, 4, 5, 6, 7, 0 };
    uint32_t first_load_mask[8] = { 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
    uint32_t last_load_mask[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    const auto neighbourhood = n / 2;

    for (uint32_t neighbour = 0; neighbour < neighbourhood; ++neighbour)
    {
        first_load_mask[neighbour] ^= 0x80000000;
        last_load_mask[neighbour] ^= 0x80000000;
    }

    // Permuters and masks.
    const __m256i forward_permuter = _mm256_loadu_si256((__m256i*)permutation_mask);
    const __m256i first_masker = _mm256_loadu_si256((__m256i*)first_load_mask);
    const __m256i last_masker = _mm256_loadu_si256((__m256i*)last_load_mask);

    for (uint32_t batch = input_view->view_begin.t[NN_DATA_COORD_n]; batch <= input_view->view_end.t[NN_DATA_COORD_n]; ++batch)
    {
        for (uint32_t row = input_view->view_begin.t[NN_DATA_COORD_y], out_row = output_view->view_begin.t[NN_DATA_COORD_y]; 
             row <= input_view->view_end.t[NN_DATA_COORD_y]; 
             ++row, ++out_row)
        {
            for (uint32_t column = input_view->view_begin.t[NN_DATA_COORD_x], out_column = output_view->view_begin.t[NN_DATA_COORD_x]; 
                 column <= input_view->view_end.t[NN_DATA_COORD_x]; 
                 ++column, ++out_column)
            {
                const auto input_address = &input_buffer[batch*input_batch_size + row*input_row_size + column*input_column_size];
                const auto output_address = &output_buffer[batch*output_batch_size + out_row*output_row_size + out_column*output_column_size];

                // Prepare first data chunk.
                __m256 source_tmp = _mm256_maskload_ps(input_address - neighbourhood, first_masker);
                source_tmp = _mm256_mul_ps(source_tmp, source_tmp);

                for (uint32_t feature_map = input_view->view_begin.t[NN_DATA_COORD_z], out_feature_map = output_view->view_begin.t[NN_DATA_COORD_z]; 
                     feature_map <= input_view->view_end.t[NN_DATA_COORD_z];
                     feature_map += C_simd_width, out_feature_map += C_simd_width)
                {
                    // Initialize accumulator.
                    __m256 acc = _mm256_setzero_ps();

                    // Move previous saved chunk to first and load new one as a next.
                    __m256 source_first = source_tmp;
                    __m256 source_second = 
                        (feature_map + C_simd_width <= input_view->view_end.t[NN_DATA_COORD_z])
                            ? _mm256_loadu_ps(input_address + feature_map - neighbourhood + C_simd_width) 
                            : _mm256_maskload_ps(input_address + feature_map - neighbourhood + C_simd_width, last_masker);

                    // Square of new chunk and save for next iteration.
                    source_tmp = source_second = _mm256_mul_ps(source_second, source_second);

                    // Required for final computation.
                    __m256 source_raw = _mm256_loadu_ps(input_address + feature_map);

                    // Forward permute - five times.
                    for (int i = 0; i < n; ++i)
                    {
                        acc = _mm256_add_ps(source_first, acc);
                        source_first = _mm256_permutevar8x32_ps(source_first, forward_permuter);
                        source_second = _mm256_permutevar8x32_ps(source_second, forward_permuter);
                        source_first = _mm256_blend_ps(source_first, source_second, 0x80);
                    }

                    // Do k + alpha * acc.
                    acc = _mm256_fmadd_ps(acc, _mm256_set1_ps(alpha), _mm256_set1_ps(k));

                    // Magic happens here. (acc^-0.75)
                    acc = _inner_mm256_invpow075_ps(acc);

                    // Multiply with input data.
                    acc = _mm256_mul_ps(acc, source_raw);

                    // Save data.
                    _mm256_storeu_ps(output_address + out_feature_map, acc);
                }
            }
        }
    }
}

struct normalization_response_across_maps_f32_request_handle {
    normalization_response_across_maps_f32 *primitive;
    const nn::nn_workload_data_t<float> *input_view;
    nn::nn_workload_data_t<float> *output_view;
};

void unpack_3d_normalization_callback_handle(
    void* void_handle)
{
    auto handle = reinterpret_cast<normalization_response_across_maps_f32_request_handle *>(void_handle);
    handle->primitive->run_3d_normalization_work_item(handle->input_view, handle->output_view);
}

void normalization_response_across_maps_f32::run_multithreaded_3d_normalization_work_item(
    const nn::nn_workload_data_t<float> *input, nn::nn_workload_data_t<float> *output) {
    auto num_hardware_threads = std::min(device->thread_pool.get_num_threads(), max_threads);

    const auto item_view_length =
        output->view_end.t[NN_DATA_COORD_y] - output->view_begin.t[NN_DATA_COORD_y] + 1;

    const auto items_per_thread = item_view_length / num_hardware_threads;
    const auto items_modulo = item_view_length % num_hardware_threads;

    // Check if we have enough data to cover all threads.
    if (items_per_thread == 0 && items_modulo < 2)
    {
        // Its tiny data - just do it singlethreaded way.
        run_3d_normalization_work_item(input, output);
    }
    else
    {
        // Full cores utilization version.
        // Not all threads will be used.
        if (items_per_thread == 0)
            num_hardware_threads = items_modulo;

        std::vector<normalization_response_across_maps_f32_request_handle> request_handles(num_hardware_threads);
        std::vector<const nn::nn_workload_data_t<float> *> input_views(num_hardware_threads);
        std::vector<nn::nn_workload_data_t<float> *> output_views(num_hardware_threads);

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

        // Fill slave work items.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            auto work_begin = 0u;
            if (thread_id > 0u)
                work_begin = thread_items_sums[thread_id - 1];

            auto work_end = thread_items_sums[thread_id] - 1;

            // Replace nn_workload_datas pointers with views.
            nn_workload_data_coords_t nn_view_begin =
            {
                0,
                0,
                work_begin,
                0,
                0,
                0
            };

            nn_workload_data_coords_t nn_view_end =
            {
                input->get_length(NN_DATA_COORD_n) - 1,
                input->get_length(NN_DATA_COORD_x) - 1,
                work_end,
                input->get_length(NN_DATA_COORD_z) - 1,
                input->get_length(NN_DATA_COORD_p) - 1,
                input->get_length(NN_DATA_COORD_q) - 1
            };

            input_views[thread_id] = new nn::nn_workload_data_t<float>(*input, nn_view_begin, nn_view_end);

            output_views[thread_id] = new nn::nn_workload_data_t<float>(*output, nn_view_begin, nn_view_end);
        }

        // Run threads.
        std::vector<nn_multithreaded_request> job(num_hardware_threads);

        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            request_handles[thread_id].primitive = this;
            request_handles[thread_id].input_view = input_views[thread_id];
            request_handles[thread_id].output_view = output_views[thread_id];

            job[thread_id].callback = unpack_3d_normalization_callback_handle;
            job[thread_id].request_handle = &request_handles[thread_id];
        }

        // Wait for all sub threads.
        device->thread_pool.push_job(job);


        // Cleanup dynamic memory.
        for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
        {
            delete input_views[thread_id];
            delete output_views[thread_id];
        }
    }
}

void normalization_response_across_maps_f32::forward(const nn::nn_workload_data_t<float> *input,
                                                     nn::nn_workload_data_t<float> *output) {
    if (device->thread_pool.get_num_threads() > 1) {
        run_multithreaded_3d_normalization_work_item(input, output);
    }
    else {
        run_3d_normalization_work_item(input, output);
    }
}

void wrapper_normalization_work_item(nn_workload_item *const work_item, nn_device_internal *device) {
    switch (work_item->arguments.forward_normalization.mode) {
    case NN_NORMALIZATION_MODE_LINEAR_SINGLE: {
        auto primitive = static_cast<normalization_elementwise_linear_f32 *>(work_item->primitive);
        primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->input[0]->output),
                           reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->output));
        break;
    }
    case NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS:
        auto primitive = static_cast<normalization_response_across_maps_f32 *>(work_item->primitive);
        primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->input[0]->output),
                           reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->output));
        break;
    }
}

normalization_elementwise_linear_f32 *normalization_elementwise_linear_f32::create(float alpha,
                                                                                   float beta,
                                                                                   size_t image_size_x,
                                                                                   size_t image_size_y,
                                                                                   size_t image_size_z,
                                                                                   size_t batch_size,
                                                                                   nn_device_t *device) {
    return new normalization_elementwise_linear_f32(alpha,
                                                    beta,
                                                    image_size_x,
                                                    image_size_y,
                                                    image_size_z,
                                                    batch_size,
                                                    reinterpret_cast<nn_device_internal *>(device));
}

normalization_elementwise_linear_f32::normalization_elementwise_linear_f32(float alpha,
                                                                           float beta,
                                                                           size_t image_size_x,
                                                                           size_t image_size_y,
                                                                           size_t image_size_z,
                                                                           size_t batch_size,
                                                                           nn_device_internal *device)
    : primitive_zxyn_f32_base(batch_size, image_size_z, image_size_x, image_size_y, image_size_z, device),
      normalization_mode(NN_NORMALIZATION_MODE_LINEAR_SINGLE),
      alpha(alpha),
      beta(beta) {}

size_t normalization_elementwise_linear_f32::get_required_input_w() { return output_size_x; }

size_t normalization_elementwise_linear_f32::get_required_input_h() { return output_size_y; }

normalization_response_across_maps_f32 *normalization_response_across_maps_f32::create(float alpha,
                                                                                       float beta,
                                                                                       uint32_t k,
                                                                                       uint32_t n,
                                                                                       size_t image_size_x,
                                                                                       size_t image_size_y,
                                                                                       size_t image_size_z,
                                                                                       size_t batch_size,
                                                                                       nn_device_t *device) {
    return new normalization_response_across_maps_f32(alpha,
                                                      beta,
                                                      k,
                                                      n,
                                                      image_size_x,
                                                      image_size_y,
                                                      image_size_z,
                                                      batch_size,
                                                      reinterpret_cast<nn_device_internal *>(device));
}

normalization_response_across_maps_f32::normalization_response_across_maps_f32(float alpha,
                                                                               float beta,
                                                                               uint32_t k,
                                                                               uint32_t n,
                                                                               size_t image_size_x,
                                                                               size_t image_size_y,
                                                                               size_t image_size_z,
                                                                               size_t batch_size,
                                                                               nn_device_internal *device)
    : primitive_zxyn_f32_base(batch_size, image_size_z, image_size_x, image_size_y, image_size_z, device),
      normalization_mode(NN_NORMALIZATION_MODE_LINEAR_SINGLE),
      alpha(alpha),
      beta(beta),
      k(k),
      n(n) {}

size_t normalization_response_across_maps_f32::get_required_input_w() { return output_size_x; }

size_t normalization_response_across_maps_f32::get_required_input_h() { return output_size_y; }

namespace normalization_elementwise_linear_f32_impl {
nn_event_t NN_API_CALL_CONVENTION forward_async(nn_primitive_handle_t handle,
                                                nn_opaque_data_t *input,
                                                nn_opaque_data_t *output,
                                                size_t dependencies_count,
                                                nn_event_t *dependencies,
                                                NN_API_STATUS *status) {
    auto primitive = static_cast<layer::normalization_elementwise_linear_f32 *>(handle);
    primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(input),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(output));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}

nn_primitive_handle_t NN_API_CALL_CONVENTION create(nn_device_t *device,
                                                    float alpha,
                                                    float beta,
                                                    size_t image_size_x,
                                                    size_t image_size_y,
                                                    size_t image_size_z,
                                                    size_t batch_size,
                                                    NN_API_STATUS *status) {
    SET_STATUS(NN_API_STATUS_OK);
    return layer::normalization_elementwise_linear_f32::create(
        alpha, beta, image_size_x, image_size_y, image_size_z, batch_size, device);
}
}

namespace normalization_response_across_maps_f32_impl {
nn_event_t NN_API_CALL_CONVENTION forward_async(nn_primitive_handle_t handle,
                                                nn_opaque_data_t *input,
                                                nn_opaque_data_t *output,
                                                size_t dependencies_count,
                                                nn_event_t *dependencies,
                                                NN_API_STATUS *status) {
    auto primitive = static_cast<layer::normalization_response_across_maps_f32 *>(handle);
    primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(input),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(output));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}

nn_primitive_handle_t NN_API_CALL_CONVENTION create(nn_device_t *device,
                                                    float alpha,
                                                    float beta,
                                                    uint32_t k,
                                                    uint32_t n,
                                                    size_t image_size_x,
                                                    size_t image_size_y,
                                                    size_t image_size_z,
                                                    size_t batch_size,
                                                    NN_API_STATUS *status) {
    SET_STATUS(NN_API_STATUS_OK);
    return layer::normalization_response_across_maps_f32::create(
        alpha, beta, k, n, image_size_x, image_size_y, image_size_z, batch_size, device);
}
}

} // namespace layer

nn_primitives_normalization_elementwise_linear_f32_0_t nn_primitives_normalization_elementwise_linear_f32_0{
    layer::normalization_elementwise_linear_f32_impl::create,
    layer::helper_zxyn_f32::create_input,
    layer::helper_zxyn_f32::validate_input,
    layer::helper_zxyn_f32::create_output,
    nullptr, // create_view
    layer::normalization_elementwise_linear_f32_impl::forward_async,
    layer::helper_zxyn_f32::copy_output_async };

nn_primitives_normalization_response_across_maps_f32_0_t nn_primitives_normalization_response_across_maps_f32_0{
    layer::normalization_response_across_maps_f32_impl::create,
    layer::helper_zxyn_f32::create_input,
    layer::helper_zxyn_f32::validate_input,
    layer::helper_zxyn_f32::create_output,
    layer::helper_zxyn_f32::create_output_with_padding,
    nullptr, // create_view
    layer::normalization_response_across_maps_f32_impl::forward_async,
    layer::helper_zxyn_f32::copy_output_async };
