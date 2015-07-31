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
#include "layer_convert_float_to_int16_fixedpoint_avx2.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>
#include <stdexcept>

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

// SIMD width for this implementation
static const auto C_max_acc = 12u;
static const auto C_batch_size = C_simd_width;

namespace int16_fixedpoint
{
    convert_float_to_int16::convert_float_to_int16(
        const size_t num_output,
        size_t input_w,
        size_t input_h,
        size_t batch_size,
        size_t output_padding_left,
        size_t output_padding_right,
        size_t output_padding_top,
        size_t output_padding_bottom,
        int8_t out_fraction,
        nn_device_internal *device)
        :
        num_input(num_output),
        input_w(input_w),
        input_h(input_h),
        batch_size(batch_size),
        output_padding_left(output_padding_left),
        output_padding_right(output_padding_right),
        output_padding_top(output_padding_top),
        output_padding_bottom(output_padding_bottom),
        out_fraction(out_fraction),
        device(device)
    {
    }

    void convert_float_to_int16::forward(
        const nn::workload_data<float> *input_view,
        nn::workload_data<int16_t> *output_view)
    {
        auto input_ptr = reinterpret_cast<float *>(input_view->parent->data_buffer);
        auto output_ptr = reinterpret_cast<std::int16_t *>(output_view->parent->data_buffer);

        auto input_width = input_view->parent->buffer_size / input_view->parent->data_type_size;

        auto are_coords_equal = [](const nn_workload_data_coords_t &val, const nn_workload_data_coords_t &coords) {
            return coords.t[NN_DATA_COORD_n] == val.t[NN_DATA_COORD_n] &&
                coords.t[NN_DATA_COORD_x] == val.t[NN_DATA_COORD_x] &&
                coords.t[NN_DATA_COORD_y] == val.t[NN_DATA_COORD_y] &&
                coords.t[NN_DATA_COORD_z] == val.t[NN_DATA_COORD_z] &&
                coords.t[NN_DATA_COORD_p] == val.t[NN_DATA_COORD_p] &&
                coords.t[NN_DATA_COORD_q] == val.t[NN_DATA_COORD_q];
        };

        if ((input_view->get_length(NN_DATA_COORD_n) * input_view->get_length(NN_DATA_COORD_x) *
            input_view->get_length(NN_DATA_COORD_y) * input_view->get_length(NN_DATA_COORD_z) *
            input_view->get_length(NN_DATA_COORD_p) * input_view->get_length(NN_DATA_COORD_q) ==
            input_width) &&
            (output_view->get_length(NN_DATA_COORD_n) * output_view->get_length(NN_DATA_COORD_x) *
            output_view->get_length(NN_DATA_COORD_y) * output_view->get_length(NN_DATA_COORD_z) *
            output_view->get_length(NN_DATA_COORD_p) * output_view->get_length(NN_DATA_COORD_q) ==
            input_width)) {
            convert_float_to_int16_fixedpoint_contiguous(input_width, input_ptr, output_ptr);
        }
        else if ((input_view->parent->lengths.t[NN_DATA_COORD_z] == 3 &&
            input_view->parent->lengths.t[NN_DATA_COORD_p] == 1 &&
            input_view->parent->lengths.t[NN_DATA_COORD_q] == 1) &&
            are_coords_equal({ NN_DATA_COORD_z,
            NN_DATA_COORD_x,
            NN_DATA_COORD_y,
            NN_DATA_COORD_n,
            NN_DATA_COORD_p,
            NN_DATA_COORD_q },
            input_view->parent->layout.ordering) &&

            are_coords_equal({ NN_DATA_COORD_p,
            NN_DATA_COORD_x,
            NN_DATA_COORD_y,
            NN_DATA_COORD_z,
            NN_DATA_COORD_n,
            NN_DATA_COORD_q },
            output_view->parent->layout.ordering) &&
            are_coords_equal({ input_view->parent->lengths.t[NN_DATA_COORD_n],
            input_view->parent->lengths.t[NN_DATA_COORD_x],
            input_view->parent->lengths.t[NN_DATA_COORD_y],
            1,
            4,
            1 },
            output_view->parent->lengths)) {

            size_t output_z_block_size = output_view->parent->lengths.t[NN_DATA_COORD_p];
            const size_t batch_window_start = output_view->view_begin.t[NN_DATA_COORD_n];
            const size_t batch_window_size = output_view->view_end.t[NN_DATA_COORD_n] - output_view->view_begin.t[NN_DATA_COORD_n] + 1;

            // outputs
            const size_t output_size_x = output_view->parent->lengths.t[NN_DATA_COORD_x],
                output_stride_x = output_z_block_size;
            const size_t output_size_y = output_view->parent->lengths.t[NN_DATA_COORD_y],
                output_stride_y = output_stride_x * output_size_x;
            const size_t output_size_z = output_view->parent->lengths.t[NN_DATA_COORD_z] * output_z_block_size,
                output_stride_z_block = output_stride_y * output_size_y;
            const size_t output_stride_batch = output_size_x * output_size_y * output_size_z;

            const size_t output_window_start_x = output_view->view_begin.t[NN_DATA_COORD_x],
                output_window_size_x = output_view->view_end.t[NN_DATA_COORD_x] - output_window_start_x + 1;
            const size_t output_window_start_y = output_view->view_begin.t[NN_DATA_COORD_y],
                output_window_size_y = output_view->view_end.t[NN_DATA_COORD_y] - output_window_start_y + 1;
            const size_t output_window_start_z = output_view->view_begin.t[NN_DATA_COORD_z],
                output_window_size_z =
                (output_view->view_end.t[NN_DATA_COORD_z] - output_window_start_z + 1) * output_z_block_size;

            // inputs
            const size_t input_size_x = input_view->parent->lengths.t[NN_DATA_COORD_x],
                input_stride_x = input_view->parent->lengths.t[NN_DATA_COORD_z],
                input_window_size_x = input_view->view_end.t[NN_DATA_COORD_x] - input_view->view_begin.t[NN_DATA_COORD_x] + 1;
            const size_t input_size_y = input_view->parent->lengths.t[NN_DATA_COORD_y],
                input_stride_y = input_stride_x * input_size_x,
                input_window_size_y = input_view->view_end.t[NN_DATA_COORD_y] - input_view->view_begin.t[NN_DATA_COORD_y] + 1;
            const size_t input_size_z = input_view->parent->lengths.t[NN_DATA_COORD_z],
                input_stride_z_block = input_stride_y * input_size_y,
                input_window_size_z_blocks = input_view->view_end.t[NN_DATA_COORD_z] - input_view->view_begin.t[NN_DATA_COORD_z] + 1;
            const size_t input_stride_batch = input_size_x * input_size_y * input_size_z;

            const size_t input_start_x = input_view->view_begin.t[NN_DATA_COORD_x];
            const size_t input_start_y = input_view->view_begin.t[NN_DATA_COORD_y];
            const size_t input_start_z_block = input_view->view_begin.t[NN_DATA_COORD_z];

            const auto threadpool_size = device->thread_pool.get_num_threads();
            const auto task_count = batch_window_size;

            // Single thread
            if (threadpool_size < 2 || task_count < 2)
            {
                int16_t *output_window = (int16_t *)output_view->parent->data_buffer
                    + output_window_start_x * output_stride_x
                    + output_window_start_y * output_stride_y
                    + output_window_start_z * output_stride_z_block
                    + batch_window_start * output_stride_batch;

                float *input_window = (float *)input_view->parent->data_buffer
                    + input_start_x * input_stride_x
                    + input_start_y * input_stride_y
                    + input_start_z_block * input_stride_z_block
                    + batch_window_start * input_stride_batch;

                input_width = input_window_size_x * input_window_size_y * input_window_size_z_blocks * batch_window_size;
                convert_float_to_int16_fixedpoint_rgb(
                    input_window,
                    output_window,
                    batch_window_size,
                    input_window_size_z_blocks,
                    input_window_size_x,
                    input_window_size_y,
                    input_stride_batch,
                    output_stride_batch,
                    input_stride_y,
                    output_stride_y,
                    out_fraction);
            }
            else // Multi-threaded
            {
                std::vector<nn_multithreaded_request> jobs(task_count);
                std::vector<convert_float_to_int16_request_handle> request_handles(task_count);

                for (size_t it_batch = 0; it_batch < batch_window_size; ++it_batch)
                {
                    int16_t *output_window = (int16_t *)output_view->parent->data_buffer
                        + output_window_start_x * output_stride_x
                        + output_window_start_y * output_stride_y
                        + output_window_start_z * output_stride_z_block
                        + (batch_window_start + it_batch) * output_stride_batch;

                    float *input_window = (float *)input_view->parent->data_buffer
                        + input_start_x * input_stride_x
                        + input_start_y * input_stride_y
                        + input_start_z_block * input_stride_z_block
                        + (batch_window_start + it_batch) * input_stride_batch;

                    request_handles[it_batch].input_window = input_window;
                    request_handles[it_batch].output_window = output_window;
                    request_handles[it_batch].batch_window_size = 1;
                    request_handles[it_batch].input_window_size_z_blocks = input_window_size_z_blocks;
                    request_handles[it_batch].input_window_size_x = input_window_size_x;
                    request_handles[it_batch].input_window_size_y = input_window_size_y;
                    request_handles[it_batch].input_stride_batch = input_stride_batch;
                    request_handles[it_batch].output_stride_batch = output_stride_batch;
                    request_handles[it_batch].input_stride_y = input_stride_y;
                    request_handles[it_batch].output_stride_y = output_stride_y;
                    request_handles[it_batch].output_fraction = out_fraction;

                    jobs[it_batch].callback = unpack_convert_float_to_int16_callback_handle;
                    jobs[it_batch].request_handle = &request_handles[it_batch];
                }

                // Wait for all sub threads.
                device->thread_pool.push_job(jobs);
            }
        }
        else
        {
            throw std::invalid_argument(
                "convert float to int16 only works with contiguous buffer or RGB to RGB0 conversion");
        }
    }

    void convert_float_to_int16::forward(
        const std::vector<const nn_workload_data_t *> &inputs,
        const std::vector<const nn_workload_data_t *> &parameters,
        const std::vector<nn_workload_data_t *> &outputs) {
        assert(inputs.size() == 1);
        assert(outputs.size() == 1);

        forward(reinterpret_cast<const nn::workload_data<float> *>(inputs[0]),
            reinterpret_cast<nn::workload_data<int16_t> *>(outputs[0]));
    }

    bool convert_float_to_int16::validate_input(size_t index, nn_workload_data_t *data)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    std::vector<nn_workload_data_t *> convert_float_to_int16::create_inputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, float>::create(
            device, input_w, input_h, num_input, batch_size, 0, 0, 0, 0, allocate_delta) };
    }

    std::vector<nn_workload_data_t *> convert_float_to_int16::create_outputs(bool allocate_delta) {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, int16_t>::create(
            device,
            input_w,
            input_h,
            num_input,
            batch_size,
            block_size,
            output_padding_left,
            output_padding_right,
            output_padding_top,
            output_padding_bottom) };
    }

#define CONVERT_TO_SAT_INT16(N, V1, V2) \
    if (T_SIZE < N) \
    V1 = _mm256_castsi256_ps(_mm256_packs_epi32(_mm256_castps_si256(V1), _mm256_setzero_si256())); \
else \
    V1 = _mm256_castsi256_ps(_mm256_packs_epi32(_mm256_castps_si256(V1), _mm256_castps_si256(V2)))

#define PERMUTE_4x64(N, V1) \
    if (T_SIZE >= 1) \
    acc0 = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(acc0), 0xd8))

#define STORE(ADR, N, V1) \
    if (T_SIZE >= (N + 1)) \
    { \
    if (T_SIZE < (N + 2)) \
    _mm256_maskstore_epi32((int32_t *)ADR + (N / 2) * C_batch_size, _mm256_castps_si256(V1), mask); \
    else \
    _mm256_storeu_si256((__m256i *)ADR + (N / 2) * C_batch_size, _mm256_castps_si256(V1)); \
    }

    template<uint32_t T_SIZE>
    void convert_float_toint16_compute_block_shiftR(
        const float* &input_ptr,
        int16_t* &output_ptr,
        __m256 &multiplier)
    {
        // We are not using table of registers and unroll pragmas
        // due to compiler which have issues with register allocation
        // and needs special, obvious treatment. Template immediate
        // arguments matching will remove all conditions in this code.
        __m256  acc0, acc1, acc2, acc3, acc4,
            acc5, acc6, acc7, acc8, acc9,
            acc10, acc11, acc12, acc13, acc14, acc15;

        // Load inputs and perform e^x
        if (T_SIZE >= 1)  {
            acc0 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 0 * C_batch_size), multiplier);
            acc1 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 1 * C_batch_size), multiplier);
        }
        if (T_SIZE >= 3)  {
            acc2 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 2 * C_batch_size), multiplier);
            acc3 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 3 * C_batch_size), multiplier);
        }
        if (T_SIZE >= 5)  {
            acc4 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 4 * C_batch_size), multiplier);
            acc5 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 5 * C_batch_size), multiplier);
        }
        if (T_SIZE >= 7)  {
            acc6 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 6 * C_batch_size), multiplier);
            acc7 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 7 * C_batch_size), multiplier);
        }
        if (T_SIZE >= 9)  {
            acc8 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 8 * C_batch_size), multiplier);
            acc9 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 9 * C_batch_size), multiplier);
        }
        if (T_SIZE >= 11) {
            acc10 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 10 * C_batch_size), multiplier);
            acc11 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 11 * C_batch_size), multiplier);
        }
        if (T_SIZE >= 13) {
            acc12 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 12 * C_batch_size), multiplier);
            acc13 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 13 * C_batch_size), multiplier);
        }
        if (T_SIZE >= 15){
            acc14 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 14 * C_batch_size), multiplier);
            acc15 = _mm256_div_ps(_mm256_loadu_ps(input_ptr + 15 * C_batch_size), multiplier);
        }

        // round results.
        if (T_SIZE >= 1){
            acc0 = _mm256_round_ps(acc0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            acc1 = _mm256_round_ps(acc1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        if (T_SIZE >= 3) {
            acc2 = _mm256_round_ps(acc2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            acc3 = _mm256_round_ps(acc3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        if (T_SIZE >= 5) {
            acc4 = _mm256_round_ps(acc4, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            acc5 = _mm256_round_ps(acc5, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        if (T_SIZE >= 7) {
            acc6 = _mm256_round_ps(acc6, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            acc7 = _mm256_round_ps(acc7, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        if (T_SIZE >= 9) {
            acc8 = _mm256_round_ps(acc8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            acc9 = _mm256_round_ps(acc9, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        if (T_SIZE >= 11) {
            acc10 = _mm256_round_ps(acc10, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            acc11 = _mm256_round_ps(acc11, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        if (T_SIZE >= 13) {
            acc12 = _mm256_round_ps(acc12, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            acc13 = _mm256_round_ps(acc13, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        if (T_SIZE >= 15) {
            acc14 = _mm256_round_ps(acc14, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            acc15 = _mm256_round_ps(acc15, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }

        // convert to int32
        if (T_SIZE >= 1) {
            acc0 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc0));
            acc1 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc1));
        }
        if (T_SIZE >= 3){
            acc2 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc2));
            acc3 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc3));
        }
        if (T_SIZE >= 5) {
            acc4 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc4));
            acc5 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc5));
        }
        if (T_SIZE >= 7) {
            acc6 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc6));
            acc7 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc7));
        }
        if (T_SIZE >= 9) {
            acc8 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc8));
            acc9 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc9));
        }
        if (T_SIZE >= 11) {
            acc10 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc10));
            acc11 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc11));
        }
        if (T_SIZE >= 13){
            acc12 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc12));
            acc13 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc13));
        }
        if (T_SIZE >= 15){
            acc14 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc14));
            acc15 = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc15));
        }

        ////convert to saturated int16
        if (T_SIZE >= 1)
        {
            CONVERT_TO_SAT_INT16(2, acc0, acc1);
        }
        if (T_SIZE >= 3)
        {
            CONVERT_TO_SAT_INT16(4, acc2, acc3);
        }
        if (T_SIZE >= 5)
        {
            CONVERT_TO_SAT_INT16(6, acc4, acc5);
        }
        if (T_SIZE >= 7)
        {
            CONVERT_TO_SAT_INT16(8, acc6, acc7);
        }
        if (T_SIZE >= 9)
        {
            CONVERT_TO_SAT_INT16(10, acc8, acc9);
        }
        if (T_SIZE >= 11)
        {
            CONVERT_TO_SAT_INT16(12, acc10, acc11);
        }
        if (T_SIZE >= 13)
        {
            CONVERT_TO_SAT_INT16(14, acc12, acc13);
        }
        if (T_SIZE >= 15)
        {
            CONVERT_TO_SAT_INT16(16, acc14, acc15);
        }

        PERMUTE_4x64(1, acc0);
        PERMUTE_4x64(3, acc2);
        PERMUTE_4x64(5, acc4);
        PERMUTE_4x64(7, acc6);
        PERMUTE_4x64(9, acc8);
        PERMUTE_4x64(11, acc10);
        PERMUTE_4x64(13, acc12);
        PERMUTE_4x64(15, acc14);

        __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
        // Store results.

        STORE(output_ptr, 0, acc0);
        STORE(output_ptr, 2, acc2);
        STORE(output_ptr, 4, acc4);
        STORE(output_ptr, 6, acc6);
        STORE(output_ptr, 8, acc8);
        STORE(output_ptr, 10, acc10);
        STORE(output_ptr, 12, acc12);
        STORE(output_ptr, 14, acc14);

        input_ptr += C_batch_size*T_SIZE;
        output_ptr += C_batch_size*T_SIZE;
    }

    void convert_float_to_int16::convert_float_to_int16_fixedpoint_contiguous(
        size_t output_width,
        const float *input_ptr,
        std::int16_t *output_ptr)
    {
        const auto num_full_blocks = output_width / C_max_acc;
        const auto partial_block_size = output_width % C_max_acc;

        __m256 multiplier;
        if (out_fraction > 8)
            multiplier = _mm256_set1_ps((float)(1 << out_fraction));
        else
            multiplier = _mm256_set1_ps((float)(1 / (1 << -out_fraction)));

        for (auto block = 0u; block < num_full_blocks; ++block) {
            // Run computation.
            convert_float_toint16_compute_block_shiftR<C_max_acc>(input_ptr, output_ptr, multiplier);
        }

        switch (partial_block_size) {
        case 0:
            break;
        case 1:
            convert_float_toint16_compute_block_shiftR<1>(input_ptr, output_ptr, multiplier);
            break;
        case 2:
            convert_float_toint16_compute_block_shiftR<2>(input_ptr, output_ptr, multiplier);
            break;
        case 3:
            convert_float_toint16_compute_block_shiftR<3>(input_ptr, output_ptr, multiplier);
            break;
        case 4:
            convert_float_toint16_compute_block_shiftR<4>(input_ptr, output_ptr, multiplier);
            break;
        case 5:
            convert_float_toint16_compute_block_shiftR<5>(input_ptr, output_ptr, multiplier);
            break;
        case 6:
            convert_float_toint16_compute_block_shiftR<6>(input_ptr, output_ptr, multiplier);
            break;
        case 7:
            convert_float_toint16_compute_block_shiftR<7>(input_ptr, output_ptr, multiplier);
            break;
        case 8:
            convert_float_toint16_compute_block_shiftR<8>(input_ptr, output_ptr, multiplier);
            break;
        case 9:
            convert_float_toint16_compute_block_shiftR<9>(input_ptr, output_ptr, multiplier);
            break;
        case 10:
            convert_float_toint16_compute_block_shiftR<10>(input_ptr, output_ptr, multiplier);
            break;
        case 11:
            convert_float_toint16_compute_block_shiftR<11>(input_ptr, output_ptr, multiplier);
            break;
        case 12:
            convert_float_toint16_compute_block_shiftR<12>(input_ptr, output_ptr, multiplier);
            break;
        case 13:
            convert_float_toint16_compute_block_shiftR<13>(input_ptr, output_ptr, multiplier);
            break;
        case 14:
            convert_float_toint16_compute_block_shiftR<14>(input_ptr, output_ptr, multiplier);
            break;
        default:
            NN_UNREACHABLE_CODE;
        }
    }

    void convert_float_to_int16_fixedpoint_rgb(
        const float *const input_ptr,
        int16_t *const output_ptr,
        size_t batch_window_size,
        size_t input_view_num_feature_maps,
        size_t input_view_width,
        size_t input_view_height,
        size_t input_stride_batch,
        size_t output_stride_batch,
        size_t input_stride_y,
        size_t output_stride_y,
        int8_t output_fraction
        )
    {
        const size_t input_block_size = sizeof(__m256) / sizeof(float)* 3;
        const size_t output_block_size = sizeof(__m256i) / sizeof(std::int16_t) * 2;
        const float scale = (1.0f * (1 << output_fraction));
        const __m256 scaler = _mm256_set1_ps(scale);
        size_t i;

        const float *input_ptr_batch_base = input_ptr;
        int16_t *output_ptr_batch_base = output_ptr;

        for (size_t it_batch = 0; it_batch < batch_window_size; ++it_batch)
        {
            const float *input_ptr_row_base = input_ptr_batch_base;
            int16_t *output_ptr_row_base = output_ptr_batch_base;

            for (int y = 0; y < input_view_height; y++)
            {
                const float *input_ptr_in_row = input_ptr_row_base;
                int16_t *output_ptr_in_row = output_ptr_row_base;

                for (i = 0; i < input_view_width * input_view_num_feature_maps - input_block_size + 1; i += input_block_size)
                {
                    __m256 in1 = _mm256_loadu_ps(input_ptr_in_row);
                    __m256 in2 = _mm256_loadu_ps(input_ptr_in_row + sizeof(__m256) / sizeof(float));
                    __m256 in3 = _mm256_loadu_ps(input_ptr_in_row + 2 * sizeof(__m256) / sizeof(float));

                    // convert to fixed point int32
                    __m256i tmp76543210 = _mm256_cvtps_epi32(_mm256_mul_ps(in1, scaler));
                    __m256i tmpFEDCBA98 = _mm256_cvtps_epi32(_mm256_mul_ps(in2, scaler));
                    __m256i tmpNMLKJIHG = _mm256_cvtps_epi32(_mm256_mul_ps(in3, scaler));

                    // swizzle to correct subpixel order
                    __m256i tmpBA987654 = _mm256_permute2x128_si256(tmp76543210, tmpFEDCBA98, 0x21);
                    __m256i tmpJIHGFEDC = _mm256_permute2x128_si256(tmpFEDCBA98, tmpNMLKJIHG, 0x21);

                    __m256i tmp76x8x210 = _mm256_blend_epi32(tmp76543210, tmpBA987654, /*00x1'x000*/ 0x10);
                    __m256i tmpBA9x3x54 = _mm256_blend_epi32(tmp76543210, tmpBA987654, /*111x'0x11*/ 0xf7);
                    __m256i tmpJIxKxEDC = _mm256_blend_epi32(tmpNMLKJIHG, tmpJIHGFEDC, /*11x0'x111*/ 0xef);
                    __m256i tmpNMLxFxHG = _mm256_blend_epi32(tmpNMLKJIHG, tmpJIHGFEDC, /*000x'1x00*/ 0x08);

                    __m256i tmpBA9x76x83x54x210 = _mm256_packs_epi32(tmp76x8x210, tmpBA9x3x54);
                    __m256i tmpNMLxJIxKFxHGxEDC = _mm256_packs_epi32(tmpJIxKxEDC, tmpNMLxFxHG);

                    __m256i tmpFxHGxEDC3x54x210 = _mm256_permute2x128_si256(tmpBA9x76x83x54x210, tmpNMLxJIxKFxHGxEDC, 0x20);
                    __m256i tmpNMLxJIxKBA9x76x8 = _mm256_permute2x128_si256(tmpBA9x76x83x54x210, tmpNMLxJIxKFxHGxEDC, 0x31);

                    __m256i tmpxHGFxEDCx543x210 = _mm256_shufflehi_epi16(tmpFxHGxEDC3x54x210, /*2103*/ 0x93);
                    __m256i tmpxNMLxKJIxBA9x876 =
                        _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(tmpNMLxJIxKBA9x76x8, /*1032*/ 0x4e), /*0321*/ 0x39);

                    __m256i zero = _mm256_setzero_si256();

                    __m256i tmpzHGFzEDCz543z210 = _mm256_blend_epi16(tmpxHGFxEDCx543x210, zero, /*1000'1000'1000'1000*/ 0x88);
                    __m256i tmpzNMLzKJIzBA9z876 = _mm256_blend_epi16(tmpxNMLxKJIxBA9x876, zero, /*1000'1000'1000'1000*/ 0x88);

                    __m256i tmpzBA9z876z543z210 = _mm256_permute2x128_si256(tmpzHGFzEDCz543z210, tmpzNMLzKJIzBA9z876, 0x20);
                    __m256i tmpzNMLzKJIzHGFzEDC = _mm256_permute2x128_si256(tmpzHGFzEDCz543z210, tmpzNMLzKJIzBA9z876, 0x31);

                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output_ptr_in_row), tmpzBA9z876z543z210);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output_ptr_in_row)+1, tmpzNMLzKJIzHGFzEDC);

                    input_ptr_in_row += input_block_size;
                    output_ptr_in_row += output_block_size;
                }

                for (; i < input_view_width * input_view_num_feature_maps; i += 3)
                {
                    output_ptr_in_row[0] = _mm_extract_epi16(
                        _mm_packs_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_set1_ps(input_ptr_in_row[0]), _mm256_castps256_ps128(scaler))),
                        _mm_setzero_si128()),
                        0);
                    output_ptr_in_row[1] = _mm_extract_epi16(
                        _mm_packs_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_set1_ps(input_ptr_in_row[1]), _mm256_castps256_ps128(scaler))),
                        _mm_setzero_si128()),
                        0);
                    output_ptr_in_row[2] = _mm_extract_epi16(
                        _mm_packs_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_set1_ps(input_ptr_in_row[2]), _mm256_castps256_ps128(scaler))),
                        _mm_setzero_si128()),
                        0);
                    output_ptr_in_row[3] = 0;
                    input_ptr_in_row += 3;
                    output_ptr_in_row += 4;
                }

                input_ptr_row_base += input_stride_y;
                output_ptr_row_base += output_stride_y;
            }

            input_ptr_batch_base += input_stride_batch;
            output_ptr_batch_base += output_stride_batch;
        }
    }

    void unpack_convert_float_to_int16_callback_handle(void *void_handle)
    {
        convert_float_to_int16::convert_float_to_int16_request_handle* handle = reinterpret_cast<convert_float_to_int16::convert_float_to_int16_request_handle*>(void_handle);

        convert_float_to_int16_fixedpoint_rgb(
            handle->input_window,
            handle->output_window,
            handle->batch_window_size,
            handle->input_window_size_z_blocks,
            handle->input_window_size_x,
            handle->input_window_size_y,
            handle->input_stride_batch,
            handle->output_stride_batch,
            handle->input_stride_y,
            handle->output_stride_y,
            handle->output_fraction);
    }

    void run_convert_float_to_int16_fp_work_item(nn_workload_item *const work_item, nn_device_internal* device)
    {
        auto input_view = reinterpret_cast<nn::workload_data<float> *>(work_item->input[0].get_data_view());
        auto output_view = reinterpret_cast<nn::workload_data<std::int16_t> *>(work_item->output[0]);

        static_cast<convert_float_to_int16 *>(work_item->primitive)->forward(input_view, output_view);
    }

} // namepace

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_convert_float_to_i16_create_0(
    nn_device_t *device,  /* IDLF device handle */
    size_t image_size_x,  /* image width */
    size_t image_size_y,  /* image height */
    size_t image_size_z,  /* number of feature maps */
    size_t batch_size,    /* size of input batch */
    const nn_primitives_convert_float_to_i16_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);

    nn_primitives_softmax_hints_t hints_ = {};
    hints_.fixed_point_fraction_bits.output = 0; // set output_fraction default value

    if (hints != nullptr)
        hints_ = *hints;

    return new int16_fixedpoint::convert_float_to_int16(
        image_size_z,
        image_size_x,
        image_size_y,
        batch_size,
        hints_.output_padding.left,
        hints_.output_padding.right,
        hints_.output_padding.top,
        hints_.output_padding.bottom,
        hints_.fixed_point_fraction_bits.output,
        reinterpret_cast<nn_device_internal *>(device));
}