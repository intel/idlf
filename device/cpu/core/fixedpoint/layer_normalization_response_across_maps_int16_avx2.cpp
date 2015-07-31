/*
Copyright (c) 2015, Intel Corporation

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
#include "device/cpu/api_internal/data_helper.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "layer_normalization_response_across_maps_int16_avx2.h"

#include <immintrin.h>
#include <string.h>
#include <thread>
#include <vector>

#define EXP_POLY_DEGREE 3
#define LOG_POLY_DEGREE 5

namespace int16_fixedpoint {
    ///////////////////////////////////////////////////////////////////////////////////////////////////

#define POLY0_AVX(x, c0) _mm256_set1_ps(c0)
#define POLY1_AVX(x, c0, c1) _mm256_fmadd_ps(POLY0_AVX(x, c1), x, _mm256_set1_ps(c0))
#define POLY2_AVX(x, c0, c1, c2) _mm256_fmadd_ps(POLY1_AVX(x, c1, c2), x, _mm256_set1_ps(c0))
#define POLY3_AVX(x, c0, c1, c2, c3) _mm256_fmadd_ps(POLY2_AVX(x, c1, c2, c3), x, _mm256_set1_ps(c0))
#define POLY4_AVX(x, c0, c1, c2, c3, c4) _mm256_fmadd_ps(POLY3_AVX(x, c1, c2, c3, c4), x, _mm256_set1_ps(c0))
#define POLY5_AVX(x, c0, c1, c2, c3, c4, c5) _mm256_fmadd_ps(POLY4_AVX(x, c1, c2, c3, c4, c5), x, _mm256_set1_ps(c0))

    inline __m256 _inner_mm256_exp2_ps(__m256 x)
    {
        __m256i ipart;
        __m256 fpart, expipart, expfpart;

        x = _mm256_min_ps(x, _mm256_set1_ps(129.00000f));
        x = _mm256_max_ps(x, _mm256_set1_ps(-126.99999f));

        /* ipart = int(x - 0.5) */
        ipart = _mm256_cvtps_epi32(_mm256_sub_ps(x, _mm256_set1_ps(0.5f)));

        /* fpart = x - ipart */
        fpart = _mm256_sub_ps(x, _mm256_cvtepi32_ps(ipart));

        /* expipart = (float) (1 << ipart) */
        expipart = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ipart, _mm256_set1_epi32(127)), 23));

        /* minimax polynomial fit of 2**x, in range [-0.5, 0.5[ */
#if EXP_POLY_DEGREE == 5
        expfpart = POLY5_AVX(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
#elif EXP_POLY_DEGREE == 4
        expfpart = POLY4_AVX(fpart, 1.0000026f, 6.9300383e-1f, 2.4144275e-1f, 5.2011464e-2f, 1.3534167e-2f);
#elif EXP_POLY_DEGREE == 3
        expfpart = POLY3_AVX(fpart, 9.9992520e-1f, 6.9583356e-1f, 2.2606716e-1f, 7.8024521e-2f);
#elif EXP_POLY_DEGREE == 2
        expfpart = POLY2_AVX(fpart, 1.0017247f, 6.5763628e-1f, 3.3718944e-1f);
#else
#error
#endif
        return _mm256_mul_ps(expipart, expfpart);
    }

    //log2

    inline __m256 _inner_mm256_log2_ps(__m256 x)
    {
        __m256i exp = _mm256_set1_epi32(0x7F800000);
        __m256i mant = _mm256_set1_epi32(0x007FFFFF);

        __m256 one = _mm256_set1_ps(1.0f);

        __m256i i = _mm256_castps_si256(x);

        __m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(_mm256_and_si256(i, exp), 23), _mm256_set1_epi32(127)));

        __m256 m = _mm256_or_ps(_mm256_castsi256_ps(_mm256_and_si256(i, mant)), one);

        __m256 p;

        /* Minimax polynomial fit of log2(x)/(x - 1), for x in range [1, 2[ */
#if LOG_POLY_DEGREE == 6
        p = POLY5_AVX(m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f, 3.1821337e-1f, -3.4436006e-2f);
#elif LOG_POLY_DEGREE == 5
        p = POLY4_AVX(m, 2.8882704548164776201f, -2.52074962577807006663f, 1.48116647521213171641f, -0.465725644288844778798f, 0.0596515482674574969533f);
#elif LOG_POLY_DEGREE == 4
        p = POLY3_AVX(m, 2.61761038894603480148f, -1.75647175389045657003f, 0.688243882994381274313f, -0.107254423828329604454f);
#elif LOG_POLY_DEGREE == 3
        p = POLY2_AVX(m, 2.28330284476918490682f, -1.04913055217340124191f, 0.204446009836232697516f);
#else
#error
#endif

        /* This effectively increases the polynomial degree by one, but ensures that log2(1) == 0*/
        p = _mm256_mul_ps(p, _mm256_sub_ps(m, one));

        return _mm256_add_ps(p, e);
    }

    static inline __m256 _inner_mm256_pow_ps(__m256 x, __m256 y)
    {
        return _inner_mm256_exp2_ps(_mm256_mul_ps(_inner_mm256_log2_ps(x), y));
    }

    static inline __m256 _inner_mm256_invpow075_ps(__m256 arg)
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
            _mm256_set1_epi32(1 << 24)),
            _mm256_set1_epi32(0))));

        __m256 p2 = _mm256_blendv_ps(
            _mm256_set1_ps(0.35355339059327376220042218105f),
            _mm256_set1_ps(1.0f),
            _mm256_castsi256_ps(
            _mm256_cmpeq_epi32(
            _mm256_and_si256(
            e,
            _mm256_set1_epi32(2 << 24)),
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

    static inline __m256 finalize_lrn_computation_p075(__m256 acc,
                                                       const float alpha,
                                                       const float k,
                                                       __m256 source_raw) {
        // Do k + alpha * acc.
        __m256 result = _mm256_fmadd_ps(acc, _mm256_set1_ps(alpha), _mm256_set1_ps(k));

        // Magic happens here. (acc^-0.75)
        result = _inner_mm256_invpow075_ps(result);

        // Multiply with input data.
        result = _mm256_mul_ps(result, source_raw);
        return result;
    }

    struct lrn_fixedpoint_request_handle_p075 {
        const nn::workload_data<int16_t> *input_view;
        nn::workload_data<int16_t> *output_view;
        float k;
        float alpha;
        float scale_in;
        float scale_out;
    };

    normalization_response_across_maps_i16::normalization_response_across_maps_i16(
        uint32_t k,
        uint32_t n,
        float alpha,
        float beta,
        float scale_in,
        float scale_out,
        size_t image_size_x,
        size_t image_size_y,
        size_t image_size_z,
        size_t batch_size,
        size_t output_padding_left,
        size_t output_padding_right,
        size_t output_padding_top,
        size_t output_padding_bottom,
        nn_device_internal *device)
        : primitive_z_block_xyz_i16_base(
            batch_size,
            image_size_z,
            image_size_x,
            image_size_y,
            image_size_z,
            output_padding_left,
            output_padding_right,
            output_padding_top,
            output_padding_bottom,
            device),
        k(k),
        n(n),
        alpha(alpha),
        beta(beta),
        scale_in(scale_in),
        scale_out(scale_out),
        image_size_x(image_size_x),
        image_size_y(image_size_y),
        image_size_z(image_size_z),
        batch_size(batch_size),
        output_padding_left(output_padding_left),
        output_padding_right(output_padding_right),
        output_padding_top(output_padding_top),
        output_padding_bottom(output_padding_bottom),
        device(device),
        in_out_layout(nn::workload_data<int16_t>::layout.pxyznq)
    {}

    void normalization_response_across_maps_i16::forward(
        const nn::workload_data<int16_t> *input,
        nn::workload_data<int16_t> *output)
    {
        assert(n == 5);
        assert(beta == 0.75f);

        if (device == nullptr || device->thread_pool.get_num_threads() == 1) {
            process_lrn_p075<5>(input, output, alpha, k, scale_in, scale_out);
            return;
        }

        const auto size_y = input->get_length(NN_DATA_COORD_y);

        const size_t num_work_chunks = size_y;
        std::vector<lrn_fixedpoint_request_handle_p075> request_handles(num_work_chunks);
        std::vector<nn_multithreaded_request> jobs(num_work_chunks);

        nn_workload_data_coords_t view_begin( 0, 0, 0, 0, 0, 0 );
        nn_workload_data_coords_t view_end(input->get_length(NN_DATA_COORD_n) - 1,
            input->get_length(NN_DATA_COORD_x) - 1,
            0,
            input->get_length(NN_DATA_COORD_z) - 1,
            input->get_length(NN_DATA_COORD_p) - 1,
            input->get_length(NN_DATA_COORD_q) - 1 );

        for (size_t y = 0; y < size_y; ++y){
            //for (size_t x = 0; x < size_x; ++x) {
            const size_t thread_id = y;

            //view_begin.t[NN_DATA_COORD_x] = view_end.t[NN_DATA_COORD_x] = static_cast<uint32_t>(x);
            view_begin.t[NN_DATA_COORD_y] = view_end.t[NN_DATA_COORD_y] = static_cast<uint32_t>(y);

            auto &request_handle = request_handles[thread_id];

            request_handle.input_view = new nn::workload_data<int16_t>(*input, view_begin, view_end);
            request_handle.output_view = new nn::workload_data<int16_t>(*output, view_begin, view_end);

            request_handle.alpha = alpha;
            request_handle.k = k;
            request_handle.scale_in = scale_in;
            request_handle.scale_out = scale_out;

            auto &job = jobs[thread_id];
            job.request_handle = &request_handle;
            job.callback = unpack<5>;
        }

        device->thread_pool.push_job(jobs);

        // cleanup
        for (auto &request_handle : request_handles)
        {
            delete request_handle.input_view;
            delete request_handle.output_view;
        }
    }

    void normalization_response_across_maps_i16::forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs)
    {
        assert(inputs.size() == 1);
        assert(outputs.size() == 1);

        forward(reinterpret_cast<const nn::workload_data<int16_t> *>(inputs[0]),
            reinterpret_cast<nn::workload_data<int16_t> *>(outputs[0]));
    }

    bool normalization_response_across_maps_i16::validate_input(size_t index, nn_workload_data_t *data)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    template <uint32_t T_n> void unpack(void *void_handle) {
        auto handle = reinterpret_cast<lrn_fixedpoint_request_handle_p075 *>(void_handle);
        process_lrn_p075<T_n>(
            handle->input_view, handle->output_view, handle->alpha, handle->k, handle->scale_in, handle->scale_out);
    }

    template <uint32_t T_n>
    void process_lrn_p075(const nn::workload_data<int16_t> *input_view,
                          nn::workload_data<int16_t> *output_view,
                          // const uint32_t n,
                          const float alpha,
                          const float k,
                          const float scale_in,
                          const float scale_out) {
        const size_t size_z_block = 16;
        const size_t storage_simd_width = 16;
        const size_t compute_simd_width = 8;

        const size_t iterations_in_z_block = size_z_block / storage_simd_width;

        static_assert(size_z_block % storage_simd_width == 0, "z block size must be multiply of SIMD width");
        assert(size_z_block == input_view->parent->lengths.t[NN_DATA_COORD_p]);
        assert(size_z_block == output_view->parent->lengths.t[NN_DATA_COORD_p]);

        const size_t num_z_blocks = input_view->parent->lengths.t[NN_DATA_COORD_z];
        assert(num_z_blocks == output_view->parent->lengths.t[NN_DATA_COORD_z]);

        // dont support views on z
        assert(input_view->view_begin.t[NN_DATA_COORD_p] == 0);
        assert(output_view->view_begin.t[NN_DATA_COORD_p] == 0);

        assert(input_view->view_end.t[NN_DATA_COORD_p] + 1 == size_z_block);
        assert(output_view->view_end.t[NN_DATA_COORD_p] + 1 == size_z_block);

        assert(input_view->view_begin.t[NN_DATA_COORD_z] == 0);
        assert(output_view->view_begin.t[NN_DATA_COORD_z] == 0);

        assert(input_view->view_end.t[NN_DATA_COORD_z] + 1 == num_z_blocks);
        assert(output_view->view_end.t[NN_DATA_COORD_z] + 1 == num_z_blocks);

        const auto input_stride_x = input_view->parent->lengths.t[NN_DATA_COORD_p];
        const auto input_stride_y = input_view->parent->lengths.t[NN_DATA_COORD_x] * input_stride_x;
        const auto input_stride_z_block = input_view->parent->lengths.t[NN_DATA_COORD_y] * input_stride_y;
        const auto input_stride_batch = input_view->parent->lengths.t[NN_DATA_COORD_z] * input_stride_z_block;

        const auto output_stride_x = output_view->parent->lengths.t[NN_DATA_COORD_p];
        const auto output_stride_y = output_view->parent->lengths.t[NN_DATA_COORD_x] * output_stride_x;
        const auto output_stride_z_block = output_view->parent->lengths.t[NN_DATA_COORD_y] * output_stride_y;
        const auto output_stride_batch = output_view->parent->lengths.t[NN_DATA_COORD_z] * output_stride_z_block;

        const auto input_view_begin_batch = input_view->view_begin.t[NN_DATA_COORD_n];
        const auto input_view_end_batch = input_view->view_end.t[NN_DATA_COORD_n] + 1;

        const auto input_view_begin_y = input_view->view_begin.t[NN_DATA_COORD_y];
        const auto output_view_begin_y = output_view->view_begin.t[NN_DATA_COORD_y];
        const auto input_view_end_y = input_view->view_end.t[NN_DATA_COORD_y] + 1;

        const auto input_view_begin_x = input_view->view_begin.t[NN_DATA_COORD_x];
        const auto output_view_begin_x = output_view->view_begin.t[NN_DATA_COORD_x];
        const auto input_view_end_x = input_view->view_end.t[NN_DATA_COORD_x] + 1;

        auto input_buffer = static_cast<int16_t *>(input_view->parent->data_buffer);
        auto output_buffer = static_cast<int16_t *>(output_view->parent->data_buffer);

        const auto n = T_n;
        const auto neighbourhood = n / 2; // n / 2
        assert(neighbourhood < compute_simd_width);

        uint32_t forward_permutation_mask[compute_simd_width];
        uint32_t backward_permutation_mask[compute_simd_width];
        const uint8_t backward_blend_mask = (uint8_t)~(0xff << neighbourhood);

        for (uint32_t neighbour = 0; neighbour < neighbourhood; ++neighbour) {
            // const_cast<uint8_t &>(backward_blend_mask) |= 1 << (compute_simd_width - neighbourhood + neighbour);
        }

        for (size_t i = 0; i < compute_simd_width; ++i) {
            forward_permutation_mask[i] = (i + 1) % compute_simd_width;
            backward_permutation_mask[i] = (compute_simd_width + i - neighbourhood) % compute_simd_width;
        }

        // Permuters and masks.
        const __m256i forward_permuter = _mm256_loadu_si256((__m256i *)forward_permutation_mask);
        const __m256i backward_permuter = _mm256_loadu_si256((__m256i *)backward_permutation_mask);

        const __m256 input_scaler = _mm256_set1_ps(1.0f / scale_in);
        const __m256 output_scaler = _mm256_set1_ps(scale_out);

        for (uint32_t batch = input_view_begin_batch; batch < input_view_end_batch; ++batch) {

            for (uint32_t row = input_view_begin_y, out_row = output_view_begin_y; row < input_view_end_y;
                 ++row, ++out_row) {
                for (uint32_t column = input_view_begin_x, out_column = output_view_begin_x; column < input_view_end_x;
                     ++column, ++out_column) {
                    auto input_z_block_start =
                        &input_buffer[batch * input_stride_batch + row * input_stride_y + column * input_stride_x];
                    auto output_z_block_start = &output_buffer[batch * output_stride_batch + out_row * output_stride_y +
                                                               out_column * output_stride_x];

                    __m256 source_previous_rotated_squared = _mm256_setzero_ps();
                    __m256 source_raw = _mm256_mul_ps(
                        input_scaler, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(*(__m128i *)(input_z_block_start))));

                    __m256 source_rotated = _mm256_permutevar8x32_ps(source_raw, backward_permuter);
                    __m256 source_rotated_squared = _mm256_mul_ps(source_rotated, source_rotated);

                    __m256 source_next_raw, source_next_rotated_squared, source_first_squared, source_second_squared;

                    size_t z_block = 0;
                    size_t z_next = compute_simd_width, z = 0;

                    for (; z_block < num_z_blocks; ++z_block, input_z_block_start += input_stride_z_block) {
                        for (; z_next < size_z_block; z_next += compute_simd_width) {
                            source_next_raw = _mm256_mul_ps(
                                input_scaler,
                                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(*(__m128i *)(input_z_block_start + z_next))));
                            __m256 source_next_rotated = _mm256_permutevar8x32_ps(source_next_raw, backward_permuter);
                            source_next_rotated_squared = _mm256_mul_ps(source_next_rotated, source_next_rotated);

                            source_first_squared = _mm256_blend_ps(
                                source_rotated_squared, source_previous_rotated_squared, backward_blend_mask);
                            source_second_squared = _mm256_blend_ps(
                                source_next_rotated_squared, source_rotated_squared, backward_blend_mask);

                            __m256 acc = _mm256_setzero_ps();

#pragma unroll(n)
                            for (size_t i = 0; i < n; ++i) {
                                acc = _mm256_add_ps(source_first_squared, acc);
                                source_first_squared = _mm256_permutevar8x32_ps(source_first_squared, forward_permuter);
                                source_second_squared =
                                    _mm256_permutevar8x32_ps(source_second_squared, forward_permuter);
                                source_first_squared =
                                    _mm256_blend_ps(source_first_squared, source_second_squared, 0x80);
                            }

                            __m256 result = finalize_lrn_computation_p075(acc, alpha, k, source_raw);
                            __m256i result32 = _mm256_cvttps_epi32(_mm256_mul_ps(output_scaler, result));
                            __m256i result32swizzle = _mm256_permute4x64_epi64(result32, 0x0e);
                            _mm_store_si128((__m128i *)(output_z_block_start + z),
                                            _mm256_castsi256_si128(_mm256_packs_epi32(result32, result32swizzle)));

                            z += compute_simd_width;
                            if (z >= size_z_block) {
                                output_z_block_start += output_stride_z_block;
                                z = 0;
                            }

                            source_previous_rotated_squared = source_rotated_squared;
                            source_raw = source_next_raw;
                            source_rotated_squared = source_next_rotated_squared;
                        }

                        z_next = 0;
                    }

                    // compute last elements
                    source_next_rotated_squared = _mm256_setzero_ps();
                    source_first_squared =
                        _mm256_blend_ps(source_rotated_squared, source_previous_rotated_squared, backward_blend_mask);
                    source_second_squared =
                        _mm256_blend_ps(source_next_rotated_squared, source_rotated_squared, backward_blend_mask);

                    __m256 acc = _mm256_setzero_ps();

#pragma unroll(n)
                    for (size_t i = 0; i < n; ++i) {
                        acc = _mm256_add_ps(source_first_squared, acc);
                        source_first_squared = _mm256_permutevar8x32_ps(source_first_squared, forward_permuter);
                        source_second_squared = _mm256_permutevar8x32_ps(source_second_squared, forward_permuter);
                        source_first_squared = _mm256_blend_ps(source_first_squared, source_second_squared, 0x80);
                    }

                    __m256 result = finalize_lrn_computation_p075(acc, alpha, k, source_raw);
                    __m256i result32 = _mm256_cvttps_epi32(_mm256_mul_ps(output_scaler, result));
                    __m256i result32swizzle = _mm256_permute4x64_epi64(result32, 0x0e);
                    _mm_store_si128((__m128i *)(output_z_block_start + z),
                                    _mm256_castsi256_si128(_mm256_packs_epi32(result32, result32swizzle)));
                }
            }
        }
    }

    void wrapper_lrn_fixedpoint_work_item(nn_workload_item *const work_item, nn_device_internal *device)
    {

        auto are_coords_equal = [](const nn_workload_data_coords_t &val, const nn_workload_data_coords_t &coords) {
            return coords.t[NN_DATA_COORD_n] == val.t[NN_DATA_COORD_n] &&
                   coords.t[NN_DATA_COORD_x] == val.t[NN_DATA_COORD_x] &&
                   coords.t[NN_DATA_COORD_y] == val.t[NN_DATA_COORD_y] &&
                   coords.t[NN_DATA_COORD_z] == val.t[NN_DATA_COORD_z] &&
                   coords.t[NN_DATA_COORD_p] == val.t[NN_DATA_COORD_p] &&
                   coords.t[NN_DATA_COORD_q] == val.t[NN_DATA_COORD_q];
        };

        nn::workload_data<int16_t> *input = reinterpret_cast<nn::workload_data<int16_t> *>(work_item->input[0].get_data_view());
        nn::workload_data<int16_t> *output = reinterpret_cast<nn::workload_data<int16_t> *>(work_item->output[0]);

        assert(input->parent->lengths.t[NN_DATA_COORD_p] % 16 == 0);
        assert(are_coords_equal({ NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, input->parent->layout.ordering));
        assert(are_coords_equal({ NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, output->parent->layout.ordering));

        assert(are_coords_equal(input->get_length(), output->get_length()));

        static_cast<normalization_response_across_maps_i16 *>(work_item->primitive)->forward(input, output);
    }

    size_t normalization_response_across_maps_i16::get_required_input_w() { return output_size_x; }
    size_t normalization_response_across_maps_i16::get_required_input_h() { return output_size_y; }

} // namespace int16_fixedpoint

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_normalization_response_across_maps_i16_create_0(
    nn_device_t *device, /* IDLF device handle */
    float alpha,         /* sum scale */
    float beta,          /* sum power */
    uint32_t k,          /* square sum weight */
    uint32_t n,          /* size of moving window on the feature maps */
    size_t image_size_x, /* image width */
    size_t image_size_y, /* image height */
    size_t image_size_z, /* number of feature maps */
    size_t batch_size,   /* size of input batch */
    const nn_primitives_normalization_response_across_maps_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */) {
    SET_STATUS(NN_API_STATUS_OK);

    nn_primitives_softmax_hints_t hints_ = {};
    hints_.fixed_point_fraction_bits.accumulator = 8; // set scale_in default value
    hints_.fixed_point_fraction_bits.output = 8; // set scale_out default value

    if (hints != nullptr)
        hints_ = *hints;

    return new int16_fixedpoint::normalization_response_across_maps_i16(
        k,
        n,
        alpha,
        beta,
        static_cast<float>(1 << hints_.fixed_point_fraction_bits.accumulator), // scale_in
        static_cast<float>(1 << hints_.fixed_point_fraction_bits.output),      // scale_out
        image_size_x,
        image_size_y,
        image_size_z,
        batch_size,
        hints_.output_padding.left,
        hints_.output_padding.right,
        hints_.output_padding.top,
        hints_.output_padding.bottom,
        reinterpret_cast<nn_device_internal *>(device));
}