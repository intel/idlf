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

#include "../../../common/nn_workload_data.h"
#include "../../api_internal/nn_device_interface_0_internal.h"
#include "layer_normalization_response_across_maps_int16_avx2.h"
//#include "fastmath.h"

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

    inline __m256 exp2f4(__m256 x)
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

        //__m256 sum = _mm256_fmadd_ps(_mm256_set1_ps(7.8024521e-2f), x, _mm256_set1_ps(2.2606716e-1f));
        //sum = _mm256_fmadd_ps(sum, x, _mm256_set1_ps(6.9583356e-1f));
        //sum = _mm256_fmadd_ps(sum, x, _mm256_set1_ps(9.9992520e-1f));

        return _mm256_mul_ps(expipart, expfpart);
    }

    //log2

    inline __m256 log2f4(__m256 x)
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

    static inline __m256 powf4(__m256 x, __m256 y)
    {
        return exp2f4(_mm256_mul_ps(log2f4(x), y));
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

    inline void transpose8_ps(__m256 * inout)
    {
        __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
        __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
        __t0 = _mm256_unpacklo_ps(inout[0], inout[1]);
        __t1 = _mm256_unpackhi_ps(inout[0], inout[1]);
        __t2 = _mm256_unpacklo_ps(inout[2], inout[3]);
        __t3 = _mm256_unpackhi_ps(inout[2], inout[3]);
        __t4 = _mm256_unpacklo_ps(inout[4], inout[5]);
        __t5 = _mm256_unpackhi_ps(inout[4], inout[5]);
        __t6 = _mm256_unpacklo_ps(inout[6], inout[7]);
        __t7 = _mm256_unpackhi_ps(inout[6], inout[7]);
        __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
        __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
        __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
        __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
        __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
        __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
        __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
        __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
        inout[0] = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
        inout[1] = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
        inout[2] = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
        inout[3] = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
        inout[4] = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
        inout[5] = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
        inout[6] = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
        inout[7] = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
    }

    //template <const uint32_t NoStores>
    //inline void StoreResultsT(__m256 * LRN_result, int16_t * output_buffer, __m256 scale_out)
    //{
    //    transpose8_ps((__m256 *)LRN_result);
    //    for (auto itrF2S = 0; itrF2S < NoStores; ++itrF2S)
    //    {
    //        __m256 acc = _mm256_div_ps(LRN_result[itrF2S + 0], scale_out);
    //        acc = _mm256_round_ps(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    //        acc = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc));
    //        acc = _mm256_castsi256_ps(_mm256_packs_epi32(_mm256_castps_si256(acc), _mm256_setzero_si256()));
    //        acc = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(acc), 0xd8));
    //        _mm_storeu_si128((__m128i *)output_buffer + itrF2S * C_simd_width, _mm256_castsi256_si128(acc));
    //    }
    //}

    /*inline void StoreResults(__m256 * LRN_result, int16_t * output_buffer, __m256 scale_out, uint32_t NoStores)
    {
    transpose8_ps((__m256 *)LRN_result);
    for (auto itrF2S = 0; itrF2S < NoStores; ++itrF2S)
    {
    __m256 acc = _mm256_mul_ps(LRN_result[itrF2S + 0], scale_out);
    acc = _mm256_round_ps(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    acc = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc));
    acc = _mm256_castsi256_ps(_mm256_packs_epi32(_mm256_castps_si256(acc), _mm256_setzero_si256()));
    acc = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(acc), 0xd8));
    _mm_storeu_si128((__m128i *)(output_buffer + itrF2S * C_simd_width), _mm256_castsi256_si128(_mm256_castps_si256(acc)));
    }
    }*/

    inline void StoreResult(__m256 LRN_result, int16_t * output_buffer, const __m256 scale_out)
    {
        __m256 acc = _mm256_mul_ps(LRN_result, scale_out);
        acc = _mm256_round_ps(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        acc = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc));
        acc = _mm256_castsi256_ps(_mm256_packs_epi32(_mm256_castps_si256(acc), _mm256_setzero_si256()));
        acc = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(acc), 0xd8));
        _mm_storeu_si128((__m128i *)(output_buffer), _mm256_castsi256_si128(_mm256_castps_si256(acc)));
    }

    inline void StoreResultModulo(__m256 LRN_result, int16_t * output_buffer, __m256 scale_out, uint8_t mask)
    {
        __m256 acc = _mm256_mul_ps(LRN_result, scale_out);
        acc = _mm256_round_ps(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        acc = _mm256_castsi256_ps(_mm256_cvttps_epi32(acc));
        acc = _mm256_castsi256_ps(_mm256_packs_epi32(_mm256_castps_si256(acc), _mm256_setzero_si256()));
        acc = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(acc), 0xd8));
        __m128i acc16 = _mm256_castsi256_si128(_mm256_castps_si256(acc));
        memcpy((int16_t *)output_buffer, &acc16, mask * sizeof(int16_t));
    }

    inline __m256 _mm256_calculate_LRN_item(__m256i sum, const __m256 scale_v2, const __m256 coeff_a, const __m256 coeff_k, const __m256 coeff_b, __m256i middle_value, const __m256 scale_v)
    {
        __m256 sum_f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(sum), scale_v2);
        sum_f2 = _mm256_fmadd_ps(sum_f2, coeff_a, coeff_k);
        sum_f2 = _inner_mm256_invpow075_ps(sum_f2);
        sum_f2 = _mm256_mul_ps((_mm256_mul_ps(_mm256_cvtepi32_ps(middle_value), scale_v)), sum_f2);

        return sum_f2;
    }

    inline void LoadInitModuloData(__m256i LRN_numerator[][8], int16_t *input_window_start, const size_t it_block_offset, const size_t it_z_block, const size_t view_width, const size_t x_stride, const size_t y_stride, const size_t z_block_stride, uint32_t num_blocks_modulo)
    {
#pragma unroll
        for (auto itrPix8 = 0; itrPix8 < num_blocks_modulo; ++itrPix8)
        {
            const size_t xy = it_block_offset + itrPix8;
            const size_t x = xy % view_width;
            const size_t y = xy / view_width;

            LRN_numerator[0][itrPix8] = _mm256_setzero_si256();
            LRN_numerator[1][itrPix8] = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i *)(
                input_window_start
                + it_z_block * z_block_stride
                + x * x_stride
                + y * y_stride)));
        }
#pragma unroll
        for (auto itrPix8 = num_blocks_modulo; itrPix8 < C_simd_width; ++itrPix8)
        {
            LRN_numerator[0][itrPix8] = _mm256_setzero_si256();
            LRN_numerator[1][itrPix8] = _mm256_setzero_si256();
        }
    }

    inline void LoadNextModuloData(__m256i LRN_numerator[][8], int16_t *input_window_start, const size_t it_block_offset, const size_t it_z_block, const size_t view_width, const size_t x_stride, const size_t y_stride, const size_t z_block_stride, uint32_t num_blocks_modulo)
    {
#pragma unroll
        for (auto itrPix8 = 0; itrPix8 < num_blocks_modulo; ++itrPix8)
        {
            const size_t xy = it_block_offset + itrPix8;
            const size_t x = xy % view_width;
            const size_t y = xy / view_width;

            LRN_numerator[1][itrPix8] = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i *)(
                input_window_start
                + it_z_block * z_block_stride
                + x * x_stride
                + y * y_stride)));
        }

        for (auto itrPix8 = num_blocks_modulo; itrPix8 < C_simd_width; ++itrPix8)
        {
            LRN_numerator[1][itrPix8] = _mm256_setzero_si256();
        }
    }

    inline void LoadInitFullData(__m256i LRN_numerator[][8], int16_t *input_window_start, const size_t it_block_offset, const size_t it_z_block, const size_t view_width, const size_t x_stride, const size_t y_stride, const size_t z_block_stride)
    {
#pragma unroll
        for (auto itrPix8 = 0; itrPix8 < C_simd_width; ++itrPix8)
        {
            const size_t xy = it_block_offset + itrPix8;
            const size_t x = xy % view_width;
            const size_t y = xy / view_width;

            LRN_numerator[0][itrPix8] = _mm256_setzero_si256();
            LRN_numerator[1][itrPix8] = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i *)(
                input_window_start
                + it_z_block * z_block_stride
                + x * x_stride
                + y * y_stride)));
        }
    }

    inline void LoadNextFullData(__m256i LRN_numerator[][8], int16_t *input_window_start, const size_t it_block_offset, const size_t it_z_block, const size_t view_width, const size_t x_stride, const size_t y_stride, const size_t z_block_stride)
    {
#pragma unroll
        for (auto itrPix8 = 0; itrPix8 < C_simd_width; ++itrPix8)
        {
            const size_t xy = it_block_offset + itrPix8;
            const size_t x = xy % view_width;
            const size_t y = xy / view_width;

            LRN_numerator[1][itrPix8] = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i *)(
                input_window_start
                + it_z_block * z_block_stride
                + x * x_stride
                + y * y_stride)));
        }
    }

    inline void UpdatPixelSqared(__m256i LRN_numerator[][8], __m256i & sum, __m256i * Pixel_sqared, uint32_t pos, uint32_t &PosOfLastSqr)
    {
        const auto i = 4u;
        auto acc4 = LRN_numerator[(pos + i) / 8][(pos + i) % 8];
        sum = _mm256_sub_epi32(sum, Pixel_sqared[PosOfLastSqr]);
        Pixel_sqared[PosOfLastSqr] = _mm256_madd_epi16(_mm256_abs_epi32(acc4), _mm256_abs_epi32(acc4));       //X^2
        sum = _mm256_add_epi32(sum, Pixel_sqared[PosOfLastSqr]);
        PosOfLastSqr = ++PosOfLastSqr % 5u;
    }
    inline void UpdatEndingPixelSqared(__m256i LRN_numerator[][8], __m256i & sum, __m256i * Pixel_sqared, uint32_t pos, uint32_t &PosOfLastSqr)
    {
        auto acc4 = _mm256_setzero_si256();
        sum = _mm256_sub_epi32(sum, Pixel_sqared[PosOfLastSqr]);
        PosOfLastSqr++;
        PosOfLastSqr = PosOfLastSqr % 5;
    }



    void run_lrn_accros_maps_fixedpoint_work_item_linear_single_latency(
        nn_workload_item *const work_item,
        nn_workload_data_t *input_view,
        nn_workload_data_t *output_view)
    {
        const auto &arguments = work_item->arguments.normalization_response_across_maps_forward_i16qn;

        //output_width = input_width
        //N = 5 just like for AlexK network
        const auto batch_view_begin = output_view->view_begin.t[NN_DATA_COORD_n];
        const auto batch_size = output_view->view_end.t[NN_DATA_COORD_n] - output_view->view_begin.t[NN_DATA_COORD_n] + 1;

        const auto output_width = output_view->parent->lengths.t[NN_DATA_COORD_x];
        const auto output_height = output_view->parent->lengths.t[NN_DATA_COORD_y];
        const auto output_view_width = output_view->view_end.t[NN_DATA_COORD_x] - output_view->view_begin.t[NN_DATA_COORD_x] + 1;
        const auto output_view_height = output_view->view_end.t[NN_DATA_COORD_y] - output_view->view_begin.t[NN_DATA_COORD_y] + 1;
        const auto input_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
        const auto input_height = input_view->parent->lengths.t[NN_DATA_COORD_y];
        const size_t input_view_width = input_view->view_end.t[NN_DATA_COORD_x] - input_view->view_begin.t[NN_DATA_COORD_x] + 1;

        auto FMBlockSize = output_view->parent->lengths.t[NN_DATA_COORD_p];
        auto NoFMBlocks = output_view->parent->lengths.t[NN_DATA_COORD_z];      //NoFMs = FMBlockSize * NoFMBlocks
        auto NoFMaps = FMBlockSize * NoFMBlocks;

        //auto NoFM_view = output_view->view_end.t[NN_DATA_COORD_z] - output_view->view_begin.t[NN_DATA_COORD_z] + 1;

        assert(FMBlockSize == C_simd_width);

        const auto num_full_blocks = output_view_width * output_view_height / C_simd_width;
        const auto num_blocks_modulo = output_view_width * output_view_height % C_simd_width;

        const size_t out_x_stride = FMBlockSize;
        const size_t in_x_stride = FMBlockSize;

        const size_t out_y_stride = output_view->parent->lengths.t[NN_DATA_COORD_x] * out_x_stride;
        const size_t in_y_stride = input_view->parent->lengths.t[NN_DATA_COORD_x] * in_x_stride;

        const size_t out_z_block_stride = out_y_stride * output_view->parent->lengths.t[NN_DATA_COORD_y];
        const size_t in_z_block_stride = in_y_stride * input_view->parent->lengths.t[NN_DATA_COORD_y];

        const auto output_view_offset = output_view->view_begin.t[NN_DATA_COORD_y] * out_y_stride + output_view->view_begin.t[NN_DATA_COORD_x] * out_x_stride;
        const auto input_view_offset = input_view->view_begin.t[NN_DATA_COORD_y] * in_y_stride + input_view->view_begin.t[NN_DATA_COORD_x] * in_x_stride;

        //auto NoFM_offset = output_view->view_begin.t[NN_DATA_COORD_z];

        const __m256 coeff_a = _mm256_set1_ps(arguments.alpha);
        const __m256 coeff_b = _mm256_set1_ps(arguments.beta);
        const __m256 coeff_k = _mm256_set1_ps(arguments.k);
        const auto coeff_N = arguments.n;                                  //at least at this moment coeff_N = 5

        //with only one acc simple implementation
        auto acc_fraction = arguments.fractions.input;
        auto scale = 1.0 / (float)(1 << acc_fraction);
        auto scale_v = _mm256_set1_ps(scale);
        auto scale_v2 = _mm256_set1_ps(scale * scale);
        auto scale_out = _mm256_set1_ps((float)(1 << arguments.fractions.output));

        const auto NoPixels = 8u;
        __m256i  LRN_numerator[2][NoPixels];
        __m256   LRN_result[NoPixels];


        for (uint32_t itrBatch = 0; itrBatch < batch_size; ++itrBatch)
        {
            auto input_buffer = &static_cast<int16_t*>(input_view->parent->data_buffer)[(batch_view_begin + itrBatch) * input_width * input_height * NoFMaps + input_view_offset];
            auto output_buffer = &static_cast<int16_t*>(output_view->parent->data_buffer)[(batch_view_begin + itrBatch) * output_width * output_height * NoFMaps + output_view_offset];

            auto itrResult = 0u;

            //for full blocks of inputs
            for (size_t it_block_offset = 0; it_block_offset < num_full_blocks * C_simd_width; it_block_offset += C_simd_width)
            {
                uint32_t it_z_block = 0;
                auto pos = 6u;
                const auto i2 = C_simd_width - pos; // 2

                LoadInitFullData(LRN_numerator, input_buffer, it_block_offset, it_z_block, input_view_width, in_x_stride, in_y_stride, in_z_block_stride);
                ++it_z_block;
                transpose8_ps((__m256 *)LRN_numerator[1]);

                __m256i Pixel_sqared[5];
                Pixel_sqared[0] = _mm256_setzero_si256();
                Pixel_sqared[1] = _mm256_setzero_si256();
                __m256i sum = _mm256_setzero_si256();
                __m256i acc0;

#pragma unroll
                for (auto i = 0; i < 2; ++i)
                {
                    auto acc0 = LRN_numerator[1][i];
                    Pixel_sqared[i + 2] = _mm256_madd_epi16(_mm256_abs_epi32(acc0), _mm256_abs_epi32(acc0));           //X^2
                    sum = _mm256_add_epi32(sum, Pixel_sqared[i + 2]);
                }
                Pixel_sqared[4] = _mm256_setzero_si256();

                uint32_t PosOfLastSqr = 4;

                //FMs from 0 to N-2
                for (auto itrFM = 0; itrFM < FMBlockSize * NoFMBlocks - coeff_N / 2; ++itrFM)
                {
                    UpdatPixelSqared(LRN_numerator, sum, Pixel_sqared, pos, PosOfLastSqr);

                    auto acc2 = LRN_numerator[(pos + i2) / 8][(pos + i2) % 8];

#pragma forceinline recursive
                    LRN_result[itrResult] = _mm256_calculate_LRN_item(sum, scale_v2, coeff_a, coeff_k, coeff_b, acc2, scale_v);

                    if (!(++itrResult % C_simd_width))
                    {
                        itrResult = 0;
                        transpose8_ps((__m256 *)LRN_result);
                        for (size_t it_in_block = 0; it_in_block < C_simd_width; ++it_in_block){
                            const size_t it_x = (it_block_offset + it_in_block) % output_view_width;
                            const size_t it_y = (it_block_offset + it_in_block) / output_view_width;
                            const size_t index = (itrFM / C_simd_width) * out_z_block_stride + it_x*out_x_stride + it_y*out_y_stride;
                            StoreResult(LRN_result[it_in_block], &output_buffer[index], scale_out);
                        }
                    }

                    pos = (++pos) % 8;
                    if (pos == 0){
                        for (auto itrPix8 = 0; itrPix8 < C_simd_width; ++itrPix8)
                            LRN_numerator[0][itrPix8] = LRN_numerator[1][itrPix8];
                        if (it_z_block < NoFMBlocks){
                            LoadNextFullData(LRN_numerator, input_buffer, it_block_offset, it_z_block, input_view_width, in_x_stride, in_y_stride, in_z_block_stride);
                            transpose8_ps((__m256*)LRN_numerator[1]);
                        }
                        ++it_z_block;
                    }
                }

                //FMs from N-2 to N - for last 2 sum of squared values isnt incremented because of lack of ending FMs
                for (auto itrFM = FMBlockSize * NoFMBlocks - coeff_N / 2; itrFM < FMBlockSize * NoFMBlocks; ++itrFM)
                {
                    UpdatEndingPixelSqared(LRN_numerator, sum, Pixel_sqared, pos, PosOfLastSqr);

                    const auto i2 = 2u;
                    auto acc2 = LRN_numerator[(pos + i2) / 8][(pos + i2) % 8];

#pragma forceinline recursive
                    LRN_result[itrResult] = _mm256_calculate_LRN_item(sum, scale_v2, coeff_a, coeff_k, coeff_b, acc2, scale_v);

                    if (!(++itrResult % C_simd_width))
                    {
                        itrResult = 0;
                        transpose8_ps((__m256 *)LRN_result);
                        for (size_t it_in_block = 0; it_in_block < C_simd_width; ++it_in_block){
                            const size_t it_x = (it_block_offset + it_in_block) % output_view_width;
                            const size_t it_y = (it_block_offset + it_in_block) / output_view_width;
                            const size_t index = (itrFM / C_simd_width) * out_z_block_stride + it_x*out_x_stride + it_y*out_y_stride;
                            StoreResult(LRN_result[it_in_block], &output_buffer[index], scale_out);
                        }
                    }

                    pos = (++pos) % 8;
                    if (pos == 0){
                        LoadNextFullData(LRN_numerator, input_buffer, it_block_offset, it_z_block, input_view_width, in_x_stride, in_y_stride, in_z_block_stride);
                        transpose8_ps((__m256*)LRN_numerator[1]);
                        ++it_z_block;
                    }
                }
            }

            //________________________________________________________
            //
            //  and for the rest of inputs - modulo
            //________________________________________________________

            auto it_block_offset = num_full_blocks * C_simd_width;
            uint32_t it_z_block = 0;

            LoadInitModuloData(LRN_numerator, input_buffer, it_block_offset, it_z_block, input_view_width, in_x_stride, in_y_stride, in_z_block_stride, num_blocks_modulo);

            ++it_z_block;

            transpose8_ps((__m256 *)LRN_numerator[1]);

            __m256i Pixel_sqared[5];
            __m256i acc0;
            Pixel_sqared[0] = _mm256_setzero_si256();
            Pixel_sqared[1] = _mm256_setzero_si256();
            __m256i sum = _mm256_setzero_si256();

#pragma unroll
            for (auto i = 0; i < 3; ++i)
            {
                auto acc0 = LRN_numerator[1][i];
                Pixel_sqared[i + 2] = _mm256_madd_epi16(_mm256_abs_epi32(acc0), _mm256_abs_epi32(acc0));       //X^2
                sum = _mm256_add_epi32(sum, Pixel_sqared[i + 2]);
            }

            uint32_t PosOfLastSqr = 4;
            auto pos = 6u;

            //FMs from 0 to N-2
            for (auto itrFM = 0; itrFM < FMBlockSize * NoFMBlocks - coeff_N / 2; ++itrFM)
            {
                UpdatPixelSqared(LRN_numerator, sum, Pixel_sqared, pos, PosOfLastSqr);

                const auto i2 = 2u;
                auto acc2 = LRN_numerator[(pos + i2) / 8][(pos + i2) % 8];

                LRN_result[itrResult] = _mm256_calculate_LRN_item(sum, scale_v2, coeff_a, coeff_k, coeff_b, acc2, scale_v);

                if (!(++itrResult % C_simd_width))
                {
                    itrResult = 0;
                    transpose8_ps((__m256 *)LRN_result);
                    for (size_t it_in_block = 0; it_in_block < num_blocks_modulo; ++it_in_block){
                        const size_t it_x = (it_block_offset + it_in_block) % output_view_width;
                        const size_t it_y = (it_block_offset + it_in_block) / output_view_width;
                        const size_t index = (itrFM / C_simd_width) * out_z_block_stride + it_x*out_x_stride + it_y*out_y_stride;
                        StoreResult(LRN_result[it_in_block], &output_buffer[index], scale_out);
                    }
                }

                pos = (++pos) % 8;
                if (pos == 0){
                    for (auto itrPix8 = 0; itrPix8 < C_simd_width; ++itrPix8)
                        LRN_numerator[0][itrPix8] = LRN_numerator[1][itrPix8];
                    if (it_z_block < NoFMBlocks){
                        LoadNextModuloData(LRN_numerator, input_buffer, it_block_offset, it_z_block, input_view_width, in_x_stride, in_y_stride, in_z_block_stride, num_blocks_modulo);
                        transpose8_ps((__m256*)LRN_numerator[1]);
                    }
                    ++it_z_block;
                }
            }

            //FMs from N-2 to N - for last 2 sum of squared values isnt incremented because of lack of ending FMs
            for (auto itrFM = FMBlockSize * NoFMBlocks - coeff_N / 2; itrFM < FMBlockSize * NoFMBlocks; ++itrFM)
            {
                UpdatEndingPixelSqared(LRN_numerator, sum, Pixel_sqared, pos, PosOfLastSqr);

                const auto i2 = 2u;
                auto acc2 = LRN_numerator[(pos + i2) / 8][(pos + i2) % 8];
                LRN_result[itrResult] = _mm256_calculate_LRN_item(sum, scale_v2, coeff_a, coeff_k, coeff_b, acc2, scale_v);

                if (!(++itrResult % C_simd_width))
                {
                    itrResult = 0;
                    transpose8_ps((__m256 *)LRN_result);
                    for (size_t it_in_block = 0; it_in_block < num_blocks_modulo; ++it_in_block){
                        const size_t it_x = (it_block_offset + it_in_block) % output_view_width;
                        const size_t it_y = (it_block_offset + it_in_block) / output_view_width;
                        const size_t index = (itrFM / C_simd_width) * out_z_block_stride + it_x*out_x_stride + it_y*out_y_stride;
                        StoreResult(LRN_result[it_in_block], &output_buffer[index], scale_out);
                    }
                }

                pos = (++pos) % 8;
            }
        }
    }

    inline __m128i _mm_loadmask_epi16(int16_t * input, uint8_t mask)
    {
        int16_t buf[C_simd_width] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        memcpy(buf, input, mask * sizeof(int16_t));
        return _mm_setr_epi16(
            buf[0], buf[1], buf[2], buf[3],
            buf[4], buf[5], buf[6], buf[7]);
    }

    void run_lrn_accros_maps_fixedpoint_work_item_linear_single_batching(
        nn_workload_item *const work_item,
        nn_workload_data_t *input_view,
        nn_workload_data_t *output_view)
    {
        uint32_t batchsize = work_item->input[0]->output->view_end.t[NN_DATA_COORD_n] - work_item->input[0]->output->view_begin.t[NN_DATA_COORD_n] + 1;
        run_lrn_accros_maps_fixedpoint_work_item_linear_single_latency(work_item, input_view, output_view);
    }



    void choose_normalization_work_item_lrn_accros_maps_fixedpoint_single_batching_mode(
        nn_workload_item *const work_item,
        nn_workload_data_t *input_view,
        nn_workload_data_t *output_view)
    {
        auto batch_size = input_view->parent->lengths.t[NN_DATA_COORD_n];
        switch (batch_size)
        {
        case 1:
            run_lrn_accros_maps_fixedpoint_work_item_linear_single_latency(work_item, input_view, output_view);
            break;
        case 8:
        case 16:
        case 24:
        case 32:
            run_lrn_accros_maps_fixedpoint_work_item_linear_single_batching(work_item, input_view, output_view);
            break;

        default:
            break;
        }
    }

    void run_singlethreaded_lrn_fixedpoint_work_item(
        nn_workload_item *const work_item,
        nn_workload_data_t *input_view,
        nn_workload_data_t *output_view)
    {
        choose_normalization_work_item_lrn_accros_maps_fixedpoint_single_batching_mode(work_item, input_view, output_view);
    }

    void wrapper_lrn_fixedpoint_work_item(nn_workload_item *const work_item)
    {
        const auto num_hardware_threads = std::thread::hardware_concurrency();

        auto are_coords_equal = [](const nn_workload_data_coords_t &val, const nn_workload_data_coords_t &coords) {
            return coords.t[NN_DATA_COORD_n] == val.t[NN_DATA_COORD_n] &&
                coords.t[NN_DATA_COORD_x] == val.t[NN_DATA_COORD_x] &&
                coords.t[NN_DATA_COORD_y] == val.t[NN_DATA_COORD_y] &&
                coords.t[NN_DATA_COORD_z] == val.t[NN_DATA_COORD_z] &&
                coords.t[NN_DATA_COORD_p] == val.t[NN_DATA_COORD_p] &&
                coords.t[NN_DATA_COORD_q] == val.t[NN_DATA_COORD_q];
        };

        // make sure that block size is 8 and check the ordering
        assert(work_item->input[0]->output->parent->lengths.t[NN_DATA_COORD_p] == 8);
        assert(are_coords_equal({ NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, work_item->input[0]->output->parent->layout.ordering));
        assert(are_coords_equal({ NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, work_item->output->parent->layout.ordering));

        run_singlethreaded_lrn_fixedpoint_work_item(work_item, work_item->input[0]->output, work_item->output);
    }

} // namepace int16_fixedpoint
