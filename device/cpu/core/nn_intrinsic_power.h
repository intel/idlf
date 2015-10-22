
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

#pragma once

#include "device/common/nn_asmjit_compilation.h"

namespace nn
{
namespace intrinsics
{

//domain is numbers greater than 0 (for negative it is undefined, for 0 it is large value, but not infinity)
inline __m256 invpow_075(__m256 arg)
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
                           _mm256_set1_epi32(0)))
                      );

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

    return _mm256_mul_ps(
                _mm256_mul_ps(p0, p1),
                _mm256_mul_ps(p2, intermediate_result));
}

inline void invpow_075(float* input, float* output)
{
    auto in = _mm256_loadu_ps(input);
    auto out = invpow_075(in);
    _mm256_storeu_ps(output, out);
}

inline __m256 _inner_mm256_exp_ps(__m256 arg)
{
    __m256 mask = _mm256_cmp_ps(arg, _mm256_set1_ps(-87.336f), _CMP_GT_OQ);

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

    return _mm256_and_ps(res, mask);
}

} //namespace intrinsics
} //namespace nn

