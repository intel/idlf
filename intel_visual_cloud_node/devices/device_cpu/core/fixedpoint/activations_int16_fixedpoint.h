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
#include <cstdint>
#include <immintrin.h>

namespace activations {

namespace int16_fixedpoint {

enum class ShiftDirection { Left, Right };

namespace impl {
template <typename OutputType> struct ActivationTypeBase {
    struct ImplBase {
        using output_type = OutputType;
    };
};

/*************************************************************************************************/

template <bool Batched, ShiftDirection Shift, typename OutputType> struct None;
template <>
struct None<false, ShiftDirection::Right, std::int32_t> : public ActivationTypeBase<std::int32_t>::ImplBase {
    static inline void store_activation(
        output_type *addr, __m256i val1, __m256i val2, const std::int8_t shift_in, const std::int8_t shift_out) {
        _mm256_stream_si256((__m256i *)addr + 0, _mm256_srai_epi32(val1, shift_in - shift_out));
        _mm256_stream_si256((__m256i *)addr + 1, _mm256_srai_epi32(val2, shift_in - shift_out));
    }
    static inline void store_activation(
        output_type *addr, __m256i val1, const std::int8_t shift_in, const std::int8_t shift_out) {
        _mm256_stream_si256((__m256i *)addr + 0, _mm256_srai_epi32(val1, shift_in - shift_out));
    }
};
template <> struct None<false, ShiftDirection::Left, std::int32_t> : public ActivationTypeBase<std::int32_t>::ImplBase {
    static inline void store_activation(
        output_type *addr, __m256i val1, __m256i val2, const std::int8_t shift_in, const std::int8_t shift_out) {
        _mm256_stream_si256((__m256i *)addr + 0, _mm256_slli_epi32(val1, shift_out - shift_in));
        _mm256_stream_si256((__m256i *)addr + 1, _mm256_slli_epi32(val2, shift_out - shift_in));
    }
    static inline void store_activation(
        output_type *addr, __m256i val1, const std::int8_t shift_in, const std::int8_t shift_out) {
        _mm256_stream_si256((__m256i *)addr + 0, _mm256_slli_epi32(val1, shift_out - shift_in));
    }

};

template <> struct None<true, ShiftDirection::Right, std::int32_t> : public ActivationTypeBase<std::int32_t>::ImplBase {
    static inline void store_activation(
        output_type *addr, __m256i val1, __m256i val2, const std::int8_t shift_in, const std::int8_t shift_out) {
        __m256i out5410 = _mm256_unpacklo_epi32(val1, val2);
        __m256i out7632 = _mm256_unpackhi_epi32(val1, val2);

        _mm256_stream_si256((__m256i *)addr + 0,
                            _mm256_srai_epi32(_mm256_permute2x128_si256(out5410, out7632, 0x20), shift_in - shift_out));
        _mm256_stream_si256((__m256i *)addr + 1,
                            _mm256_srai_epi32(_mm256_permute2x128_si256(out5410, out7632, 0x31), shift_in - shift_out));
    }
};

template <> struct None<true, ShiftDirection::Left, std::int32_t> : public ActivationTypeBase<std::int32_t>::ImplBase {
    static inline void store_activation(
        output_type *addr, __m256i val1, __m256i val2, const std::int8_t shift_in, const std::int8_t shift_out) {
        __m256i out5410 = _mm256_unpacklo_epi32(val1, val2);
        __m256i out7632 = _mm256_unpackhi_epi32(val1, val2);

        _mm256_stream_si256((__m256i *)addr + 0,
                            _mm256_slli_epi32(_mm256_permute2x128_si256(out5410, out7632, 0x20), shift_out - shift_in));
        _mm256_stream_si256((__m256i *)addr + 1,
                            _mm256_slli_epi32(_mm256_permute2x128_si256(out5410, out7632, 0x31), shift_out - shift_in));
    }

};


/*************************************************************************************************/
template <bool Batched, ShiftDirection Shift, typename OutputType> struct ReLu;
template <>
struct ReLu<false, ShiftDirection::Right, std::int16_t> : public ActivationTypeBase<std::int16_t>::ImplBase {
    static inline void store_activation(
        output_type *addr, __m256i val1, __m256i val2, const std::int8_t shift_in, const std::int8_t shift_out) {
        _mm256_stream_si256((__m256i *)(addr) + 0,
                            _mm256_max_epi16((_mm256_permute4x64_epi64(
                                                 _mm256_packs_epi32(_mm256_srai_epi32(val1, shift_in - shift_out),
                                                                    _mm256_srai_epi32(val2, shift_in - shift_out)),
                                                 0xd8)),
                                             _mm256_setzero_si256()));
    }

    static inline void store_activation(
        output_type *addr, __m256i val1, const std::int8_t shift_in, const std::int8_t shift_out) {
        _mm_stream_si128(
            (__m128i *)(addr) + 0,
            _mm256_castsi256_si128(_mm256_max_epi16(
                (_mm256_permute4x64_epi64(
                    _mm256_packs_epi32(_mm256_srai_epi32(val1, shift_in - shift_out), _mm256_setzero_si256()), 0xd8)),
                _mm256_setzero_si256())));
    }
};

template <> struct ReLu<false, ShiftDirection::Left, std::int16_t> : public ActivationTypeBase<std::int16_t>::ImplBase {
    static inline void store_activation(
        output_type *addr, __m256i val1, __m256i val2, const std::int8_t shift_in, const std::int8_t shift_out) {
        _mm256_stream_si256((__m256i *)(addr) + 0,
                            _mm256_max_epi16((_mm256_permute4x64_epi64(
                                                 _mm256_packs_epi32(_mm256_slli_epi32(val1, shift_out - shift_in),
                                                                    _mm256_slli_epi32(val2, shift_out - shift_in)),
                                                 0xd8)),
                                             _mm256_setzero_si256()));
    }

    static inline void store_activation(
        output_type *addr, __m256i val1, const std::int8_t shift_in, const std::int8_t shift_out) {
        _mm_stream_si128(
            (__m128i *)(addr) + 0,
            _mm256_castsi256_si128(_mm256_max_epi16(
                (_mm256_permute4x64_epi64(
                    _mm256_packs_epi32(_mm256_slli_epi32(val1, shift_out - shift_in), _mm256_setzero_si256()), 0xd8)),
                _mm256_setzero_si256())));
    }
};

template <> struct ReLu<true, ShiftDirection::Right, std::int16_t> : public ActivationTypeBase<std::int16_t>::ImplBase {
    static inline void store_activation(
        output_type *addr, __m256i val1, __m256i val2, const std::int8_t shift_in, const std::int8_t shift_out) {
        __m256i out5410 = _mm256_unpacklo_epi32(val1, val2);
        __m256i out7632 = _mm256_unpackhi_epi32(val1, val2);

        _mm256_stream_si256((__m256i *)(addr) + 0,
                            _mm256_max_epi16(_mm256_packs_epi32(_mm256_srai_epi32(out5410, shift_in - shift_out),
                                                                _mm256_srai_epi32(out7632, shift_in - shift_out)),
                                             _mm256_setzero_si256()));
    }
};

template <> struct ReLu<true, ShiftDirection::Left, std::int16_t> : public ActivationTypeBase<std::int16_t>::ImplBase {
    static inline void store_activation(
        output_type *addr, __m256i val1, __m256i val2, const std::int8_t shift_in, const std::int8_t shift_out) {
        __m256i out5410 = _mm256_unpacklo_epi32(val1, val2);
        __m256i out7632 = _mm256_unpackhi_epi32(val1, val2);

        _mm256_stream_si256((__m256i *)(addr) + 0,
                            _mm256_max_epi16(_mm256_packs_epi32(_mm256_slli_epi32(out5410, shift_out - shift_in),
                                                                _mm256_slli_epi32(out7632, shift_out - shift_in)),
                                             _mm256_setzero_si256()));
    }
};




/*************************************************************************************************/
template <bool Batched, typename OutputType> struct Logistic;

inline __m256 _inner_mm256_exp_ps(__m256 arg) {
    arg = _mm256_mul_ps(arg, _mm256_set1_ps(1.4426950408889634073599246810018921374266459541529859f));

    __m256i e = _mm256_add_epi32(_mm256_castps_si256(_mm256_cmp_ps(arg, _mm256_set1_ps(0.0f), _CMP_LT_OQ)),
                                 _mm256_cvttps_epi32(arg));

    arg = _mm256_sub_ps(arg, _mm256_cvtepi32_ps(e));

    __m256 intermediate_result;
    intermediate_result =
        _mm256_fmadd_ps(_mm256_set1_ps(0.0136779459179717f), arg, _mm256_set1_ps(0.0517692205767896f));
    intermediate_result = _mm256_fmadd_ps(intermediate_result, arg, _mm256_set1_ps(0.241554388295527f));
    intermediate_result = _mm256_fmadd_ps(intermediate_result, arg, _mm256_set1_ps(0.692998430056128f));
    intermediate_result = _mm256_fmadd_ps(intermediate_result, arg, _mm256_set1_ps(0.999999804292074f));
    arg = intermediate_result;

    __m256 res = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(127)), 23));

    res = _mm256_mul_ps(res, arg);

    return res;
}

template <> struct Logistic<false, std::int16_t> : public ActivationTypeBase<std::int16_t>::ImplBase {
    static inline void store_activation(
        output_type *addr, __m256i val1, __m256i val2, const std::int8_t shift_in, const std::int8_t shift_out) {
        const __m256 _one = _mm256_set1_ps(1.0f);
        const __m256 _scale_in = _mm256_set1_ps(shift_in >= 0 ? (1.0f / (1 << shift_in)) : 1 << (-shift_in));
        const __m256 _scale_out = _mm256_set1_ps(shift_out >= 0 ? (1 << shift_out) : 1 / (1 << (-shift_out)));
        _mm256_stream_si256(
            (__m256i *)(addr) + 0,
            _mm256_permute4x64_epi64(
                _mm256_packs_epi32(
                    _mm256_cvtps_epi32(_mm256_mul_ps(
                        _mm256_sub_ps(_one,
                                      _mm256_div_ps(_one,
                                                    _mm256_add_ps(_one,
                                                                  _inner_mm256_exp_ps(_mm256_mul_ps(
                                                                      _mm256_cvtepi32_ps(val1), _scale_in))))),
                        _scale_out)),
                    _mm256_cvtps_epi32(_mm256_mul_ps(
                        _mm256_sub_ps(_one,
                                      _mm256_div_ps(_one,
                                                    _mm256_add_ps(_one,
                                                                  _inner_mm256_exp_ps(_mm256_mul_ps(
                                                                      _mm256_cvtepi32_ps(val2), _scale_in))))),
                        _scale_out))),
                0xd8));
    }

    // untested
    static inline void store_activation(
        output_type *addr, __m256i val1, const std::int8_t shift_in, const std::int8_t shift_out) {
        const __m256 _one = _mm256_set1_ps(1.0f);
        const __m256 _scale_in = _mm256_set1_ps(shift_in >= 0 ? (1.0f / (1 << shift_in)) : 1 << (-shift_in));
        const __m256 _scale_out = _mm256_set1_ps(shift_out >= 0 ? (1 << shift_out) : 1 / (1 << (-shift_out)));
        __m256i out5410 = _mm256_unpacklo_epi32(val1, _mm256_setzero_si256());
        __m256i out7632 = _mm256_unpackhi_epi32(val1, _mm256_setzero_si256());

        _mm_stream_si128(
            (__m128i *)(addr) + 0,
            _mm256_castsi256_si128(_mm256_permute4x64_epi64(
                _mm256_packs_epi32(
                    _mm256_cvtps_epi32(_mm256_mul_ps(
                        _mm256_sub_ps(_one,
                                      _mm256_div_ps(_one,
                                                    _mm256_add_ps(_one,
                                                                  _inner_mm256_exp_ps(_mm256_mul_ps(
                                                                      _mm256_cvtepi32_ps(val1), _scale_in))))),
                        _scale_out)),
                    _mm256_cvtps_epi32(_mm256_mul_ps(
                        _mm256_sub_ps(
                            _one,
                            _mm256_div_ps(_one,
                                          _mm256_add_ps(_one,
                                                        _inner_mm256_exp_ps(_mm256_mul_ps(
                                                            _mm256_cvtepi32_ps(_mm256_setzero_si256()), _scale_in))))),
                        _scale_out))),
                0xd8)));
    }

};

template <> struct Logistic<true, std::int16_t> : public ActivationTypeBase<std::int16_t>::ImplBase {
    static inline void store_activation(
        output_type *addr, __m256i val1, __m256i val2, const std::int8_t shift_in, const std::int8_t shift_out) {
        const __m256 _one = _mm256_set1_ps(1.0f);
        const __m256 _scale_in = _mm256_set1_ps(shift_in >= 0 ? (1.0f / (1 << shift_in)) : 1 << (-shift_in));
        const __m256 _scale_out = _mm256_set1_ps(shift_out >= 0 ? (1 << shift_out) : 1 / (1 << (-shift_out)));
        __m256i out5410 = _mm256_unpacklo_epi32(val1, val2);
        __m256i out7632 = _mm256_unpackhi_epi32(val1, val2);

        _mm256_stream_si256(
            (__m256i *)(addr) + 0,
            _mm256_packs_epi32(
                _mm256_cvtps_epi32(_mm256_mul_ps(
                    _mm256_sub_ps(_one,
                                  _mm256_div_ps(_one,
                                                _mm256_add_ps(_one,
                                                              _inner_mm256_exp_ps(_mm256_mul_ps(
                                                                  _mm256_cvtepi32_ps(out5410), _scale_in))))),
                    _scale_out)),
                _mm256_cvtps_epi32(_mm256_mul_ps(
                    _mm256_sub_ps(_one,
                                  _mm256_div_ps(_one,
                                                _mm256_add_ps(_one,
                                                              _inner_mm256_exp_ps(_mm256_mul_ps(
                                                                  _mm256_cvtepi32_ps(out7632), _scale_in))))),
                    _scale_out))));
    }

};
};

/*
    Batched versions place values in pairs and then in batches
*/

template <typename OutputType> struct None : public impl::ActivationTypeBase<OutputType> {
    template <bool Batched, ShiftDirection Shift> struct Impl : public impl::None<Batched, Shift, OutputType> {};
};

template <typename OutputType> struct ReLu : public impl::ActivationTypeBase<OutputType> {
    template <bool Batched, ShiftDirection Shift> struct Impl : public impl::ReLu<Batched, Shift, OutputType> {};
};

template <typename OutputType> struct Logistic : public impl::ActivationTypeBase<OutputType> {
    template <bool Batched, ShiftDirection Shift> struct Impl : public impl::Logistic<Batched, OutputType> {};
};
};
};
