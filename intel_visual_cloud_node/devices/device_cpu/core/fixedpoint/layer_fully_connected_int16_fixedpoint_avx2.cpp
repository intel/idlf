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
#include "layer_fully_connected_int16_fixedpoint_avx2.h"
#include "activations_int16_fixedpoint.h"

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
const auto C_simd_width = sizeof(__m256) / sizeof(uint32_t);
const auto OUT_GROUPING = 8;

namespace int16_fixedpoint {

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // validation code
    /*#define SETUP_CHECK bool valid = true
    #define CHECK_AND_SAVE(expr) valid &= (!!(expr))
    #define RETURN_ERROR(status) if(!valid) {return status;}

    NN_API_STATUS validate_fully_connected_int16_fixedpoint_work_item(nn_workload_item *const work_item)
    {
    auto &arguments = work_item->arguments.forward_fully_connected;
    if (arguments.input != nullptr &&
    arguments.input->parent != nullptr &&
    arguments.input->parent->data_buffer != nullptr &&
    arguments.input->parent->buffer_size != 0 &&
    arguments.output != nullptr &&
    arguments.output->parent != nullptr &&
    arguments.output->parent->data_buffer != nullptr &&
    arguments.output->parent->buffer_size != 0 &&
    arguments.weights != nullptr &&
    arguments.weights->parent != nullptr &&
    arguments.weights->parent->data_buffer != nullptr &&
    arguments.weights->parent->buffer_size != 0)
    {
    nn_workload_data_layout_t in_layout =
    {
    { 0, 0, 0, 0, 0, 0 }, // tile_log2
    { 0, 0, 0, 0, 0, 0 }, // alignment
    { NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
    NN_DATATYPE_INT16
    };
    nn_workload_data_layout_t out_layout =
    {
    { 0, 0, 0, 0, 0, 0 }, // tile_log2
    { 0, 0, 0, 0, 0, 0 }, // alignment
    { NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
    NN_DATATYPE_INT16
    };

    nn_workload_data_layout_t weight_layout =
    {
    { 0, 0, 0, 0, 0, 0 }, // tile_log2
    { 0, 0, 0, 0, 0, 0 }, // alignment
    { NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
    NN_DATATYPE_INT16
    };

    SETUP_CHECK;

    CHECK_AND_SAVE(arguments.input->parent->lengths.t[0] == 1);
    CHECK_AND_SAVE(arguments.input->parent->lengths.t[4] == 1);
    CHECK_AND_SAVE(arguments.input->parent->lengths.t[5] == 1);
    CHECK_AND_SAVE(arguments.output->parent->lengths.t[0] == 1);
    CHECK_AND_SAVE(arguments.output->parent->lengths.t[4] == 1);
    CHECK_AND_SAVE(arguments.output->parent->lengths.t[5] == 1);
    CHECK_AND_SAVE(arguments.weights->parent->lengths.t[0] == 1);
    CHECK_AND_SAVE(arguments.weights->parent->lengths.t[5] == 1);

    RETURN_ERROR(NN_API_STATUS_ERROR_INVALID_MEMORY_LAYOUT);

    bool use_biases = (arguments.biases != nullptr);

    // Check biases.
    if (use_biases)
    {
    CHECK_AND_SAVE(arguments.biases->parent != nullptr);
    RETURN_ERROR(NN_API_STATUS_ERROR_INVALID_POINTER);

    CHECK_AND_SAVE(arguments.biases->parent->data_buffer != nullptr);
    RETURN_ERROR(NN_API_STATUS_ERROR_INVALID_POINTER);

    CHECK_AND_SAVE(memcmp(&out_layout, &arguments.biases->parent->layout, sizeof(nn_workload_data_layout_t)) == 0);
    CHECK_AND_SAVE(arguments.biases->parent->data_type_size == sizeof(int32_t));
    RETURN_ERROR(NN_API_STATUS_ERROR_INVALID_MEMORY_LAYOUT);

    CHECK_AND_SAVE(arguments.biases->parent->buffer_size != 0);
    CHECK_AND_SAVE(arguments.biases->parent->lengths.t[NN_DATA_COORD_x] == arguments.output->parent->lengths.t[NN_DATA_COORD_x]);
    RETURN_ERROR(NN_API_STATUS_ERROR_DATA_NOT_CONSISTENT);
    }

    // Check memory layouts.
    CHECK_AND_SAVE(memcmp(&in_layout, &arguments.input->parent->layout, sizeof(nn_workload_data_layout_t)) == 0);
    CHECK_AND_SAVE(memcmp(&out_layout, &arguments.output->parent->layout, sizeof(nn_workload_data_layout_t)) == 0);
    CHECK_AND_SAVE(memcmp(&weight_layout, &arguments.weights->parent->layout, sizeof(nn_workload_data_layout_t)) == 0);
    CHECK_AND_SAVE(arguments.input->parent->data_type_size == sizeof(int16_t));
    CHECK_AND_SAVE(arguments.output->parent->data_type_size == sizeof(int16_t));
    CHECK_AND_SAVE(arguments.weights->parent->data_type_size == sizeof(int16_t));
    RETURN_ERROR(NN_API_STATUS_ERROR_INVALID_MEMORY_LAYOUT);

    // Check data consistence.
    CHECK_AND_SAVE(arguments.input->parent->lengths.t[NN_DATA_COORD_y]  == arguments.weights->parent->lengths.t[NN_DATA_COORD_y]);
    CHECK_AND_SAVE(arguments.output->parent->lengths.t[NN_DATA_COORD_y] * 2 == arguments.weights->parent->lengths.t[NN_DATA_COORD_p] * C_simd_width);
    RETURN_ERROR(NN_API_STATUS_ERROR_DATA_NOT_CONSISTENT);

    // Check current (temporary) constraints.
    // TODO - dependencies support.
    // TODO - latency implementation.

    CHECK_AND_SAVE(arguments.activation.function == NN_ACTIVATION_FUNCTION_RELU ||
    arguments.activation.function == NN_ACTIVATION_FUNCTION_NONE);

    CHECK_AND_SAVE(arguments.input->parent->lengths.t[NN_DATA_COORD_z] == 8);
    CHECK_AND_SAVE(work_item->dependency_count == 0);
    RETURN_ERROR(NN_API_STATUS_ERROR_UNSUPPORTED_VERSION);

    return NN_API_STATUS_OK;
    }
    else
    {
    return NN_API_STATUS_ERROR_INVALID_POINTER;
    }
    }*/

    // forward implementation
    template<class Activation, bool T_NEED_BIAS_COPY, bool FullOutGrouping>
    static inline void process_fully_connected_int16_fixedpoint_AVX2_output_b1(
        const int16_t* const input,
        const int16_t* const weights,
        typename Activation::output_type* output,
        uint32_t numInputs,
        const uint32_t numAcc,
        const int32_t* const bias,
        int8_t in_shift,
        int8_t out_shift)
    {
        if (FullOutGrouping)
        {
            __m256i sums[OUT_GROUPING];

#pragma unroll
            for (uint32_t out_it = 0; out_it < OUT_GROUPING; ++out_it)
            if (T_NEED_BIAS_COPY)
                sums[out_it] = _mm256_load_si256((__m256i*)(bias + C_simd_width * out_it));
            else
                sums[out_it] = _mm256_setzero_si256();

            for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
            {

                __m256i load_i = _mm256_set1_epi32(*(int32_t*)(input + in_it));

#pragma unroll
                for (int out_it = 0; out_it < OUT_GROUPING; ++out_it)
                {
                    __m256i load_w = _mm256_loadu_si256((__m256i*)(weights + in_it * C_simd_width + out_it * C_simd_width * numInputs));
                    __m256i tempOut = _mm256_madd_epi16(load_i, load_w);
                    sums[out_it] = _mm256_add_epi32(sums[out_it], tempOut);
                }
            }

#pragma unroll
            for (uint32_t out_it = 0; out_it < OUT_GROUPING; out_it += 2)
            {
                Activation::store_activation(output + 8 * out_it, sums[out_it + 0], sums[out_it + 1], in_shift, out_shift);
            }
        }
        else
        {
            //__m256i sums[numAcc];
            __m256i *sums = (__m256i *)_mm_malloc(numAcc*sizeof(__m256i), 64);
#pragma unroll
            for (uint32_t out_it = 0; out_it < numAcc; ++out_it)
            if (T_NEED_BIAS_COPY)
                sums[out_it] = _mm256_load_si256((__m256i*)(bias + C_simd_width * out_it));
            else
                sums[out_it] = _mm256_setzero_si256();

            for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
            {

                __m256i load_i = _mm256_set1_epi32(*(int32_t*)(input + in_it));

#pragma unroll
                for (uint32_t out_it = 0; out_it < numAcc; ++out_it)
                {
                    __m256i load_w = _mm256_stream_load_si256((__m256i*)(weights + in_it * C_simd_width + out_it * C_simd_width * numInputs));
                    __m256i tempOut = _mm256_madd_epi16(load_i, load_w);
                    sums[out_it] = _mm256_add_epi32(sums[out_it], tempOut);
                }
            }

#pragma unroll
            for (uint32_t out_it = 0; out_it < (numAcc / 2) * 2; out_it += 2)
            {
                Activation::store_activation(output + 8 * out_it, sums[out_it + 0], sums[out_it + 1], in_shift, out_shift);
            }

            if (numAcc % 2)
            {
                Activation::store_activation(output + 8 * (numAcc - 1), sums[numAcc - 1], in_shift, out_shift);
            }

            _mm_free(sums);
        }
    }

    // forward implementation
    template<class Activation, bool T_NEED_BIAS_COPY>
    static inline void process_fully_connected_int16_fixedpoint_AVX2_output_b8(
        const int16_t* const input,
        const int16_t* const weights,
        typename Activation::output_type* output,
        uint32_t numInputs,
        uint32_t numOutputs,
        const int32_t* const bias,
        int8_t in_shift,
        int8_t out_shift)
    {
        __m256i sums[OUT_GROUPING];

#pragma unroll
        for (uint32_t out_it = 0; out_it < OUT_GROUPING; ++out_it)
        if (T_NEED_BIAS_COPY)
            sums[out_it] = _mm256_set1_epi32(*(bias + out_it));
        else
            sums[out_it] = _mm256_setzero_si256();

        for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
        {
            __m256i load_i = _mm256_stream_load_si256((__m256i*)(input + in_it * 8));

#pragma unroll
            for (int out_it = 0; out_it < OUT_GROUPING; ++out_it)
            {
                __m256i load_w = _mm256_set1_epi32(*(int32_t*)(weights + in_it*OUT_GROUPING + out_it * 2));
                __m256i tempOut = _mm256_madd_epi16(load_i, load_w);
                sums[out_it] = _mm256_add_epi32(sums[out_it], tempOut);
            }
        }

#pragma unroll
        for (uint32_t out_it = 0; out_it < OUT_GROUPING; out_it += 2)
        {
            Activation::store_activation(output, sums[out_it], sums[out_it + 1], in_shift, out_shift);
            output += 16;
        }
    }

    // forward implementation
    template<class Activation, bool T_NEED_BIAS_COPY>
    static inline void process_fully_connected_int16_fixedpoint_AVX2_output_b16(
        const int16_t* const input,
        const int16_t* const weights,
        typename Activation::output_type* output,
        uint32_t numInputs,
        uint32_t numOutputs,
        const int32_t* const bias,
        int8_t in_shift,
        int8_t out_shift)
    {
        __m256i sums[2 * OUT_GROUPING];

#pragma unroll
        for (uint32_t out_it = 0; out_it < OUT_GROUPING; ++out_it)
        if (T_NEED_BIAS_COPY)
        {
            sums[2 * out_it + 0] = _mm256_set1_epi32(*(bias + out_it));
            sums[2 * out_it + 1] = _mm256_set1_epi32(*(bias + out_it));
        }
        else
        {
            sums[2 * out_it + 0] = _mm256_setzero_si256();
            sums[2 * out_it + 1] = _mm256_setzero_si256();
        }

        for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
        {
            __m256i load_iL = _mm256_stream_load_si256((__m256i*)(input + 2 * in_it * 8 + 0));
            __m256i load_iH = _mm256_stream_load_si256((__m256i*)(input + 2 * in_it * 8 + C_simd_width * 2));

#pragma unroll
            for (int out_it = 0; out_it < OUT_GROUPING; ++out_it)
            {
                __m256i load_w = _mm256_set1_epi32(*(int32_t*)(weights + in_it * OUT_GROUPING + out_it * 2));
                __m256i tempOut = _mm256_madd_epi16(load_iL, load_w);
                sums[2 * out_it + 0] = _mm256_add_epi32(sums[2 * out_it + 0], tempOut);
                tempOut = _mm256_madd_epi16(load_iH, load_w);
                sums[2 * out_it + 1] = _mm256_add_epi32(sums[2 * out_it + 1], tempOut);
            }
        }

#pragma unroll
         for (uint32_t out_it = 0; out_it < 2 * OUT_GROUPING; out_it += 4)
        {
            Activation::store_activation(output, sums[out_it + 0], sums[out_it + 2], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 1], sums[out_it + 3], in_shift, out_shift);
            output += (2 * C_simd_width);
        }
    }


    // forward implementation
    template<class Activation, bool T_NEED_BIAS_COPY>
    static inline void process_fully_connected_int16_fixedpoint_AVX2_output_b24(
        const int16_t* const input,
        const int16_t* const weights,
        typename Activation::output_type* output,
        uint32_t numInputs,
        uint32_t numOutputs,
        const int32_t* const bias,
        int8_t in_shift,
        int8_t out_shift)
    {
        __m256i sums[3 * OUT_GROUPING];

#pragma unroll
        for (uint32_t out_it = 0; out_it < OUT_GROUPING; ++out_it)
        if (T_NEED_BIAS_COPY)
        {
            sums[3 * out_it + 0] = _mm256_set1_epi32(*(bias + out_it));
            sums[3 * out_it + 1] = _mm256_set1_epi32(*(bias + out_it));
            sums[3 * out_it + 2] = _mm256_set1_epi32(*(bias + out_it));
        }
        else
        {
            sums[3 * out_it + 0] = _mm256_setzero_si256();
            sums[3 * out_it + 1] = _mm256_setzero_si256();
            sums[3 * out_it + 2] = _mm256_setzero_si256();
        }

        for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
        {
            __m256i load_iL = _mm256_stream_load_si256((__m256i*)(input + 3 * in_it * 8 + 0));
            __m256i load_iM = _mm256_stream_load_si256((__m256i*)(input + 3 * in_it * 8 + 1 * C_simd_width * 2));
            __m256i load_iH = _mm256_stream_load_si256((__m256i*)(input + 3 * in_it * 8 + 2 * C_simd_width * 2));

#pragma unroll
            for (int out_it = 0; out_it < OUT_GROUPING; ++out_it)
            {
                __m256i load_w = _mm256_set1_epi32(*(int32_t*)(weights + in_it * OUT_GROUPING + out_it * 2));

                __m256i tempOut = _mm256_madd_epi16(load_iL, load_w);
                sums[3 * out_it + 0] = _mm256_add_epi32(sums[3 * out_it + 0], tempOut);

                tempOut = _mm256_madd_epi16(load_iM, load_w);
                sums[3 * out_it + 1] = _mm256_add_epi32(sums[3 * out_it + 1], tempOut);

                tempOut = _mm256_madd_epi16(load_iH, load_w);
                sums[3 * out_it + 2] = _mm256_add_epi32(sums[3 * out_it + 2], tempOut);
            }
        }

#pragma unroll
        for (uint32_t out_it = 0; out_it < 3 * OUT_GROUPING; out_it += 6)
        {
            Activation::store_activation(output, sums[out_it + 0], sums[out_it + 3], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 1], sums[out_it + 4], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 2], sums[out_it + 5], in_shift, out_shift);
            output += (2 * C_simd_width);
        }
    }

    // forward implementation
    template<class Activation, bool T_NEED_BIAS_COPY>
    static inline void process_fully_connected_int16_fixedpoint_AVX2_output_b32(
        const int16_t* const input,
        const int16_t* const weights,
        typename Activation::output_type* output,
        uint32_t numInputs,
        uint32_t numOutputs,
        const int32_t* const bias,
        int8_t in_shift,
        int8_t out_shift)
    {
        __m256i sums[4 * OUT_GROUPING];

#pragma unroll
        for (uint32_t out_it = 0; out_it < OUT_GROUPING; ++out_it)
        if (T_NEED_BIAS_COPY)
        {
            sums[4 * out_it + 0] = _mm256_set1_epi32(*(bias + out_it));
            sums[4 * out_it + 1] = _mm256_set1_epi32(*(bias + out_it));
            sums[4 * out_it + 2] = _mm256_set1_epi32(*(bias + out_it));
            sums[4 * out_it + 3] = _mm256_set1_epi32(*(bias + out_it));
        }
        else
        {
            sums[4 * out_it + 0] = _mm256_setzero_si256();
            sums[4 * out_it + 1] = _mm256_setzero_si256();
            sums[4 * out_it + 2] = _mm256_setzero_si256();
            sums[4 * out_it + 3] = _mm256_setzero_si256();
        }

        for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
        {
            __m256i load_iLL = _mm256_stream_load_si256((__m256i*)(input + 4 * in_it * 8 + 0));
            __m256i load_iLH = _mm256_stream_load_si256((__m256i*)(input + 4 * in_it * 8 + 1 * C_simd_width * 2));
            __m256i load_iHL = _mm256_stream_load_si256((__m256i*)(input + 4 * in_it * 8 + 2 * C_simd_width * 2));
            __m256i load_iHH = _mm256_stream_load_si256((__m256i*)(input + 4 * in_it * 8 + 3 * C_simd_width * 2));

#pragma unroll
            for (int out_it = 0; out_it < OUT_GROUPING; ++out_it)
            {
                __m256i load_w = _mm256_set1_epi32(*(int32_t*)(weights + in_it * OUT_GROUPING + out_it * 2));

                __m256i tempOut = _mm256_madd_epi16(load_iLL, load_w);
                sums[4 * out_it + 0] = _mm256_add_epi32(sums[4 * out_it + 0], tempOut);

                tempOut = _mm256_madd_epi16(load_iLH, load_w);
                sums[4 * out_it + 1] = _mm256_add_epi32(sums[4 * out_it + 1], tempOut);

                tempOut = _mm256_madd_epi16(load_iHL, load_w);
                sums[4 * out_it + 2] = _mm256_add_epi32(sums[4 * out_it + 2], tempOut);

                tempOut = _mm256_madd_epi16(load_iHH, load_w);
                sums[4 * out_it + 3] = _mm256_add_epi32(sums[4 * out_it + 3], tempOut);
            }
        }

#pragma unroll
        for (uint32_t out_it = 0; out_it < 4 * OUT_GROUPING; out_it += 8)
        {
            Activation::store_activation(output, sums[out_it + 0], sums[out_it + 4], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 1], sums[out_it + 5], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 2], sums[out_it + 6], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 3], sums[out_it + 7], in_shift, out_shift);
            output += (2 * C_simd_width);
        }
    }

    template <typename OutputType> struct get_arguments;

    template <> struct get_arguments<std::int16_t> {
        static inline const nn::arguments_fully_connected_forward_i16qn_i16qn &
        get(const nn_workload_item *const &work_item) {
            return work_item->arguments.fully_connected_forward_i16qn_i16qn;
        }
    };

    template <> struct get_arguments<std::int32_t> {
        static inline const nn::arguments_fully_connected_forward_i16qn_i32qn &
        get(const nn_workload_item *const &work_item) {
            return work_item->arguments.fully_connected_forward_i16qn_i32qn;
        }
    };

    template<class ActivationType, bool T_NEED_BIAS_COPY>
    void run_fully_connected_int16_fixedpoint_work_item_internal(
        nn_workload_item *const work_item,
        nn_workload_data_t *input_view,
        nn_workload_data_t *output_view)
    {
        if(std::is_same<typename ActivationType::ImplBase::output_type, std::int32_t>::value)
            assert(work_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN);
        else
            assert(work_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN);

        const auto &arguments = get_arguments<typename ActivationType::ImplBase::output_type>::get(work_item);

        const auto numInputNeurons = input_view->parent->lengths.t[NN_DATA_COORD_z] * input_view->parent->lengths.t[NN_DATA_COORD_p];
        const auto numOutputNeurons = arguments.weights->parent->lengths.t[NN_DATA_COORD_p] * C_simd_width;
        const auto outputWidth = (output_view->view_end.t[NN_DATA_COORD_z] - output_view->view_begin.t[NN_DATA_COORD_z] + 1) * output_view->parent->lengths.t[NN_DATA_COORD_p];

        const auto outputStart = output_view->view_begin.t[NN_DATA_COORD_z] * output_view->parent->lengths.t[NN_DATA_COORD_p];

        const auto input_buffer = static_cast<int16_t*>(input_view->parent->data_buffer);
        auto output_buffer = static_cast<typename ActivationType::ImplBase::output_type*>(output_view->parent->data_buffer);
        const auto biases_buffer = static_cast<int32_t*>(arguments.biases->parent->data_buffer);
        const auto weights_buffer = static_cast<int16_t*>(arguments.weights->parent->data_buffer);

        const auto C_batch_size = output_view->parent->lengths.t[NN_DATA_COORD_n];

        auto output_ptr = output_buffer + outputStart * C_batch_size;
        auto bias_ptr = biases_buffer + outputStart;
        auto weights_ptr = weights_buffer + outputStart * numInputNeurons;

        const auto acc_fraction = arguments.activation.fractions.accumulator;
        const auto out_fraction = arguments.activation.fractions.output;
        const auto shift = acc_fraction - out_fraction;

        using namespace activations::int16_fixedpoint;

        switch (C_batch_size)
        {
        case 1:
        {
                  auto NumOfFullItr = outputWidth / (C_simd_width * OUT_GROUPING);
                  for (uint32_t n = 0; n < NumOfFullItr; ++n)
                  {
                      if (shift >= 0)
                          process_fully_connected_int16_fixedpoint_AVX2_output_b1<
                              typename ActivationType::template Impl<false, ShiftDirection::Right>,
                              T_NEED_BIAS_COPY,
                              true>(input_buffer,
                                    weights_ptr,
                                    output_ptr,
                                    numInputNeurons,
                                    OUT_GROUPING,
                                    bias_ptr,
                                    acc_fraction,
                                    out_fraction);
                      else if (shift < 0)
                          process_fully_connected_int16_fixedpoint_AVX2_output_b1<
                              typename ActivationType::template Impl<false, ShiftDirection::Left>,
                              T_NEED_BIAS_COPY,
                              true>(input_buffer,
                                    weights_ptr,
                                    output_ptr,
                                    numInputNeurons,
                                    OUT_GROUPING,
                                    bias_ptr,
                                    acc_fraction,
                                    out_fraction);

                      weights_ptr += numInputNeurons * C_simd_width * OUT_GROUPING;
                      output_ptr += C_simd_width * OUT_GROUPING;
                      bias_ptr += C_simd_width * OUT_GROUPING;
                  }

                  if (outputWidth % (C_simd_width * OUT_GROUPING))
                  {
                      auto group = (outputWidth - (outputWidth / (C_simd_width * OUT_GROUPING)) * (C_simd_width * OUT_GROUPING)) / C_simd_width;

                      if (shift >= 0)
                          process_fully_connected_int16_fixedpoint_AVX2_output_b1<
                              typename ActivationType::template Impl<false, ShiftDirection::Right>, T_NEED_BIAS_COPY, false>(
                              input_buffer, weights_ptr, output_ptr, numInputNeurons, group, bias_ptr, acc_fraction, out_fraction);
                      else if (shift < 0)
                          process_fully_connected_int16_fixedpoint_AVX2_output_b1<
                              typename ActivationType::template Impl<false, ShiftDirection::Left>, T_NEED_BIAS_COPY, false>(
                              input_buffer, weights_ptr, output_ptr, numInputNeurons, group, bias_ptr, acc_fraction, out_fraction);
                  }
        }
            break;

        case 8:
        {
                  for (uint32_t n = 0; n < outputWidth; n += OUT_GROUPING)
                  {
                      if (shift >= 0)
                          process_fully_connected_int16_fixedpoint_AVX2_output_b8<
                              typename ActivationType::template Impl<true, ShiftDirection::Right>, T_NEED_BIAS_COPY>(
                              input_buffer, weights_ptr, output_ptr, numInputNeurons, outputWidth, bias_ptr,
                              acc_fraction, out_fraction);
                      else if (shift < 0)
                          process_fully_connected_int16_fixedpoint_AVX2_output_b8<
                              typename ActivationType::template Impl<true, ShiftDirection::Left>, T_NEED_BIAS_COPY>(
                              input_buffer, weights_ptr, output_ptr, numInputNeurons, outputWidth, bias_ptr,
                              acc_fraction, out_fraction);

                      weights_ptr += numInputNeurons * OUT_GROUPING;
                      output_ptr += OUT_GROUPING * C_batch_size;
                      bias_ptr += OUT_GROUPING;
                  }
        }
            break;

        case 16:
        {
                   for (uint32_t n = 0; n < outputWidth; n += OUT_GROUPING)
                  {
                       if (shift >= 0)
                           process_fully_connected_int16_fixedpoint_AVX2_output_b16<
                           typename ActivationType::template Impl<true, ShiftDirection::Right>, T_NEED_BIAS_COPY>(
                           input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                           acc_fraction, out_fraction);

                       else if (shift < 0)
                           process_fully_connected_int16_fixedpoint_AVX2_output_b16<
                           typename ActivationType::template Impl<true, ShiftDirection::Left>, T_NEED_BIAS_COPY>(
                           input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                          acc_fraction, out_fraction);

                      weights_ptr += numInputNeurons * OUT_GROUPING;
                      output_ptr += (OUT_GROUPING) * C_batch_size;
                      bias_ptr += OUT_GROUPING;
                  }
        }
            break;

        case 24:
        {
                   for (uint32_t n = 0; n < outputWidth; n += OUT_GROUPING)
                   {
                       if (shift >= 0)
                           process_fully_connected_int16_fixedpoint_AVX2_output_b24<
                           typename ActivationType::template Impl<true, ShiftDirection::Right>, T_NEED_BIAS_COPY>(
                           input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                           acc_fraction, out_fraction);
                       else if (shift < 0)
                           process_fully_connected_int16_fixedpoint_AVX2_output_b24<
                           typename ActivationType::template Impl<true, ShiftDirection::Left>, T_NEED_BIAS_COPY>(
                           input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                           acc_fraction, out_fraction);

                       weights_ptr += numInputNeurons * OUT_GROUPING;
                       output_ptr += (OUT_GROUPING)* C_batch_size;
                       bias_ptr += OUT_GROUPING;
                   }
        }
            break;

        case 32:
        {
                   for (uint32_t n = 0; n < outputWidth; n += OUT_GROUPING)
                   {
                       if (shift >= 0)
                           process_fully_connected_int16_fixedpoint_AVX2_output_b32<
                           typename ActivationType::template Impl<true, ShiftDirection::Right>, T_NEED_BIAS_COPY>(
                           input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                           acc_fraction, out_fraction);
                       else if (shift < 0)
                           process_fully_connected_int16_fixedpoint_AVX2_output_b32<
                           typename ActivationType::template Impl<true, ShiftDirection::Left>, T_NEED_BIAS_COPY>(
                           input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                           acc_fraction, out_fraction);

                       weights_ptr += numInputNeurons * OUT_GROUPING;
                       output_ptr += (OUT_GROUPING)* C_batch_size;
                       bias_ptr += OUT_GROUPING;
                   }
        }
            break;



        default:
            break;
        }
    }

    void run_fully_connected_fixedpoint_work_item(
        nn_workload_item *const work_item,
        nn_workload_data_t *input_view,
        nn_workload_data_t *output_view)
    {
        NN_ACTIVATION_FUNCTION function =
            work_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN
                ? work_item->arguments.fully_connected_forward_i16qn_i16qn.activation.basic_arguments.function
                : work_item->arguments.fully_connected_forward_i16qn_i32qn.activation.basic_arguments.function;

        const auto &biases = work_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN
                                 ? work_item->arguments.fully_connected_forward_i16qn_i16qn.biases
                                 : work_item->arguments.fully_connected_forward_i16qn_i32qn.biases;

        bool need_bias_copy = (biases != nullptr);

        using namespace activations::int16_fixedpoint;

        if (need_bias_copy) {
            if (function == NN_ACTIVATION_FUNCTION_NONE) {
                run_fully_connected_int16_fixedpoint_work_item_internal<None<std::int32_t>, true>(work_item, input_view,
                                                                                                  output_view);
            } else if (function == NN_ACTIVATION_FUNCTION_RELU) {
                run_fully_connected_int16_fixedpoint_work_item_internal<ReLu<std::int16_t>, true>(work_item, input_view,
                                                                                                  output_view);
            }else if(function == NN_ACTIVATION_FUNCTION_LOGISTIC){
                run_fully_connected_int16_fixedpoint_work_item_internal<Logistic<std::int16_t>, true>(
                    work_item, input_view, output_view);
            }else{
                assert(false);
            }
        } else {
            if (function == NN_ACTIVATION_FUNCTION_NONE) {
                run_fully_connected_int16_fixedpoint_work_item_internal<None<std::int32_t>, false>(
                    work_item, input_view, output_view);
            } else if (function == NN_ACTIVATION_FUNCTION_RELU) {
                run_fully_connected_int16_fixedpoint_work_item_internal<ReLu<std::int16_t>, false>(
                    work_item, input_view, output_view);
            } else if (function == NN_ACTIVATION_FUNCTION_LOGISTIC) {
                run_fully_connected_int16_fixedpoint_work_item_internal<Logistic<std::int16_t>, false>(
                    work_item, input_view, output_view);
            }else{
                assert(false);
            }
        }
    }

    void unpack_fully_connected_fixedpoint_callback_handle(
        void* void_handle)
    {
        nn_cpu_request_handle* handle = reinterpret_cast<nn_cpu_request_handle*>(void_handle);
        run_fully_connected_fixedpoint_work_item(handle->work_item, handle->input_view, handle->output_view);
    }


    /*
     * Inputs layout required:
     *      nn_workload_data_layout_t in_view_layout =
     *      {
     *          { 0, 0, 0, 0, 0, 0 }, // tile_log2
     *          { 0, 0, 0, 0, 0, 0 }, // alignment
     *          { NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q }, // ordering
     *          NN_DATATYPE_INT16
     *      };
     *
     *      nn_workload_data_coords_t input_view_coords =
     *      {
     *          NumBatch,
     *          1,
     *          1,
     *          NumInputs/2,
     *          2,
     *          1
     *      };
     *
     * Output is configured similarly, with outputs either Int16 or Int32
     *
     */
    void run_multithreaded_fully_connected_fixedpoint_work_item(nn_workload_item *const work_item, nn_device_internal* device)
    {
        auto num_hardware_threads = std::min(device->thread_pool.get_num_threads(), max_threads);

        const auto &weights = work_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN
                                  ? work_item->arguments.fully_connected_forward_i16qn_i16qn.weights
                                  : work_item->arguments.fully_connected_forward_i16qn_i32qn.weights;
        const auto &biases = work_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN
                                 ? work_item->arguments.fully_connected_forward_i16qn_i16qn.biases
                                 : work_item->arguments.fully_connected_forward_i16qn_i32qn.biases;

        const auto& input_view = work_item->input[0]->output;
        const auto& output_view = work_item->output;

        const auto numInputNeurons = input_view->parent->lengths.t[NN_DATA_COORD_p] * input_view->parent->lengths.t[NN_DATA_COORD_z];

        const auto outputWidth = (output_view->view_end.t[NN_DATA_COORD_z] - output_view->view_begin.t[NN_DATA_COORD_z] + 1) * output_view->parent->lengths.t[NN_DATA_COORD_p];
        const auto outputStart = output_view->view_begin.t[NN_DATA_COORD_z] * output_view->parent->lengths.t[NN_DATA_COORD_p];

        const auto batch_size = output_view->parent->lengths.t[NN_DATA_COORD_n];

        int itemsGroups_per_thread = (outputWidth / OUT_GROUPING) / num_hardware_threads;
        int itemsGroups_per_thread_modulo = (outputWidth / OUT_GROUPING) % num_hardware_threads;

        // Check if we have enough data to cover all threads.
        if ((itemsGroups_per_thread == 0) || (num_hardware_threads == 1) || (batch_size == 1))
        {
            // Its tiny data - just do it singlethreaded way.
            run_fully_connected_fixedpoint_work_item(work_item, input_view, output_view);
        }
        else
        {
            // Full cores utilization version.
            std::vector<nn_cpu_request_handle*> request_handles;
            std::vector<nn_workload_item*> slaves_work_items;

            // Allocate slave work items.
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                slaves_work_items.push_back(new nn_workload_item());
            }

            auto per_thread_multiplier = OUT_GROUPING;
            if (batch_size == 1)
                per_thread_multiplier = OUT_GROUPING * C_simd_width;

            //uint32_t output_z_per_thread = itemsGroups_per_thread * per_thread_multiplier;
            //if (batch_size != 1)
            //    output_z_per_thread = itemsGroups_per_thread * OUT_GROUPING;

            uint32_t* thread_items_sums = static_cast<uint32_t*>(alloca(num_hardware_threads * sizeof(uint32_t)));

            if (thread_items_sums == nullptr) throw std::bad_alloc();

            // Distribute elements more evenly.
            auto elements_left = itemsGroups_per_thread_modulo;
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                thread_items_sums[thread_id] = itemsGroups_per_thread;
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
                thread_items_sums[thread_id] *= per_thread_multiplier;
            }

            // Fill slave work items.
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                auto slave = slaves_work_items[thread_id];

                // Copy all data from master.
                slave->type = work_item->type;
                slave->arguments = work_item->arguments;

                const nn::nn_workload_data_t<int16_t>* cpp_master_output16;
                const nn::nn_workload_data_t<int32_t>* cpp_master_output32;
                bool output32 = output_view->parent->data_type_size == 4;
                if(output32)
                    cpp_master_output32  = reinterpret_cast<nn::nn_workload_data_t<int32_t>*>(output_view);
                else
                    cpp_master_output16  = reinterpret_cast<nn::nn_workload_data_t<int16_t>*>(output_view);

                auto work_begin = 0u;
                if (thread_id > 0u)
                    work_begin = thread_items_sums[thread_id - 1] / 2;

                auto work_end = thread_items_sums[thread_id] / 2 - 1;

                nn_workload_data_coords_t output_view_begin =
                {
                    0,
                    0,
                    0,
                    work_begin,
                    0,
                    0
                };

                nn_workload_data_coords_t output_view_end = {
                    batch_size - 1,
                    0,
                    0,
                    // Last one gets all remaining.
                    work_end,
                    1,
                    0
                };

                (output32 ? slave->arguments.fully_connected_forward_i16qn_i32qn.weights
                          : slave->arguments.fully_connected_forward_i16qn_i16qn.weights) = weights;

                if (output32)
                    slave->output = new nn::nn_workload_data_t<int32_t>(
                        *(reinterpret_cast<nn::nn_workload_data_t<int32_t> *>(output_view)), output_view_begin,
                        output_view_end);
                else
                    slave->output = new nn::nn_workload_data_t<int16_t>(
                        *(reinterpret_cast<nn::nn_workload_data_t<int16_t> *>(output_view)), output_view_begin,
                        output_view_end);

                if (biases != nullptr)
                {
                    (output32 ? slave->arguments.fully_connected_forward_i16qn_i32qn.biases
                              : slave->arguments.fully_connected_forward_i16qn_i16qn.biases) = biases;
                }
            }

            // Run threads.
            std::vector<nn_multithreaded_request> job(num_hardware_threads);
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                request_handles.push_back(new nn_cpu_request_handle);
            }

            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                request_handles[thread_id]->work_item = slaves_work_items[thread_id];
                request_handles[thread_id]->input_view = work_item->input[0]->output;
                request_handles[thread_id]->output_view = slaves_work_items[thread_id]->output;

                job[thread_id].callback = unpack_fully_connected_fixedpoint_callback_handle;
                job[thread_id].request_handle = request_handles[thread_id];
            }

            // Wait for all sub threads.
            device->thread_pool.push_job(job);

            // Cleanup dynamic memory.
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                delete request_handles[thread_id];

                //delete input_views[thread_id];

                delete slaves_work_items[thread_id]->output;
                //delete slaves_work_items[thread_id]->arguments.forward_fully_connected.biases;
                //delete slaves_work_items[thread_id]->arguments.forward_fully_connected.weights;

                delete slaves_work_items[thread_id];
            }

            // Cleanup vectors.
            request_handles.clear();
            //input_views.clear();
            slaves_work_items.clear();

        }
    }

} // namepace device_int16
