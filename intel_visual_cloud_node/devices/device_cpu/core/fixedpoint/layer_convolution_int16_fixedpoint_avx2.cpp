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
#include "layer_convolution_int16_fixedpoint_avx2.h"

#include <iostream>
#include <assert.h>
#include <string.h>
#include <algorithm>
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

namespace int16_fixedpoint
{

#define SETGET_PARAMETERS \
    const auto OFMOutBlock = work_item->output->parent->lengths.t[NN_DATA_COORD_p]; \
    const auto num_output_feature_maps = work_item->output->parent->lengths.t[NN_DATA_COORD_z] * OFMOutBlock; \
    const auto num_input_feature_maps = input_view->parent->lengths.t[NN_DATA_COORD_z] * input_view->parent->lengths.t[NN_DATA_COORD_p]; \
    const auto output_feature_map_width = work_item->output->parent->lengths.t[NN_DATA_COORD_x]; \
    const auto output_feature_map_height = work_item->output->parent->lengths.t[NN_DATA_COORD_y]; \
    const auto input_feature_map_width = input_view->parent->lengths.t[NN_DATA_COORD_x]; \
    const auto input_feature_map_height = input_view->parent->lengths.t[NN_DATA_COORD_y]; \
    const auto output_view_width = work_item->output->view_end.t[NN_DATA_COORD_x] - work_item->output->view_begin.t[NN_DATA_COORD_x] + 1; \
    const auto output_view_height = work_item->output->view_end.t[NN_DATA_COORD_y] - work_item->output->view_begin.t[NN_DATA_COORD_y] + 1; \
    const auto input_view_width = input_view->view_end.t[NN_DATA_COORD_x] - input_view->view_begin.t[NN_DATA_COORD_x] + 1; \
    const auto input_view_height = input_view->view_end.t[NN_DATA_COORD_y] - input_view->view_begin.t[NN_DATA_COORD_y] + 1; \
    const auto kernel_width = arguments->weights->parent->lengths.t[NN_DATA_COORD_n]; \
    const auto kernel_height = arguments->weights->parent->lengths.t[NN_DATA_COORD_x]; \
    const auto kernel_stride_x = arguments->stride[0]; \
    const auto kernel_stride_y = arguments->stride[1]; \
    const auto acc_fraction = arguments->activation.fractions.accumulator; \
    const auto out_fraction = arguments->activation.fractions.output

    inline void AddNoBias(__m256i & val, void * addr)
    {
        val = _mm256_setzero_si256();
    }

    inline void AddBias16(__m256i & val, void * addr)
    {
        val = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i *)(addr)));
    }
    inline void AddBias32(__m256i & val, void * addr)
    {
        val = _mm256_load_si256((__m256i *)(addr));
    }

    inline void Store16_wRELu_shiftR(void * addr, __m256i val, uint8_t shift)
    {
        _mm_stream_si128((__m128i *)(addr), _mm_max_epi16(_mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(_mm256_srai_epi32(val, shift), _mm256_setzero_si256()), 0xd8)), _mm_setzero_si128()));
    }
    inline void Store16_wRELu_shiftL(void * addr, __m256i val, uint8_t shift)
    {
        _mm_stream_si128((__m128i *)(addr), _mm_max_epi16(_mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(_mm256_slli_epi32(val, -shift), _mm256_setzero_si256()), 0xd8)), _mm_setzero_si128()));
    }

    inline void Store16_woActiv_shiftR(void * addr, __m256i val, uint8_t shift)
    {
        _mm_stream_si128((__m128i *)(addr), _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(_mm256_srai_epi32(val, shift), _mm256_setzero_si256()), 0xd8)));
    }
    inline void Store16_woActiv_shiftL(void * addr, __m256i val, uint8_t shift)
    {
        _mm_stream_si128((__m128i *)(addr), _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(_mm256_slli_epi32(val, -shift), _mm256_setzero_si256()), 0xd8)));
    }

    inline __m256i ***Create3DBuf256(uint32_t Dim1, uint32_t Dim2, uint32_t Dim3)
    {
        __m256i *** dest = new __m256i **[Dim1];

        dest[0] = new __m256i *[Dim1*Dim2];

        dest[0][0] = (__m256i *)_mm_malloc(Dim1*Dim2*Dim3*sizeof(__m256i), 64);

        uint32_t i, j;

        for (i = 0; i < Dim1; i++)
        {
            if (i < Dim1 - 1)
            {
                dest[0][(i + 1)*Dim2] = &(dest[0][0][(i + 1)*Dim3*Dim2]);
                dest[i + 1] = &(dest[0][(i + 1)*Dim2]);
            }

            for (j = 0; j < Dim2; j++)
            {
                if (j > 0) dest[i][j] = dest[i][j - 1] + Dim3;
            }
        }
        return dest;
    }

    static void Delete3DBuf256(__m256i ***dist)
    {
        _mm_free(dist[0][0]);
        delete[] dist[0];
        delete[] dist;
    }

    template <class T> T ***Create3DBuf(uint32_t Dim1, uint32_t Dim2, uint32_t Dim3)
    {
        T *** dest = new T **[Dim1];

        dest[0] = new T *[Dim1*Dim2];

        dest[0][0] = new T[Dim1*Dim2*Dim3];

        uint32_t i, j;

        for (i = 0; i < Dim1; i++)
        {
            if (i < Dim1 - 1)
            {
                dest[0][(i + 1)*Dim2] = &(dest[0][0][(i + 1)*Dim3*Dim2]);
                dest[i + 1] = &(dest[0][(i + 1)*Dim2]);
            }

            for (j = 0; j < Dim2; j++)
            {
                if (j > 0) dest[i][j] = dest[i][j - 1] + Dim3;
            }
        }
        return dest;
    }


    template <class T> void Delete3DBuf(T ***dist)
    {
        delete[] dist[0][0];
        delete[] dist[0];
        delete[] dist;
    }

    template <const int T_input_z_block_size,
        const int T_compute_ofm_block_size,
        const int T_compute_x_block_size,
        const int T_compute_y_block_size,
        const int T_compute_x_sub_block_size,
        const int T_compute_y_sub_block_size,
        FActiveShift FStoreActiv,
        FBias FBias>
        void NN_ConvolveAVX2_INT16_fixedpoint(nn_workload_item *const work_item, nn_workload_data_t *_input_view) {
            const size_t ifm_sub_block_size = 2;
            const size_t output_z_block_size = 8;

            auto input_view = reinterpret_cast<nn::nn_workload_data_t<int16_t> *>(_input_view);
            auto output_view = reinterpret_cast<nn::nn_workload_data_t<int16_t> *>(work_item->output);

            const size_t batch_window_size =
                output_view->view_end.t[NN_DATA_COORD_n] - output_view->view_begin.t[NN_DATA_COORD_n] + 1;
            const size_t batch_window_start = output_view->view_begin.t[NN_DATA_COORD_n];

            // outputs
            assert(output_view->parent->lengths.t[NN_DATA_COORD_p] == output_z_block_size);
            const size_t output_size_x = output_view->parent->lengths.t[NN_DATA_COORD_x],
                output_stride_x = output_z_block_size;
            const size_t output_size_y = output_view->parent->lengths.t[NN_DATA_COORD_y],
                output_stride_y = output_stride_x * output_size_x;
            const size_t output_size_z = output_view->parent->lengths.t[NN_DATA_COORD_z] * output_z_block_size,
                output_stride_z = output_stride_y * output_size_y;
            const size_t output_stride_batch = output_size_x * output_size_y * output_size_z;

            const size_t output_window_start_x = output_view->view_begin.t[NN_DATA_COORD_x],
                output_window_size_x = output_view->view_end.t[NN_DATA_COORD_x] - output_window_start_x + 1;
            const size_t output_window_start_y = output_view->view_begin.t[NN_DATA_COORD_y],
                output_window_size_y = output_view->view_end.t[NN_DATA_COORD_y] - output_window_start_y + 1;
            const size_t output_window_start_z = output_view->view_begin.t[NN_DATA_COORD_z],
                output_window_size_z =
                (output_view->view_end.t[NN_DATA_COORD_z] - output_window_start_z + 1) * output_z_block_size;

            // inputs
            if (input_view->parent->lengths.t[NN_DATA_COORD_p] != T_input_z_block_size)
                assert(input_view->parent->lengths.t[NN_DATA_COORD_p] == T_input_z_block_size);

            const size_t input_size_x = input_view->parent->lengths.t[NN_DATA_COORD_x],
                input_stride_x = T_input_z_block_size;
            const size_t input_size_y = input_view->parent->lengths.t[NN_DATA_COORD_y],
                input_stride_y = input_stride_x * input_size_x;
            const size_t input_size_z = input_view->parent->lengths.t[NN_DATA_COORD_z] * T_input_z_block_size,
                input_stride_z = input_stride_y * input_size_y,
                input_window_size_z = (input_view->view_end.t[NN_DATA_COORD_z] - input_view->view_begin.t[NN_DATA_COORD_z] + 1) * T_input_z_block_size;
            const size_t input_stride_batch = input_size_x * input_size_y * input_size_z;

            const size_t input_start_x = input_view->view_begin.t[NN_DATA_COORD_x];
            const size_t input_start_y = input_view->view_begin.t[NN_DATA_COORD_y];
            const size_t input_start_z = input_view->view_begin.t[NN_DATA_COORD_z];

            const auto& arguments = work_item->arguments.forward_convolution_fixedpoint;

            // weights
            assert(arguments.weights->parent->lengths.t[NN_DATA_COORD_p] == T_compute_ofm_block_size);
            assert(arguments.weights->parent->lengths.t[NN_DATA_COORD_y] == ifm_sub_block_size);
            // TODO assert weights ifm view matches input z view
            const size_t weights_start_ofm = arguments.weights->view_begin.t[NN_DATA_COORD_q];
            const size_t weights_stride_ifm = ifm_sub_block_size * T_compute_ofm_block_size;
            const size_t weights_stride_x = weights_stride_ifm * arguments.weights->parent->lengths.t[NN_DATA_COORD_z];
            const size_t weights_size_x = arguments.weights->parent->lengths.t[NN_DATA_COORD_n];
            const size_t weights_stride_y = weights_stride_x * arguments.weights->parent->lengths.t[NN_DATA_COORD_n];
            const size_t weights_size_y = arguments.weights->parent->lengths.t[NN_DATA_COORD_x];
            const size_t weights_stride_ofm = weights_stride_y * arguments.weights->parent->lengths.t[NN_DATA_COORD_x];

            // bias
            const size_t bias_start = arguments.biases->view_begin.t[NN_DATA_COORD_z];

            for (uint32_t batch_it = 0; batch_it < batch_window_size; ++batch_it)
            {
                int16_t *output_window = (int16_t *)output_view->parent->data_buffer
                    + output_window_start_x * output_stride_x
                    + output_window_start_y * output_stride_y
                    + output_window_start_z * output_stride_z
                    + (batch_window_start + batch_it) * output_stride_batch;

                int16_t *input_window = (int16_t *)input_view->parent->data_buffer
                    + input_start_x * input_stride_x
                    + input_start_y * input_stride_y
                    + input_start_z * input_stride_z
                    + (batch_window_start + batch_it) * input_stride_batch;

                int16_t* weights_window = (int16_t*)arguments.weights->parent->data_buffer
                    + weights_stride_ofm * weights_start_ofm;

                int32_t* bias_window = (int32_t*)arguments.biases->parent->data_buffer
                    + bias_start;

                const size_t center_x = arguments.center_offset[0];
                const size_t center_y = arguments.center_offset[1];

                const size_t kernel_stride_x = arguments.stride[0];
                const size_t kernel_stride_y = arguments.stride[1];

                const size_t shift = arguments.activation.fractions.accumulator - arguments.activation.fractions.output;

                //it is assumed that inputs and kernel layout is ready
                for (size_t BlockOffsetY = 0; BlockOffsetY < output_window_size_y; BlockOffsetY += T_compute_y_block_size)
                {
                    for (size_t BlockOffsetX = 0; BlockOffsetX < output_window_size_x; BlockOffsetX += T_compute_x_block_size)
                    {
                        for (size_t BlockOffsetOFM = 0; BlockOffsetOFM < output_window_size_z; BlockOffsetOFM += T_compute_ofm_block_size) {
                            __m256i OutBlock[T_compute_y_block_size][T_compute_x_block_size][T_compute_ofm_block_size / 8];
                            for (size_t OBlockYItr = 0; OBlockYItr < T_compute_y_block_size; OBlockYItr += 1)
                            {
#pragma unroll
                                for (size_t OBlockXItr = 0; OBlockXItr < T_compute_x_block_size; OBlockXItr += 1)
                                {
#pragma unroll
                                    for (size_t OFMItr = 0; OFMItr < T_compute_ofm_block_size / 8; ++OFMItr)
                                    {
                                        FBias(OutBlock[OBlockYItr][OBlockXItr][OFMItr],
                                            (__m128i *)(bias_window + BlockOffsetOFM + OFMItr * 8));
                                    }
                                }
                            }

                            for (size_t ifm_block_offset = 0; ifm_block_offset < input_window_size_z; ifm_block_offset += T_input_z_block_size) {
                                for (size_t KernelYItr = 0; KernelYItr < weights_size_y; KernelYItr += 1)
                                {
                                    for (size_t KernelXItr = 0; KernelXItr < weights_size_x; KernelXItr += 1)
                                    {
#pragma unroll
                                        for (size_t OBlockYItr = 0; OBlockYItr < T_compute_y_block_size; OBlockYItr += T_compute_y_sub_block_size)
                                        {
#pragma unroll
                                            for (size_t OBlockXItr = 0; OBlockXItr < T_compute_x_block_size; OBlockXItr += T_compute_x_sub_block_size)
                                            {
#pragma unroll (T_compute_y_sub_block_size)
                                                for (size_t OpBlockYItr = 0; OpBlockYItr < T_compute_y_sub_block_size; OpBlockYItr += 1)
                                                {
#pragma unroll (T_compute_x_sub_block_size)
                                                    for (size_t OpBlockXItr = 0; OpBlockXItr < T_compute_x_sub_block_size; OpBlockXItr += 1)
                                                    {
                                                        __m256i vi[T_input_z_block_size / ifm_sub_block_size];
#pragma unroll (T_input_z_block_size / ifm_sub_block_size)
                                                        for (size_t IFMSubItr = 0; IFMSubItr < T_input_z_block_size / ifm_sub_block_size; IFMSubItr += 1)
                                                        {
                                                            vi[IFMSubItr] = _mm256_set1_epi32(*(int32_t *)(input_window
                                                                + ((BlockOffsetX + OBlockXItr + OpBlockXItr - center_x) * kernel_stride_x + KernelXItr) * input_stride_x
                                                                + ((BlockOffsetY + OBlockYItr + OpBlockYItr - center_y) * kernel_stride_y + KernelYItr) * input_stride_y
                                                                + (ifm_block_offset) / T_input_z_block_size * input_stride_z
                                                                + IFMSubItr * ifm_sub_block_size));
                                                        }
#pragma unroll (T_input_z_block_size / ifm_sub_block_size)
                                                        for (size_t IFMSubItr = 0; IFMSubItr < T_input_z_block_size / ifm_sub_block_size; IFMSubItr += 1)
                                                        {
#pragma unroll (T_compute_ofm_block_size / 8)
                                                            for (size_t OFMItr = 0; OFMItr < T_compute_ofm_block_size / 8; OFMItr += 1)
                                                            {
                                                                __m256i vw = _mm256_load_si256((__m256i *)(weights_window
                                                                    + ifm_sub_block_size * 8 * OFMItr
                                                                    + (ifm_block_offset + IFMSubItr * ifm_sub_block_size) / ifm_sub_block_size * weights_stride_ifm
                                                                    + (BlockOffsetOFM) / T_compute_ofm_block_size * weights_stride_ofm
                                                                    + KernelXItr * weights_stride_x
                                                                    + KernelYItr * weights_stride_y));

                                                                __m256i tmp =
                                                                    _mm256_madd_epi16(
                                                                    vw
                                                                    , vi[IFMSubItr]);

                                                                OutBlock[OBlockYItr + OpBlockYItr][OBlockXItr + OpBlockXItr][OFMItr] =
                                                                    _mm256_add_epi32(OutBlock[OBlockYItr + OpBlockYItr][OBlockXItr + OpBlockXItr][OFMItr], tmp);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            for (size_t OBlockYItr = 0; OBlockYItr < T_compute_y_block_size; OBlockYItr += 1)
                            {
                                for (size_t OBlockXItr = 0; OBlockXItr < T_compute_x_block_size; OBlockXItr += 1)
                                {
                                    for (size_t OFMItr = 0; OFMItr < T_compute_ofm_block_size; OFMItr += 8)
                                    {
                                        FStoreActiv((void *)(output_window
                                            + (BlockOffsetOFM + OFMItr) / 8 * output_stride_z
                                            + (BlockOffsetY + OBlockYItr) * output_stride_y
                                            + (BlockOffsetX + OBlockXItr) * output_stride_x),
                                            OutBlock[OBlockYItr][OBlockXItr][OFMItr / 8],
                                            shift);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    template <FActiveShift FStoreActiv, FBias FBias>
    void NN_ConvolveAVX2_INT16_fixedpoint_Flex(
        nn_workload_item *const work_item,
        nn_workload_data_t *input_view,
        nn_workload_data_t *output_view,
        nn::arguments_forward_convolution_fixedpoint* arguments)
    {
        // Get required data from call argument.
        int16_t* output = (int16_t *)output_view->parent->data_buffer;
        int16_t* input = (int16_t *)input_view->parent->data_buffer;
        int16_t* kernel = (int16_t*)arguments->weights->parent->data_buffer;
        int32_t* bias = (int32_t*)arguments->biases->parent->data_buffer;

        SETGET_PARAMETERS;

        const auto center_x = arguments->center_offset[0];
        const auto center_y = arguments->center_offset[1];
        const auto OutStartX = output_view->view_begin.t[NN_DATA_COORD_x];
        const auto OutStartY = output_view->view_begin.t[NN_DATA_COORD_y];

        auto BlockOffsetOFMStart = OFMOutBlock * output_view->view_begin.t[NN_DATA_COORD_z];
        auto BlockOffsetOFMEnd = OFMOutBlock * (output_view->view_end.t[NN_DATA_COORD_z] + 1);

        const auto shift = acc_fraction - out_fraction;

        const uint32_t OXBlock = output_view_width;
        const uint32_t OXpBlock = output_view_width;
        const uint32_t OYBlock = output_view_height;
        const uint32_t OYpBlock = (output_view_height <= 16 && output_view_height % 2 == 0) ? 2 : 1;
        const uint32_t IFMBlock = input_view->parent->lengths.t[NN_DATA_COORD_p];
        const uint32_t OFMBlock = arguments->weights->parent->lengths.t[NN_DATA_COORD_p];

        __m256i *** OutBlock = Create3DBuf256(OYBlock, OXBlock, OFMBlock / 8);
        __m256i *vi = (__m256i *)_mm_malloc((IFMBlock / 2) * sizeof(__m256i), 64);


        //it is assumed that inputs and kernal layout is ready
        for (uint32_t BlockOffsetY = OutStartY; BlockOffsetY < OutStartY + output_view_height; BlockOffsetY += OYBlock)
        {
            for (uint32_t BlockOffsetX = OutStartX; BlockOffsetX < OutStartX + output_view_width; BlockOffsetX += OXBlock)
            {
                for (uint32_t BlockOffsetOFM = BlockOffsetOFMStart; BlockOffsetOFM < BlockOffsetOFMEnd;
                    BlockOffsetOFM += OFMBlock)
                {
                    for (uint32_t OBlockYItr = 0; OBlockYItr < OYBlock; OBlockYItr += 1)
                    {
#pragma unroll
                        for (uint32_t OBlockXItr = 0; OBlockXItr < OXBlock; OBlockXItr += 1)
                        {
#pragma unroll
                            for (uint32_t OFMItr = 0; OFMItr < OFMBlock; OFMItr += 8)
                            {
                                FBias(OutBlock[OBlockYItr][OBlockXItr][OFMItr / 8], (__m128i *)(bias + BlockOffsetOFM + OFMItr));
                            }
                        }
                    }

                    for (uint32_t IFMItr = 0; IFMItr < num_input_feature_maps / IFMBlock; IFMItr += 1)
                    {
                        for (uint32_t KernelYItr = 0; KernelYItr < kernel_height; KernelYItr += 1)
                        {
                            for (uint32_t KernelXItr = 0; KernelXItr < kernel_width; KernelXItr += 1)
                            {
#pragma unroll
                                for (uint32_t OBlockYItr = 0; OBlockYItr < OYBlock; OBlockYItr += OYpBlock)
                                {
#pragma unroll
                                    for (uint32_t OBlockXItr = 0; OBlockXItr < OXBlock; OBlockXItr += OXpBlock)
                                    {
#pragma unroll
                                        for (uint32_t OpBlockYItr = 0; OpBlockYItr < OYpBlock; OpBlockYItr += 1)
                                        {
#pragma unroll
                                            for (uint32_t OpBlockXItr = 0; OpBlockXItr < OXpBlock; OpBlockXItr += 1)
                                            {

#pragma unroll
                                                for (uint32_t IFMSubItr = 0; IFMSubItr < IFMBlock / 2; IFMSubItr += 1)
                                                {
                                                    vi[IFMSubItr]
                                                        = _mm256_set1_epi32(*(int32_t *)(input + IFMItr * IFMBlock * input_feature_map_width * input_feature_map_height +
                                                        IFMSubItr * 2 +
                                                        (OBlockXItr + OpBlockXItr + BlockOffsetX - center_x) * IFMBlock * kernel_stride_x + KernelXItr * IFMBlock +
                                                        (OBlockYItr + OpBlockYItr + BlockOffsetY - center_y) * input_feature_map_width * IFMBlock * kernel_stride_y + KernelYItr * input_feature_map_width * IFMBlock));
                                                }
#pragma unroll
                                                for (uint32_t IFMSubItr = 0; IFMSubItr < IFMBlock / 2; IFMSubItr += 1)
                                                {
#pragma unroll
                                                    for (uint32_t OFMItr = 0; OFMItr < OFMBlock / 8; OFMItr += 1)
                                                    {
                                                        __m256i vw = _mm256_load_si256((__m256i *)(kernel + 2 * 8 * OFMItr + (IFMItr * IFMBlock + IFMSubItr * 2) * OFMBlock +
                                                            KernelXItr * (num_input_feature_maps * OFMBlock) +
                                                            KernelYItr * (num_input_feature_maps * OFMBlock) * kernel_width +
                                                            num_input_feature_maps * kernel_width * kernel_height * BlockOffsetOFM));

                                                        __m256i tmp =
                                                            _mm256_madd_epi16(
                                                            vw
                                                            , vi[IFMSubItr]);
                                                        OutBlock[OBlockYItr + OpBlockYItr][OBlockXItr + OpBlockXItr][OFMItr] =
                                                            _mm256_add_epi32(OutBlock[OBlockYItr + OpBlockYItr][OBlockXItr + OpBlockXItr][OFMItr], tmp);

                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (uint32_t OBlockYItr = 0; OBlockYItr < OYBlock; OBlockYItr += 1)
                    {
                        for (uint32_t OBlockXItr = 0; OBlockXItr < OXBlock; OBlockXItr += 1)
                        {
                            for (uint32_t OFMItr = 0; OFMItr < OFMBlock; OFMItr += 8)
                            {
                                FStoreActiv((void *)(output +
                                    output_feature_map_width * output_feature_map_height * (BlockOffsetOFM + OFMItr) +
                                    (OBlockXItr + BlockOffsetX +
                                    OBlockYItr * output_feature_map_width +
                                    BlockOffsetY * output_feature_map_width) * OFMOutBlock),
                                    OutBlock[OBlockYItr][OBlockXItr][OFMItr / 8]
                                    , shift);

                            }
                        }
                    }

                }
            }
        }
        _mm_free(vi);
        Delete3DBuf256(OutBlock);
    }

    template <FActiveShift FStoreActiv, FBias FBias>
    struct convolution_avx2_int16_fixedpoint_selector{

        using convolve_call_params = std::tuple<nn_workload_item *const, nn_workload_data_t *>;

        template<int IFMBlock, int OFMBlock, int OXBlock, int OXpBlock>
        struct SelectOYBlock{
            inline static void choose(int num_ifm, int num_ofm, int out_width, int out_height, convolve_call_params const& call_params){
                if (out_height % 2 == 0)
                    NN_ConvolveAVX2_INT16_fixedpoint<IFMBlock, OFMBlock, OXBlock, 2, OXpBlock, 2, FStoreActiv, FBias>(std::get<0>(call_params), std::get<1>(call_params));
                else
                    NN_ConvolveAVX2_INT16_fixedpoint<IFMBlock, OFMBlock, OXBlock, 1, OXpBlock, 1, FStoreActiv, FBias>(std::get<0>(call_params), std::get<1>(call_params));
            }
        };

        template<int IFMBlock, int OFMBlock>
        struct SelectOXBlock{
            inline static void choose(int num_ifm, int num_ofm, int out_width, int out_height, convolve_call_params const& call_params){
                if (out_width % 14 == 0)
                    SelectOYBlock<IFMBlock, OFMBlock, 14, 2>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else if (out_width % 13 == 0)
                    SelectOYBlock<IFMBlock, OFMBlock, 13, 1>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else if (out_width % 12 == 0)
                    SelectOYBlock<IFMBlock, OFMBlock, 12, 2>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else if (out_width % 9 == 0)
                    SelectOYBlock<IFMBlock, OFMBlock, 9, 3>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else if (out_width % 2 == 0)
                    SelectOYBlock<IFMBlock, OFMBlock, 2, 2>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else
                    SelectOYBlock<IFMBlock, OFMBlock, 1, 1>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
            }
        };

        template <int IFMBlock>
        struct SelectOXBlock<IFMBlock, 384> {
            inline static void choose(int num_ifm, int num_ofm, int out_width, int out_height, convolve_call_params const& call_params){
                //    template <const int IFMBlock, const int OFMBlock, const int OXBlock, const int OYBlock, const int OXpBlock, const int OYpBlock, FActiveShift FStoreActiv, FBias FBias>

                NN_ConvolveAVX2_INT16_fixedpoint<IFMBlock, 384, 1, 1, 1, 1, FStoreActiv, FBias>(std::get<0>(call_params), std::get<1>(call_params));
            }
        };

        template<int IFMBlock>
        struct SelectOFMBlock{
            inline static void choose(int num_ifm, int num_ofm, int out_width, int out_height, convolve_call_params const& call_params){
                if (out_width == 1 && out_height == 1 && num_ofm % 384 == 0)
                    SelectOXBlock<IFMBlock, 384>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else if (num_ofm % 32 == 0)
                    SelectOXBlock<IFMBlock, 32>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else
                    assert(0);
            }
        };

        struct SelectIFMBlock{
            inline static void choose(int num_ifm, int num_ofm, int out_width, int out_height, convolve_call_params const& call_params){
                if (num_ifm % 8 == 0)
                    SelectOFMBlock<8>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else if (num_ifm % 4 == 0)
                    SelectOFMBlock<4>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else
                    assert(0);
            }
        };

        inline static void choose(
            nn_workload_item *const work_item,
            nn_workload_data_t *input_view,
            nn_workload_data_t *output_view,
            nn::arguments_forward_convolution_fixedpoint* arguments)
        {
            SETGET_PARAMETERS;
            const auto acc_shift = acc_fraction - out_fraction;

            SelectIFMBlock::choose(input_view->parent->lengths.t[NN_DATA_COORD_p], num_output_feature_maps, output_view_width, output_view_height, std::forward_as_tuple(work_item, input_view));
        }
    };

    void run_convolve_fixedpoint_work_item(
        nn_workload_item *const work_item,
        nn_workload_data_t *input_view,
        nn_workload_data_t *output_view)
    {
        auto &arguments = work_item->arguments.forward_convolution_fixedpoint;

        const auto acc_shift = arguments.activation.fractions.accumulator - arguments.activation.fractions.output;
        // Copy biases.
        if (arguments.biases != nullptr &&
            arguments.biases->parent != nullptr &&
            arguments.biases->parent->data_buffer != nullptr)
        {
            // Invoke convolution.
            if (arguments.activation.basic_arguments.function == NN_ACTIVATION_FUNCTION_NONE)
            {
                //convolution_avx2_int16_fixedpoint_wBias_woActiv(&arguments);
                if (acc_shift >= 0)
                    convolution_avx2_int16_fixedpoint_selector<Store16_woActiv_shiftR, AddBias32>::choose(work_item, input_view, output_view, &arguments);
                else
                    convolution_avx2_int16_fixedpoint_selector<Store16_woActiv_shiftL, AddBias32>::choose(work_item, input_view, output_view, &arguments);
            }
            else if (arguments.activation.basic_arguments.function == NN_ACTIVATION_FUNCTION_RELU)
            {
                //convolution_avx2_int16_fixedpoint_wBias_wRELu(&arguments);
                if (acc_shift >= 0)
                    convolution_avx2_int16_fixedpoint_selector<Store16_wRELu_shiftR, AddBias32>::choose(work_item, input_view, output_view, &arguments);
                else
                    convolution_avx2_int16_fixedpoint_selector<Store16_wRELu_shiftL, AddBias32>::choose(work_item, input_view, output_view, &arguments);
            }
        }
        else
        {
            // Invoke convolution.
            if (arguments.activation.basic_arguments.function == NN_ACTIVATION_FUNCTION_NONE)
            {
                //convolution_avx2_int16_fixedpoint_woBias_woActiv(&arguments);
                if (acc_shift >= 0)
                    convolution_avx2_int16_fixedpoint_selector<Store16_woActiv_shiftR, AddNoBias>::choose(work_item, input_view, output_view, &arguments);
                else
                    convolution_avx2_int16_fixedpoint_selector<Store16_woActiv_shiftL, AddNoBias>::choose(work_item, input_view, output_view, &arguments);
            }
            else if (arguments.activation.basic_arguments.function == NN_ACTIVATION_FUNCTION_RELU)
            {
                //convolution_avx2_int16_fixedpoint_woBias_wRELu(&arguments);
                if (acc_shift >= 0)
                    convolution_avx2_int16_fixedpoint_selector<Store16_wRELu_shiftR, AddNoBias>::choose(work_item, input_view, output_view, &arguments);
                else
                    convolution_avx2_int16_fixedpoint_selector<Store16_wRELu_shiftL, AddNoBias>::choose(work_item, input_view, output_view, &arguments);
            }
        }
    }


    void unpack_convolve_fixedpoint_callback_handle(
        void* void_handle)
    {
        nn_cpu_request_handle* handle = reinterpret_cast<nn_cpu_request_handle*>(void_handle);
        run_convolve_fixedpoint_work_item(handle->work_item, handle->input_view, handle->output_view);
    }

    /*
    * Inputs layout required:
    *      nn_workload_data_layout_t in_view_layout =
    *      {
    *          { 0, 0, 0, 0, 0, 0 }, // tile_log2
    *          { 0, 0, 0, 0, 0, 0 }, // alignment
    *          { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
    *          NN_DATATYPE_INT16
    *      };
    *
    *      nn_workload_data_coords_t input_view_coords =
    *      {
    *          NumBatch,
    *          Width,
    *          Height,
    *          NumInputs/FMBlock,
    *          FMBlock,
    *          1
    *      };
    *
    * Output is configured similarly, with outputs either Int16 or Int32
    *
    */
    void run_multithreaded_convolve_fixedpoint_work_item(nn_workload_item *const work_item, nn_device_internal* device)
    {
        const auto threadpool_size = device->thread_pool.get_num_threads();

        const auto &master_arguments = work_item->arguments.forward_convolution_fixedpoint;

        const auto OFMBlock = master_arguments.weights->parent->lengths.t[NN_DATA_COORD_p];
        const auto ofm_out_block_size = work_item->output->parent->lengths.t[NN_DATA_COORD_p];

        const auto num_output_feature_maps =
            (work_item->output->view_end.t[NN_DATA_COORD_z] - work_item->output->view_begin.t[NN_DATA_COORD_z] + 1) *
            ofm_out_block_size;

        const auto batch_size = work_item->output->parent->lengths.t[NN_DATA_COORD_n];

        const auto ofm_group_size = OFMBlock;

        const auto ofm_groups_per_batch = num_output_feature_maps / ofm_group_size;

        const auto task_count = batch_size * ofm_groups_per_batch;

        // Check if we have enough data to cover all threads.
        if (threadpool_size < 2 || task_count < 2)
        {
            // Its tiny data - just do it singlethreaded way.
            run_convolve_fixedpoint_work_item(work_item, work_item->input[0]->output, work_item->output);
        }
        else
        {
            std::vector<nn_workload_item> slave_work_items;
            std::vector<nn_workload_data_t*> input_views;

            slave_work_items.resize(task_count);

            // Fill slave work items.
            for (auto it_ofm_group = 0u; it_ofm_group < ofm_groups_per_batch; ++it_ofm_group) {
                for (auto it_batch = 0u; it_batch < batch_size; ++it_batch) {

                    auto& slave = slave_work_items[it_batch + it_ofm_group * batch_size];
                    auto& slave_arguments = slave.arguments.forward_convolution_fixedpoint;

                    // Copy all data from master.
                    slave.type = work_item->type;
                    slave_arguments = master_arguments;

                    const auto cpp_master_input = reinterpret_cast<nn::nn_workload_data_t<int16_t>*>(work_item->input[0]->output);
                    const auto cpp_master_output = reinterpret_cast<nn::nn_workload_data_t<int16_t>*>(work_item->output);
                    const auto cpp_master_weights = reinterpret_cast<nn::nn_workload_data_t<int16_t>*>(master_arguments.weights);

                    auto work_begin_batch = it_batch;
                    auto work_begin_ofm_out_block = it_ofm_group * ofm_group_size / ofm_out_block_size;

                    auto work_end_batch = it_batch + 1;
                    auto work_end_ofm_out_block = work_begin_ofm_out_block + ofm_group_size / ofm_out_block_size;

                    nn_workload_data_coords_t output_view_begin =
                    {
                        work_begin_batch,
                        0,
                        0,
                        work_begin_ofm_out_block,
                        0,
                        0
                    };

                    nn_workload_data_coords_t output_view_end =
                    {
                        work_end_batch - 1,
                        cpp_master_output->get_length(NN_DATA_COORD_x) - 1,
                        cpp_master_output->get_length(NN_DATA_COORD_y) - 1,
                        work_end_ofm_out_block - 1,
                        cpp_master_output->get_length(NN_DATA_COORD_p) - 1,
                        0
                    };

                    nn_workload_data_coords_t input_view_begin =
                    {
                        work_begin_batch,
                        0,
                        0,
                        0,
                        0,
                        0
                    };

                    nn_workload_data_coords_t input_view_end = {
                        work_end_batch - 1,
                        cpp_master_input->get_length(NN_DATA_COORD_x) - 1,
                        cpp_master_input->get_length(NN_DATA_COORD_y) - 1,
                        cpp_master_input->get_length(NN_DATA_COORD_z) - 1,
                        cpp_master_input->get_length(NN_DATA_COORD_p) - 1,
                        cpp_master_input->get_length(NN_DATA_COORD_q) - 1
                    };


                    nn_workload_data_coords_t weights_view_begin =
                    {
                        0,
                        0,
                        0,
                        0,
                        0,
                        work_begin_ofm_out_block * ofm_out_block_size / ofm_group_size
                    };

                    nn_workload_data_coords_t weights_view_end =
                    {
                        cpp_master_weights->get_length(NN_DATA_COORD_n) - 1,
                        cpp_master_weights->get_length(NN_DATA_COORD_x) - 1,
                        cpp_master_weights->get_length(NN_DATA_COORD_y) - 1,
                        cpp_master_weights->get_length(NN_DATA_COORD_z) - 1,
                        cpp_master_weights->get_length(NN_DATA_COORD_p) - 1,
                        work_end_ofm_out_block * ofm_out_block_size / ofm_group_size - 1
                    };

                    input_views.push_back(new nn::nn_workload_data_t<int16_t>(
                        *(reinterpret_cast<nn::nn_workload_data_t<int16_t> *>(work_item->input[0]->output)),
                        input_view_begin,
                        input_view_end));

                    slave.output = new nn::nn_workload_data_t<int16_t>(
                        *(reinterpret_cast<nn::nn_workload_data_t<int16_t> *>(work_item->output)),
                        output_view_begin,
                        output_view_end);

                    slave_arguments.weights = new nn::nn_workload_data_t<int16_t>(
                        *(reinterpret_cast<nn::nn_workload_data_t<int16_t> *>(master_arguments.weights)),
                        weights_view_begin,
                        weights_view_end);

                    // Use biases.
                    if (master_arguments.biases != nullptr)
                    {
                        const auto cpp_master_biases = reinterpret_cast<nn::nn_workload_data_t<int32_t>*>(master_arguments.biases);

                        nn_workload_data_coords_t bias_view_begin =
                        {
                            0,
                            0,
                            0,
                            work_begin_ofm_out_block * ofm_out_block_size,
                            0,
                            0
                        };
                        nn_workload_data_coords_t bias_view_end =
                        {
                            cpp_master_biases->get_length(NN_DATA_COORD_n) - 1,
                            cpp_master_biases->get_length(NN_DATA_COORD_x) - 1,
                            cpp_master_biases->get_length(NN_DATA_COORD_y) - 1,
                            work_end_ofm_out_block * ofm_out_block_size - 1,
                            cpp_master_biases->get_length(NN_DATA_COORD_p) - 1,
                            cpp_master_biases->get_length(NN_DATA_COORD_q) - 1
                        };

                        slave_arguments.biases = new nn::nn_workload_data_t<int32_t>(
                            *(reinterpret_cast<nn::nn_workload_data_t<int32_t> *>(master_arguments.biases)),
                            bias_view_begin,
                            bias_view_end);
                    }
                }
            }

            std::vector<nn_multithreaded_request> jobs(task_count);
            std::vector<nn_cpu_request_handle> request_handles(task_count);

            for (auto it_task = 0; it_task < task_count; ++it_task)
            {
                request_handles[it_task].work_item = &slave_work_items[it_task];
                request_handles[it_task].input_view = input_views[it_task];
                request_handles[it_task].output_view = slave_work_items[it_task].output;

                jobs[it_task].callback = unpack_convolve_fixedpoint_callback_handle;
                jobs[it_task].request_handle = &request_handles[it_task];
            }

            // Wait for all sub threads.
            device->thread_pool.push_job(jobs);

            for (auto it_task = 0; it_task < task_count; ++it_task)
            {
                delete input_views[it_task];

                delete slave_work_items[it_task].output;

                auto &slave_arguments = slave_work_items[it_task].arguments.forward_convolution_fixedpoint;
                delete slave_arguments.biases;
                delete slave_arguments.weights;
            }
        }
    }
} // namepace device_int16
