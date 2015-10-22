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
#include "layer_convert_data_layout.h"
#include "layer_convert_data_to_batch_block_layout.h"
#include "layer_convert_data_from_batch_block_layout.h"
#include "device/cpu/api_internal/data_helper.h"

namespace
{
uint32_t C_simd_size = sizeof(__m256) / sizeof(float);
} //namespace

namespace layer
{

    template<
        bool T_fully_optimized,
        uint32_t T_batch = 0,
        uint32_t T_image_size = 0>
    void batching_conversion(
        const nn_workload_data_t* input_view,
        nn_workload_data_t* output_view)
    {
        if (T_fully_optimized)
        {
            const __m256i offset_reg = _mm256_set_epi32(
                                            T_image_size * 7,
                                            T_image_size * 6,
                                            T_image_size * 5,
                                            T_image_size * 4,
                                            T_image_size * 3,
                                            T_image_size * 2,
                                            T_image_size * 1,
                                            T_image_size * 0);

            float* in_data_ptr = reinterpret_cast<float*>(input_view->parent->data_buffer);
            float* out_data_ptr = reinterpret_cast<float*>(output_view->parent->data_buffer);

            if (T_batch == 8)
            {
                const uint32_t num_full_passes = T_image_size / 12;
                const uint32_t num_part_passes = T_image_size % 12;

                for (uint32_t pass = 0; pass < num_full_passes; ++pass)
                {
                    __m256 data_reg0  = _mm256_i32gather_ps(in_data_ptr +  0, offset_reg, 4);
                    __m256 data_reg1  = _mm256_i32gather_ps(in_data_ptr +  1, offset_reg, 4);
                    __m256 data_reg2  = _mm256_i32gather_ps(in_data_ptr +  2, offset_reg, 4);
                    __m256 data_reg3  = _mm256_i32gather_ps(in_data_ptr +  3, offset_reg, 4);
                    __m256 data_reg4  = _mm256_i32gather_ps(in_data_ptr +  4, offset_reg, 4);
                    __m256 data_reg5  = _mm256_i32gather_ps(in_data_ptr +  5, offset_reg, 4);
                    __m256 data_reg6  = _mm256_i32gather_ps(in_data_ptr +  6, offset_reg, 4);
                    __m256 data_reg7  = _mm256_i32gather_ps(in_data_ptr +  7, offset_reg, 4);
                    __m256 data_reg8  = _mm256_i32gather_ps(in_data_ptr +  8, offset_reg, 4);
                    __m256 data_reg9  = _mm256_i32gather_ps(in_data_ptr +  9, offset_reg, 4);
                    __m256 data_reg10 = _mm256_i32gather_ps(in_data_ptr + 10, offset_reg, 4);
                    __m256 data_reg11 = _mm256_i32gather_ps(in_data_ptr + 11, offset_reg, 4);

                    _mm256_store_ps(out_data_ptr + C_simd_size *  0,  data_reg0);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  1,  data_reg1);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  2,  data_reg2);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  3,  data_reg3);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  4,  data_reg4);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  5,  data_reg5);

                    _mm256_store_ps(out_data_ptr + C_simd_size *  6,  data_reg6);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  7,  data_reg7);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  8,  data_reg8);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  9,  data_reg9);
                    _mm256_store_ps(out_data_ptr + C_simd_size * 10, data_reg10);
                    _mm256_store_ps(out_data_ptr + C_simd_size * 11, data_reg11);

                    in_data_ptr += 12;
                    out_data_ptr += C_simd_size * 12;
                }

                for (uint32_t pass = 0; pass < num_part_passes; ++pass)
                {
                    __m256 data_reg0  = _mm256_i32gather_ps(in_data_ptr, offset_reg, 1);

                    _mm256_store_ps(out_data_ptr, data_reg0);

                    in_data_ptr += 1;
                    out_data_ptr += C_simd_size;
                }
            }
            else if (T_batch == 48)
            {
                const uint32_t num_full_passes = T_image_size / 2;
                const uint32_t num_part_passes = T_image_size % 2;

                for (uint32_t pass = 0; pass < num_full_passes; ++pass)
                {
                    __m256 data_reg0  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 0, offset_reg, 4);
                    __m256 data_reg1  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 1, offset_reg, 4);
                    __m256 data_reg2  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 2, offset_reg, 4);
                    __m256 data_reg3  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 3, offset_reg, 4);
                    __m256 data_reg4  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 4, offset_reg, 4);
                    __m256 data_reg5  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 5, offset_reg, 4);

                    __m256 data_reg6  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 0 + 1, offset_reg, 4);
                    __m256 data_reg7  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 1 + 1, offset_reg, 4);
                    __m256 data_reg8  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 2 + 1, offset_reg, 4);
                    __m256 data_reg9  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 3 + 1, offset_reg, 4);
                    __m256 data_reg10 = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 4 + 1, offset_reg, 4);
                    __m256 data_reg11 = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 5 + 1, offset_reg, 4);

                    _mm256_store_ps(out_data_ptr + C_simd_size *  0,  data_reg0);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  1,  data_reg1);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  2,  data_reg2);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  3,  data_reg3);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  4,  data_reg4);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  5,  data_reg5);

                    _mm256_store_ps(out_data_ptr + C_simd_size *  6,  data_reg6);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  7,  data_reg7);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  8,  data_reg8);
                    _mm256_store_ps(out_data_ptr + C_simd_size *  9,  data_reg9);
                    _mm256_store_ps(out_data_ptr + C_simd_size * 10, data_reg10);
                    _mm256_store_ps(out_data_ptr + C_simd_size * 11, data_reg11);

                    in_data_ptr += 2;
                    out_data_ptr += C_simd_size * 12;
                }

                for (uint32_t pass = 0; pass < num_part_passes; ++pass)
                {
                    __m256 data_reg0  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 0, offset_reg, 4);
                    __m256 data_reg1  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 1, offset_reg, 4);
                    __m256 data_reg2  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 2, offset_reg, 4);
                    __m256 data_reg3  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 3, offset_reg, 4);
                    __m256 data_reg4  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 4, offset_reg, 4);
                    __m256 data_reg5  = _mm256_i32gather_ps(in_data_ptr + T_image_size * C_simd_size * 5, offset_reg, 4);

                    _mm256_store_ps(out_data_ptr + C_simd_size * 0, data_reg0);
                    _mm256_store_ps(out_data_ptr + C_simd_size * 1, data_reg1);
                    _mm256_store_ps(out_data_ptr + C_simd_size * 2, data_reg2);
                    _mm256_store_ps(out_data_ptr + C_simd_size * 3, data_reg3);
                    _mm256_store_ps(out_data_ptr + C_simd_size * 4, data_reg4);
                    _mm256_store_ps(out_data_ptr + C_simd_size * 5, data_reg5);
                }
            }
        }
        else
        {
            const auto input_buffer = static_cast<float*>(input_view->parent->data_buffer);
            const auto output_buffer = static_cast<float*>(output_view->parent->data_buffer);

            const auto input_squashed_x_length = input_view->parent->buffer_size / sizeof(float) / input_view->parent->lengths.t[NN_DATA_COORD_n];
            const auto output_squashed_x_length = output_view->parent->buffer_size / sizeof(float) / output_view->parent->lengths.t[NN_DATA_COORD_n];

            if(input_squashed_x_length != output_squashed_x_length)
                throw std::runtime_error("batching conversion: different data sizes");

            if((input_view->parent->layout == nn::layout_t<nn::layout_nzxypq_f32>::layout && output_view->parent->layout == nn::layout_t<nn::layout_zxynpq_f32>::layout) || // N-ZXY -> ZXY-N
               (input_view->parent->layout == nn::layout_t<nn::layout_nxyzpq_f32>::layout && output_view->parent->layout == nn::layout_t<nn::layout_zxynpq_f32>::layout) || // N-X   -> ZXY-N
               (input_view->parent->layout == nn::layout_t<nn::layout_nxyzpq_f32>::layout && output_view->parent->layout == nn::layout_t<nn::layout_xyzpqn_f32>::layout))   // N-XYZ -> XYZ-N
            {
                // N.. to ..N
                const auto load_data_stride = T_batch;
                const auto store_data_stride = output_squashed_x_length;

                for(uint32_t x = 0; x < input_squashed_x_length; ++x)
                {
                    #pragma unroll (T_batch)
                    for(uint32_t n = 0; n < T_batch; ++n)
                        *(output_buffer + n * store_data_stride + x) = *(input_buffer + x * load_data_stride + n);
                }
            }
            else if((input_view->parent->layout == nn::layout_t<nn::layout_zxynpq_f32>::layout && output_view->parent->layout == nn::layout_t<nn::layout_nzxypq_f32>::layout) || // ZXY-N -> N-ZXY
                    (input_view->parent->layout == nn::layout_t<nn::layout_zxynpq_f32>::layout && output_view->parent->layout == nn::layout_t<nn::layout_nxyzpq_f32>::layout) || // ZXY-N -> N-X
                    (input_view->parent->layout == nn::layout_t<nn::layout_xyzpqn_f32>::layout && output_view->parent->layout == nn::layout_t<nn::layout_nxyzpq_f32>::layout))   // XYZ-N -> N-XYZ
            {
                // ..N to N..
                const auto load_data_stride = input_squashed_x_length;
                const auto store_data_stride = T_batch;

                for(uint32_t x = 0; x < input_squashed_x_length; ++x)
                {
                    #pragma unroll (T_batch)
                    for(uint32_t n = 0; n < T_batch; ++n)
                        *(output_buffer + x * store_data_stride + n) = *(input_buffer + n * load_data_stride + x);
                }
            }
        }
    }

    void run_convert_to_data_layout_work_item(nn_workload_item *const work_item) {
        const auto &master_arguments = work_item->arguments.convert_data_layout;
        const auto &input_view = work_item->input[0].get_data_view();
        const auto &output_view = work_item->output[0];
        const auto &type = master_arguments.type;

        auto batchsize = output_view->parent->lengths.t[NN_DATA_COORD_n];
        auto size = output_view->parent->lengths.t[NN_DATA_COORD_z];
        auto size2 = output_view->parent->lengths.t[NN_DATA_COORD_p];

        switch (type) {
        case 0: // non-batched convolution -> fully connected
        {
            auto primitive = static_cast<layer::convert_z_block_xyz_z2nz *>(work_item->primitive);
            primitive->forward(reinterpret_cast<nn::workload_data<int16_t> *>(input_view),
                reinterpret_cast<nn::workload_data<int16_t> *>(output_view));

            //const auto input_buffer = static_cast<int16_t *>(input_view->parent->data_buffer);
            //auto output_buffer = static_cast<int16_t *>(output_view->parent->data_buffer);

            //const size_t width = input_view->parent->lengths.t[NN_DATA_COORD_x];
            //const size_t height = input_view->parent->lengths.t[NN_DATA_COORD_y];
            //const size_t fm_sub_block = output_view->parent->lengths.t[NN_DATA_COORD_p];
            //const size_t fm_block = input_view->parent->lengths.t[NN_DATA_COORD_p];
            //const size_t fm_count = input_view->parent->lengths.t[NN_DATA_COORD_p] * input_view->parent->lengths.t[NN_DATA_COORD_z];

            //for (size_t it_fm_block_number = 0; it_fm_block_number < fm_count / fm_block; ++it_fm_block_number)
            //    for (size_t it_fm_in_block = 0; it_fm_in_block < fm_block; ++it_fm_in_block)
            //        for (size_t it_xy = 0; it_xy < width * height / fm_sub_block; ++it_xy)
            //                for (size_t it_fm_in_sub_block = 0; it_fm_in_sub_block < fm_sub_block; ++it_fm_in_sub_block) {
            //                    output_buffer[it_fm_in_sub_block +
            //                                  fm_sub_block * (it_xy +
            //                                                  (width * height / fm_sub_block) *
            //                                                      (it_fm_in_block + it_fm_block_number * fm_block))] =
            //                        input_buffer[it_fm_in_sub_block * fm_block + it_xy * fm_sub_block * fm_block +
            //                                     it_fm_in_block + it_fm_block_number * fm_block * width * height];
            //                }
        } break;

        case 2: // fully connected -> softmax
        {
            const auto input_buffer = static_cast<int32_t *>(input_view->parent->data_buffer);
            auto output_buffer = static_cast<int32_t *>(output_view->parent->data_buffer);

            for (auto itrBatch = 0u; itrBatch < batchsize / 8; ++itrBatch)
                for (auto itrSize = 0u; itrSize < size; ++itrSize)
                    for (auto itrBatch8 = 0; itrBatch8 < 8; ++itrBatch8)
                        for (auto itrSize2 = 0u; itrSize2 < size2; ++itrSize2)
                            output_buffer[itrSize2 * 8 + itrBatch8 + itrSize * size2 * 8 +
                                          itrBatch * size * 8 * size2] =
                                input_buffer[itrSize2 + itrBatch8 * size2 + itrSize * size2 * batchsize +
                                             itrBatch * 8 * size2];
        } break;

        case 3: // convolution -> fully connected
        {
            auto primitive = static_cast<layer::convert_z_block_xyz_z2nz *>(work_item->primitive);
            primitive->forward(reinterpret_cast<nn::workload_data<int16_t> *>(input_view),
                reinterpret_cast<nn::workload_data<int16_t> *>(output_view));

            /*const auto input_buffer = static_cast<int16_t *>(input_view->parent->data_buffer);
            auto output_buffer = static_cast<int16_t *>(output_view->parent->data_buffer);

            const size_t in_x_size = input_view->parent->lengths.t[NN_DATA_COORD_x];
            const size_t in_y_size = input_view->parent->lengths.t[NN_DATA_COORD_y];
            const size_t out_z_block_size = output_view->parent->lengths.t[NN_DATA_COORD_p];
            const size_t in_z_block_size = input_view->parent->lengths.t[NN_DATA_COORD_p],
                         in_z_block_count = input_view->parent->lengths.t[NN_DATA_COORD_z];
            const size_t z_total_size = in_x_size * in_y_size * in_z_block_count * in_z_block_size;
            assert(output_view->parent->lengths.t[NN_DATA_COORD_z] == (z_total_size - 1) / out_z_block_size + 1);
            assert(input_view->parent->lengths.t[NN_DATA_COORD_n] == batchsize);
            assert(out_z_block_size < in_x_size && in_x_size % out_z_block_size == 0);

            const size_t in_x_stride = in_z_block_size,
                         in_y_stride = in_x_stride * in_x_size,
                         in_z_block_stride = in_y_stride * in_y_size,
                         in_batch_stride = in_z_block_stride * in_z_block_count;

            const size_t out_batch_stride = out_z_block_size,
                         out_z_block_stride = out_batch_stride * batchsize;

            auto dst = output_buffer;
            for (size_t it_z_block = 0; it_z_block < in_z_block_count; ++it_z_block)
                for (size_t it_z_in_block = 0; it_z_in_block < in_z_block_size; ++it_z_in_block)
                    for (size_t it_in_y = 0; it_in_y < in_y_size; ++it_in_y)
                        for (size_t it_in_x = 0; it_in_x < in_x_size; it_in_x += out_z_block_size)
                            for (size_t it_batch = 0; it_batch < batchsize; ++it_batch)
                                for (size_t it_in_out_z_block = 0; it_in_out_z_block < out_z_block_size; ++it_in_out_z_block)
                                {
                                    *(dst++) =
                                        input_buffer[it_batch * in_batch_stride
                                                    + it_z_block * in_z_block_stride
                                                    + it_in_y * in_y_stride
                                                    + (it_in_x + it_in_out_z_block) * in_x_stride
                                                    + it_z_in_block];
                                }*/
        } break;

        case 4: // conv->fc in float batch8/48
        case 7: // fc->conv in backward batch8/48
        {
            // It won't change anything if batch == 1.
            if(batchsize != 1)
            {
                auto image_size = input_view->parent->lengths.t[NN_DATA_COORD_x] * input_view->parent->lengths.t[NN_DATA_COORD_y] * input_view->parent->lengths.t[NN_DATA_COORD_z];

                if      (batchsize ==  8 && image_size ==  9216 && type == 4) batching_conversion< true,  8,  9216>(input_view, output_view);
                else if (batchsize == 48 && image_size ==  9216 && type == 4) batching_conversion< true, 48,  9216>(input_view, output_view);
                else if (batchsize ==  8 && image_size == 36864 && type == 4) batching_conversion< true,  8, 36864>(input_view, output_view);
                else if (batchsize == 48 && image_size == 36864 && type == 4) batching_conversion< true, 48, 36864>(input_view, output_view);
                else if (batchsize ==  8)                                     batching_conversion<false,  8       >(input_view, output_view);
                else if (batchsize == 32)                                     batching_conversion<false, 32       >(input_view, output_view);
                else if (batchsize == 48)                                     batching_conversion<false, 48       >(input_view, output_view);
                else
                    throw std::runtime_error("batch_conversion: unknown batch");
            }
            else
            {
                // It should be view-style mapping.
                if(input_view->parent->data_buffer != output_view->parent->data_buffer)
                {
                    // It can be possible if INPUT is before conversion and its buffer changed. In such case
                    // we'll copy data_buffer ptr. It's also only case that should allow going into this IF.
                    if(work_item->input[0].item->type == NN_WORK_ITEM_TYPE_INPUT)
                    {
                        output_view->parent->data_buffer = input_view->parent->data_buffer;
                    }
                    else
                        throw std::runtime_error("batch1_conversion: invalid view mapping");
                }
            }
            break;
        }

        case 5: // input zxyn -> z<block>xyzn for int16 convolutions
        {
            const auto input_data = static_cast<nn::workload_data<int16_t>*>(input_view);
            auto output_data = reinterpret_cast<nn::workload_data<int16_t>*>(output_view);
            const uint32_t z_block = static_cast<uint32_t>(output_view->parent->lengths.t[NN_DATA_COORD_p]);
            assert(z_block == 16 || z_block == 4);
            const uint32_t z_block_count = static_cast<uint32_t>(output_view->parent->lengths.t[NN_DATA_COORD_z]);
            const uint32_t x_size        = static_cast<uint32_t>(output_view->parent->lengths.t[NN_DATA_COORD_x]);
            const uint32_t y_size        = static_cast<uint32_t>(output_view->parent->lengths.t[NN_DATA_COORD_y]);
            const uint32_t batch_size    = static_cast<uint32_t>(output_view->parent->lengths.t[NN_DATA_COORD_n]);

            for (uint32_t it_batch = 0; it_batch < batch_size; ++it_batch)
                for (uint32_t it_z_block = 0; it_z_block < z_block_count; ++it_z_block)
                    for (uint32_t it_y = 0; it_y < y_size; ++it_y)
                        for (uint32_t it_x = 0; it_x < x_size; ++it_x)
                            for (uint32_t z_in_block = 0; z_in_block < z_block; z_in_block++)
                                output_data->at(it_batch, it_x, it_y, it_z_block, z_in_block, 0) = input_data->at(it_batch, it_x, it_y, it_z_block * z_block + z_in_block, 0, 0);
            break;
        }

        case 6: // z<block>xyzn -> zxyn
        {
            const auto input_data = static_cast<nn::workload_data<int16_t>*>(input_view);
            auto output_data = reinterpret_cast<nn::workload_data<int16_t>*>(output_view);

            auto input_lenght = input_data->get_length();
            const uint32_t z_block = input_lenght.t[NN_DATA_COORD_p];
            const uint32_t z_block_count = input_lenght.t[NN_DATA_COORD_z];
            const uint32_t x_size = input_lenght.t[NN_DATA_COORD_x];
            const uint32_t y_size = input_lenght.t[NN_DATA_COORD_y];
            const uint32_t batch_size = input_lenght.t[NN_DATA_COORD_n];

            for (uint32_t it_batch = 0; it_batch < batch_size; ++it_batch)
                for (uint32_t it_z_block = 0; it_z_block < z_block_count; ++it_z_block)
                    for (uint32_t it_y = 0; it_y < y_size; ++it_y)
                        for (uint32_t it_x = 0; it_x < x_size; ++it_x)
                            for (uint32_t z_in_block = 0; z_in_block < z_block; z_in_block++)
                                (*output_data)(it_batch, it_x, it_y, it_z_block * z_block + z_in_block, 0, 0)= (*input_data)(it_batch, it_x, it_y, it_z_block, z_in_block, 0);
            break;
        }
        }
    }

    bool convert_zxyn_nx_f32::validate_input(size_t index, nn_workload_data_t *data) {
        switch (index) {
        case 0:
            return nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::validate<false>(
                data,
                static_cast<uint32_t>(input_size_x),
                static_cast<uint32_t>(input_size_y),
                static_cast<uint32_t>(input_size_z),
                static_cast<uint32_t>(batch_size),
                0,
                0,
                0,
                0);
        }

        throw std::invalid_argument("index out of range");
    }

    void convert_zxyn_nx_f32::forward(const nn::workload_data<> *input, nn::workload_data<> *output)
    {
        if(batch_size == 1)
            // no batching, interleaving batches does no change to the data layout
            memcpy(output->parent->data_buffer, input->parent->data_buffer, output->parent->buffer_size);
        else if (batch_size == 8 && output_size == 9216)
            batching_conversion<true, 8, 9216>(input, output);
        else if (batch_size == 48 && output_size == 9216)
            batching_conversion<true, 48, 9216>(input, output);
        else if (batch_size == 8 && output_size == 36864)
            batching_conversion<true, 8, 36864>(input, output);
        else if (batch_size == 48 && output_size == 36864)
            batching_conversion<true, 48, 36864>(input, output);
        else{
            // reinterpret output as nzxy to interleave batches and produce single dimensional output correctly
            nn_workload_data_layout_t nzxy_layout = nn::layout_t<nn::layout_nzxypq_f32>::layout;
            nn::workload_data<nn::layout_f32> tmp_output(
                NN_WORKLOAD_DATA_TAG_UNKNOWN, output->parent->data_buffer, input->parent->lengths, nzxy_layout);
            batching_conversion<false>(input, &tmp_output);
        }
    }

    void convert_zxyn_nx_f32::forward(const std::vector<const nn_workload_data_t *> &inputs,
                                      const std::vector<const nn_workload_data_t *> &parameters,
                                      const std::vector<nn_workload_data_t *> &outputs) {
        assert(inputs.size() == 1);
        assert(parameters.size() == 0);
        assert(outputs.size() == 1);

        forward(nn::workload_data_cast<>(inputs[0]),
                nn::workload_data_cast<>(outputs[0]));
    }

    void convert_zxyn_nx_f32::backward(
        const std::vector<nn_workload_data_t *> &inputs,
        const std::vector<const nn_workload_data_t *> &parameters,
        const std::vector<const nn_workload_data_t *> &outputs)
    {
        assert(inputs.size() == 1);
        assert(parameters.size() == 0);
        assert(outputs.size() == 1);
        
        auto input_data =  nn::workload_data_cast<>(outputs[0]);
        auto output_data = nn::workload_data_cast<>(inputs[0]);

        nn::workload_data<> input(
            input_data->parent->delta_buffer,
            input_data->parent->lengths, input_data->parent->layout);

        nn::workload_data<> output(
            output_data->parent->delta_buffer,
            output_data->parent->lengths, output_data->parent->layout);

        if (batch_size == 1)
            // no batching, interleaving batches does no change to the data layout
            memcpy(output.parent->data_buffer, input.parent->data_buffer, output.parent->buffer_size);
        else if (batch_size == 8)
            batching_conversion<false,  8>(&input, &output);
        else if (batch_size == 32)
            batching_conversion<false, 32>(&input, &output);
        else if (batch_size == 48)
            batching_conversion<false, 48>(&input, &output);
        else{
            throw std::runtime_error("batch size not supported");
        }
        
    }

    std::vector<nn_workload_data_t *> convert_zxyn_nx_f32::create_inputs(bool allocate_delta) {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, nn::layout_zxyn_f32>::create(
                                                                          device,
                                                                          static_cast<uint32_t>(input_size_x),
                                                                          static_cast<uint32_t>(input_size_y),
                                                                          static_cast<uint32_t>(input_size_z),
                                                                          static_cast<uint32_t>(batch_size),
                                                                          0,
                                                                          0,
                                                                          0,
                                                                          0,
                                                                          allocate_delta)};
    }

    std::vector<nn_workload_data_t *> convert_zxyn_nx_f32::create_outputs(bool allocate_delta) {
        return {nn::data_helper<NN_WORKLOAD_DATA_TAG_NX, nn::layout_nx_f32>::create(
                                                                        device,
                                                                        static_cast<uint32_t>(output_size),
                                                                        static_cast<uint32_t>(batch_size),
                                                                        allocate_delta
                                                                        )};
    }

    convert_zxyn_nx_f32::convert_zxyn_nx_f32(
        size_t input_size_x, size_t input_size_y, size_t input_size_z, size_t batch_size, nn_device_internal *device)
        : input_size_x(input_size_x),
          input_size_y(input_size_y),
          input_size_z(input_size_z),
          batch_size(batch_size),
          device(device),
          output_size(input_size_x * input_size_y * input_size_z) {}

    bool convert_z_block_xyz_z2nz::validate_input(size_t index, nn_workload_data_t *data)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    void convert_z_block_xyz_z2nz::forward(const nn::workload_data<int16_t> *input_view, nn::workload_data<int16_t> *output_view)
    {
        // validate input

        if (batch_size == 1)
        {
            const auto input_buffer = static_cast<int16_t *>(input_view->parent->data_buffer);
            auto output_buffer = static_cast<int16_t *>(output_view->parent->data_buffer);

            const size_t width = input_view->parent->lengths.t[NN_DATA_COORD_x];
            const size_t height = input_view->parent->lengths.t[NN_DATA_COORD_y];
            const size_t fm_sub_block = output_view->parent->lengths.t[NN_DATA_COORD_p];
            const size_t fm_block = input_view->parent->lengths.t[NN_DATA_COORD_p];
            const size_t fm_count = input_view->parent->lengths.t[NN_DATA_COORD_p] * input_view->parent->lengths.t[NN_DATA_COORD_z];

            for (size_t it_fm_block_number = 0; it_fm_block_number < fm_count / fm_block; ++it_fm_block_number)
            for (size_t it_fm_in_block = 0; it_fm_in_block < fm_block; ++it_fm_in_block)
            for (size_t it_xy = 0; it_xy < width * height / fm_sub_block; ++it_xy)
            for (size_t it_fm_in_sub_block = 0; it_fm_in_sub_block < fm_sub_block; ++it_fm_in_sub_block) {
                output_buffer[it_fm_in_sub_block +
                    fm_sub_block * (it_xy +
                    (width * height / fm_sub_block) *
                    (it_fm_in_block + it_fm_block_number * fm_block))] =
                    input_buffer[it_fm_in_sub_block * fm_block + it_xy * fm_sub_block * fm_block +
                    it_fm_in_block + it_fm_block_number * fm_block * width * height];
            }
        }
        else
        {
            const auto input_buffer = static_cast<int16_t *>(input_view->parent->data_buffer);
            auto output_buffer = static_cast<int16_t *>(output_view->parent->data_buffer);

            const size_t in_x_size = input_view->parent->lengths.t[NN_DATA_COORD_x];
            const size_t in_y_size = input_view->parent->lengths.t[NN_DATA_COORD_y];
            const size_t out_z_block_size = output_view->parent->lengths.t[NN_DATA_COORD_p];
            const size_t in_z_block_size = input_view->parent->lengths.t[NN_DATA_COORD_p],
                in_z_block_count = input_view->parent->lengths.t[NN_DATA_COORD_z];
            const size_t z_total_size = in_x_size * in_y_size * in_z_block_count * in_z_block_size;
            assert(output_view->parent->lengths.t[NN_DATA_COORD_z] == (z_total_size - 1) / out_z_block_size + 1);
            assert(input_view->parent->lengths.t[NN_DATA_COORD_n] == batch_size);
            assert(out_z_block_size < in_x_size && in_x_size % out_z_block_size == 0);

            const size_t in_x_stride = in_z_block_size,
                in_y_stride = in_x_stride * in_x_size,
                in_z_block_stride = in_y_stride * in_y_size,
                in_batch_stride = in_z_block_stride * in_z_block_count;

            const size_t out_batch_stride = out_z_block_size,
                out_z_block_stride = out_batch_stride * batch_size;

            auto dst = output_buffer;
            for (size_t it_z_block = 0; it_z_block < in_z_block_count; ++it_z_block)
            for (size_t it_z_in_block = 0; it_z_in_block < in_z_block_size; ++it_z_in_block)
            for (size_t it_in_y = 0; it_in_y < in_y_size; ++it_in_y)
            for (size_t it_in_x = 0; it_in_x < in_x_size; it_in_x += out_z_block_size)
            for (size_t it_batch = 0; it_batch < batch_size; ++it_batch)
            for (size_t it_in_out_z_block = 0; it_in_out_z_block < out_z_block_size; ++it_in_out_z_block)
            {
                *(dst++) =
                    input_buffer[it_batch * in_batch_stride
                    + it_z_block * in_z_block_stride
                    + it_in_y * in_y_stride
                    + (it_in_x + it_in_out_z_block) * in_x_stride
                    + it_z_in_block];
            }
        }
    }

    void convert_z_block_xyz_z2nz::forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs)
    {
        assert(inputs.size() == 1);
        assert(outputs.size() == 1);

        forward(reinterpret_cast<const nn::workload_data<int16_t> *>(inputs[0]),
            reinterpret_cast<nn::workload_data<int16_t> *>(outputs[0]));
    }

    convert_z_block_xyz_z2nz::convert_z_block_xyz_z2nz(
        size_t input_size_x, size_t input_size_y, size_t input_size_z, size_t batch_size, nn_device_internal *device)
        : input_size_x(input_size_x),
        input_size_y(input_size_y),
        input_size_z(input_size_z),
        batch_size(batch_size),
        device(device),
        output_size(input_size_x * input_size_y * input_size_z) {}

    std::vector<nn_workload_data_t *> convert_z_block_xyz_z2nz::create_outputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_X2NX, nn::layout_x2nx_f32>::create(device,
                                                                          static_cast<uint32_t>(output_size),
                                                                          static_cast<uint32_t>(batch_size),
                                                                          out_block_size) };
    }

    std::vector<nn_workload_data_t *> convert_z_block_xyz_z2nz::create_inputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, nn::layout_zblockxyzn_f32>::create(
                                                                                device,
                                                                                static_cast<uint32_t>(input_size_x),
                                                                                static_cast<uint32_t>(input_size_y),
                                                                                static_cast<uint32_t>(input_size_z),
                                                                                static_cast<uint32_t>(batch_size),
                                                                                static_cast<uint32_t>(in_block_size),
                                                                                0,
                                                                                0,
                                                                                0,
                                                                                0) };
    }

    bool convert_z2nz_n8xn::validate_input(size_t index, nn_workload_data_t *data)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    void convert_z2nz_n8xn::forward(const nn::workload_data<int32_t> *input_view, nn::workload_data<int32_t> *output_view)
    {
        auto batchsize = output_view->parent->lengths.t[NN_DATA_COORD_n];
        auto size_z = output_view->parent->lengths.t[NN_DATA_COORD_z];
        auto size_p = output_view->parent->lengths.t[NN_DATA_COORD_p];

        // fully connected -> softmax
        // zblocknz -> n8zn
        const auto input_buffer = static_cast<int32_t *>(input_view->parent->data_buffer);
        auto output_buffer = static_cast<int32_t *>(output_view->parent->data_buffer);

        for (auto n = 0u; n < batchsize / 8; ++n)
        for (auto z = 0u; z < size_z; ++z)
        for (auto i = 0; i < 8; ++i)
        for (auto p = 0u; p < size_p; ++p)
            output_buffer[i + 8 * (p + size_p *(z + n * size_z))] =
            input_buffer[p + size_p * ((i + n * 8) + z * batchsize)];
    }

    void convert_z2nz_n8xn::forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs)
    {
        assert(inputs.size() == 1);
        assert(outputs.size() == 1);

        assert(inputs[0]->parent->layout.data_type == NN_DATATYPE_INT32);
        assert(outputs[0]->parent->layout.data_type == NN_DATATYPE_INT32);

        forward(reinterpret_cast<const nn::workload_data<int32_t> *>(inputs[0]),
            reinterpret_cast<nn::workload_data<int32_t> *>(outputs[0]));
    }

    convert_z2nz_n8xn::convert_z2nz_n8xn(
        size_t input_size_x, size_t batch_size, nn_device_internal *device)
        :
        input_size_x(input_size_x),
        batch_size(batch_size),
        device(device),
        output_size(input_size_x) {}

    std::vector<nn_workload_data_t *> convert_z2nz_n8xn::create_outputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_X2NX, int32_t>::create(
                                                                            device,
                                                                            static_cast<uint32_t>(output_size),
                                                                            static_cast<uint32_t>(batch_size),
                                                                            static_cast<uint32_t>(in_block_size)) };
    }

    std::vector<nn_workload_data_t *> convert_z2nz_n8xn::create_inputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_X2NX, int32_t>::create(
                                                                            device,
                                                                            static_cast<uint32_t>(output_size),
                                                                            static_cast<uint32_t>(batch_size),
                                                                            static_cast<uint32_t>(in_block_size)) };
    }

} //namespace layer

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_convert_zxyn_nx_f32_create_0(
    nn_device_t *device, /* IDLF device handle */
    size_t input_size_x,
    size_t input_size_y,
    size_t input_size_z,
    size_t batch_size,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);
    return new layer::convert_zxyn_nx_f32(
        input_size_x, input_size_y, input_size_z, batch_size, reinterpret_cast<nn_device_internal *>(device));
}

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_convert_z_block_xyz_x2nx_i16_create_0(
    nn_device_t *device, /* IDLF device handle */
    size_t input_size_x,
    size_t input_size_y,
    size_t input_size_z,
    size_t batch_size,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);
    return new layer::convert_z_block_xyz_z2nz(
        input_size_x, input_size_y, input_size_z, batch_size, reinterpret_cast<nn_device_internal *>(device));
}

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_convert_z2nz_n8xn_create_0(
    nn_device_t *device, /* IDLF device handle */
    size_t input_size_z,
    size_t batch_size,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);
    return new layer::convert_z2nz_n8xn(
        input_size_z, batch_size, reinterpret_cast<nn_device_internal *>(device));
}

nn_primitive_handle_t NN_API_CALL_CONVENTION
 nn_primitives_convert_from_zxyn_to_batch_block_format_nzxyn_create(
     nn_device_t *device,
     size_t input_size_x,
     size_t input_size_y,
     size_t input_size_z,
     size_t batch_size)
{
    return new layer::convert_from_zxyn_to_batch_block_format_nzxyn(
        batch_size, input_size_x, input_size_y, input_size_z, static_cast<nn_device_internal*>(device));
}

nn_primitive_handle_t NN_API_CALL_CONVENTION
 nn_primitives_convert_from_batch_block_format_to_zxyn_create(
     nn_device_t *device,
     size_t input_size_x,
     size_t input_size_y,
     size_t input_size_z,
     size_t batch_size)
{
    return new layer::convert_from_batch_block_format_to_zxyn(
        batch_size, input_size_x, input_size_y, input_size_z, static_cast<nn_device_internal*>(device));
}

