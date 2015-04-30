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
#include "layer_convert_data_layout.h"

uint32_t C_simd_size = sizeof(__m256) / sizeof(float);

namespace layer{

    template<
        bool T_optimized,
        uint32_t T_batch = 0,
        uint32_t T_image_size = 0>
    void batching_conversion(
        const nn_workload_data_t* input_view, 
        nn_workload_data_t* output_view)
    {
        if (T_optimized)
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
            nn_workload_data_copy(output_view, input_view);
        }
    }

    void run_convert_to_data_layout_work_item(nn_workload_item *const work_item) {
        const auto &master_arguments = work_item->arguments.convert_data_layout;
        const auto &input_view = work_item->input[0]->output;
        const auto &output_view = work_item->output;
        const auto &type = master_arguments.type;

        auto batchsize = output_view->parent->lengths.t[NN_DATA_COORD_n];
        auto size = output_view->parent->lengths.t[NN_DATA_COORD_z];
        auto size2 = output_view->parent->lengths.t[NN_DATA_COORD_p];

        switch (type) {
        case 0: // non-batched convolution -> fully connected
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
        } break;

        case 2: // fully connected -> softmax
        {
            const auto input_buffer = static_cast<int32_t *>(input_view->parent->data_buffer);
            auto output_buffer = static_cast<int32_t *>(output_view->parent->data_buffer);

            for (auto itrBatch = 0; itrBatch < batchsize / 8; ++itrBatch)
                for (auto itrSize = 0; itrSize < size; ++itrSize)
                    for (auto itrBatch8 = 0; itrBatch8 < 8; ++itrBatch8)
                        for (auto itrSize2 = 0; itrSize2 < size2; ++itrSize2)
                            output_buffer[itrSize2 * 8 + itrBatch8 + itrSize * size2 * 8 +
                                          itrBatch * size * 8 * size2] =
                                input_buffer[itrSize2 + itrBatch8 * size2 + itrSize * size2 * batchsize +
                                             itrBatch * 8 * size2];
        } break;

        case 3: // convolution -> fully connected
        {
            const auto input_buffer = static_cast<int16_t *>(input_view->parent->data_buffer);
            auto output_buffer = static_cast<int16_t *>(output_view->parent->data_buffer);

            const size_t in_x_size = input_view->parent->lengths.t[NN_DATA_COORD_x];
            const size_t in_y_size = input_view->parent->lengths.t[NN_DATA_COORD_y];
            const size_t out_z_block_size = output_view->parent->lengths.t[NN_DATA_COORD_p];
            const size_t in_z_block_size = input_view->parent->lengths.t[NN_DATA_COORD_p],
                         in_z_block_count = input_view->parent->lengths.t[NN_DATA_COORD_z];
            const size_t z_total_size = in_x_size * in_y_size * input_view->parent->lengths.t[NN_DATA_COORD_z] * in_z_block_size;
            assert(output_view->parent->lengths.t[NN_DATA_COORD_z] == (z_total_size - 1) / out_z_block_size + 1);
            assert(input_view->parent->lengths.t[NN_DATA_COORD_n] == batchsize);
            assert(out_z_block_size < in_z_block_size && in_z_block_size % out_z_block_size == 0);

            const size_t in_x_stride = in_z_block_size,
                         in_y_stride = in_x_stride * in_x_size,
                         in_z_block_stride = in_y_stride * in_y_size,
                         in_batch_stride = in_z_block_stride * in_z_block_count;

            const size_t out_batch_stride = out_z_block_size,
                         out_z_block_stride = out_batch_stride * batchsize;

            for(size_t it_in_z_block = 0; it_in_z_block < in_z_block_count; ++it_in_z_block)
                for(size_t it_in_y = 0; it_in_y < in_y_size; ++it_in_y)
                    for(size_t it_in_x = 0; it_in_x < in_x_size; ++it_in_x)
                        for (size_t it_batch = 0; it_batch < batchsize; ++it_batch)
                            for (size_t it_in_z_in_block = 0; it_in_z_in_block < in_z_block_size; ++it_in_z_in_block){
                                const size_t out_z = it_in_x
                                                        + it_in_y * in_x_size
                                                        + (it_in_z_block * in_z_block_size + it_in_z_in_block) * in_x_size * in_y_size;
                                const size_t out_z_block = out_z / out_z_block_size;
                                const size_t out_z_in_block = out_z % out_z_block_size;
                                output_buffer[out_z_block * out_z_block_stride + it_batch * out_batch_stride +
                                              out_z_in_block] =
                                    input_buffer[it_batch * in_batch_stride + it_in_z_block * in_z_block_stride +
                                                 it_in_y * in_y_stride + it_in_x * in_x_stride + it_in_z_in_block];
                                }
        } break;

        case 4: // conv->fc in float batch8/48
        {
            auto image_size = input_view->parent->lengths.t[NN_DATA_COORD_x] * input_view->parent->lengths.t[NN_DATA_COORD_y] * input_view->parent->lengths.t[NN_DATA_COORD_z];

            if      (batchsize ==  8 && image_size ==  9216) batching_conversion<true,  8,  9216>(input_view, output_view);
            else if (batchsize == 48 && image_size ==  9216) batching_conversion<true, 48,  9216>(input_view, output_view);
            else if (batchsize ==  8 && image_size == 36864) batching_conversion<true,  8, 36864>(input_view, output_view);
            else if (batchsize == 48 && image_size == 36864) batching_conversion<true, 48, 36864>(input_view, output_view);
            else batching_conversion<false>(input_view, output_view);

            break;
        }

        case 5: // input zxyn -> z<block>xyzn for int16 convolutions
        {
            const auto input_data = static_cast<nn::nn_workload_data_t<int16_t>*>(input_view);
            auto output_data = reinterpret_cast<nn::nn_workload_data_t<int16_t>*>(output_view);
            const size_t z_block = output_view->parent->lengths.t[NN_DATA_COORD_p];
            assert(z_block == 8 || z_block == 4);
            const size_t z_block_count = output_view->parent->lengths.t[NN_DATA_COORD_z];
            const size_t x_size = output_view->parent->lengths.t[NN_DATA_COORD_x];
            const size_t y_size = output_view->parent->lengths.t[NN_DATA_COORD_y];
            const size_t batch_size = output_view->parent->lengths.t[NN_DATA_COORD_n];

            for (size_t it_batch = 0; it_batch < batch_size; ++it_batch)
                for (size_t it_z_block = 0; it_z_block < z_block_count; ++it_z_block)
                    for (size_t it_y = 0; it_y < y_size; ++it_y)
                        for (size_t it_x = 0; it_x < x_size; ++it_x)
                            for (size_t z_in_block = 0; z_in_block < z_block; z_in_block++)
                                (*output_data)(it_batch, it_x, it_y, it_z_block, z_in_block, 0) = (*input_data)(it_batch, it_x, it_y, it_z_block * z_block + z_in_block, 0, 0)();
            break;
        }

        case 6: // z<block>xyzn -> zxyn
        {
            const auto input_data = static_cast<nn::nn_workload_data_t<int16_t>*>(input_view);
            auto output_data = reinterpret_cast<nn::nn_workload_data_t<int16_t>*>(output_view);

            auto input_lenght = input_data->get_length();
            const size_t z_block = input_lenght.t[NN_DATA_COORD_p];
            const size_t z_block_count = input_lenght.t[NN_DATA_COORD_z];
            const size_t x_size = input_lenght.t[NN_DATA_COORD_x];
            const size_t y_size = input_lenght.t[NN_DATA_COORD_y];
            const size_t batch_size = input_lenght.t[NN_DATA_COORD_n];

            for (size_t it_batch = 0; it_batch < batch_size; ++it_batch)
                for (size_t it_z_block = 0; it_z_block < z_block_count; ++it_z_block)
                    for (size_t it_y = 0; it_y < y_size; ++it_y)
                        for (size_t it_x = 0; it_x < x_size; ++it_x)
                            for (size_t z_in_block = 0; z_in_block < z_block; z_in_block++)
                                (*output_data)(it_batch, it_x, it_y, it_z_block * z_block + z_in_block, 0, 0)= (*input_data)(it_batch, it_x, it_y, it_z_block, z_in_block, 0);
            break;
        }
        }
    }

    convert_zxyn_nx_f32 *convert_zxyn_nx_f32::create(
        size_t input_size_x, size_t input_size_y, size_t input_size_z, size_t batch_size, nn_device_t *device) {
        return new convert_zxyn_nx_f32(
            input_size_x, input_size_y, input_size_z, batch_size, reinterpret_cast<nn_device_internal *>(device));
    }

    nn::nn_workload_data_t<float> *convert_zxyn_nx_f32::create_input(const nn::data<float, 4> &input) {
        if (input.size[3] != batch_size)
            throw std::invalid_argument("input batch size doesn't match");

        if (input.size[0] != input_size_z)
            throw std::invalid_argument("number of input feature maps doesn't match");

        if (input.size[1] != input_size_x)
            throw std::invalid_argument("input width doesn't match");

        if (input.size[2] != input_size_y)
            throw std::invalid_argument("input height doesn't match");

        nn_workload_data_coords_t size = {static_cast<uint32_t>(input.size[3]), // n
                                          static_cast<uint32_t>(input.size[1]), // x
                                          static_cast<uint32_t>(input.size[2]), // y
                                          static_cast<uint32_t>(input.size[0]), // z
                                          1,
                                          1};

        auto buffer = new nn::nn_workload_data_t<float>(size, helper_zxyn_f32::primitive_zxyn_f32_base::in_out_layout);

        memcpy(buffer->parent->data_buffer, input.buffer, input.count() * input.sizeof_value);

        return buffer;
    }

    bool convert_zxyn_nx_f32::validate_input(const nn::nn_workload_data_t<float> &input) {
        if (0 != memcmp(&input.parent->layout,
                        &helper_zxyn_f32::primitive_zxyn_f32_base::in_out_layout,
                        sizeof(nn_workload_data_layout_t)))
            return false;

        // views are not supported by this layer
        if (input.parent->buffer_size / input.parent->data_type_size !=
            input_size_x * input_size_y * input_size_z * batch_size)
            return false;

        const auto view_size = input.get_length();

        if (view_size.t[NN_DATA_COORD_n] != batch_size)
            return false;

        if (view_size.t[NN_DATA_COORD_x] != input_size_x)
            return false;

        if (view_size.t[NN_DATA_COORD_y] != input_size_y)
            return false;

        if (view_size.t[NN_DATA_COORD_z] != input_size_z)
            return false;

        return true;
    }

    nn::nn_workload_data_t<float> *convert_zxyn_nx_f32::create_output() {
        nn_workload_data_coords_t size = {
            static_cast<uint32_t>(batch_size), static_cast<uint32_t>(output_size), 1, 1, 1, 1};
        return new nn::nn_workload_data_t<float>(size, out_layout);
    }

    void convert_zxyn_nx_f32::copy_output(nn::data<float, 2> &destination,
                                          const nn::nn_workload_data_t<float> &source) {
        assert(destination.size[0] == output_size);
        assert(destination.size[1] == batch_size);

        assert(memcmp(&source.parent->layout, &out_layout, sizeof(nn_workload_data_layout_t)) == 0);
        const auto view_size = source.get_length();
        assert(view_size.t[NN_DATA_COORD_n] == batch_size);
        assert(view_size.t[NN_DATA_COORD_z] == batch_size);

        assert(source.parent->buffer_size == destination.count() * destination.sizeof_value);

        for(size_t x = 0; x < output_size; ++x)
            for(size_t n = 0; n < batch_size; ++n)
                ((float *)destination.buffer)[x + n * output_size] =
                    ((float *)source.parent->data_buffer)[n + x * batch_size];
    }

    void convert_zxyn_nx_f32::forward(const nn::nn_workload_data_t<float> *input, nn::nn_workload_data_t<float> *output)
    {
        assert(validate_input(*input));

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
            nn_workload_data_layout_t nzxy_layout = {{0, 0, 0, 0, 0, 0}, // tile in log2(size)
                                                     {0, 0, 0, 0, 0, 0}, // alignment
                                                     {NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_p, NN_DATA_COORD_q}, // ordering
                                                     NN_DATATYPE_FLOAT};
            nn::nn_workload_data_t<float> tmp_output(output->parent->data_buffer, input->parent->lengths, nzxy_layout);
            batching_conversion<false>(input, &tmp_output);
        }
    }

    // out_layout is the same as in fully connected layers
    const nn_workload_data_layout_t convert_zxyn_nx_f32::out_layout = {{ 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
                                                                       {0, 0, 0, 0, 0, 0}, // alignment
                                                                       {NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q}, // ordering
                                                                       NN_DATATYPE_FLOAT};

    convert_zxyn_nx_f32::convert_zxyn_nx_f32(
        size_t input_size_x, size_t input_size_y, size_t input_size_z, size_t batch_size, nn_device_internal *device)
        : input_size_x(input_size_x),
          input_size_y(input_size_y),
          input_size_z(input_size_z),
          batch_size(batch_size),
          device(device),
          output_size(input_size_x * input_size_y * input_size_z) {}

    namespace convert_zxyn_nx_f32_impl {

    nn_opaque_data_t *NN_API_CALL_CONVENTION
    create_input(nn_primitive_handle_t handle, const nn_data_t *input, NN_API_STATUS *status) {
        auto primitive = static_cast<layer::convert_zxyn_nx_f32 *>(handle);
        auto result = primitive->create_input(*nn::data_cast<float, 4>(input));
        SET_STATUS(NN_API_STATUS_OK);
        return reinterpret_cast<nn_opaque_data_t *>(result);
    }

    nn_opaque_data_t *NN_API_CALL_CONVENTION create_output(nn_primitive_handle_t handle, NN_API_STATUS *status) {
        auto primitive = static_cast<layer::convert_zxyn_nx_f32 *>(handle);
        auto result = primitive->create_output();
        SET_STATUS(NN_API_STATUS_OK);
        return reinterpret_cast<nn_opaque_data_t *>(result);
    }

    int NN_API_CALL_CONVENTION
    validate_input(nn_primitive_handle_t handle, /* primitive handle */
                   nn_opaque_data_t *opaque_data /* internal data storage handle to validate */) {
        auto primitive = static_cast<layer::convert_zxyn_nx_f32 *>(handle);
        return primitive->validate_input(*reinterpret_cast<nn::nn_workload_data_t<float> *>(opaque_data));
    }

    nn_event_t NN_API_CALL_CONVENTION copy_output_async(nn_primitive_handle_t handle,
                                                        nn_data_t *output,
                                                        nn_opaque_data_t *output_buffer,
                                                        size_t dependencies_count,
                                                        nn_event_t *dependencies,
                                                        NN_API_STATUS *status) {
        auto primitive = static_cast<layer::convert_zxyn_nx_f32 *>(handle);
        primitive->copy_output(*nn::data_cast<float, 2>(output),
                               *reinterpret_cast<nn::nn_workload_data_t<float> *>(output_buffer));
        SET_STATUS(NN_API_STATUS_OK);
        return {};
    }

    nn_event_t NN_API_CALL_CONVENTION forward_async(nn_primitive_handle_t handle,
                                                    nn_opaque_data_t *input,
                                                    nn_opaque_data_t *output,
                                                    size_t dependencies_count,
                                                    nn_event_t *dependencies,
                                                    NN_API_STATUS *status) {
        auto primitive = static_cast<layer::convert_zxyn_nx_f32 *>(handle);
        primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(input),
                           reinterpret_cast<nn::nn_workload_data_t<float> *>(output));
        SET_STATUS(NN_API_STATUS_OK);
        return {};
    }

    nn_primitive_handle_t NN_API_CALL_CONVENTION create(nn_device_t *device, /* IDLF device handle */
                                                        size_t input_size_x,
                                                        size_t input_size_y,
                                                        size_t input_size_z,
                                                        size_t batch_size,
                                                        NN_API_STATUS *status /* NN_API_STATUS_OK on success */
                                                        ) {
        SET_STATUS(NN_API_STATUS_OK);
        return layer::convert_zxyn_nx_f32::create(input_size_x, input_size_y, input_size_z, batch_size, device);
    }
    } // namespace convert_zxyn_nx_f32_impl

} // namespace layer

nn_primitives_convert_zxyn_nx_f32_0_t nn_primitives_convert_zxyn_nx_f32_0{
    layer::convert_zxyn_nx_f32_impl::create,
    layer::convert_zxyn_nx_f32_impl::create_input,
    layer::convert_zxyn_nx_f32_impl::validate_input,
    layer::convert_zxyn_nx_f32_impl::create_output,
    layer::convert_zxyn_nx_f32_impl::forward_async,
    layer::convert_zxyn_nx_f32_impl::copy_output_async,
};
