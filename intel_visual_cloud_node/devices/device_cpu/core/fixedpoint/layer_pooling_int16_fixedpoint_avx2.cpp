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
#include "layer_pooling_int16_fixedpoint_avx2.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>

namespace int16_fixedpoint {

    void NN_Pool_INT16_fixedpoint(
        int16_t* output,
        int16_t* input,
        size_t input_num_z_blocks,
        size_t input_view_width,
        size_t input_view_height,
        size_t input_stride_x,
        size_t input_stride_y,
        size_t input_stride_z_blocks,
        size_t output_stride_x,
        size_t output_stride_y,
        size_t output_stride_z,
        size_t pool_size_x,
        size_t pool_size_y,
        size_t pool_stride_x,
        size_t pool_stride_y)
    {
        const size_t IFMBlock = 8;

        for (int zBlock = 0; zBlock < input_num_z_blocks; zBlock++)
        {
            for (int y = 0; y + pool_stride_y <= input_view_height; y += pool_stride_y)
            {
                for (int x = 0; x + pool_stride_x <= input_view_width; x += pool_stride_x)
                {
                    size_t coord0 = y * input_stride_y + x * input_stride_x + zBlock * input_stride_z_blocks;
                    __m128i max_val_avx = _mm_load_si128((__m128i *)(input + coord0));

                    for (int pool_y = 0; pool_y < pool_size_y; pool_y++)
                    {
                        for (int pool_x = 0; pool_x < pool_size_x; pool_x++)
                        {
                            size_t coord_next = coord0 + pool_y *input_stride_y + pool_x * input_stride_x;
                            __m128i next_val_avx = _mm_load_si128((__m128i *)(input + coord_next));
                            max_val_avx = _mm_max_epi16(next_val_avx, max_val_avx);
                        }
                    }

                    coord0 = (x / pool_stride_x) * output_stride_x + (y / pool_stride_y) * output_stride_y + zBlock * output_stride_z;
                    _mm_stream_si128((__m128i *)(output + coord0), max_val_avx);
                }
            }
        }
    }

    void maxpool_avx2_int16_fixedpoint(
        nn_workload_item *const work_item,
        nn_workload_data_t *input_view,
        nn_arguments_forward_pooling_fixedpoint *arguments)
    {
        auto output_view = reinterpret_cast<nn::nn_workload_data_t<int16_t> *>(work_item->output);

        const size_t batch_window_size = output_view->view_end.t[NN_DATA_COORD_n] - output_view->view_begin.t[NN_DATA_COORD_n] + 1;
        const size_t batch_window_start = output_view->view_begin.t[NN_DATA_COORD_n];

        size_t pool_size_x = arguments->pool_size[0];
        size_t pool_size_y = arguments->pool_size[1];
        size_t pool_stride_x = arguments->pool_stride[0];
        size_t pool_stride_y = arguments->pool_stride[1];

        size_t output_z_block_size = 8;
        size_t input_z_block_size = input_view->parent->lengths.t[NN_DATA_COORD_p];

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
            input_stride_x = input_z_block_size,
            input_window_size_x = input_view->view_end.t[NN_DATA_COORD_x] - input_view->view_begin.t[NN_DATA_COORD_x] + 1;
        const size_t input_size_y = input_view->parent->lengths.t[NN_DATA_COORD_y],
            input_stride_y = input_stride_x * input_size_x,
            input_window_size_y = input_view->view_end.t[NN_DATA_COORD_y] - input_view->view_begin.t[NN_DATA_COORD_y] + 1;
        const size_t input_size_z = input_view->parent->lengths.t[NN_DATA_COORD_z] * input_z_block_size,
            input_stride_z_block = input_stride_y * input_size_y,
            input_window_size_z_blocks = input_view->view_end.t[NN_DATA_COORD_z] - input_view->view_begin.t[NN_DATA_COORD_z] + 1;
        const size_t input_stride_batch = input_size_x * input_size_y * input_size_z;

        const size_t input_start_x = input_view->view_begin.t[NN_DATA_COORD_x];
        const size_t input_start_y = input_view->view_begin.t[NN_DATA_COORD_y];
        const size_t input_start_z_block = input_view->view_begin.t[NN_DATA_COORD_z];

        for (size_t it_batch = 0; it_batch < batch_window_size; ++it_batch)
        {
            int16_t *output_window = (int16_t *)output_view->parent->data_buffer
                + output_window_start_x * output_stride_x
                + output_window_start_y * output_stride_y
                + output_window_start_z * output_stride_z_block
                + (batch_window_start + it_batch) * output_stride_batch;

            int16_t *input_window = (int16_t *)input_view->parent->data_buffer
                + input_start_x * input_stride_x
                + input_start_y * input_stride_y
                + input_start_z_block * input_stride_z_block
                + (batch_window_start + it_batch) * input_stride_batch;

            NN_Pool_INT16_fixedpoint(
                output_window,
                input_window,
                input_window_size_z_blocks,
                input_window_size_x,
                input_window_size_y,
                input_stride_x,
                input_stride_y,
                input_stride_z_block,
                output_stride_x,
                output_stride_y,
                output_stride_z_block,
                pool_size_x,
                pool_size_y,
                pool_stride_x,
                pool_stride_y);
        }
    }


    void run_pooling_work_item(nn_workload_item *const work_item)
    {
        nn_workload_data_t *input_view = work_item->input[0]->output;
        nn_arguments_forward_pooling_fixedpoint &arguments = work_item->arguments.forward_pooling_fixedpoint;

        maxpool_avx2_int16_fixedpoint(work_item, reinterpret_cast<nn::nn_workload_data_t<int16_t> *>(input_view), &arguments);
    }

} // namespace
