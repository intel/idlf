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
#include "device/cpu/api_internal/data_helper.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "layer_pooling_int16_fixedpoint_avx2.h"

#include <immintrin.h>
#include <string.h>
#include <algorithm>

namespace int16_fixedpoint
{
    pooling_i16::pooling_i16(
        NN_POOLING_MODE pooling_mode,
        const size_t num_output,
        size_t output_w,
        size_t output_h,
        size_t pool_size_x,
        size_t pool_size_y,
        size_t pool_stride_x,
        size_t pool_stride_y,
        const int32_t center_offset_x,
        const int32_t center_offset_y,
        size_t batch_size,
        size_t output_padding_left,
        size_t output_padding_right,
        size_t output_padding_top,
        size_t output_padding_bottom,
        nn_device_internal *device)
        :
        pooling_mode(pooling_mode),
        num_output(num_output),
        output_w(output_w),
        output_h(output_h),
        pool_size_x(pool_size_x),
        pool_size_y(pool_size_y),
        pool_stride_x(pool_stride_x),
        pool_stride_y(pool_stride_y),
        center_offset_x(center_offset_x),
        center_offset_y(center_offset_y),
        batch_size(batch_size),
        output_padding_left(output_padding_left),
        output_padding_right(output_padding_right),
        output_padding_top(output_padding_top),
        output_padding_bottom(output_padding_bottom),
        device(device)
    {
    }

    void pooling_i16::forward(
        const nn::workload_data<int16_t> *input_view,
        nn::workload_data<int16_t> *output_view)
    {
        const size_t batch_window_size = output_view->view_end.t[NN_DATA_COORD_n] - output_view->view_begin.t[NN_DATA_COORD_n] + 1;
        const size_t batch_window_start = output_view->view_begin.t[NN_DATA_COORD_n];

        const size_t output_z_block_size = output_view->parent->lengths.t[NN_DATA_COORD_p];
        const size_t input_z_block_size = input_view->parent->lengths.t[NN_DATA_COORD_p];
        assert(output_z_block_size == input_z_block_size);
        const size_t z_block_size = output_z_block_size;

        // outputs
        const size_t output_size_x = output_view->parent->lengths.t[NN_DATA_COORD_x],
            output_stride_x = output_z_block_size;
        const size_t output_size_y = output_view->parent->lengths.t[NN_DATA_COORD_y],
            output_stride_y = output_stride_x * output_size_x;
        const size_t output_size_z = output_view->parent->lengths.t[NN_DATA_COORD_z] * output_z_block_size,
            output_stride_z_block = output_stride_y * output_size_y;
        const size_t output_stride_batch = output_view->parent->lengths.t[NN_DATA_COORD_z] * output_stride_z_block;

        const size_t output_window_start_x = output_view->view_begin.t[NN_DATA_COORD_x],
            output_window_size_x = output_view->view_end.t[NN_DATA_COORD_x] - output_window_start_x + 1;
        const size_t output_window_start_y = output_view->view_begin.t[NN_DATA_COORD_y],
            output_window_size_y = output_view->view_end.t[NN_DATA_COORD_y] - output_window_start_y + 1;
        const size_t output_window_start_z_block = output_view->view_begin.t[NN_DATA_COORD_z];

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

        const auto threadpool_size = device->thread_pool.get_num_threads();
        const auto task_count = batch_window_size;

        // Single thread
        if (threadpool_size < 2 || task_count < 2)
        {
            int16_t *output_window = (int16_t *)output_view->parent->data_buffer
                + output_window_start_x * output_stride_x
                + output_window_start_y * output_stride_y
                + output_window_start_z_block * output_stride_z_block
                + batch_window_start * output_stride_batch;

            int16_t *input_window = (int16_t *)input_view->parent->data_buffer
                + input_start_x * input_stride_x
                + input_start_y * input_stride_y
                + input_start_z_block * input_stride_z_block
                + batch_window_start * input_stride_batch;

            if (pool_size_x == 3 && pool_size_y == 3)
            {
                NN_Pool_INT16_fixedpoint_optimized<16, 3, 3>(
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
                    pool_stride_x,
                    pool_stride_y);
            }
            else
            {
                NN_Pool_INT16_fixedpoint(
                    output_window,
                    input_window,
                    input_window_size_z_blocks,
                    input_window_size_x,
                    input_window_size_y,
                    z_block_size,
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
        else // Multi-threaded
        {
            std::vector<nn_multithreaded_request> jobs(task_count);
            std::vector<pooling_request_handle> request_handles(task_count);

            for (size_t it_batch = 0; it_batch < batch_window_size; ++it_batch)
            {
                int16_t *output_window = (int16_t *)output_view->parent->data_buffer
                    + output_window_start_x * output_stride_x
                    + output_window_start_y * output_stride_y
                    + output_window_start_z_block * output_stride_z_block
                    + (batch_window_start + it_batch) * output_stride_batch;

                int16_t *input_window = (int16_t *)input_view->parent->data_buffer
                    + input_start_x * input_stride_x
                    + input_start_y * input_stride_y
                    + input_start_z_block * input_stride_z_block
                    + (batch_window_start + it_batch) * input_stride_batch;

                request_handles[it_batch].output_window = output_window;
                request_handles[it_batch].input_window = input_window;
                request_handles[it_batch].input_window_size_z_blocks = input_window_size_z_blocks;
                request_handles[it_batch].input_window_size_x = input_window_size_x;
                request_handles[it_batch].input_window_size_y = input_window_size_y;
                request_handles[it_batch].z_block_size = z_block_size;
                request_handles[it_batch].input_stride_x = input_stride_x;
                request_handles[it_batch].input_stride_y = input_stride_y;
                request_handles[it_batch].input_stride_z_block = input_stride_z_block;
                request_handles[it_batch].output_stride_x = output_stride_x;
                request_handles[it_batch].output_stride_y = output_stride_y;
                request_handles[it_batch].output_stride_z_block = output_stride_z_block;
                request_handles[it_batch].pool_size_x = pool_size_x;
                request_handles[it_batch].pool_size_y = pool_size_y;
                request_handles[it_batch].pool_stride_x = pool_stride_x;
                request_handles[it_batch].pool_stride_y = pool_stride_y;

                jobs[it_batch].callback = unpack_pooling_fixedpoint_callback_handle;
                jobs[it_batch].request_handle = &request_handles[it_batch];
            }

            // Wait for all sub threads.
            device->thread_pool.push_job(jobs);
        }
    }

    void pooling_i16::forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs)
    {
        assert(inputs.size() == 1);
        assert(outputs.size() == 1);

        forward(reinterpret_cast<const nn::workload_data<int16_t> *>(inputs[0]),
            reinterpret_cast<nn::workload_data<int16_t> *>(outputs[0]));
    }

    bool pooling_i16::validate_input(size_t index, nn_workload_data_t *data)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    std::vector<nn_workload_data_t *> pooling_i16::create_outputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, int16_t>::create(
            device,
            output_w,
            output_h,
            num_output,
            batch_size,
            block_size,
            output_padding_left,
            output_padding_right,
            output_padding_top,
            output_padding_bottom) };
    }

    std::vector<nn_workload_data_t *> pooling_i16::create_inputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, int16_t>::create(
            device,
            output_w,
            output_h,
            num_output,
            batch_size,
            block_size,
            output_padding_left,
            output_padding_right,
            output_padding_top,
            output_padding_bottom) };
    }

    std::vector<nn_workload_data_t *> pooling_i16::create_parameters(bool allocate_delta)
    {
        throw std::logic_error("The method or operation is not implemented.");
    }

    //nn::workload_data<int16_t> *pooling_i16::create_output(
    //    size_t padding_left,
    //    size_t padding_right,
    //    size_t padding_top,
    //    size_t padding_bottom)
    //{
    //    nn_workload_data_layout_t layout = {
    //        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
    //        { 0, 0, 0, 0, 0, 0 }, // alignment
    //        { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
    //        NN_DATATYPE_INT16
    //    };

    //    nn_workload_data_coords_t size = {
    //        static_cast<uint32_t>(batch_size),
    //        static_cast<uint32_t>(output_w),
    //        static_cast<uint32_t>(output_h),
    //        static_cast<uint32_t>(num_output / OFMOutBlock),
    //        OFMOutBlock,
    //        1 };

    //    return new nn::workload_data<int16_t>(size, out_layout, padding_left, padding_right, padding_top, padding_bottom);
    //}

    void NN_Pool_INT16_fixedpoint(
        int16_t* output,
        const int16_t* input,
        const size_t input_num_z_blocks,
        const size_t input_view_width,
        const size_t input_view_height,
        const size_t z_block_size,
        const size_t input_stride_x,
        const size_t input_stride_y,
        const size_t input_stride_z_blocks,
        const size_t output_stride_x,
        const size_t output_stride_y,
        const size_t output_stride_z_blocks,
        const size_t pool_size_x,
        const size_t pool_size_y,
        const size_t pool_stride_x,
        const size_t pool_stride_y)
    {
        const size_t SIMD_width = 16;

        for (size_t z_block = 0; z_block < input_num_z_blocks; z_block++)
        {
            for (size_t y = 0; y + pool_stride_y <= input_view_height; y += pool_stride_y)
            {
                for (size_t x = 0; x + pool_stride_x <= input_view_width; x += pool_stride_x)
                {
                    for (size_t z = 0; z < z_block_size; z += SIMD_width){
                        size_t index = y * input_stride_y + x * input_stride_x + z_block * input_stride_z_blocks + z;
                        __m256i max_val_avx = _mm256_load_si256((__m256i *)(input + index));

                        for (size_t pool_y = 0; pool_y < pool_size_y; pool_y++)
                        {
                            for (size_t pool_x = 0; pool_x < pool_size_x; pool_x++)
                            {
                                size_t index_next = index + pool_y * input_stride_y + pool_x * input_stride_x;
                                __m256i next_val_avx = _mm256_load_si256((__m256i *)(input + index_next));
                                max_val_avx = _mm256_max_epi16(next_val_avx, max_val_avx);
                            }
                        }

                        index = (x / pool_stride_x) * output_stride_x + (y / pool_stride_y) * output_stride_y + z_block * output_stride_z_blocks + z;
                        _mm256_store_si256((__m256i *)(output + index), max_val_avx);
                    }
                }
            }
        }
    }

    template <const size_t z_block_size, const size_t pool_size_x, const size_t pool_size_y>
    void NN_Pool_INT16_fixedpoint_optimized(
        int16_t* output,
        const int16_t* input,
        const size_t input_num_z_blocks,
        const size_t input_view_width,
        const size_t input_view_height,
        const size_t input_stride_x,
        const size_t input_stride_y,
        const size_t input_stride_z_blocks,
        const size_t output_stride_x,
        const size_t output_stride_y,
        const size_t output_stride_z_blocks,
        const size_t pool_stride_x,
        const size_t pool_stride_y)
    {
        const size_t SIMD_width = 16;

        for (size_t z_block = 0; z_block < input_num_z_blocks; z_block++)
        {
            for (size_t y = 0; y + pool_stride_y <= input_view_height; y += pool_stride_y)
            {
                for (size_t x = 0; x + pool_stride_x <= input_view_width; x += pool_stride_x)
                {
#pragma unroll (z_block_size / SIMD_width)
                    for (size_t z = 0; z < z_block_size; z += SIMD_width){
                        size_t index = y * input_stride_y + x * input_stride_x + z_block * input_stride_z_blocks + z;
                        __m256i max_val_avx = _mm256_load_si256((__m256i *)(input + index));

#pragma unroll (pool_size_y)
                        for (size_t pool_y = 0; pool_y < pool_size_y; pool_y++)
                        {
#pragma unroll (pool_size_x)
                            for (size_t pool_x = 0; pool_x < pool_size_x; pool_x++)
                            {
                                size_t index_next = index + pool_y * input_stride_y + pool_x * input_stride_x;
                                __m256i next_val_avx = _mm256_load_si256((__m256i *)(input + index_next));
                                max_val_avx = _mm256_max_epi16(next_val_avx, max_val_avx);
                            }
                        }

                        index = (x / pool_stride_x) * output_stride_x + (y / pool_stride_y) * output_stride_y + z_block * output_stride_z_blocks + z;
                        _mm256_store_si256((__m256i *)(output + index), max_val_avx);
                    }
                }
            }
        }
    }

    void unpack_pooling_fixedpoint_callback_handle(void *void_handle) {
        pooling_i16::pooling_request_handle* handle = reinterpret_cast<pooling_i16::pooling_request_handle*>(void_handle);

        if (handle->pool_size_x == 3 && handle->pool_size_y == 3)
        {
            NN_Pool_INT16_fixedpoint_optimized<16, 3, 3>(
                handle->output_window,
                handle->input_window,
                handle->input_window_size_z_blocks,
                handle->input_window_size_x,
                handle->input_window_size_y,
                handle->input_stride_x,
                handle->input_stride_y,
                handle->input_stride_z_block,
                handle->output_stride_x,
                handle->output_stride_y,
                handle->output_stride_z_block,
                handle->pool_stride_x,
                handle->pool_stride_y);
        }
        else
        {
            NN_Pool_INT16_fixedpoint(
                handle->output_window,
                handle->input_window,
                handle->input_window_size_z_blocks,
                handle->input_window_size_x,
                handle->input_window_size_y,
                handle->z_block_size,
                handle->input_stride_x,
                handle->input_stride_y,
                handle->input_stride_z_block,
                handle->output_stride_x,
                handle->output_stride_y,
                handle->output_stride_z_block,
                handle->pool_size_x,
                handle->pool_size_y,
                handle->pool_stride_x,
                handle->pool_stride_y);
        }
    }

    void run_pooling_work_item(nn_workload_item *const work_item, nn_device_internal* device)
    {
        nn::workload_data<int16_t>* input_view = reinterpret_cast<nn::workload_data<int16_t> *>(work_item->input[0].get_data_view());
        nn::workload_data<int16_t>* output_view = reinterpret_cast<nn::workload_data<int16_t> *>(work_item->output[0]);

        static_cast<pooling_i16 *>(work_item->primitive)->forward(input_view, output_view);
    }
} // namespace

nn_primitive_handle_t NN_API_CALL_CONVENTION
nn_primitives_pooling_i16_create_0(
    nn_device_t *device,
    NN_POOLING_MODE pooling_mode,
    size_t pool_size_x,
    size_t pool_size_y,
    size_t pool_stride_x,
    size_t pool_stride_y,
    size_t num_feature_maps,
    size_t output_w,
    size_t output_h,
    const int32_t center_offset_x,
    const int32_t center_offset_y,
    size_t batch_size,
    const nn_primitives_pooling_hints_t *hints,
    NN_API_STATUS *status)
{
    SET_STATUS(NN_API_STATUS_OK);

    std::remove_const<std::remove_pointer<decltype(hints)>::type>::type hints_ = {};
    if (hints != nullptr)
        hints_ = *hints;

    return new int16_fixedpoint::pooling_i16(
        pooling_mode,
        num_feature_maps,
        output_w,
        output_h,
        pool_size_x,
        pool_size_y,
        pool_stride_x,
        pool_stride_y,
        center_offset_x,
        center_offset_y,
        batch_size,
        hints_.output_padding.left,
        hints_.output_padding.right,
        hints_.output_padding.top,
        hints_.output_padding.bottom,
        reinterpret_cast<nn_device_internal *>(device));
}