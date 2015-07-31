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
#include "device/api/nn_device_interface_0.h"
#include "device/api/nn_primitives_api_0.h"

struct nn_workload_item;
struct nn_device_internal;

namespace int16_fixedpoint
{
    class pooling_i16 : public nn_primitive_t
    {
    public:
        pooling_i16(
            const size_t num_output,
            size_t output_w,
            size_t output_h,
            size_t pool_size_x,
            size_t pool_size_y,
            size_t pool_stride_x,
            size_t pool_stride_y,
            size_t batch_size,
            size_t output_padding_left,
            size_t output_padding_right,
            size_t output_padding_top,
            size_t output_padding_bottom,
            nn_device_internal *device);

        virtual ~pooling_i16() {}

        virtual void forward(
            const nn::workload_data<int16_t> *input,
            nn::workload_data<int16_t> *output);

        virtual void forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs) override;

        virtual bool validate_input(size_t index, nn_workload_data_t *data) override;

        template <const size_t z_block_size, const size_t pool_size_x, const size_t pool_size_y>
        friend void NN_Pool_INT16_fixedpoint_optimized(
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
            const size_t pool_stride_y);

        friend void NN_Pool_INT16_fixedpoint(
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
            const size_t pool_stride_y);

        friend void unpack_pooling_fixedpoint_callback_handle(void *void_handle);

        friend void run_pooling_work_item(nn_workload_item *const work_item, nn_device_internal* device);

        virtual std::vector<nn_workload_data_t *> create_parameters(bool allocate_delta) override;

        virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta) override;

        virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta) override;

    protected:
        const uint32_t block_size = 16;
        const size_t batch_size;
        const size_t num_output;
        const size_t pool_size_x;
        const size_t pool_size_y;
        const size_t pool_stride_x;
        const size_t pool_stride_y;
        const size_t output_w, output_h;
        const size_t output_padding_left;
        const size_t output_padding_right;
        const size_t output_padding_top;
        const size_t output_padding_bottom;
        nn_device_internal * const device;
        static const nn_workload_data_layout_t out_layout;

        struct pooling_request_handle
        {
            int16_t* output_window;
            int16_t* input_window;
            size_t input_window_size_z_blocks;
            size_t input_window_size_x;
            size_t input_window_size_y;
            size_t z_block_size;
            size_t input_stride_x;
            size_t input_stride_y;
            size_t input_stride_z_block;
            size_t output_stride_x;
            size_t output_stride_y;
            size_t output_stride_z_block;
            size_t pool_size_x;
            size_t pool_size_y;
            size_t pool_stride_x;
            size_t pool_stride_y;
        };
    };

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
        const size_t pool_stride_y);

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
        const size_t pool_stride_y);

    void unpack_pooling_fixedpoint_callback_handle(void *void_handle);
} //namespace device_int16
