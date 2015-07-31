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
#include "device/api/nn_primitives_api_0.h"

struct nn_workload_item;
struct nn_device_internal;

namespace int16_fixedpoint {
    class convert_float_to_int16 : public nn_primitive_t
    {
    public:
        convert_float_to_int16(
            const size_t num_input,
            size_t input_w,
            size_t input_h,
            size_t batch_size,
            size_t output_padding_left,
            size_t output_padding_right,
            size_t output_padding_top,
            size_t output_padding_bottom,
            int8_t out_fraction,
            nn_device_internal *device);

        virtual ~convert_float_to_int16() {}

        virtual void forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs) override;

        virtual bool validate_input(size_t index, nn_workload_data_t *data) override;

        void convert_float_to_int16_fixedpoint_contiguous(size_t input_width, const float *input_ptr, std::int16_t *output_ptr);

        friend void unpack_convert_float_to_int16_callback_handle(void *void_handle);

        friend void run_convert_float_to_int16_fp_work_item(nn_workload_item *const work_item, nn_device_internal* device);

        friend void convert_float_to_int16_fixedpoint_rgb(
            const float *const input_ptr,
            int16_t *const output_ptr,
            size_t batch_window_size,
            size_t input_view_num_feature_maps,
            size_t input_view_width,
            size_t input_view_height,
            size_t input_stride_batch,
            size_t output_stride_batch,
            size_t input_stride_y,
            size_t output_stride_y,
            int8_t out_fraction
            );

        virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta) override;

        virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta) override;

        const int8_t out_fraction;
        const size_t batch_size;
        const size_t num_input;
        const size_t input_w, input_h;
        nn_device_internal * const device;
        static const nn_workload_data_layout_t out_layout;

    protected:
        virtual void forward(
            const nn::workload_data<float> *input,
            nn::workload_data<int16_t> *output);

        const uint32_t block_size = 4;
        const size_t
            output_padding_left,
            output_padding_right,
            output_padding_top,
            output_padding_bottom;

        struct convert_float_to_int16_request_handle
        {
            float *input_window;
            int16_t* output_window;
            size_t batch_window_size;
            size_t input_window_size_z_blocks;
            size_t input_window_size_x;
            size_t input_window_size_y;
            size_t input_stride_batch;
            size_t output_stride_batch;
            size_t input_stride_y;
            size_t output_stride_y;
            int8_t output_fraction;
        };
    };

    void unpack_convert_float_to_int16_callback_handle(void *void_handle);

    void convert_float_to_int16_fixedpoint_rgb(
        const float *const input_ptr,
        int16_t *const output_ptr,
        size_t batch_window_size,
        size_t input_view_num_feature_maps,
        size_t input_view_width,
        size_t input_view_height,
        size_t input_stride_batch,
        size_t output_stride_batch,
        size_t input_stride_y,
        size_t output_stride_y,
        int8_t output_fraction);

    void run_convert_float_to_int16_fp_work_item(nn_workload_item *const work_item, nn_device_internal* device);
}
