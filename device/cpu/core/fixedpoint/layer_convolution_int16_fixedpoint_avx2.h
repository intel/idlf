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
#include "helper_z_block_xyz_i16.h"

struct nn_workload_item;
struct nn_device_internal;

namespace int16_fixedpoint
{
    typedef void(*FActiveShift)(void *, __m256i, uint8_t);
    typedef void(*FBias)(__m256i &, void *);

    class convolution_i16 : public helper_z_block_xyz_i16::primitive_z_block_xyz_i16_base
    {
    public:
        convolution_i16(
            const size_t kernel_w,
            const size_t kernel_h,
            const size_t num_input,
            const size_t num_output,
            const size_t output_w,
            const size_t output_h,
            const int32_t center_offset_x,
            const int32_t center_offset_y,
            const size_t stride_x,
            const size_t stride_y,
            const nn_argument_activation_fixedpoint_t &activation,
            size_t batch_size,
            const size_t output_padding_left,
            const size_t output_padding_right,
            const size_t output_padding_top,
            const size_t output_padding_bottom,
            nn_device_internal *device);

        virtual ~convolution_i16() {}

        virtual void forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs) override;

        virtual nn::workload_data<int16_t> *create_weights(const nn::data<int16_t, 4> &weights);
        virtual nn::workload_data<int32_t> *create_bias(const nn::data<int32_t, 1> &bias);

        void run_convolve_fixedpoint_work_item(
            const nn::workload_data<int16_t> *input_buffer,
            const nn::workload_data<int16_t> *weights_buffer,
            const nn::workload_data<int32_t> *bias_buffer,
            nn::workload_data<int16_t> *output_buffer);

        template <const int IFMBlock, const int OFMBlock, const int OXBlock, const int OYBlock, const int OXpBlock, const int OYpBlock, FActiveShift FStoreActiv, FBias FBias>
        friend void NN_ConvolveAVX2_INT16_fixedpoint(
            convolution_i16 *primitive,
            const nn::workload_data<int16_t> *input_view,
            const nn::workload_data<int16_t> *weights_buffer,
            const nn::workload_data<int32_t> *bias_buffer,
            nn::workload_data<int16_t> *output_view);

        friend void run_multithreaded_convolve_fixedpoint_work_item(
            nn_workload_item *const work_item,
            nn_device_internal* device);

        friend void unpack_convolve_fixedpoint_callback_handle(void *void_handle);

        virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta) override;
        virtual std::vector<nn_workload_data_t *> create_parameters(bool allocate_delta) override;

    protected:
        virtual void forward(
            const nn::workload_data<int16_t> *input_buffer,
            const nn::workload_data<int16_t> *weights_buffer,
            const nn::workload_data<int32_t> *bias_buffer,
            nn::workload_data<int16_t> *output_buffer);

        virtual size_t get_required_input_w() override;
        virtual size_t get_required_input_h() override;

        const uint32_t block_size = 16;
        const size_t kernel_w;
        const size_t kernel_h;
        const NN_PADDING_MODE padding;
        const int32_t center_offset_x;
        const int32_t center_offset_y;
        const size_t stride_x;
        const size_t stride_y;
        const nn_argument_activation_fixedpoint_t activation;

        struct request_handle
        {
            convolution_i16 *primitive;
            const nn::workload_data<int16_t> *input_view;
            const nn::workload_data<int16_t> *weights_view;
            const nn::workload_data<int32_t> *biases_view;
            nn::workload_data<int16_t> *output_view;
        };
    };

    void unpack_convolve_fixedpoint_callback_handle(void* void_handle);
    void run_multithreaded_convolve_fixedpoint_work_item(nn_workload_item *const work_item, nn_device_internal* device);
} //namespace layer
