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
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "helper_z_block_xyz_i16.h"

struct nn_workload_item;

namespace int16_fixedpoint {
    class normalization_response_across_maps_i16 : public helper_z_block_xyz_i16::primitive_z_block_xyz_i16_base
    {
    public:
        normalization_response_across_maps_i16(
            uint32_t k,
            uint32_t n,
            float alpha,
            float beta,
            float scale_in,
            float scale_out,
            size_t image_size_x,
            size_t image_size_y,
            size_t image_size_z,
            size_t batch_size,
            size_t output_padding_left,
            size_t output_padding_right,
            size_t output_padding_top,
            size_t output_padding_bottom,
            nn_device_internal *device);

        virtual ~normalization_response_across_maps_i16() {}

        virtual void forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs) override;

        virtual bool validate_input(size_t index, nn_workload_data_t *data) override;

        template <uint32_t T_n> friend void unpack(void *void_handle);

        friend void wrapper_lrn_fixedpoint_work_item(nn_workload_item *const work_item, nn_device_internal *device);

    protected:
        virtual void forward(const nn::workload_data<int16_t> *input, nn::workload_data<int16_t> *output);
        virtual size_t get_required_input_w() override;
        virtual size_t get_required_input_h() override;

        template <uint32_t T_n>
        friend void process_lrn_p075(const nn::workload_data<int16_t> *input_view,
            nn::workload_data<int16_t> *output_view,
            // const uint32_t n,
            const float alpha,
            const float k,
            const float scale_in,
            const float scale_out);

        float alpha;
        float beta;
        uint32_t k;
        uint32_t n;
        uint32_t z_block = 16;
        size_t image_size_x;
        size_t image_size_y;
        size_t image_size_z;
        const size_t output_padding_left;
        const size_t output_padding_right;
        const size_t output_padding_top;
        const size_t output_padding_bottom;
        const size_t batch_size;
        const float scale_in;
        const float scale_out;
        nn_device_internal *const device;

        const nn_workload_data_layout_t in_out_layout;
    };

    void wrapper_lrn_fixedpoint_work_item(nn_workload_item *const work_item, nn_device_internal *device);
} //namespace layer
