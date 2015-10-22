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
#include "device/api/nn_primitives_api_0.h"

struct nn_workload_item;

namespace int16_fixedpoint {
    class softmax_i32 : public nn_primitive_t
    {
    public:
        softmax_i32(
            size_t num_features,
            size_t batch_size,
            int8_t input_fraction,
            nn_device_internal *device);

        virtual ~softmax_i32() {}

        void forward(const nn::workload_data<int32_t> *input, nn::workload_data<> *output);

        virtual void forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs) override;

        virtual bool validate_input(size_t index, nn_workload_data_t *data) override;

        virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta) override;

        virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta) override;

        friend void run_softmax_int32_float_work_item(nn_workload_item *const work_item, nn_device_internal* device);

    protected:
        int8_t input_fraction;
        nn_device_internal *const device;
        const size_t num_features, batch_size;

    private:
        void run_softmax_int32_float_work_item_batch8(const nn::workload_data<int32_t> *input_view,
            nn::workload_data<> *output_view);
        void run_softmax_int32_float_work_item_batch8x(const nn::workload_data<int32_t> *input_view,
            nn::workload_data<> *output_view, uint16_t NoBatch8);
        void run_softmax_int32_float_work_item_latency(const nn::workload_data<int32_t> *input_view,
            nn::workload_data<> *output_view);
    };
} //namespace device_int16
