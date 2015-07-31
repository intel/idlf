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
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "device/api/nn_primitives_api_0.h"

struct nn_workload_item;
struct nn_device_internal;

namespace int16_fixedpoint
{
    template<typename T_output_type>
    class fully_connected_i16 : public nn_primitive_t
    {
    public:
        fully_connected_i16(
            size_t num_input,
            size_t num_output,
            const nn_argument_activation_fixedpoint_t &activation,
            size_t batch_size,
            nn_device_internal *device);

        virtual ~fully_connected_i16() {}

        virtual void forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs) override;

        virtual bool validate_input(size_t index, nn_workload_data_t *data) override;

        virtual std::vector<nn_workload_data_t *> create_parameters(bool allocate_delta) override;

        virtual nn::workload_data<int16_t> *create_weights(const nn::data<int16_t, 2> &weights);

        virtual nn::workload_data<int16_t> *create_weights(const nn::data<int16_t, 4> &weights);

        virtual nn::workload_data<int32_t> *create_bias(const nn::data<int32_t, 1> &bias);

        virtual void copy_output(nn::data<int16_t, 2> &destination, const nn::workload_data<int16_t> &source);

        virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta) override;

        virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta) override;

    protected:
        void forward(
            const nn::workload_data<int16_t> *input,
            const nn::workload_data<int16_t> *weights,
            const nn::workload_data<int32_t> *bias,
            nn::workload_data<T_output_type> *output);

        typedef void(*FActiveFCShift)(void *, __m256i, __m256i, uint8_t);

        void run_fully_connected_fixedpoint_work_item(
            const nn::workload_data<int16_t> *input,
            const nn::workload_data<int16_t> *weights,
            const nn::workload_data<int32_t> *biases,
            nn::workload_data<T_output_type> *output);

        template<class ActivationType, bool T_NEED_BIAS_COPY>
        void run_fully_connected_int16_fixedpoint_work_item_internal(
            const nn::workload_data<int16_t> *input,
            const nn::workload_data<int16_t> *weights,
            const nn::workload_data<int32_t> *biases,
            const nn_argument_activation_fixedpoint_t activation,
            nn::workload_data<T_output_type> *output);

        template<typename>
        friend void wrapper_fully_connected_fixedpoint_work_item(
            nn_workload_item *const work_item,
            nn_device_internal* device);

        template<typename>
        friend void unpack_fully_connected_callback_handle(void *void_handle);

        friend void run_multithreaded_fully_connected_fixedpoint_work_item(
            nn_workload_item *const work_item,
            nn_device_internal* device);

        nn_device_internal *device;
        const size_t block_size = 2;
        const size_t num_input, num_output, batch_size;
        const nn_argument_activation_fixedpoint_t activation;
        static const nn_workload_data_layout_t &in_layout;
        static const nn_workload_data_layout_t &out_layout;

        struct request_handle
        {
            fully_connected_i16<T_output_type> *primitive;
            const nn::workload_data<int16_t> *input_view;
            const nn::workload_data<int16_t> *weights;
            const nn::workload_data<int32_t> *biases;
            nn::workload_data<T_output_type> *output_view;
        };
    };

    template class fully_connected_i16<int16_t>;
    template class fully_connected_i16<int32_t>;

    void run_multithreaded_fully_connected_fixedpoint_work_item(
        nn_workload_item *const work_item,
        nn_device_internal* device);
} //namespace int16_fixedpoint
