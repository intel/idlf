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

#include "layer_parameter_update.h"

namespace layer
{
    template<class T> inline void read_wrapper(T& reg, float* data);
    template<> inline void read_wrapper<__m256>(__m256& reg, float* data) {reg = _mm256_loadu_ps(data);}
    template<> inline void read_wrapper<float>(float& reg, float* data) {reg = *data;}

    template<class T> inline void store_wrapper(T& reg, float* data);
    template<> inline void store_wrapper<__m256>(__m256& reg, float* data) {_mm256_storeu_ps(data, reg);}
    template<> inline void store_wrapper<float>(float& reg, float* data) {*data = reg;}

    template<class T> inline T op_wrapper(T& reg1, T& reg2, float data);
    template<> inline __m256 op_wrapper<__m256>(__m256& reg1, __m256& reg2, float data) { return _mm256_sub_ps(reg1, _mm256_mul_ps(reg2, _mm256_set1_ps(data))); }
    template<> inline float op_wrapper<float>(float& reg1, float& reg2, float data) { return reg1 - reg2*data; }

    template <uint32_t block_size, class T = __m256>
    inline void inner_update_processing(
        float* data,
        float* delta,
        float learning_rate)
    {
        T data_acc_array[block_size];
        T delta_acc_array[block_size];

#pragma unroll(block_size)
        for(auto acc = 0u; acc < block_size; ++acc)
            read_wrapper<T>(data_acc_array[acc], data + sizeof(T) / sizeof(float) * acc);

#pragma unroll(block_size)
        for(auto acc = 0u; acc < block_size; ++acc)
            read_wrapper<T>(delta_acc_array[acc], delta + sizeof(T) / sizeof(float) * acc);

#pragma unroll(block_size)
        for(auto acc = 0u; acc < block_size; ++acc)
            data_acc_array[acc] = op_wrapper<T>(data_acc_array[acc], delta_acc_array[acc], learning_rate);

#pragma unroll(block_size)
        for(auto acc = 0u; acc < block_size; ++acc)
            store_wrapper<T>(data_acc_array[acc], data + sizeof(T) / sizeof(float) * acc);
    }

    void param_update_data(
        nn::workload_data<float> *forward_data,
        nn::workload_data<float> *delta_data,
        float learning_rate) 
    {
        const auto C_max_block_size = 7;
        const auto C_simd_size = 8;

        assert(forward_data->parent->buffer_size == delta_data->parent->buffer_size);

        auto data_start = reinterpret_cast<float*>(forward_data->parent->data_buffer);
        auto delta_start = reinterpret_cast<float*>(delta_data->parent->data_buffer);

        const auto total_image_size = forward_data->parent->buffer_size / 4;

        const auto full_blocks = total_image_size / (C_max_block_size * C_simd_size);
        const auto partial_block = (total_image_size % (C_max_block_size * C_simd_size) / C_simd_size);
        const auto partial_subblock = total_image_size % C_simd_size;

        auto data_ptr = data_start;
        auto delta_ptr = delta_start;

        #pragma forceinline recursive
        {
            // Full blocks processing (all accumulators used).
            for (auto block = 0u; block < full_blocks; ++block)
            {
                inner_update_processing<C_max_block_size>(data_ptr, delta_ptr, learning_rate);

                data_ptr += C_simd_size * C_max_block_size;
                delta_ptr += C_simd_size * C_max_block_size;
            }

            // Partial blocks processing (only part of accumulators).
            switch(partial_block)
            {
            case  0: break;
            case  1: inner_update_processing< 1>(data_ptr, delta_ptr, learning_rate); break;
            case  2: inner_update_processing< 2>(data_ptr, delta_ptr, learning_rate); break;
            case  3: inner_update_processing< 3>(data_ptr, delta_ptr, learning_rate); break;
            case  4: inner_update_processing< 4>(data_ptr, delta_ptr, learning_rate); break;
            case  5: inner_update_processing< 5>(data_ptr, delta_ptr, learning_rate); break;
            case  6: inner_update_processing< 6>(data_ptr, delta_ptr, learning_rate); break;
            }
            data_ptr += C_simd_size * partial_block;
            delta_ptr += C_simd_size * partial_block;

            // Processing of sub blocks (only part of one SIMD).
            switch(partial_subblock)
            {
            case  0: break;
            case  1: inner_update_processing<1, float>(data_ptr, delta_ptr, learning_rate); break;
            case  2: inner_update_processing<2, float>(data_ptr, delta_ptr, learning_rate); break;
            case  3: inner_update_processing<3, float>(data_ptr, delta_ptr, learning_rate); break;
            case  4: inner_update_processing<4, float>(data_ptr, delta_ptr, learning_rate); break;
            case  5: inner_update_processing<5, float>(data_ptr, delta_ptr, learning_rate); break;
            case  6: inner_update_processing<6, float>(data_ptr, delta_ptr, learning_rate); break;
            case  7: inner_update_processing<7, float>(data_ptr, delta_ptr, learning_rate); break;
            }
        }
    }

    void run_parameter_update(nn_workload_item *const item)
    {
        auto updated_item = item->forward_item;
        auto rate = item->arguments.update.learning_rate;
        auto weights_delta = reinterpret_cast<nn::workload_data<float>*>(item->input[0].get_data_view());
        auto biases_delta = reinterpret_cast<nn::workload_data<float>*>(item->input[1].get_data_view());
        decltype(weights_delta) weights = nullptr;
        decltype(biases_delta) biases = nullptr;

        switch (updated_item->type)
        {
        case NN_WORK_ITEM_TYPE_CONVOLUTION:
        case NN_WORK_ITEM_TYPE_FULLY_CONNECTED:
            weights = reinterpret_cast<nn::workload_data<float>*>(updated_item->parameters[0]);
            biases = reinterpret_cast<nn::workload_data<float>*>(updated_item->parameters[1]);
            break;
        default:
            assert(0);
            break;
        }

        param_update_data(weights, weights_delta, rate);
        param_update_data(biases, biases_delta, rate);
    }
}