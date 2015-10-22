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

const auto C_simd_width = sizeof(__m256) / sizeof(float);

namespace layer
{
    struct update_parameters_f32_avx2_handle
    {
        nn::workload_data<nn::layout_f32> *forward_data;
        nn::workload_data<nn::layout_f32> *delta_data;
        float learning_rate;
    };

    template<class T> inline T read_mul_wrapper(float* data, float mul);
    template<> inline __m256 read_mul_wrapper<__m256>(float* data, float mul) { return _mm256_mul_ps(_mm256_loadu_ps(data), _mm256_set1_ps(mul)); }
    template<> inline float read_mul_wrapper<float>(float* data, float mul) { return (*data) * mul; }

    template<class T> inline void store_wrapper(T& reg, float* data);
    template<> inline void store_wrapper<__m256>(__m256& reg, float* data) { _mm256_storeu_ps(data, reg); }
    template<> inline void store_wrapper<float>(float& reg, float* data) { *data = reg; }

    template<class T> inline T read_sub_wrapper(T& reg1, float* data);
    template<> inline __m256 read_sub_wrapper<__m256>(__m256& reg1, float* data) { return _mm256_sub_ps(_mm256_loadu_ps(data), reg1); }
    template<> inline float read_sub_wrapper<float>(float& reg1, float* data) { return (*data) - reg1; }

    template <uint32_t T_block_size, class T = __m256>
    inline void inner_update_processing(
        float* data,
        float* delta,
        float learning_rate)
    {
        T delta_acc_array[T_block_size];

        #pragma unroll (T_block_size)
        for(uint32_t acc = 0; acc < T_block_size; ++acc)
            delta_acc_array[acc] = read_mul_wrapper<T>(delta + sizeof(T) / sizeof(float) * acc, learning_rate);

        #pragma unroll (T_block_size)
        for(uint32_t acc = 0; acc < T_block_size; ++acc)
            delta_acc_array[acc] = read_sub_wrapper<T>(delta_acc_array[acc], data + sizeof(T) / sizeof(float) * acc);

        #pragma unroll (T_block_size)
        for(uint32_t acc = 0; acc < T_block_size; ++acc)
            store_wrapper<T>(delta_acc_array[acc], data + sizeof(T) / sizeof(float) * acc);
    }

    void param_update_data_internal(
        nn::workload_data<nn::layout_f32> *forward_data,
        nn::workload_data<nn::layout_f32> *delta_data,
        float learning_rate)
    {
        const auto C_max_block_size = 15;
        const auto C_simd_size = 8;

        assert(forward_data->parent->buffer_size == delta_data->parent->buffer_size);

        auto data_start = reinterpret_cast<float*>(forward_data->parent->data_buffer);
        auto delta_start = reinterpret_cast<float*>(delta_data->parent->data_buffer);

        const auto total_image_size = forward_data->get_length(NN_DATA_COORD_x);
        const auto view_start = forward_data->view_begin.t[NN_DATA_COORD_x];

        const auto full_blocks = total_image_size / (C_max_block_size * C_simd_size);
        const auto partial_block = (total_image_size % (C_max_block_size * C_simd_size) / C_simd_size);
        const auto partial_subblock = total_image_size % C_simd_size;

        auto data_ptr = data_start + view_start;
        auto delta_ptr = delta_start + view_start;

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
            case  7: inner_update_processing< 7>(data_ptr, delta_ptr, learning_rate); break;
            case  8: inner_update_processing< 8>(data_ptr, delta_ptr, learning_rate); break;
            case  9: inner_update_processing< 9>(data_ptr, delta_ptr, learning_rate); break;
            case 10: inner_update_processing<10>(data_ptr, delta_ptr, learning_rate); break;
            case 11: inner_update_processing<11>(data_ptr, delta_ptr, learning_rate); break;
            case 12: inner_update_processing<12>(data_ptr, delta_ptr, learning_rate); break;
            case 13: inner_update_processing<13>(data_ptr, delta_ptr, learning_rate); break;
            case 14: inner_update_processing<14>(data_ptr, delta_ptr, learning_rate); break;
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

    void unpack_parameter_update_handle(void* handle)
    {
        auto internal_handle = reinterpret_cast<update_parameters_f32_avx2_handle*>(handle);
        param_update_data_internal(internal_handle->forward_data, internal_handle->delta_data, internal_handle->learning_rate);
    }

    void param_update_data(
        nn_thread_worker_pool& thread_pool,
        nn::workload_data<nn::layout_f32> *forward_data,
        nn::workload_data<nn::layout_f32> *delta_data,
        float learning_rate)
    {
        // Split it for multi threading.
        auto num_hardware_threads = thread_pool.get_num_threads();
        auto item_length = forward_data->parent->buffer_size / sizeof(float) / C_simd_width;
        auto item_length_remainder = forward_data->parent->buffer_size / sizeof(float) % C_simd_width;

        const auto items_per_thread = item_length / num_hardware_threads;
        const auto items_modulo = item_length % num_hardware_threads;

        // Check if we have enough data to cover all threads.
        if (items_per_thread == 0 && items_modulo < 2)
        {
            param_update_data_internal(forward_data, delta_data, learning_rate);
        }
        else
        {
            // Not all threads will be used.
            if (items_per_thread == 0)
                num_hardware_threads = items_modulo;

            uint32_t* thread_items_sums = static_cast<uint32_t*>(alloca(num_hardware_threads * sizeof(uint32_t)));

            if (thread_items_sums == nullptr) throw std::bad_alloc();

            // Distribute elements more evenly.
            auto elements_left = items_modulo;
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                thread_items_sums[thread_id] = items_per_thread;
                if (elements_left)
                {
                    ++thread_items_sums[thread_id];
                    --elements_left;
                }
            }

            // Now create table of thread sums.
            auto thread_sum = 0u;
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                thread_sum += thread_items_sums[thread_id];
                thread_items_sums[thread_id] = thread_sum;

                thread_items_sums[thread_id] *= C_simd_width;

                if (thread_id + 1 == num_hardware_threads)
                {
                    thread_items_sums[thread_id] += item_length_remainder;
                }
            }

            std::vector<nn::workload_data<nn::layout_f32>*> output_views(num_hardware_threads);
            std::vector<update_parameters_f32_avx2_handle> request_handles(num_hardware_threads);
            std::vector<nn_multithreaded_request> job(num_hardware_threads);

            // Fill slave work items.
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                auto work_begin = 0u;
                if (thread_id > 0u)
                    work_begin = thread_items_sums[thread_id - 1];

                auto work_end = thread_items_sums[thread_id] - 1;

                // Replace nn_workload_datas pointers with views.
                nn_workload_data_coords_t nn_view_begin = {0, work_begin, 0, 0, 0, 0};
                nn_workload_data_coords_t nn_view_end = {0, work_end, 0, 0, 0, 0};
                nn_workload_data_coords_t nn_view_sizes = {1, static_cast<uint32_t>(forward_data->parent->buffer_size / sizeof(float)), 1, 1, 1, 1};

                nn::workload_data<nn::layout_f32> temp_buffer_flat(forward_data->parent->data_buffer, nn_view_sizes, forward_data->parent->layout);
                output_views[thread_id] = new nn::workload_data<nn::layout_f32>(temp_buffer_flat, nn_view_begin, nn_view_end);

                request_handles[thread_id].forward_data = output_views[thread_id];
                request_handles[thread_id].delta_data = delta_data;
                request_handles[thread_id].learning_rate = learning_rate;

                job[thread_id].callback = unpack_parameter_update_handle;
                job[thread_id].request_handle = &request_handles[thread_id];
            }

            // Wait for all sub threads.
            thread_pool.push_job(job);

            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
                delete output_views[thread_id];
        }
    }

    void run_parameter_update(
        nn_workload_item *const item,
        nn_device_internal *device)
    {
        auto should_update_params = (*reinterpret_cast<nn::workload_data<int32_t>*>(item->input[3].get_data_view()))(0, 0, 0, 0, 0, 0);
        if(!should_update_params)
            return;

        auto updated_item = item->forward_item;
        auto weights_delta = nn::workload_data_cast<nn::layout_f32 >(item->input[0].get_data_view());
        auto biases_delta = nn::workload_data_cast<nn::layout_f32>(item->input[1].get_data_view());
        auto rate = nn::workload_data_cast<nn::layout_f32>(item->input[2].get_data_view());
        decltype(weights_delta) weights = nullptr;
        decltype(biases_delta) biases = nullptr;

        switch (updated_item->type)
        {
        case NN_WORK_ITEM_TYPE_CONVOLUTION:
        case NN_WORK_ITEM_TYPE_FULLY_CONNECTED:
            weights = nn::workload_data_cast<nn::layout_f32>(updated_item->parameters[0]);
            biases = nn::workload_data_cast<nn::layout_f32>(updated_item->parameters[1]);
            break;
        default:
            assert(0);
            break;
        }

        param_update_data(device->thread_pool, weights, weights_delta, (*rate)(0, 0, 0, 0, 0, 0));
        param_update_data(device->thread_pool, biases, biases_delta, (*rate)(0, 0, 0, 0, 0, 0));
    }
}
