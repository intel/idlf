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

#include "../../common/nn_workload_data.h"
#include "../api_internal/nn_device_interface_0_internal.h"
#include "layer_arithmetic_operation.h"
#include <limits>
#include <assert.h>

namespace layer
{

    const auto C_max_block_size = 16;
    const auto C_simd_size = sizeof(__m256) / sizeof(float);

    template<class T> inline void read_wrapper(T& reg, float* data);
    template<> inline void read_wrapper<__m256>(__m256& reg, float* data) {reg = _mm256_loadu_ps(data);}
    template<> inline void read_wrapper<float>(float& reg, float* data) {reg = *data;}

    template<class T> inline void store_wrapper(T& reg, float* data);
    template<> inline void store_wrapper<__m256>(__m256& reg, float* data) {_mm256_storeu_ps(data, reg);}
    template<> inline void store_wrapper<float>(float& reg, float* data) {*data = reg;}

    template<NN_ARITHMETIC_FUNCTION function, class T> struct op_wrapper {};

    template<NN_ARITHMETIC_FUNCTION function>
    struct op_wrapper<function, __m256>
    {
        static inline void body(__m256& reg, float* data)
        {
            if (function == NN_ARITHMETIC_FUNCTION_ADDITION) reg = _mm256_add_ps(reg, _mm256_loadu_ps(data));
            if (function == NN_ARITHMETIC_FUNCTION_SUBTRACTION) reg = _mm256_sub_ps(reg, _mm256_loadu_ps(data));
            if (function == NN_ARITHMETIC_FUNCTION_MULTIPLICATION) reg = _mm256_mul_ps(reg, _mm256_loadu_ps(data));
            if (function == NN_ARITHMETIC_FUNCTION_DIVISION) reg = _mm256_div_ps(reg, _mm256_loadu_ps(data));
        }
    };

    template<NN_ARITHMETIC_FUNCTION function>
    struct op_wrapper<function, float>
    {
        static inline void body(float& reg, float* data)
        {
            if (function == NN_ARITHMETIC_FUNCTION_ADDITION) reg += *data;
            if (function == NN_ARITHMETIC_FUNCTION_SUBTRACTION) reg -= *data;
            if (function == NN_ARITHMETIC_FUNCTION_MULTIPLICATION) reg *= *data;
            if (function == NN_ARITHMETIC_FUNCTION_DIVISION) reg /= *data;
        }
    };

    template <NN_ARITHMETIC_FUNCTION function, uint32_t block_size, class T = __m256>
    inline void inner_arithmetic_processing(
        float* input,
        float* factor,
        float* output)
    {
        T acc_array[block_size];

#pragma unroll(block_size)
        for (auto acc = 0u; acc < block_size; ++acc)
        {
            read_wrapper<T>(acc_array[acc], input + sizeof(T)/sizeof(float) * acc);
            op_wrapper<function, T>::body(acc_array[acc], factor +  sizeof(T)/sizeof(float)* acc);
        }

#pragma unroll(block_size)
        for (auto acc = 0u; acc < block_size; ++acc)
        {
            store_wrapper<T>(acc_array[acc], output +  sizeof(T)/sizeof(float)* acc);
        }
    }

    template <NN_ARITHMETIC_FUNCTION T_function>
    void arithmetic_f32::process_arithmetic_operation(const nn::nn_workload_data_t<float> *input,
                                                      const nn::nn_workload_data_t<float> *factor,
                                                      nn::nn_workload_data_t<float> *output) {
        auto input_start = reinterpret_cast<float*>(input->parent->data_buffer);
        auto factor_start = reinterpret_cast<float*>(factor->parent->data_buffer);
        auto output_start = reinterpret_cast<float*>(output->parent->data_buffer);

        const auto total_image_size = output->parent->lengths.t[NN_DATA_COORD_x];

        const auto total_view_size = output->get_length(NN_DATA_COORD_x);
        const auto full_blocks = total_view_size / (C_max_block_size * C_simd_size);
        const auto partial_block = (total_view_size % (C_max_block_size * C_simd_size) / C_simd_size);
        const auto partial_subblock = total_view_size % C_simd_size;

        for (auto n = output->view_begin.t[NN_DATA_COORD_n]; n <= output->view_end.t[NN_DATA_COORD_n]; ++n)
        {
            auto input_ptr = input_start + input->view_begin.t[NN_DATA_COORD_x] + n * total_image_size;
            auto output_ptr = output_start + output->view_begin.t[NN_DATA_COORD_x] + n * total_image_size;
            auto factor_ptr = factor_start + output->view_begin.t[NN_DATA_COORD_x];

#pragma forceinline recursive
            {
                // Full blocks processing (all accumulators used).
                for (auto block = 0u; block < full_blocks; ++block)
                {
                    inner_arithmetic_processing<T_function, C_max_block_size>(input_ptr, factor_ptr, output_ptr);

                    input_ptr += C_simd_size * C_max_block_size;
                    output_ptr += C_simd_size * C_max_block_size;
                    factor_ptr += C_simd_size * C_max_block_size;
                }

                // Partial blocks processing (only part of accumulators).
                switch (partial_block)
                {
                case  0: break;
                case  1: inner_arithmetic_processing<T_function,  1>(input_ptr, factor_ptr, output_ptr); break;
                case  2: inner_arithmetic_processing<T_function,  2>(input_ptr, factor_ptr, output_ptr); break;
                case  3: inner_arithmetic_processing<T_function,  3>(input_ptr, factor_ptr, output_ptr); break;
                case  4: inner_arithmetic_processing<T_function,  4>(input_ptr, factor_ptr, output_ptr); break;
                case  5: inner_arithmetic_processing<T_function,  5>(input_ptr, factor_ptr, output_ptr); break;
                case  6: inner_arithmetic_processing<T_function,  6>(input_ptr, factor_ptr, output_ptr); break;
                case  7: inner_arithmetic_processing<T_function,  7>(input_ptr, factor_ptr, output_ptr); break;
                case  8: inner_arithmetic_processing<T_function,  8>(input_ptr, factor_ptr, output_ptr); break;
                case  9: inner_arithmetic_processing<T_function,  9>(input_ptr, factor_ptr, output_ptr); break;
                case 10: inner_arithmetic_processing<T_function, 10>(input_ptr, factor_ptr, output_ptr); break;
                case 11: inner_arithmetic_processing<T_function, 11>(input_ptr, factor_ptr, output_ptr); break;
                case 12: inner_arithmetic_processing<T_function, 12>(input_ptr, factor_ptr, output_ptr); break;
                case 13: inner_arithmetic_processing<T_function, 13>(input_ptr, factor_ptr, output_ptr); break;
                case 14: inner_arithmetic_processing<T_function, 14>(input_ptr, factor_ptr, output_ptr); break;
                case 15: inner_arithmetic_processing<T_function, 15>(input_ptr, factor_ptr, output_ptr); break;
                default:
                    NN_UNREACHABLE_CODE;
                }

                input_ptr += C_simd_size * partial_block;
                output_ptr += C_simd_size * partial_block;
                factor_ptr += C_simd_size * partial_block;

                // Processing of sub blocks (only part of one SIMD).
                switch (partial_subblock)
                {
                case  0: break;
                case  1: inner_arithmetic_processing<T_function, 1, float>(input_ptr, factor_ptr, output_ptr); break;
                case  2: inner_arithmetic_processing<T_function, 2, float>(input_ptr, factor_ptr, output_ptr); break;
                case  3: inner_arithmetic_processing<T_function, 3, float>(input_ptr, factor_ptr, output_ptr); break;
                case  4: inner_arithmetic_processing<T_function, 4, float>(input_ptr, factor_ptr, output_ptr); break;
                case  5: inner_arithmetic_processing<T_function, 5, float>(input_ptr, factor_ptr, output_ptr); break;
                case  6: inner_arithmetic_processing<T_function, 6, float>(input_ptr, factor_ptr, output_ptr); break;
                case  7: inner_arithmetic_processing<T_function, 7, float>(input_ptr, factor_ptr, output_ptr); break;
                default:
                    NN_UNREACHABLE_CODE;
                }
            }
        }
    }

    void arithmetic_f32::run_arithmetic_operation_work_item(const nn::nn_workload_data_t<float> *input,
                                                            const nn::nn_workload_data_t<float> *factor,
                                                            nn::nn_workload_data_t<float> *output) {

        switch(arithmetic_function) {
        case NN_ARITHMETIC_FUNCTION_ADDITION: 
            process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_ADDITION>(input, factor, output);
                break;
        case NN_ARITHMETIC_FUNCTION_SUBTRACTION:
            process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_SUBTRACTION>(input, factor, output);
            break;
        case NN_ARITHMETIC_FUNCTION_MULTIPLICATION:
            process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_MULTIPLICATION>(input, factor, output);
            break;
        case NN_ARITHMETIC_FUNCTION_DIVISION:
            process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_DIVISION>(input, factor, output);
            break;
        default: 
            NN_UNREACHABLE_CODE;
        }
    }

    struct arithmetic_f32_request_handle {
        arithmetic_f32 *primitive;
        const nn::nn_workload_data_t<float> *input;
        const nn::nn_workload_data_t<float> *factor;
        nn::nn_workload_data_t<float> *output;
    };

    void unpack_arithmetic_callback_handle(
        void* void_handle)
    {
        auto handle = reinterpret_cast<arithmetic_f32_request_handle *>(void_handle);
        handle->primitive->run_arithmetic_operation_work_item(handle->input, handle->factor, handle->output);
    }

    void wrapper_arithmetic_operation_work_item(nn_workload_item *const work_item, nn_device_internal *device) {
        auto primitive = static_cast<arithmetic_f32*>(work_item->primitive);
        primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->input[0]->output),
                           work_item->arguments.forward_arithmetic.factor,
                           reinterpret_cast<nn::nn_workload_data_t<float> *>(work_item->output));
    }

    void arithmetic_f32::forward(const nn::nn_workload_data_t<float> *input,
                                 const nn::nn_workload_data_t<float> *factor,
                                 nn::nn_workload_data_t<float> *output) {

        {
            auto input_length = input->get_length();
            assert(input_length.t[0] * input_length.t[1] * input_length.t[2] * input_length.t[3] ==
                   input->parent->buffer_size / input->parent->data_type_size);

            auto output_length = output->get_length();
            assert(output_length.t[0] * output_length.t[1] * output_length.t[2] * output_length.t[3] ==
                   output->parent->buffer_size / output->parent->data_type_size);
        }

        // Create flat data views.
        nn_workload_data_coords_t input_coord = {input->parent->lengths.t[NN_DATA_COORD_n],
                                                 input->parent->buffer_size / static_cast<uint32_t>(sizeof(float)) /
                                                     input->parent->lengths.t[NN_DATA_COORD_n],
                                                 1,
                                                 1,
                                                 1,
                                                 1};

        nn_workload_data_coords_t factor_coord =
        {
            1,
            factor->parent->buffer_size / static_cast<uint32_t>(sizeof(float)),
            1,
            1,
            1,
            1
        };

        nn_workload_data_coords_t output_coord =
        {
            output->parent->lengths.t[NN_DATA_COORD_n],
            output->parent->buffer_size / static_cast<uint32_t>(sizeof(float)) / output->parent->lengths.t[NN_DATA_COORD_n],
            1,
            1,
            1,
            1
        };

        nn::nn_workload_data_t<float> input_flat(input->parent->data_buffer, input_coord, input->parent->layout);
        nn::nn_workload_data_t<float> factor_flat(factor->parent->data_buffer, factor_coord, factor->parent->layout);
        nn::nn_workload_data_t<float> output_flat(output->parent->data_buffer, output_coord, output->parent->layout);

        // Split it for multi threading.
        auto num_hardware_threads = device->thread_pool.get_num_threads();
        auto item_length = output_flat.get_length(NN_DATA_COORD_x) / C_simd_size;
        auto item_length_remainder = output_flat.get_length(NN_DATA_COORD_x) % C_simd_size;

        const auto items_per_thread = item_length / num_hardware_threads;
        const auto items_modulo = item_length % num_hardware_threads;

        // Check if we have enough data to cover all threads.
        if (items_per_thread == 0 && items_modulo < 2)
        {
            // Its tiny data - just do it single threaded way.
            run_arithmetic_operation_work_item(&input_flat, &factor_flat, &output_flat);
        }
        else
        {
            // Not all threads will be used.
            if (items_per_thread == 0)
                num_hardware_threads = items_modulo;

            // Full cores utilization version.
            std::vector<arithmetic_f32_request_handle> request_handles(num_hardware_threads);
            std::vector<nn_multithreaded_request> job(num_hardware_threads);
            std::vector<const nn::nn_workload_data_t<float> *> input_views(num_hardware_threads);
            std::vector<nn::nn_workload_data_t<float> *> output_views(num_hardware_threads);

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

                thread_items_sums[thread_id] *= C_simd_size;

                if (thread_id + 1 == num_hardware_threads)
                {
                    thread_items_sums[thread_id] += item_length_remainder;
                }
            }

            // Fill slave work items.
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                const auto cpp_master_input = (&input_flat);
                const auto cpp_master_output = (&output_flat);

                auto work_begin = 0u;
                if (thread_id > 0u)
                    work_begin = thread_items_sums[thread_id - 1];

                auto work_end = thread_items_sums[thread_id] - 1;

                // Replace nn_workload_datas pointers with views.
                nn_workload_data_coords_t nn_view_begin =
                {
                    0,
                    work_begin,
                    0,
                    0,
                    0,
                    0
                };

                nn_workload_data_coords_t nn_view_end =
                {
                    cpp_master_input->get_length(NN_DATA_COORD_n) - 1,
                    work_end,
                    cpp_master_input->get_length(NN_DATA_COORD_y) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_z) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_p) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_q) - 1
                };

                input_views[thread_id] =
                    new nn::nn_workload_data_t<float>(*cpp_master_input, nn_view_begin, nn_view_end);
                output_views[thread_id] =
                    new nn::nn_workload_data_t<float>(*cpp_master_output, nn_view_begin, nn_view_end);
            }

            // Run threads.
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                request_handles[thread_id].primitive = this;
                request_handles[thread_id].input = input_views[thread_id];
                request_handles[thread_id].factor = factor;
                request_handles[thread_id].output = output_views[thread_id];

                job[thread_id].callback = unpack_arithmetic_callback_handle;
                job[thread_id].request_handle = &request_handles[thread_id];
            }

            // Wait for all sub threads.
            device->thread_pool.push_job(job);

            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id) {
                delete input_views[thread_id];
                delete output_views[thread_id];
            }
        }
    }

    arithmetic_f32 *arithmetic_f32::create(size_t image_size_x,
                                           size_t image_size_y,
                                           size_t image_size_z,
                                           NN_ARITHMETIC_FUNCTION arithmetic_function,
                                           size_t batch_size,
                                           nn_device_t *device) {
        return new arithmetic_f32(image_size_x,
                                  image_size_y,
                                  image_size_z,
                                  arithmetic_function,
                                  batch_size,
                                  reinterpret_cast<nn_device_internal *>(device));
    }

    nn::nn_workload_data_t<float> *arithmetic_f32::create_factor(const nn::data<float, 0> &factor) {
        assert(output_size_x == factor.size[0]);
        assert(output_size_y == 1 || output_size_y == factor.size[1]);
        assert(output_size_z == 1 || output_size_z == factor.size[2]);

        nn_workload_data_layout_t layout = 
            {{0, 0, 0, 0, 0, 0}, // tile in log2(size)
             {0, 0, 0, 0, 0, 0}, // alignment
             {NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q}, // ordering
             NN_DATATYPE_FLOAT};

        nn_workload_data_coords_t size = {1,
                                          static_cast<uint32_t>(output_size_x),
                                          static_cast<uint32_t>(output_size_y),
                                          static_cast<uint32_t>(output_size_z),
                                          1,
                                          1};

        auto *factor_internal = new nn::nn_workload_data_t<float>(size, layout);
        for (auto x = 0; x < output_size_x; ++x)
            for (auto y = 0; y < output_size_y; ++y)
                for (auto z = 0; z < output_size_z; ++z)
                    (*factor_internal)(0, x, y, z, 0, 0) = factor.at(x, y, z);

        return factor_internal;
    }

    arithmetic_f32::arithmetic_f32(size_t image_size_x,
                                   size_t image_size_y,
                                   size_t image_size_z,
                                   NN_ARITHMETIC_FUNCTION arithmetic_function,
                                   size_t batch_size,
                                   nn_device_internal *device)
        : primitive_zxyn_f32_base(batch_size, image_size_z, image_size_x, image_size_y, image_size_z, device),
          arithmetic_function(arithmetic_function) {}

    size_t arithmetic_f32::get_required_input_w() { return output_size_x; }

    size_t arithmetic_f32::get_required_input_h() { return output_size_y; }

    nn::nn_workload_data_t<float> *arithmetic_f32::create_output() {
        return primitive_zxyn_f32_base::create_output(0, 0, 0, 0);
    }

namespace arithmetic_f32_impl {
nn_primitive_handle_t NN_API_CALL_CONVENTION create(nn_device_t *device,
                                                    size_t image_size_x,
                                                    size_t image_size_y,
                                                    size_t image_size_z,
                                                    NN_ARITHMETIC_FUNCTION arithmetic_function,
                                                    size_t batch_size,
                                                    NN_API_STATUS *status) {
    SET_STATUS(NN_API_STATUS_OK);
    return layer::arithmetic_f32::create(
        image_size_x, image_size_y, image_size_z, arithmetic_function, batch_size, device);
}

nn_opaque_data_t *NN_API_CALL_CONVENTION
create_factor(nn_primitive_handle_t handle, const nn_data_t *factor, NN_API_STATUS *status) {
    auto primitive = static_cast<layer::arithmetic_f32 *>(handle);
    auto result = primitive->create_factor(*nn::data_cast<float, 0>(factor));
    SET_STATUS(NN_API_STATUS_OK);
    return reinterpret_cast<nn_opaque_data_t *>(result);
}
nn_event_t NN_API_CALL_CONVENTION forward_async(nn_primitive_handle_t handle,
                                                nn_opaque_data_t *input,
                                                nn_opaque_data_t *factor,
                                                nn_opaque_data_t *output,
                                                size_t dependencies_count,
                                                nn_event_t *dependencies,
                                                NN_API_STATUS *status) {
    auto primitive = static_cast<layer::arithmetic_f32 *>(handle);
    primitive->forward(reinterpret_cast<nn::nn_workload_data_t<float> *>(input),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(factor),
                       reinterpret_cast<nn::nn_workload_data_t<float> *>(output));
    SET_STATUS(NN_API_STATUS_OK);
    return {};
}
} // namespace arithmetic_f32_impl
} // namespace layer

nn_primitives_arithmetic_f32_0_t nn_primitives_arithmetic_f32_0{
    layer::arithmetic_f32_impl::create,
    layer::arithmetic_f32_impl::create_factor,
    layer::helper_zxyn_f32::create_input,
    layer::helper_zxyn_f32::map_input,
    layer::helper_zxyn_f32::validate_input,
    layer::helper_zxyn_f32::create_output,
    layer::arithmetic_f32_impl::forward_async,
    layer::helper_zxyn_f32::copy_output_async
};
