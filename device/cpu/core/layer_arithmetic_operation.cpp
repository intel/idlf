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
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"
#include "layer_arithmetic_operation.h"
#include <limits>
#include <assert.h>
#include "device/cpu/api_internal/data_helper.h"

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

    template<class T> inline void mul_by_scalar(T& reg, float data);
    template<> inline void mul_by_scalar<__m256>(__m256& reg, float data) { reg = _mm256_mul_ps(reg, _mm256_set1_ps(data)); }
    template<> inline void mul_by_scalar<float>(float& reg, float data) { reg = reg * data; }
    
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

        static inline void body(__m256& reg, float* data, float beta)
        {
            if(function == NN_ARITHMETIC_FUNCTION_ADDITION) reg = _mm256_add_ps(reg, _mm256_mul_ps(_mm256_loadu_ps(data), _mm256_set1_ps(beta)));
            if(function == NN_ARITHMETIC_FUNCTION_SUBTRACTION) reg = _mm256_sub_ps(reg, _mm256_mul_ps(_mm256_loadu_ps(data), _mm256_set1_ps(beta)));

            assert(function != NN_ARITHMETIC_FUNCTION_DIVISION);
            assert(function != NN_ARITHMETIC_FUNCTION_MULTIPLICATION);
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

        static inline void body(float &reg, float* data, float beta)
        {
            if(function == NN_ARITHMETIC_FUNCTION_ADDITION) reg += *data * beta;
            if(function == NN_ARITHMETIC_FUNCTION_SUBTRACTION) reg -= *data * beta;
        
            assert(function != NN_ARITHMETIC_FUNCTION_DIVISION);
            assert(function != NN_ARITHMETIC_FUNCTION_MULTIPLICATION);
        }
    };

    template <NN_ARITHMETIC_FUNCTION function, scalar_op_type scalar_op, uint32_t block_size, class T = __m256>
    inline void inner_arithmetic_processing(
        float* input,
        float* factor,
        float* output,
        float alpha,
        float beta,
        float gamma)
    {
        T acc_array[block_size];

#pragma unroll(block_size)
        for(auto acc = 0u; acc < block_size; ++acc)
        {
            read_wrapper<T>(acc_array[acc], input + sizeof(T) / sizeof(float) * acc);

            /* Insert additional multiplication if input has to be multiplied by a scalar alpha */
            if(scalar_op == scalar_op_type::MUL_BY_ALPHA || scalar_op == scalar_op_type::MUL_BY_ALPHA_AND_BETA)
                mul_by_scalar<T>(acc_array[acc], alpha);

            /* Insert additional multiplication if factor has to be multiplied by a scalar beta */
            if(scalar_op == scalar_op_type::MUL_BY_BETA || scalar_op == scalar_op_type::MUL_BY_ALPHA_AND_BETA)
                op_wrapper<function, T>::body(acc_array[acc], factor + sizeof(T) / sizeof(float) * acc, beta);
            else
                op_wrapper<function, T>::body(acc_array[acc], factor + sizeof(T) / sizeof(float) * acc);
       
            /* Insert additional multiplication if result of the arithmetic operation has to be multiplied by a scalar gamma */
            if(scalar_op == scalar_op_type::MUL_BY_GAMMA)
                mul_by_scalar<T>(acc_array[acc], gamma);
        }

#pragma unroll(block_size)
        for(auto acc = 0u; acc < block_size; ++acc)
        {
            store_wrapper<T>(acc_array[acc], output + sizeof(T) / sizeof(float) * acc);
        }
    }

    template <NN_ARITHMETIC_FUNCTION T_function, scalar_op_type scalar_op>
    void arithmetic_f32::process_arithmetic_operation(const nn::workload_data<float> *input,
                                                      const nn::workload_data<float> *factor,
                                                      nn::workload_data<float> *output) {
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
                    inner_arithmetic_processing<T_function, scalar_op, C_max_block_size>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma);

                    input_ptr += C_simd_size * C_max_block_size;
                    output_ptr += C_simd_size * C_max_block_size;
                    factor_ptr += C_simd_size * C_max_block_size;
                }

                // Partial blocks processing (only part of accumulators).
                switch(partial_block)
                {
                case  0: break;
                case  1: inner_arithmetic_processing<T_function, scalar_op,  1>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  2: inner_arithmetic_processing<T_function, scalar_op,  2>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  3: inner_arithmetic_processing<T_function, scalar_op,  3>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  4: inner_arithmetic_processing<T_function, scalar_op,  4>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  5: inner_arithmetic_processing<T_function, scalar_op,  5>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  6: inner_arithmetic_processing<T_function, scalar_op,  6>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  7: inner_arithmetic_processing<T_function, scalar_op,  7>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  8: inner_arithmetic_processing<T_function, scalar_op,  8>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  9: inner_arithmetic_processing<T_function, scalar_op,  9>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case 10: inner_arithmetic_processing<T_function, scalar_op, 10>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case 11: inner_arithmetic_processing<T_function, scalar_op, 11>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case 12: inner_arithmetic_processing<T_function, scalar_op, 12>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case 13: inner_arithmetic_processing<T_function, scalar_op, 13>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case 14: inner_arithmetic_processing<T_function, scalar_op, 14>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case 15: inner_arithmetic_processing<T_function, scalar_op, 15>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                default:
                    NN_UNREACHABLE_CODE;
                }
                input_ptr += C_simd_size * partial_block;
                output_ptr += C_simd_size * partial_block;
                factor_ptr += C_simd_size * partial_block;

                // Processing of sub blocks (only part of one SIMD).
                switch(partial_subblock)
                {
                case  0: break;
                case  1: inner_arithmetic_processing<T_function, scalar_op, 1, float>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  2: inner_arithmetic_processing<T_function, scalar_op, 2, float>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  3: inner_arithmetic_processing<T_function, scalar_op, 3, float>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  4: inner_arithmetic_processing<T_function, scalar_op, 4, float>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  5: inner_arithmetic_processing<T_function, scalar_op, 5, float>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  6: inner_arithmetic_processing<T_function, scalar_op, 6, float>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                case  7: inner_arithmetic_processing<T_function, scalar_op, 7, float>(input_ptr, factor_ptr, output_ptr, alpha, beta, gamma); break;
                default:
                    NN_UNREACHABLE_CODE;
                }
            }
        }
    }

    void arithmetic_f32::run_arithmetic_operation_work_item(const nn::workload_data<float> *input,
                                                            const nn::workload_data<float> *factor,
                                                            nn::workload_data<float> *output) {

        /* Examples of operation performed by arithmetic primitive
            (alpha, beta, gamma are scalars, imput_A and input_B are of workload_data type):

            (1) output = gamma * ( input_A {ARITHMETIC_FUNCTION} input_B )
                Where {ARITHMETIC_FUNCTION} is addition, subtraction, multiplication or division

            (2) output = alpha * input_A {ARITHMETIC_FUNCTION} beta * input_B
                Where {ARITHMETIC_FUNCTION} is addition or subtraction
        */

        // But some combinations don't make much sense
        assert(gamma == 1.0f || (alpha == 1.0f && beta == 1.0f));
        assert((alpha == 1.0f && beta == 1.0f) || (arithmetic_function == NN_ARITHMETIC_FUNCTION_ADDITION || arithmetic_function == NN_ARITHMETIC_FUNCTION_SUBTRACTION));

        if(gamma != 1.0f)
        {
            switch(arithmetic_function) {
            case NN_ARITHMETIC_FUNCTION_ADDITION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_ADDITION, scalar_op_type::MUL_BY_GAMMA>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_SUBTRACTION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_SUBTRACTION, scalar_op_type::MUL_BY_GAMMA>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_MULTIPLICATION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_MULTIPLICATION, scalar_op_type::MUL_BY_GAMMA>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_DIVISION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_DIVISION, scalar_op_type::MUL_BY_GAMMA>(input, factor, output);
                break;
            default: 
                NN_UNREACHABLE_CODE;
            }
        }
        else if(alpha != 1.0f && beta != 1.0f)
        {
            switch(arithmetic_function) {
            case NN_ARITHMETIC_FUNCTION_ADDITION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_ADDITION, scalar_op_type::MUL_BY_ALPHA_AND_BETA>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_SUBTRACTION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_SUBTRACTION, scalar_op_type::MUL_BY_ALPHA_AND_BETA>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_MULTIPLICATION:
            case NN_ARITHMETIC_FUNCTION_DIVISION:
                throw std::invalid_argument("operation not supported");
            default: 
                NN_UNREACHABLE_CODE;
            }
        }
        else if(alpha != 1.0f)
        {
            switch(arithmetic_function) {
            case NN_ARITHMETIC_FUNCTION_ADDITION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_ADDITION, scalar_op_type::MUL_BY_ALPHA>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_SUBTRACTION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_SUBTRACTION, scalar_op_type::MUL_BY_ALPHA>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_MULTIPLICATION:
            case NN_ARITHMETIC_FUNCTION_DIVISION:
                throw std::invalid_argument("operation not supported");
            default:
                NN_UNREACHABLE_CODE;
            }
        }
        else if(beta != 1.0f)
        {
            switch(arithmetic_function) {
            case NN_ARITHMETIC_FUNCTION_ADDITION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_ADDITION, scalar_op_type::MUL_BY_BETA>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_SUBTRACTION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_SUBTRACTION, scalar_op_type::MUL_BY_BETA>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_MULTIPLICATION:
            case NN_ARITHMETIC_FUNCTION_DIVISION:
                throw std::invalid_argument("operation not supported");
            default:
                NN_UNREACHABLE_CODE;
            }
        }
        else
        {
            switch(arithmetic_function) {
            case NN_ARITHMETIC_FUNCTION_ADDITION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_ADDITION, scalar_op_type::NO_SCALAR_OPERATION>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_SUBTRACTION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_SUBTRACTION, scalar_op_type::NO_SCALAR_OPERATION>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_MULTIPLICATION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_MULTIPLICATION, scalar_op_type::NO_SCALAR_OPERATION>(input, factor, output);
                break;
            case NN_ARITHMETIC_FUNCTION_DIVISION:
                process_arithmetic_operation<NN_ARITHMETIC_FUNCTION_DIVISION, scalar_op_type::NO_SCALAR_OPERATION>(input, factor, output);
                break;
            default:
                NN_UNREACHABLE_CODE;
            }
        }
    }

    struct arithmetic_f32_request_handle {
        arithmetic_f32 *primitive;
        const nn::workload_data<float> *input;
        const nn::workload_data<float> *factor;
        nn::workload_data<float> *output;
    };

    void unpack_arithmetic_callback_handle(
        void* void_handle)
    {
        auto handle = reinterpret_cast<arithmetic_f32_request_handle *>(void_handle);
        handle->primitive->run_arithmetic_operation_work_item(handle->input, handle->factor, handle->output);
    }

    void arithmetic_f32::forward(const nn::workload_data<float> *input,
                                 const nn::workload_data<float> *factor,
                                 nn::workload_data<float> *output) {

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

        nn::workload_data<float> input_flat(
            NN_WORKLOAD_DATA_TAG_UNKNOWN, input->parent->data_buffer, input_coord, input->parent->layout);
        nn::workload_data<float> factor_flat(
            NN_WORKLOAD_DATA_TAG_UNKNOWN, factor->parent->data_buffer, factor_coord, factor->parent->layout);
        nn::workload_data<float> output_flat(
            NN_WORKLOAD_DATA_TAG_UNKNOWN, output->parent->data_buffer, output_coord, output->parent->layout);

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
            std::vector<const nn::workload_data<float> *> input_views(num_hardware_threads);
            std::vector<nn::workload_data<float> *> output_views(num_hardware_threads);

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
                    new nn::workload_data<float>(*cpp_master_input, nn_view_begin, nn_view_end);
                output_views[thread_id] =
                    new nn::workload_data<float>(*cpp_master_output, nn_view_begin, nn_view_end);
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

    void arithmetic_f32::forward(const std::vector<const nn_workload_data_t *> &inputs,
                                 const std::vector<const nn_workload_data_t *> &parameters,
                                 const std::vector<nn_workload_data_t *> &outputs) {
        assert(inputs.size() == 1);
        assert(parameters.size() == 1);
        assert(outputs.size() == 1);

        forward(reinterpret_cast<const nn::workload_data<float> *>(inputs[0]),
                reinterpret_cast<const nn::workload_data<float> *>(parameters[0]),
                reinterpret_cast<nn::workload_data<float> *>(outputs[0]));
    }

    arithmetic_f32::arithmetic_f32(size_t image_size_x,
                                   size_t image_size_y,
                                   size_t image_size_z,
                                   NN_ARITHMETIC_FUNCTION arithmetic_function,
                                   size_t batch_size,
                                   nn_device_internal *device)
        : primitive_zxyn_f32_base(
              batch_size, image_size_z, image_size_x, image_size_y, image_size_z, 0, 0, 0, 0, device),
              arithmetic_function(arithmetic_function), alpha(1.0f), beta(1.0f), gamma(1.0f) {}

    arithmetic_f32::arithmetic_f32(size_t image_size_x,
                                   size_t image_size_y,
                                   size_t image_size_z,
                                   NN_ARITHMETIC_FUNCTION arithmetic_function,
                                   size_t batch_size,
                                   float alpha,
                                   float beta,
                                   float gamma,
                                   nn_device_internal *device)
        : primitive_zxyn_f32_base(
              batch_size, image_size_z, image_size_x, image_size_y, image_size_z, 0, 0, 0, 0, device),
              arithmetic_function(arithmetic_function), alpha(alpha), beta(beta), gamma(gamma) {}

    size_t arithmetic_f32::get_required_input_w() { return output_size_x; }

    size_t arithmetic_f32::get_required_input_h() { return output_size_y; }

    std::vector<nn_workload_data_t *> arithmetic_f32::create_parameters(bool allocate_delta) {
        return {nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXY, float>::create(
            device, output_size_x, output_size_y, output_size_z, allocate_delta)};
    }

    bool arithmetic_f32::validate_input(size_t index, nn_workload_data_t *data) {
        switch (index) {
        case 0:
            return nn::data_helper<NN_WORKLOAD_DATA_TAG_ZXYN, float>::validate<false>(
                data, get_required_input_w(), get_required_input_h(), input_size_z, batch_size, 0, 0, 0, 0);
        }

        throw std::invalid_argument("index out of range");
    }
} // namespace layer

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_arithmetic_f32_create_0(nn_device_t *device,
                                                    size_t image_size_x,
                                                    size_t image_size_y,
                                                    size_t image_size_z,
                                                    NN_ARITHMETIC_FUNCTION arithmetic_function,
                                                    size_t batch_size,
                                                    NN_API_STATUS *status) {
    SET_STATUS(NN_API_STATUS_OK);
    return new layer::arithmetic_f32(image_size_x,
                                     image_size_y,
                                     image_size_z,
                                     arithmetic_function,
                                     batch_size,
                                     reinterpret_cast<nn_device_internal *>(device));
}
