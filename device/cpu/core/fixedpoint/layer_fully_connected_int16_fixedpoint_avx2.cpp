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
#include "device/cpu/api_internal/data_helper.h"
#include "layer_fully_connected_int16_fixedpoint_avx2.h"
#include "activations_int16_fixedpoint.h"

#include <immintrin.h>
#include <string.h>
#include <thread>
#include <vector>

// NN_CODE_UNREACHABLE signal to supporting compiler that specific location in code cannot be reached
#if defined _MSC_VER
#   define NN_UNREACHABLE_CODE __assume(0)
#endif

#if defined __GNUC__
#   if (__GNUC__ * 100 + __GNUC_MINOR__) >= 405
#       define NN_UNREACHABLE_CODE __builtin_unreachable()
#   else
#       define NN_UNREACHABLE_CODE
#   endif
#endif

#if defined __clang__
#   if __has_builtin(__builtin_unreachable)
#       define NN_UNREACHABLE_CODE __builtin_unreachable()
#   else
#       define NN_UNREACHABLE_CODE
#   endif
#endif

// SIMD width for this implementation
const auto C_simd_width = sizeof(__m256) / sizeof(uint32_t);
const auto OUT_GROUPING = 8;

namespace int16_fixedpoint
{
    namespace helper{
        template<typename T_datatype> struct datatype_to_enum;
        template<> struct datatype_to_enum<int16_t> : std::integral_constant<nn_workload_data_type_t, NN_DATATYPE_INT16>{};
        template<> struct datatype_to_enum<int32_t> : std::integral_constant<nn_workload_data_type_t, NN_DATATYPE_INT32>{};
    }

    template<typename T> inline nn_workload_data_layout_t &data_helper_layout_lookup_pnzxyq();
    template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pnzxyq<float>() { return nn::layout_t<nn::layout_pnzxyq_f32>::layout; }
    template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pnzxyq<int16_t>() { return nn::layout_t<nn::layout_pnzxyq_i16>::layout; }
    template<> inline nn_workload_data_layout_t &data_helper_layout_lookup_pnzxyq<int32_t>() { return nn::layout_t<nn::layout_pnzxyq_i32>::layout; }

    template<typename T_output_type>
    const nn_workload_data_layout_t& fully_connected_i16<T_output_type>::out_layout = data_helper_layout_lookup_pnzxyq<T_output_type>();

    template<typename T_output_type>
    const nn_workload_data_layout_t& fully_connected_i16<T_output_type>::in_layout = nn::layout_t<nn::layout_pnzxyq_i16>::layout;

    template<typename T_output_type>
    fully_connected_i16<T_output_type>::fully_connected_i16(
        size_t num_input,
        size_t num_output,
        const nn_argument_activation_fixedpoint_t &activation,
        size_t batch_size,
        nn_device_internal *device)
        : num_input(num_input),
        num_output(num_output),
        activation(activation),
        batch_size(batch_size),
        device(device)
    {
    }

    template<typename T_output_type>
    void unpack_fully_connected_callback_handle(void *void_handle) {
        auto handle = reinterpret_cast<typename fully_connected_i16<T_output_type>::request_handle *>(void_handle);
        handle->primitive->run_fully_connected_fixedpoint_work_item(handle->input_view, handle->weights, handle->biases, handle->output_view);
    }

    template<typename T_output_type>
    void fully_connected_i16<T_output_type>::forward(
        const nn::workload_data<int16_t> *input,
        const nn::workload_data<int16_t> *weights,
        const nn::workload_data<int32_t> *biases,
        nn::workload_data<T_output_type> *output) {

        auto num_hardware_threads = std::min(device->thread_pool.get_num_threads(), max_threads);

        const auto numInputNeurons = input->parent->lengths.t[NN_DATA_COORD_p] * input->parent->lengths.t[NN_DATA_COORD_z];

        const auto outputWidth = (output->view_end.t[NN_DATA_COORD_z] - output->view_begin.t[NN_DATA_COORD_z] + 1) * output->parent->lengths.t[NN_DATA_COORD_p];
        const auto outputStart = output->view_begin.t[NN_DATA_COORD_z] * output->parent->lengths.t[NN_DATA_COORD_p];

        const auto batch_size = output->parent->lengths.t[NN_DATA_COORD_n];

        int itemsGroups_per_thread = (outputWidth / OUT_GROUPING) / num_hardware_threads;
        int itemsGroups_per_thread_modulo = (outputWidth / OUT_GROUPING) % num_hardware_threads;

        // Check if we have enough data to cover all threads.
        if ((itemsGroups_per_thread == 0) || (num_hardware_threads == 1) || (batch_size == 1))
        {
            // Its tiny data - just do it singlethreaded way.
            run_fully_connected_fixedpoint_work_item(input, weights, biases, output);
        }
        else
        {
            // Full cores utilization version.
            std::vector<request_handle*> request_handles;

            auto per_thread_multiplier = OUT_GROUPING;
            if (batch_size == 1)
                per_thread_multiplier = OUT_GROUPING * C_simd_width;

            //uint32_t output_z_per_thread = itemsGroups_per_thread * per_thread_multiplier;
            //if (batch_size != 1)
            //    output_z_per_thread = itemsGroups_per_thread * OUT_GROUPING;

            uint32_t* thread_items_sums = static_cast<uint32_t*>(alloca(num_hardware_threads * sizeof(uint32_t)));

            if (thread_items_sums == nullptr) throw std::bad_alloc();

            // Distribute elements more evenly.
            auto elements_left = itemsGroups_per_thread_modulo;
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                thread_items_sums[thread_id] = itemsGroups_per_thread;
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
                thread_items_sums[thread_id] *= per_thread_multiplier;
            }

            std::vector<nn_multithreaded_request> job(num_hardware_threads);
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                request_handles.push_back(new request_handle);
            }

            // Fill slave work items.
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                const nn::workload_data<int16_t>* cpp_master_output16;
                const nn::workload_data<int32_t>* cpp_master_output32;
                bool output32 = output->parent->data_type_size == 4;
                if (output32)
                    cpp_master_output32 = reinterpret_cast<nn::workload_data<int32_t>*>(output);
                else
                    cpp_master_output16 = reinterpret_cast<nn::workload_data<int16_t>*>(output);

                auto work_begin = 0u;
                if (thread_id > 0u)
                    work_begin = thread_items_sums[thread_id - 1] / 2;

                auto work_end = thread_items_sums[thread_id] / 2 - 1;

                nn_workload_data_coords_t output_view_begin =
                {
                    0,
                    0,
                    0,
                    work_begin,
                    0,
                    0
                };

                nn_workload_data_coords_t output_view_end =
                {
                    batch_size - 1,
                    0,
                    0,
                    // Last one gets all remaining.
                    work_end,
                    1,
                    0
                };

                auto output_view = new nn::workload_data<T_output_type>(*(reinterpret_cast<nn::workload_data<T_output_type> *>(output)), output_view_begin, output_view_end);

                request_handles[thread_id]->primitive = this;
                request_handles[thread_id]->input_view = input;
                request_handles[thread_id]->weights = weights;
                request_handles[thread_id]->biases = biases;
                request_handles[thread_id]->output_view = output_view;

                job[thread_id].callback = unpack_fully_connected_callback_handle<T_output_type>;
                job[thread_id].request_handle = request_handles[thread_id];
            }

            // Wait for all sub threads.
            device->thread_pool.push_job(job);

            // Cleanup dynamic memory.
            for (auto thread_id = 0u; thread_id < num_hardware_threads; ++thread_id)
            {
                delete request_handles[thread_id]->output_view;
                delete request_handles[thread_id];
            }

            // Cleanup vectors.
            request_handles.clear();
        }
    }

    template<typename T_output_type>
    std::vector<nn_workload_data_t *> fully_connected_i16<T_output_type>::create_inputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_XBLOCKNX, T_output_type>::create(
            device,
            num_input,
            batch_size,
            block_size) };
    }

    template<typename T_output_type>
    std::vector<nn_workload_data_t *> fully_connected_i16<T_output_type>::create_outputs(bool allocate_delta)
    {
        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_XBLOCKNX, T_output_type>::create(
            device,
            num_output,
            batch_size,
            block_size) };
    }

    template<typename T_output_type>
    std::vector<nn_workload_data_t *> fully_connected_i16<T_output_type>::create_parameters(bool allocate_delta)
    {
        const uint32_t C_simd_width = sizeof(__m256) / sizeof(uint32_t);
        const uint32_t C_slice_size = 2 * C_simd_width;
        const uint32_t C_max_accumulators = (batch_size == 8) ? 13 : 2;

        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_I2O8IO, int16_t>::create(
            device, C_max_accumulators, num_input, num_output),
            nn::data_helper<NN_WORKLOAD_DATA_TAG_O, int32_t>::create(device, num_output, allocate_delta) };
    }

    template<typename T_output_type>
    void fully_connected_i16<T_output_type>::forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs)
    {
        assert(inputs.size() == 1);
        assert(outputs.size() == 1);
        assert(parameters.size() == 2);

        forward(
            reinterpret_cast<const nn::workload_data<int16_t> *>(inputs[0]),
            reinterpret_cast<const nn::workload_data<int16_t> *>(parameters[0]),
            reinterpret_cast<const nn::workload_data<int32_t> *>(parameters[1]),
            reinterpret_cast<nn::workload_data<T_output_type> *>(outputs[0]));
    }

    template<typename T_output_type>
    bool fully_connected_i16<T_output_type>::validate_input(size_t index, nn_workload_data_t *data)
    {
        // TODO: implement
        return false;
    }

    template<typename T_output_type>
    nn::workload_data<int16_t> *fully_connected_i16<T_output_type>::create_weights(const nn::data<int16_t, 2> &weights)
    {
        nn::workload_data<int16_t> *result = nullptr;

        assert(weights.dimension == 2);
        nn::data<int16_t, 2> *flow_weights;

        flow_weights = new nn::data<int16_t, 2>(static_cast<int16_t *>(weights.buffer),
            weights.size[0],
            weights.size[1]);

        //TODO: validate weight format
        nn_workload_data_layout_t layout = nn::layout_t<nn::layout_xzynpq_i16>::layout;

        const unsigned int IFMBlock = 2;
        const unsigned int OFMBlock = 8;

        nn_workload_data_coords_t size = { 1,
            IFMBlock,
            static_cast<uint32_t>(flow_weights->size[0]) / IFMBlock,
            OFMBlock,
            static_cast<uint32_t>(flow_weights->size[1]) / OFMBlock,
            1 };

        result = new nn::workload_data<std::int16_t>(size, layout);

        auto dst = static_cast<int16_t *>(result->parent->data_buffer);
        auto src = static_cast<int16_t *>(flow_weights->buffer);

        for (std::uint32_t p = 0; p < size.t[4]; ++p)
        for (std::uint32_t y = 0; y < size.t[2]; ++y)
        for (std::uint32_t z = 0; z < size.t[3]; ++z)
        for (std::uint32_t x = 0; x < size.t[1]; ++x)
            *(dst++) = src[(y * IFMBlock + x) + (p * OFMBlock + z) * flow_weights->size[0]];

        delete flow_weights;

        return result;
    }

    template<typename T_output_type>
    nn::workload_data<int16_t> *fully_connected_i16<T_output_type>::create_weights(const nn::data<int16_t, 4> &weights)
    {
        nn::workload_data<int16_t> *result = nullptr;

        assert(weights.dimension == 4);
        nn::data<int16_t, 2> *flow_weights;

        flow_weights = new nn::data<int16_t, 2>(static_cast<int16_t *>(weights.buffer),
            weights.size[0] * weights.size[1] *
            weights.size[2],
            weights.size[3]);

        //TODO: validate weight format
        nn_workload_data_layout_t layout = nn::layout_t<nn::layout_xzynpq_i16>::layout;

        const unsigned int IFMBlock = 2;
        const unsigned int OFMBlock = 8;

        nn_workload_data_coords_t size = { 1,
            IFMBlock,
            static_cast<uint32_t>(flow_weights->size[0]) / IFMBlock,
            OFMBlock,
            static_cast<uint32_t>(flow_weights->size[1]) / OFMBlock,
            1 };

        result = new nn::workload_data<std::int16_t>(size, layout);

        auto dst = static_cast<int16_t *>(result->parent->data_buffer);
        auto src = static_cast<int16_t *>(flow_weights->buffer);

        for (std::uint32_t p = 0; p < size.t[4]; ++p)
        for (std::uint32_t y = 0; y < size.t[2]; ++y)
        for (std::uint32_t z = 0; z < size.t[3]; ++z)
        for (std::uint32_t x = 0; x < size.t[1]; ++x)
            *(dst++) = src[(y * IFMBlock + x) + (p * OFMBlock + z) * flow_weights->size[0]];

        delete flow_weights;

        return result;
    }

    template<typename T_output_type>
    nn::workload_data<int32_t> *fully_connected_i16<T_output_type>::create_bias(const nn::data<int32_t, 1> &bias)
    {
        //TODO: validate bias format
        nn_workload_data_layout_t layout = nn::layout_t<nn::layout_zxynpq_i32>::layout;

        nn_workload_data_coords_t size = { 1, static_cast<uint32_t>(bias.size[0]), 1, 1, 1, 1 };
        auto result = new nn::workload_data<std::int32_t>(size, layout);
        for (auto index = 0u; index < result->get_length(1); ++index) {
            //             n, x, y, z      p  q  =                n, x,     y, z, p, q
            (*result)(0, index, 0, 0, 0, 0) = bias.at(index);
        }

        return result;
    }

    template<typename T_output_type>
    void fully_connected_i16<T_output_type>::copy_output(nn::data<int16_t, 2> &destination, const nn::workload_data<int16_t> &source)
    {
        assert(destination.size[1] == batch_size);
        assert(destination.size[0] == num_output);

        nn_workload_data_coords_t size = { static_cast<uint32_t>(batch_size), 1, 1, static_cast<uint32_t>(num_output), static_cast<uint32_t>(block_size), 1 };
        auto result = new nn::workload_data<int16_t>(size, out_layout);

        for (size_t it_batch = 0; it_batch < batch_size; ++it_batch)
            for (size_t it_output = 0; it_output < num_output; ++it_output)
                destination.at(it_output, it_batch) =
                    const_cast<nn::workload_data<int16_t> &>(source)(it_batch, 0, 0, it_output / block_size, it_output % block_size, 0);
    }

    // forward implementation
    template<class Activation, bool T_NEED_BIAS_COPY, bool FullOutGrouping>
    static inline void process_fully_connected_int16_fixedpoint_AVX2_output_b1(
        const int16_t* const input,
        const int16_t* const weights,
        typename Activation::output_type* output,
        uint32_t numInputs,
        const uint32_t numAcc,
        const int32_t* const bias,
        int8_t in_shift,
        int8_t out_shift)
    {
        if (FullOutGrouping)
        {
            __m256i sums[OUT_GROUPING];

#pragma unroll
            for (uint32_t out_it = 0; out_it < OUT_GROUPING; ++out_it)
            if (T_NEED_BIAS_COPY)
                sums[out_it] = _mm256_load_si256((__m256i*)(bias + C_simd_width * out_it));
            else
                sums[out_it] = _mm256_setzero_si256();

            for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
            {

                __m256i load_i = _mm256_set1_epi32(*(int32_t*)(input + in_it));

#pragma unroll
                for (int out_it = 0; out_it < OUT_GROUPING; ++out_it)
                {
                    __m256i load_w = _mm256_loadu_si256((__m256i*)(weights + in_it * C_simd_width + out_it * C_simd_width * numInputs));
                    __m256i tempOut = _mm256_madd_epi16(load_i, load_w);
                    sums[out_it] = _mm256_add_epi32(sums[out_it], tempOut);
                }
            }

#pragma unroll
            for (uint32_t out_it = 0; out_it < OUT_GROUPING; out_it += 2)
            {
                Activation::store_activation(output + 8 * out_it, sums[out_it + 0], sums[out_it + 1], in_shift, out_shift);
            }
        }
        else
        {
            //__m256i sums[numAcc];
            __m256i *sums = (__m256i *)_mm_malloc(numAcc*sizeof(__m256i), 64);
#pragma unroll
            for (uint32_t out_it = 0; out_it < numAcc; ++out_it)
            if (T_NEED_BIAS_COPY)
                sums[out_it] = _mm256_load_si256((__m256i*)(bias + C_simd_width * out_it));
            else
                sums[out_it] = _mm256_setzero_si256();

            for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
            {

                __m256i load_i = _mm256_set1_epi32(*(int32_t*)(input + in_it));

#pragma unroll
                for (uint32_t out_it = 0; out_it < numAcc; ++out_it)
                {
                    __m256i load_w = _mm256_stream_load_si256((__m256i*)(weights + in_it * C_simd_width + out_it * C_simd_width * numInputs));
                    __m256i tempOut = _mm256_madd_epi16(load_i, load_w);
                    sums[out_it] = _mm256_add_epi32(sums[out_it], tempOut);
                }
            }

#pragma unroll
            for (uint32_t out_it = 0; out_it < (numAcc / 2) * 2; out_it += 2)
            {
                Activation::store_activation(output + 8 * out_it, sums[out_it + 0], sums[out_it + 1], in_shift, out_shift);
            }

            if (numAcc % 2)
            {
                Activation::store_activation(output + 8 * (numAcc - 1), sums[numAcc - 1], in_shift, out_shift);
            }

            _mm_free(sums);
        }
    }

    // forward implementation
    template<class Activation, bool T_NEED_BIAS_COPY>
    static inline void process_fully_connected_int16_fixedpoint_AVX2_output_b8(
        const int16_t* const input,
        const int16_t* const weights,
        typename Activation::output_type* output,
        uint32_t numInputs,
        uint32_t numOutputs,
        const int32_t* const bias,
        int8_t in_shift,
        int8_t out_shift)
    {
        __m256i sums[OUT_GROUPING];

#pragma unroll
        for (uint32_t out_it = 0; out_it < OUT_GROUPING; ++out_it)
        if (T_NEED_BIAS_COPY)
            sums[out_it] = _mm256_set1_epi32(*(bias + out_it));
        else
            sums[out_it] = _mm256_setzero_si256();

        for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
        {
            __m256i load_i = _mm256_stream_load_si256((__m256i*)(input + in_it * 8));

#pragma unroll
            for (int out_it = 0; out_it < OUT_GROUPING; ++out_it)
            {
                __m256i load_w = _mm256_set1_epi32(*(int32_t*)(weights + in_it*OUT_GROUPING + out_it * 2));
                __m256i tempOut = _mm256_madd_epi16(load_i, load_w);
                sums[out_it] = _mm256_add_epi32(sums[out_it], tempOut);
            }
        }

#pragma unroll
        for (uint32_t out_it = 0; out_it < OUT_GROUPING; out_it += 2)
        {
            Activation::store_activation(output, sums[out_it], sums[out_it + 1], in_shift, out_shift);
            output += 16;
        }
    }

    // forward implementation
    template<class Activation, bool T_NEED_BIAS_COPY>
    static inline void process_fully_connected_int16_fixedpoint_AVX2_output_b16(
        const int16_t* const input,
        const int16_t* const weights,
        typename Activation::output_type* output,
        uint32_t numInputs,
        uint32_t numOutputs,
        const int32_t* const bias,
        int8_t in_shift,
        int8_t out_shift)
    {
        __m256i sums[2 * OUT_GROUPING];

#pragma unroll
        for (uint32_t out_it = 0; out_it < OUT_GROUPING; ++out_it)
        if (T_NEED_BIAS_COPY)
        {
            sums[2 * out_it + 0] = _mm256_set1_epi32(*(bias + out_it));
            sums[2 * out_it + 1] = _mm256_set1_epi32(*(bias + out_it));
        }
        else
        {
            sums[2 * out_it + 0] = _mm256_setzero_si256();
            sums[2 * out_it + 1] = _mm256_setzero_si256();
        }

        for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
        {
            __m256i load_iL = _mm256_stream_load_si256((__m256i*)(input + 2 * in_it * 8 + 0));
            __m256i load_iH = _mm256_stream_load_si256((__m256i*)(input + 2 * in_it * 8 + C_simd_width * 2));

#pragma unroll
            for (int out_it = 0; out_it < OUT_GROUPING; ++out_it)
            {
                __m256i load_w = _mm256_set1_epi32(*(int32_t*)(weights + in_it * OUT_GROUPING + out_it * 2));
                __m256i tempOut = _mm256_madd_epi16(load_iL, load_w);
                sums[2 * out_it + 0] = _mm256_add_epi32(sums[2 * out_it + 0], tempOut);
                tempOut = _mm256_madd_epi16(load_iH, load_w);
                sums[2 * out_it + 1] = _mm256_add_epi32(sums[2 * out_it + 1], tempOut);
            }
        }

#pragma unroll
         for (uint32_t out_it = 0; out_it < 2 * OUT_GROUPING; out_it += 4)
        {
            Activation::store_activation(output, sums[out_it + 0], sums[out_it + 2], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 1], sums[out_it + 3], in_shift, out_shift);
            output += (2 * C_simd_width);
        }
    }


    // forward implementation
    template<class Activation, bool T_NEED_BIAS_COPY>
    static inline void process_fully_connected_int16_fixedpoint_AVX2_output_b24(
        const int16_t* const input,
        const int16_t* const weights,
        typename Activation::output_type* output,
        uint32_t numInputs,
        uint32_t numOutputs,
        const int32_t* const bias,
        int8_t in_shift,
        int8_t out_shift)
    {
        __m256i sums[3 * OUT_GROUPING];

#pragma unroll
        for (uint32_t out_it = 0; out_it < OUT_GROUPING; ++out_it)
        if (T_NEED_BIAS_COPY)
        {
            sums[3 * out_it + 0] = _mm256_set1_epi32(*(bias + out_it));
            sums[3 * out_it + 1] = _mm256_set1_epi32(*(bias + out_it));
            sums[3 * out_it + 2] = _mm256_set1_epi32(*(bias + out_it));
        }
        else
        {
            sums[3 * out_it + 0] = _mm256_setzero_si256();
            sums[3 * out_it + 1] = _mm256_setzero_si256();
            sums[3 * out_it + 2] = _mm256_setzero_si256();
        }

        for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
        {
            __m256i load_iL = _mm256_stream_load_si256((__m256i*)(input + 3 * in_it * 8 + 0));
            __m256i load_iM = _mm256_stream_load_si256((__m256i*)(input + 3 * in_it * 8 + 1 * C_simd_width * 2));
            __m256i load_iH = _mm256_stream_load_si256((__m256i*)(input + 3 * in_it * 8 + 2 * C_simd_width * 2));

#pragma unroll
            for (int out_it = 0; out_it < OUT_GROUPING; ++out_it)
            {
                __m256i load_w = _mm256_set1_epi32(*(int32_t*)(weights + in_it * OUT_GROUPING + out_it * 2));

                __m256i tempOut = _mm256_madd_epi16(load_iL, load_w);
                sums[3 * out_it + 0] = _mm256_add_epi32(sums[3 * out_it + 0], tempOut);

                tempOut = _mm256_madd_epi16(load_iM, load_w);
                sums[3 * out_it + 1] = _mm256_add_epi32(sums[3 * out_it + 1], tempOut);

                tempOut = _mm256_madd_epi16(load_iH, load_w);
                sums[3 * out_it + 2] = _mm256_add_epi32(sums[3 * out_it + 2], tempOut);
            }
        }

#pragma unroll
        for (uint32_t out_it = 0; out_it < 3 * OUT_GROUPING; out_it += 6)
        {
            Activation::store_activation(output, sums[out_it + 0], sums[out_it + 3], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 1], sums[out_it + 4], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 2], sums[out_it + 5], in_shift, out_shift);
            output += (2 * C_simd_width);
        }
    }

    // forward implementation
    template<class Activation, bool T_NEED_BIAS_COPY>
    static inline void process_fully_connected_int16_fixedpoint_AVX2_output_b32(
        const int16_t* const input,
        const int16_t* const weights,
        typename Activation::output_type* output,
        uint32_t numInputs,
        uint32_t numOutputs,
        const int32_t* const bias,
        int8_t in_shift,
        int8_t out_shift)
    {
        __m256i sums[4 * OUT_GROUPING];

#pragma unroll
        for (uint32_t out_it = 0; out_it < OUT_GROUPING; ++out_it)
        if (T_NEED_BIAS_COPY)
        {
            sums[4 * out_it + 0] = _mm256_set1_epi32(*(bias + out_it));
            sums[4 * out_it + 1] = _mm256_set1_epi32(*(bias + out_it));
            sums[4 * out_it + 2] = _mm256_set1_epi32(*(bias + out_it));
            sums[4 * out_it + 3] = _mm256_set1_epi32(*(bias + out_it));
        }
        else
        {
            sums[4 * out_it + 0] = _mm256_setzero_si256();
            sums[4 * out_it + 1] = _mm256_setzero_si256();
            sums[4 * out_it + 2] = _mm256_setzero_si256();
            sums[4 * out_it + 3] = _mm256_setzero_si256();
        }

        for (uint32_t in_it = 0; in_it < numInputs; in_it += 2)
        {
            __m256i load_iLL = _mm256_stream_load_si256((__m256i*)(input + 4 * in_it * 8 + 0));
            __m256i load_iLH = _mm256_stream_load_si256((__m256i*)(input + 4 * in_it * 8 + 1 * C_simd_width * 2));
            __m256i load_iHL = _mm256_stream_load_si256((__m256i*)(input + 4 * in_it * 8 + 2 * C_simd_width * 2));
            __m256i load_iHH = _mm256_stream_load_si256((__m256i*)(input + 4 * in_it * 8 + 3 * C_simd_width * 2));

#pragma unroll
            for (int out_it = 0; out_it < OUT_GROUPING; ++out_it)
            {
                __m256i load_w = _mm256_set1_epi32(*(int32_t*)(weights + in_it * OUT_GROUPING + out_it * 2));

                __m256i tempOut = _mm256_madd_epi16(load_iLL, load_w);
                sums[4 * out_it + 0] = _mm256_add_epi32(sums[4 * out_it + 0], tempOut);

                tempOut = _mm256_madd_epi16(load_iLH, load_w);
                sums[4 * out_it + 1] = _mm256_add_epi32(sums[4 * out_it + 1], tempOut);

                tempOut = _mm256_madd_epi16(load_iHL, load_w);
                sums[4 * out_it + 2] = _mm256_add_epi32(sums[4 * out_it + 2], tempOut);

                tempOut = _mm256_madd_epi16(load_iHH, load_w);
                sums[4 * out_it + 3] = _mm256_add_epi32(sums[4 * out_it + 3], tempOut);
            }
        }

#pragma unroll
        for (uint32_t out_it = 0; out_it < 4 * OUT_GROUPING; out_it += 8)
        {
            Activation::store_activation(output, sums[out_it + 0], sums[out_it + 4], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 1], sums[out_it + 5], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 2], sums[out_it + 6], in_shift, out_shift);
            output += (2 * C_simd_width);
            Activation::store_activation(output, sums[out_it + 3], sums[out_it + 7], in_shift, out_shift);
            output += (2 * C_simd_width);
        }
    }

    template <typename OutputType> struct get_arguments;

    template <> struct get_arguments<std::int16_t> {
        static inline const nn::arguments_fully_connected_forward_i16qn_i16qn &
        get(const nn_workload_item *const &work_item) {
            return work_item->arguments.fully_connected_forward_i16qn_i16qn;
        }
    };

    template <> struct get_arguments<std::int32_t> {
        static inline const nn::arguments_fully_connected_forward_i16qn_i32qn &
        get(const nn_workload_item *const &work_item) {
            return work_item->arguments.fully_connected_forward_i16qn_i32qn;
        }
    };

    template<typename T_output_type>
    template<class ActivationType, bool T_NEED_BIAS_COPY>
    void fully_connected_i16<T_output_type>::run_fully_connected_int16_fixedpoint_work_item_internal(
        const nn::workload_data<int16_t> *input_view,
        const nn::workload_data<int16_t> *weights,
        const nn::workload_data<int32_t> *biases,
        const nn_argument_activation_fixedpoint_t activation,
        nn::workload_data<T_output_type> *output_view)
    {
        const auto numInputNeurons = input_view->parent->lengths.t[NN_DATA_COORD_z] * input_view->parent->lengths.t[NN_DATA_COORD_p];
        const auto numOutputNeurons = weights->parent->lengths.t[NN_DATA_COORD_p] * C_simd_width;
        const auto outputWidth = (output_view->view_end.t[NN_DATA_COORD_z] - output_view->view_begin.t[NN_DATA_COORD_z] + 1) * output_view->parent->lengths.t[NN_DATA_COORD_p];

        const auto outputStart = output_view->view_begin.t[NN_DATA_COORD_z] * output_view->parent->lengths.t[NN_DATA_COORD_p];

        const auto input_buffer = static_cast<int16_t*>(input_view->parent->data_buffer);
        auto output_buffer = static_cast<typename ActivationType::ImplBase::output_type*>(output_view->parent->data_buffer);
        const auto biases_buffer = static_cast<int32_t*>(biases->parent->data_buffer);
        const auto weights_buffer = static_cast<int16_t*>(weights->parent->data_buffer);

        const auto C_batch_size = output_view->parent->lengths.t[NN_DATA_COORD_n];

        auto output_ptr = output_buffer + outputStart * C_batch_size;
        auto bias_ptr = biases_buffer + outputStart;
        auto weights_ptr = weights_buffer + outputStart * numInputNeurons;

        const auto acc_fraction = activation.fractions.accumulator;
        const auto out_fraction = activation.fractions.output;
        const auto shift = acc_fraction - out_fraction;

        using namespace activations::int16_fixedpoint;

        switch (C_batch_size)
        {
        case 1:
        {
            auto NumOfFullItr = outputWidth / (C_simd_width * OUT_GROUPING);
            for (uint32_t n = 0; n < NumOfFullItr; ++n)
            {
                if (shift >= 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b1<
                        typename ActivationType::template Impl<false, ShiftDirection::Right>,
                        T_NEED_BIAS_COPY,
                        true>(input_buffer,
                            weights_ptr,
                            output_ptr,
                            numInputNeurons,
                            OUT_GROUPING,
                            bias_ptr,
                            acc_fraction,
                            out_fraction);
                else if (shift < 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b1<
                        typename ActivationType::template Impl<false, ShiftDirection::Left>,
                        T_NEED_BIAS_COPY,
                        true>(input_buffer,
                            weights_ptr,
                            output_ptr,
                            numInputNeurons,
                            OUT_GROUPING,
                            bias_ptr,
                            acc_fraction,
                            out_fraction);

                weights_ptr += numInputNeurons * C_simd_width * OUT_GROUPING;
                output_ptr += C_simd_width * OUT_GROUPING;
                bias_ptr += C_simd_width * OUT_GROUPING;
            }

            if (outputWidth % (C_simd_width * OUT_GROUPING))
            {
                auto group = (outputWidth - (outputWidth / (C_simd_width * OUT_GROUPING)) * (C_simd_width * OUT_GROUPING)) / C_simd_width;

                if (shift >= 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b1<
                        typename ActivationType::template Impl<false, ShiftDirection::Right>, T_NEED_BIAS_COPY, false>(
                        input_buffer, weights_ptr, output_ptr, numInputNeurons, group, bias_ptr, acc_fraction, out_fraction);
                else if (shift < 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b1<
                        typename ActivationType::template Impl<false, ShiftDirection::Left>, T_NEED_BIAS_COPY, false>(
                        input_buffer, weights_ptr, output_ptr, numInputNeurons, group, bias_ptr, acc_fraction, out_fraction);
            }
        }
            break;

        case 8:
        {
            for (uint32_t n = 0; n < outputWidth; n += OUT_GROUPING)
            {
                if (shift >= 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b8<
                        typename ActivationType::template Impl<true, ShiftDirection::Right>, T_NEED_BIAS_COPY>(
                        input_buffer, weights_ptr, output_ptr, numInputNeurons, outputWidth, bias_ptr,
                        acc_fraction, out_fraction);
                else if (shift < 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b8<
                        typename ActivationType::template Impl<true, ShiftDirection::Left>, T_NEED_BIAS_COPY>(
                        input_buffer, weights_ptr, output_ptr, numInputNeurons, outputWidth, bias_ptr,
                        acc_fraction, out_fraction);

                weights_ptr += numInputNeurons * OUT_GROUPING;
                output_ptr += OUT_GROUPING * C_batch_size;
                bias_ptr += OUT_GROUPING;
            }
        }
            break;

        case 16:
        {
            for (uint32_t n = 0; n < outputWidth; n += OUT_GROUPING)
            {
                if (shift >= 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b16<
                    typename ActivationType::template Impl<true, ShiftDirection::Right>, T_NEED_BIAS_COPY>(
                    input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                    acc_fraction, out_fraction);

                else if (shift < 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b16<
                    typename ActivationType::template Impl<true, ShiftDirection::Left>, T_NEED_BIAS_COPY>(
                    input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                    acc_fraction, out_fraction);

                weights_ptr += numInputNeurons * OUT_GROUPING;
                output_ptr += (OUT_GROUPING) * C_batch_size;
                bias_ptr += OUT_GROUPING;
            }
        }
            break;

        case 24:
        {
            for (uint32_t n = 0; n < outputWidth; n += OUT_GROUPING)
            {
                if (shift >= 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b24<
                    typename ActivationType::template Impl<true, ShiftDirection::Right>, T_NEED_BIAS_COPY>(
                    input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                    acc_fraction, out_fraction);
                else if (shift < 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b24<
                    typename ActivationType::template Impl<true, ShiftDirection::Left>, T_NEED_BIAS_COPY>(
                    input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                    acc_fraction, out_fraction);

                weights_ptr += numInputNeurons * OUT_GROUPING;
                output_ptr += (OUT_GROUPING)* C_batch_size;
                bias_ptr += OUT_GROUPING;
            }
        }
            break;

        case 32:
        {
            for (uint32_t n = 0; n < outputWidth; n += OUT_GROUPING)
            {
                if (shift >= 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b32<
                    typename ActivationType::template Impl<true, ShiftDirection::Right>, T_NEED_BIAS_COPY>(
                    input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                    acc_fraction, out_fraction);
                else if (shift < 0)
                    process_fully_connected_int16_fixedpoint_AVX2_output_b32<
                    typename ActivationType::template Impl<true, ShiftDirection::Left>, T_NEED_BIAS_COPY>(
                    input_buffer, weights_ptr, output_ptr, numInputNeurons, numOutputNeurons, bias_ptr,
                    acc_fraction, out_fraction);

                weights_ptr += numInputNeurons * OUT_GROUPING;
                output_ptr += (OUT_GROUPING)* C_batch_size;
                bias_ptr += OUT_GROUPING;
            }
        }
            break;

        default:
            break;
        }
    }

    template<typename T_output_type>
    void fully_connected_i16<T_output_type>::run_fully_connected_fixedpoint_work_item(
        const nn::workload_data<int16_t> *input,
        const nn::workload_data<int16_t> *weights,
        const nn::workload_data<int32_t> *biases,
        nn::workload_data<T_output_type> *output)
    {
        NN_ACTIVATION_FUNCTION function = activation.basic_arguments.function;

        bool need_bias_copy = (biases != nullptr);

        using namespace activations::int16_fixedpoint;

        if (need_bias_copy) {
            if (function == NN_ACTIVATION_FUNCTION_NONE) {
                run_fully_connected_int16_fixedpoint_work_item_internal<None<std::int32_t>, true>(input, weights, biases, activation, output);
            } else if (function == NN_ACTIVATION_FUNCTION_RELU) {
                run_fully_connected_int16_fixedpoint_work_item_internal<ReLu<std::int16_t>, true>(input, weights, biases, activation, output);
            }else if(function == NN_ACTIVATION_FUNCTION_LOGISTIC){
                run_fully_connected_int16_fixedpoint_work_item_internal<Logistic<std::int16_t>, true>(input, weights, biases, activation, output);
            }else{
                assert(false);
            }
        } else {
            if (function == NN_ACTIVATION_FUNCTION_NONE) {
                run_fully_connected_int16_fixedpoint_work_item_internal<None<std::int32_t>, false>(input, weights, biases, activation, output);
            } else if (function == NN_ACTIVATION_FUNCTION_RELU) {
                run_fully_connected_int16_fixedpoint_work_item_internal<ReLu<std::int16_t>, false>(input, weights, biases, activation, output);
            } else if (function == NN_ACTIVATION_FUNCTION_LOGISTIC) {
                run_fully_connected_int16_fixedpoint_work_item_internal<Logistic<std::int16_t>, false>(input, weights, biases, activation, output);
            }else{
                assert(false);
            }
        }
    }

    template<typename T_output_type>
    void wrapper_fully_connected_fixedpoint_work_item(nn_workload_item *const work_item, nn_device_internal* device)
    {
        nn::workload_data<int16_t>* input_view = reinterpret_cast<nn::workload_data<int16_t> *>(work_item->input[0].get_data_view());
        nn::workload_data<T_output_type>* output_view = reinterpret_cast<nn::workload_data<T_output_type> *>(work_item->output[0]);

        if (static_cast<fully_connected_i16<T_output_type> *>(work_item->primitive)->device->thread_pool.get_num_threads() > 1)
        {
            static_cast<fully_connected_i16<T_output_type> *>(work_item->primitive)
                ->forward(
                input_view,
                reinterpret_cast<nn::workload_data<int16_t> *>(work_item->arguments.fully_connected_forward_i16qn_i16qn.weights),
                reinterpret_cast<nn::workload_data<int32_t> *>(work_item->arguments.fully_connected_forward_i16qn_i16qn.biases),
                output_view);
        }
        else
        {
            static_cast<fully_connected_i16<T_output_type> *>(work_item->primitive)
                ->run_fully_connected_fixedpoint_work_item(
                input_view,
                reinterpret_cast<nn::workload_data<int16_t> *>(work_item->arguments.fully_connected_forward_i16qn_i16qn.weights),
                reinterpret_cast<nn::workload_data<int32_t> *>(work_item->arguments.fully_connected_forward_i16qn_i16qn.biases),
                output_view);
        }
    }

    void run_multithreaded_fully_connected_fixedpoint_work_item(nn_workload_item *const work_item, nn_device_internal* device)
    {
        if (work_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN)
            wrapper_fully_connected_fixedpoint_work_item<int16_t>(work_item, device);
        else if (work_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN)
            wrapper_fully_connected_fixedpoint_work_item<int32_t>(work_item, device);

    }
} // namepace device_int16

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_fully_connected_i16_create_0(
    nn_device_t *device,                        /* IDLF device handle */
    size_t num_input,                           /* number of input feature maps */
    size_t num_output,                          /* number of output feature maps */
    const nn_argument_activation_t *activation, /* struct parameterizing optional activation function */
    size_t batch_size,                          /* size of input batch */
    const nn_primitives_fully_connected_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);

    nn_primitives_fully_connected_hints_t hints_ = {};
    hints_.fixed_point_fraction_bits.accumulator = 16;
    hints_.fixed_point_fraction_bits.output = 8;

    if (hints != nullptr)
        hints_ = *hints;

    nn_argument_activation_fixedpoint_t activation_fixedpoint;
    activation_fixedpoint.basic_arguments = *activation;
    activation_fixedpoint.fractions.accumulator = hints_.fixed_point_fraction_bits.accumulator;
    activation_fixedpoint.fractions.output = hints_.fixed_point_fraction_bits.output;

    return new int16_fixedpoint::fully_connected_i16<int16_t>(
        num_input, num_output, activation_fixedpoint, batch_size, reinterpret_cast<nn_device_internal *>(device));
}

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_fully_connected_i16_i32_create_0(
    nn_device_t *device,                        /* IDLF device handle */
    size_t num_input,                           /* number of input feature maps */
    size_t num_output,                          /* number of output feature maps */
    const nn_argument_activation_t *activation, /* struct parameterizing optional activation function */
    size_t batch_size,                          /* size of input batch */
    const nn_primitives_fully_connected_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    ) {
    SET_STATUS(NN_API_STATUS_OK);

    nn_primitives_fully_connected_hints_t hints_ = {};
    hints_.fixed_point_fraction_bits.accumulator = 16;
    hints_.fixed_point_fraction_bits.output = 8;

    if (hints != nullptr)
        hints_ = *hints;

    nn_argument_activation_fixedpoint_t activation_fixedpoint;
    activation_fixedpoint.basic_arguments = *activation;
    activation_fixedpoint.fractions.accumulator = hints_.fixed_point_fraction_bits.accumulator;
    activation_fixedpoint.fractions.output = hints_.fixed_point_fraction_bits.output;

    return new int16_fixedpoint::fully_connected_i16<int32_t>(
        num_input, num_output, activation_fixedpoint, batch_size, reinterpret_cast<nn_device_internal *>(device));
}
