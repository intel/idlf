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
#include "layer_convolution_pooling_int16_fixedpoint_avx2.h"
#include "layer_convolution_int16_fixedpoint_avx2.h"

#include <immintrin.h>
#include <assert.h>
#include <string.h>
#include <algorithm>
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

namespace int16_fixedpoint
{
    convolution_pooling_i16::convolution_pooling_i16(
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
        nn_device_internal *device)
        : primitive_z_block_xyz_i16_base(
            batch_size,
            num_input,
            output_w,
            output_h,
            num_output,
            output_padding_left,
            output_padding_right,
            output_padding_top,
            output_padding_bottom,
            device),
        kernel_w(kernel_w),
        kernel_h(kernel_h),
        padding(NN_PADDING_MODE_DATA_OR_ZERO),
        center_offset_x(center_offset_x),
        center_offset_y(center_offset_y),
        stride_x(stride_x),
        stride_y(stride_y),
        activation(activation) {}

    void convolution_pooling_i16::forward(
        const nn::workload_data<int16_t> *input_buffer,
        const nn::workload_data<int16_t> *weights_buffer,
        const nn::workload_data<int32_t> *bias_buffer,
        nn::workload_data<int16_t> *output_buffer)
    {
        const auto threadpool_size = device->thread_pool.get_num_threads();

        const auto OFMBlock = weights_buffer->parent->lengths.t[NN_DATA_COORD_p];
        const auto ofm_out_block_size = output_buffer->parent->lengths.t[NN_DATA_COORD_p];

        const auto num_output_feature_maps =
            (output_buffer->view_end.t[NN_DATA_COORD_z] - output_buffer->view_begin.t[NN_DATA_COORD_z] + 1) *
            ofm_out_block_size;

        const auto batch_size = output_buffer->parent->lengths.t[NN_DATA_COORD_n];

        const auto ofm_group_size = OFMBlock;

        const auto ofm_groups_per_batch = num_output_feature_maps / ofm_group_size;

        const auto task_count = batch_size * ofm_groups_per_batch;

        auto& input = input_buffer;

        // Check if we have enough data to cover all threads.
        if (threadpool_size < 2 || task_count < 2)
        {
            // Its tiny data - just do it singlethreaded way.
            run_convolve_maxpool2x2_fixedpoint(input_buffer, weights_buffer, bias_buffer, output_buffer);
        }
        else
        {
            std::vector<nn::workload_data<int16_t> *> input_views;
            std::vector<nn::workload_data<int16_t> *> output_views;
            std::vector<nn::workload_data<int16_t> *> weights_views;
            std::vector<nn::workload_data<int32_t> *> biases_views;

            // Fill slave work items.
            for (auto it_ofm_group = 0u; it_ofm_group < ofm_groups_per_batch; ++it_ofm_group) {
                for (auto it_batch = 0u; it_batch < batch_size; ++it_batch) {
                    const auto cpp_master_input = input_buffer;
                    const auto cpp_master_output = output_buffer;
                    const auto cpp_master_weights = weights_buffer;

                    auto work_begin_batch = it_batch;
                    auto work_begin_ofm_out_block = it_ofm_group * ofm_group_size / ofm_out_block_size;

                    auto work_end_batch = it_batch + 1;
                    auto work_end_ofm_out_block = work_begin_ofm_out_block + ofm_group_size / ofm_out_block_size;

                    nn_workload_data_coords_t output_view_begin(
                        work_begin_batch,
                        0,
                        0,
                        work_begin_ofm_out_block,
                        0,
                        0
                    );

                    nn_workload_data_coords_t output_view_end(
                        work_end_batch - 1,
                        cpp_master_output->get_length(NN_DATA_COORD_x) - 1,
                        cpp_master_output->get_length(NN_DATA_COORD_y) - 1,
                        work_end_ofm_out_block - 1,
                        cpp_master_output->get_length(NN_DATA_COORD_p) - 1,
                        0
                    );

                    nn_workload_data_coords_t input_view_begin(
                        work_begin_batch,
                        0,
                        0,
                        0,
                        0,
                        0
                    );

                    nn_workload_data_coords_t input_view_end(
                        work_end_batch - 1,
                        cpp_master_input->get_length(NN_DATA_COORD_x) - 1,
                        cpp_master_input->get_length(NN_DATA_COORD_y) - 1,
                        cpp_master_input->get_length(NN_DATA_COORD_z) - 1,
                        cpp_master_input->get_length(NN_DATA_COORD_p) - 1,
                        cpp_master_input->get_length(NN_DATA_COORD_q) - 1
                    );

                    nn_workload_data_coords_t weights_view_begin(
                        0,
                        0,
                        0,
                        0,
                        0,
                        work_begin_ofm_out_block * ofm_out_block_size / ofm_group_size
                    );

                    nn_workload_data_coords_t weights_view_end(
                        cpp_master_weights->get_length(NN_DATA_COORD_n) - 1,
                        cpp_master_weights->get_length(NN_DATA_COORD_x) - 1,
                        cpp_master_weights->get_length(NN_DATA_COORD_y) - 1,
                        cpp_master_weights->get_length(NN_DATA_COORD_z) - 1,
                        cpp_master_weights->get_length(NN_DATA_COORD_p) - 1,
                        work_end_ofm_out_block * ofm_out_block_size / ofm_group_size - 1
                    );

                    input_views.push_back(new nn::workload_data<int16_t>(
                        *(input_buffer),
                        input_view_begin,
                        input_view_end));

                    output_views.push_back(new nn::workload_data<int16_t>(
                        *(output_buffer),
                        output_view_begin,
                        output_view_end));

                    weights_views.push_back(new nn::workload_data<int16_t>(
                        *(weights_buffer),
                        weights_view_begin,
                        weights_view_end));

                    // Use biases.
                    if (bias_buffer != nullptr)
                    {
                        const auto cpp_master_biases = bias_buffer;

                        nn_workload_data_coords_t bias_view_begin(
                            0,
                            0,
                            0,
                            work_begin_ofm_out_block * ofm_out_block_size,
                            0,
                            0
                        );
                        nn_workload_data_coords_t bias_view_end (
                            cpp_master_biases->get_length(NN_DATA_COORD_n) - 1,
                            cpp_master_biases->get_length(NN_DATA_COORD_x) - 1,
                            cpp_master_biases->get_length(NN_DATA_COORD_y) - 1,
                            work_end_ofm_out_block * ofm_out_block_size - 1,
                            cpp_master_biases->get_length(NN_DATA_COORD_p) - 1,
                            cpp_master_biases->get_length(NN_DATA_COORD_q) - 1
                        );

                        biases_views.push_back(new nn::workload_data<int32_t>(
                            *(bias_buffer),
                            bias_view_begin,
                            bias_view_end));
                    }
                }
            }

            std::vector<nn_multithreaded_request> jobs(task_count);
            std::vector<request_handle> request_handles(task_count);

            for (auto it_task = 0; it_task < task_count; ++it_task)
            {
                request_handles[it_task].primitive = this;
                request_handles[it_task].input_view = input_views[it_task];
                request_handles[it_task].output_view = output_views[it_task];
                request_handles[it_task].weights_view = weights_views[it_task];
                request_handles[it_task].biases_view = biases_views[it_task];

                jobs[it_task].callback = unpack_convolve_pooling_fixedpoint_callback_handle;
                jobs[it_task].request_handle = &request_handles[it_task];
            }

            // Wait for all sub threads.
            device->thread_pool.push_job(jobs);

            for (auto it_task = 0; it_task < task_count; ++it_task)
            {
                delete input_views[it_task];
                delete output_views[it_task];
                delete weights_views[it_task];
                delete biases_views[it_task];
            }
        }
    }

    void convolution_pooling_i16::forward(const std::vector<const nn_workload_data_t *> &inputs, const std::vector<const nn_workload_data_t *> &parameters, const std::vector<nn_workload_data_t *> &outputs)
    {
        assert(inputs.size() == 1);
        assert(outputs.size() == 1);
        assert(parameters.size() == 2);

        forward(
            reinterpret_cast<const nn::workload_data<int16_t> *>(inputs[0]),
            reinterpret_cast<const nn::workload_data<int16_t> *>(parameters[0]),
            reinterpret_cast<const nn::workload_data<int32_t> *>(parameters[1]),
            reinterpret_cast<nn::workload_data<int16_t> *>(outputs[0]));
    }

    std::vector<nn_workload_data_t *> convolution_pooling_i16::create_inputs(bool allocate_delta) {

        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN, int16_t>::create(
            device,
            get_required_input_w() - kernel_w + 1,
            get_required_input_h() - kernel_h + 1,
            input_size_z,
            batch_size,
            block_size,
            center_offset_x,
            kernel_w - center_offset_x - 1,
            center_offset_y,
            kernel_h - center_offset_y - 1) };
    }

    std::vector<nn_workload_data_t *> convolution_pooling_i16::create_parameters(bool allocate_delta)
    {
        const uint32_t C_simd_width = sizeof(__m256) / sizeof(uint32_t);
        const uint32_t C_slice_size = 2 * C_simd_width;

        return{ nn::data_helper<NN_WORKLOAD_DATA_TAG_I2O32IXYO, int16_t>::create(
            device, C_slice_size, kernel_w, kernel_h, input_size_z, output_size_z),
            nn::data_helper<NN_WORKLOAD_DATA_TAG_O, int32_t>::create(device, output_size_z, allocate_delta) };
    }

    nn::workload_data<int16_t> *convolution_pooling_i16::create_weights(const nn::data<int16_t, 4> &weights)
    {
        nn::workload_data<int16_t> *result = nullptr;

        assert(weights.dimension == 4);

        nn::data<int16_t, 4> *flow_weights = new nn::data<int16_t, 4>(static_cast<int16_t *>(weights.buffer),
            weights.size[0],
            weights.size[1],
            weights.size[2],
            weights.size[3]);

        //TODO: validate weight format
        nn_workload_data_layout_t layout = nn::workload_data<int16_t>::layout.ypznxq;

        const unsigned int OFMpBlock = 2;
        const unsigned int OFMBlock = 32;

        uint32_t i0 = static_cast<uint32_t>(weights.size[0]);
        uint32_t i1 = static_cast<uint32_t>(weights.size[1]);
        uint32_t i2 = static_cast<uint32_t>(weights.size[2]);
        uint32_t i3 = static_cast<uint32_t>(weights.size[3]);
        uint32_t i4 = OFMpBlock;
        uint32_t i5 = OFMBlock;

        const nn_workload_data_coords_t size(static_cast<uint32_t>(weights.size[0]),
            static_cast<uint32_t>(weights.size[1]),
            OFMpBlock,
            static_cast<uint32_t>(weights.size[2] - 1) / OFMpBlock + 1,
            OFMBlock,
            static_cast<uint32_t>(weights.size[3] - 1) / OFMBlock + 1 );

        result = new nn::workload_data<std::int16_t>(size, layout);

        auto dst = static_cast<int16_t *>(result->parent->data_buffer);
        auto src = static_cast<int16_t *>(weights.buffer);

        for (auto q = 0u; q < size.t[5]; ++q)
        for (auto x = 0u; x < size.t[1]; ++x)
        for (auto n = 0u; n < size.t[0]; ++n)
        for (auto z = 0u; z < size.t[3]; ++z)
        for (auto p = 0u; p < size.t[4]; ++p)
        for (auto y = 0u; y < size.t[2]; ++y)
            *(dst++) = (z * OFMpBlock + y < weights.size[2])
            ? src[n + weights.size[0] * (x + weights.size[1] * ((z * OFMpBlock + y) + flow_weights->size[2] * (q * OFMBlock + p)))]
            : 0;

        return result;
    }

    nn::workload_data<int32_t>* convolution_pooling_i16::create_bias(const nn::data<int32_t, 1> &bias)
    {
        //TODO: validate bias format
        nn_workload_data_layout_t layout = nn::workload_data<int32_t>::layout.zxynpq;
        nn_workload_data_coords_t size( 1, 1, 1, static_cast<uint32_t>(bias.size[0]), 1, 1 );
        auto result = new nn::workload_data<int32_t>(size, layout);
        for (auto index = 0u; index < result->get_length(3); ++index) {
            //             n, x, y, z      p  q  =                n, x,     y, z, p, q
            (*result)(0, 0, 0, index, 0, 0) = bias.at(index);
        }

        return result;
    }

    inline void AddNoBias(__m256i & val, void * addr)
    {
        val = _mm256_setzero_si256();
    }

    inline void AddBias16(__m256i & val, void * addr)
    {
        val = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i *)(addr)));
    }
    inline void AddBias32(__m256i & val, void * addr)
    {
        val = _mm256_load_si256((__m256i *)(addr));
    }

    inline void Store16_wRELu_shiftR(void * addr, __m256i val, uint8_t shift)
    {
        _mm_stream_si128((__m128i *)(addr), _mm_max_epi16(_mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(_mm256_srai_epi32(val, shift), _mm256_setzero_si256()), 0xd8)), _mm_setzero_si128()));
    }

    inline void Store16_wRELu_shiftL(void * addr, __m256i val, uint8_t shift)
    {
        _mm_stream_si128((__m128i *)(addr), _mm_max_epi16(_mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(_mm256_slli_epi32(val, -shift), _mm256_setzero_si256()), 0xd8)), _mm_setzero_si128()));
    }

    inline void Store16_woActiv_shiftR(void * addr, __m256i val, uint8_t shift)
    {
        _mm_stream_si128((__m128i *)(addr), _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(_mm256_srai_epi32(val, shift), _mm256_setzero_si256()), 0xd8)));
    }

    inline void Store16_woActiv_shiftL(void * addr, __m256i val, uint8_t shift)
    {
        _mm_stream_si128((__m128i *)(addr), _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packs_epi32(_mm256_slli_epi32(val, -shift), _mm256_setzero_si256()), 0xd8)));
    }

template <const int T_input_z_block_size,
    const int T_compute_ofm_block_size,
    const int T_compute_x_block_size,
    const int T_compute_y_block_size,
    const int T_compute_x_sub_block_size,
    const int T_compute_y_sub_block_size,
    FActiveShift FStoreActiv,
    FBias FBias>
    void NN_ConvolveAVX2_Pool2x2_INT16_fixedpoint(
    convolution_pooling_i16 *primitive,
    const nn::workload_data<int16_t> *input_view,
    const nn::workload_data<int16_t> *weights_buffer,
    const nn::workload_data<int32_t> *bias_buffer,
    nn::workload_data<int16_t> *output_view)
{
        const size_t pool_size_y_2 = 2;
        const size_t pool_size_x_2 = 2;
        const size_t ifm_sub_block_size = 2;
        const size_t output_z_block_size = 16;

        const size_t batch_window_size =
            output_view->view_end.t[NN_DATA_COORD_n] - output_view->view_begin.t[NN_DATA_COORD_n] + 1;
        const size_t batch_window_start = output_view->view_begin.t[NN_DATA_COORD_n];

        // outputs
        assert(output_view->parent->lengths.t[NN_DATA_COORD_p] == output_z_block_size);
        const size_t output_size_x = output_view->parent->lengths.t[NN_DATA_COORD_x],
            output_stride_x = output_z_block_size;
        const size_t output_size_y = output_view->parent->lengths.t[NN_DATA_COORD_y],
            output_stride_y = output_stride_x * output_size_x;
        const size_t output_size_z = output_view->parent->lengths.t[NN_DATA_COORD_z] * output_z_block_size,
            output_stride_z_block = output_stride_y * output_size_y;
        const size_t output_stride_batch = output_size_x * output_size_y * output_size_z;

        const size_t output_window_start_x = output_view->view_begin.t[NN_DATA_COORD_x],
            output_window_size_x = output_view->view_end.t[NN_DATA_COORD_x] - output_window_start_x + 1;
        const size_t output_window_start_y = output_view->view_begin.t[NN_DATA_COORD_y],
            output_window_size_y = output_view->view_end.t[NN_DATA_COORD_y] - output_window_start_y + 1;
        const size_t output_window_start_z = output_view->view_begin.t[NN_DATA_COORD_z],
            output_window_size_z =
            (output_view->view_end.t[NN_DATA_COORD_z] - output_window_start_z + 1) * output_z_block_size;

        // inputs
        if (input_view->parent->lengths.t[NN_DATA_COORD_p] != T_input_z_block_size)
            assert(input_view->parent->lengths.t[NN_DATA_COORD_p] == T_input_z_block_size);

        const size_t input_size_x = input_view->parent->lengths.t[NN_DATA_COORD_x],
            input_stride_x = T_input_z_block_size;
        const size_t input_size_y = input_view->parent->lengths.t[NN_DATA_COORD_y],
            input_stride_y = input_stride_x * input_size_x;
        const size_t input_size_z = input_view->parent->lengths.t[NN_DATA_COORD_z] * T_input_z_block_size,
            input_stride_z = input_stride_y * input_size_y;
        const size_t input_stride_batch = input_size_x * input_size_y * input_size_z;

        const size_t input_start_x = input_view->view_begin.t[NN_DATA_COORD_x];
        const size_t input_start_y = input_view->view_begin.t[NN_DATA_COORD_y];
        const size_t input_start_z = input_view->view_begin.t[NN_DATA_COORD_z];

        // weights
        assert(weights_buffer->parent->lengths.t[NN_DATA_COORD_p] == T_compute_ofm_block_size);
        assert(weights_buffer->parent->lengths.t[NN_DATA_COORD_y] == ifm_sub_block_size);
        const size_t weights_start_ofm = weights_buffer->view_begin.t[NN_DATA_COORD_q];
        const size_t weights_stride_ifm = ifm_sub_block_size * T_compute_ofm_block_size;
        const size_t weights_stride_x = weights_stride_ifm * weights_buffer->parent->lengths.t[NN_DATA_COORD_z];
        const size_t weights_size_x = weights_buffer->parent->lengths.t[NN_DATA_COORD_n];
        const size_t weights_stride_y = weights_stride_x * weights_buffer->parent->lengths.t[NN_DATA_COORD_n];
        const size_t weights_size_y = weights_buffer->parent->lengths.t[NN_DATA_COORD_x];
        const size_t weights_stride_ofm = weights_stride_y * weights_buffer->parent->lengths.t[NN_DATA_COORD_x];

        // bias
        const size_t bias_start = bias_buffer->view_begin.t[NN_DATA_COORD_z];

        for (uint32_t batch_it = 0; batch_it < batch_window_size; ++batch_it)
        {
            int16_t *output_window = (int16_t *)output_view->parent->data_buffer
                + output_window_start_x * output_stride_x
                + output_window_start_y * output_stride_y
                + output_window_start_z * output_stride_z_block
                + (batch_window_start + batch_it) * output_stride_batch;

            int16_t *input_window = (int16_t *)input_view->parent->data_buffer
                + input_start_x * input_stride_x
                + input_start_y * input_stride_y
                + input_start_z * input_stride_z
                + (batch_window_start + batch_it) * input_stride_batch;

            int16_t* weights_window = (int16_t*)weights_buffer->parent->data_buffer
                + weights_stride_ofm * weights_start_ofm;

            int32_t* bias_window = (int32_t*)bias_buffer->parent->data_buffer
                + bias_start;

            const size_t center_x = primitive->center_offset_x;
            const size_t center_y = primitive->center_offset_y;

            const size_t kernel_stride_x = primitive->stride_x;
            const size_t kernel_stride_y = primitive->stride_y;

            const size_t shift = primitive->activation.fractions.accumulator - primitive->activation.fractions.output;

            //it is assumed that inputs and kernal layout is ready
            for (size_t BlockOffsetY = 0; BlockOffsetY < output_window_size_y; BlockOffsetY += T_compute_y_block_size / pool_size_y_2)
            {
                for (size_t BlockOffsetX = 0; BlockOffsetX < output_window_size_x; BlockOffsetX += T_compute_x_block_size / pool_size_x_2)
                {
                    for (size_t BlockOffsetOFM = 0; BlockOffsetOFM < output_window_size_z; BlockOffsetOFM += T_compute_ofm_block_size) {
                        __m256i OutBlock[T_compute_y_block_size][T_compute_x_block_size][T_compute_ofm_block_size / 8];
                        for (size_t OBlockYItr = 0; OBlockYItr < T_compute_y_block_size; OBlockYItr += 1)
                        {
#pragma unroll
                            for (size_t OBlockXItr = 0; OBlockXItr < T_compute_x_block_size; OBlockXItr += 1)
                            {
#pragma unroll
                                for (size_t OFMItr = 0; OFMItr < T_compute_ofm_block_size / 8; ++OFMItr)
                                {
                                    FBias(OutBlock[OBlockYItr][OBlockXItr][OFMItr],
                                        (__m128i *)(bias_window + BlockOffsetOFM + OFMItr * 8));
                                }
                            }
                        }

                        for (size_t ifm_block_offset = 0; ifm_block_offset < input_size_z; ifm_block_offset += T_input_z_block_size) {
                            for (size_t KernelYItr = 0; KernelYItr < weights_size_y; KernelYItr += 1)
                            {
                                for (size_t KernelXItr = 0; KernelXItr < weights_size_x; KernelXItr += 1)
                                {
#pragma unroll
                                    for (size_t OBlockYItr = 0; OBlockYItr < T_compute_y_block_size; OBlockYItr += T_compute_y_sub_block_size)
                                    {
#pragma unroll
                                        for (size_t OBlockXItr = 0; OBlockXItr < T_compute_x_block_size; OBlockXItr += T_compute_x_sub_block_size)
                                        {
#pragma unroll (T_compute_y_sub_block_size)
                                            for (size_t OpBlockYItr = 0; OpBlockYItr < T_compute_y_sub_block_size; OpBlockYItr += 1)
                                            {
#pragma unroll (T_compute_x_sub_block_size)
                                                for (size_t OpBlockXItr = 0; OpBlockXItr < T_compute_x_sub_block_size; OpBlockXItr += 1)
                                                {
                                                    __m256i vi[T_input_z_block_size / ifm_sub_block_size];
#pragma unroll (T_input_z_block_size / ifm_sub_block_size)
                                                    for (size_t IFMSubItr = 0; IFMSubItr < T_input_z_block_size / ifm_sub_block_size; IFMSubItr += 1)
                                                    {
                                                        vi[IFMSubItr] = _mm256_set1_epi32(*(int32_t *)(input_window
                                                            + ((BlockOffsetX*pool_size_x_2 + OBlockXItr + OpBlockXItr - center_x) * kernel_stride_x + KernelXItr) * input_stride_x
                                                            + ((BlockOffsetY*pool_size_y_2 + OBlockYItr + OpBlockYItr - center_y) * kernel_stride_y + KernelYItr) * input_stride_y
                                                            + (ifm_block_offset) / T_input_z_block_size * input_stride_z
                                                            + IFMSubItr * ifm_sub_block_size));
                                                    }
#pragma unroll (T_input_z_block_size / ifm_sub_block_size)
                                                    for (size_t IFMSubItr = 0; IFMSubItr < T_input_z_block_size / ifm_sub_block_size; IFMSubItr += 1)
                                                    {
#pragma unroll (T_compute_ofm_block_size / 8)
                                                        for (size_t OFMItr = 0; OFMItr < T_compute_ofm_block_size / 8; OFMItr += 1)
                                                        {
                                                            __m256i vw = _mm256_load_si256((__m256i *)(weights_window
                                                                + ifm_sub_block_size * 8 * OFMItr
                                                                + (ifm_block_offset + IFMSubItr * ifm_sub_block_size) / ifm_sub_block_size * weights_stride_ifm
                                                                + (BlockOffsetOFM) / T_compute_ofm_block_size * weights_stride_ofm
                                                                + KernelXItr * weights_stride_x
                                                                + KernelYItr * weights_stride_y));

                                                            __m256i tmp =
                                                                _mm256_madd_epi16(
                                                                vw
                                                                , vi[IFMSubItr]);
                                                            OutBlock[OBlockYItr + OpBlockYItr][OBlockXItr + OpBlockXItr][OFMItr] =
                                                                _mm256_add_epi32(OutBlock[OBlockYItr + OpBlockYItr][OBlockXItr + OpBlockXItr][OFMItr], tmp);

                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        for (size_t OBlockYItr = 0; OBlockYItr < T_compute_y_block_size; OBlockYItr += pool_size_y_2)
                        {
                            for (size_t OBlockXItr = 0; OBlockXItr < T_compute_x_block_size; OBlockXItr += pool_size_x_2)
                            {
                                for (size_t OFMItr = 0; OFMItr < T_compute_ofm_block_size; OFMItr += 8)
                                {
                                    //with MaxPol2x2
                                    __m256i Max1 = _mm256_max_epi32(OutBlock[OBlockYItr + 0][OBlockXItr + 0][OFMItr / 8], OutBlock[OBlockYItr + 0][OBlockXItr + 1][OFMItr / 8]);
                                    __m256i Max2 = _mm256_max_epi32(OutBlock[OBlockYItr + 1][OBlockXItr + 0][OFMItr / 8], OutBlock[OBlockYItr + 1][OBlockXItr + 1][OFMItr / 8]);
                                    Max1 = _mm256_max_epi32(Max1, Max2);

                                    FStoreActiv((void *)(output_window
                                        + (BlockOffsetOFM + OFMItr) / output_z_block_size * output_stride_z_block + OFMItr % output_z_block_size
                                        + (BlockOffsetY + OBlockYItr / pool_size_y_2) * output_stride_y
                                        + (BlockOffsetX + OBlockXItr / pool_size_x_2) * output_stride_x),
                                        Max1,
                                        shift);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    inline __m256i ***Create3DBuf256(uint32_t Dim1, uint32_t Dim2, uint32_t Dim3)
    {
        __m256i *** dest = new __m256i **[Dim1];

        dest[0] = new __m256i *[Dim1*Dim2];

        dest[0][0] = (__m256i *)_mm_malloc(Dim1*Dim2*Dim3*sizeof(__m256i), 64);

        uint32_t i, j;

        for (i = 0; i < Dim1; i++)
        {
            if (i < Dim1 - 1)
            {
                dest[0][(i + 1)*Dim2] = &(dest[0][0][(i + 1)*Dim3*Dim2]);
                dest[i + 1] = &(dest[0][(i + 1)*Dim2]);
            }

            for (j = 0; j < Dim2; j++)
            {
                if (j > 0) dest[i][j] = dest[i][j - 1] + Dim3;
            }
        }
        return dest;
    }

    static void Delete3DBuf256(__m256i ***dist)
    {
        _mm_free(dist[0][0]);
        delete[] dist[0];
        delete[] dist;
    }

    template <FActiveShift FStoreActiv, FBias FBias>
    struct convolution_avx2_int16_fixedpoint_selector{

        using convolve_maxpool_call_params = std::tuple<
        convolution_pooling_i16 *,
        const nn::workload_data<int16_t> *,
        const nn::workload_data<int16_t> *,
        const nn::workload_data<int32_t> *,
        nn::workload_data<int16_t> *>;

        template<int IFMBlock, int OFMBlock, int OXBlock, int OXpBlock>
        struct SelectOYBlock{
            inline static void choose(int num_ifm, int num_ofm, int out_width, int out_height, convolve_maxpool_call_params const& call_params){
                if ((out_height * 2) % 2 == 0)
                    NN_ConvolveAVX2_Pool2x2_INT16_fixedpoint<IFMBlock, OFMBlock, OXBlock, 2, OXpBlock, 2, FStoreActiv, FBias>(
                    std::get<0>(call_params),
                    std::get<1>(call_params),
                    std::get<2>(call_params),
                    std::get<3>(call_params),
                    std::get<4>(call_params)); 
                else
                    NN_ConvolveAVX2_Pool2x2_INT16_fixedpoint<IFMBlock, OFMBlock, OXBlock, 1, OXpBlock, 1, FStoreActiv, FBias>(
                    std::get<0>(call_params),
                    std::get<1>(call_params),
                    std::get<2>(call_params),
                    std::get<3>(call_params),
                    std::get<4>(call_params));
            }
        };

        template<int IFMBlock, int OFMBlock>
        struct SelectOXBlock{
            inline static void choose(int num_ifm, int num_ofm, int out_width, int out_height, convolve_maxpool_call_params const& call_params){
                if ((out_width * 2) % 14 == 0)
                    SelectOYBlock<IFMBlock, OFMBlock, 14, 2>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else if ((out_width * 2) % 12 == 0)
                    SelectOYBlock<IFMBlock, OFMBlock, 12, 2>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else if ((out_width * 2) % 2 == 0)
                    SelectOYBlock<IFMBlock, OFMBlock, 2, 2>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else
                    SelectOYBlock<IFMBlock, OFMBlock, 1, 1>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
            }
        };

        template <int IFMBlock>
        struct SelectOXBlock<IFMBlock, 384> {
            inline static void choose(int num_ifm, int num_ofm, int out_width, int out_height, convolve_maxpool_call_params const& call_params){
                NN_ConvolveAVX2_Pool2x2_INT16_fixedpoint<IFMBlock, 384, 1, 1, 1, 1, FStoreActiv, FBias>(
                    std::get<0>(call_params),
                    std::get<1>(call_params),
                    std::get<2>(call_params),
                    std::get<3>(call_params),
                    std::get<4>(call_params));
            }
        };

        template<int IFMBlock>
        struct SelectOFMBlock{
            inline static void choose(int num_ifm, int num_ofm, int out_width, int out_height, convolve_maxpool_call_params const& call_params){
                if (num_ofm % 32 == 0)
                    SelectOXBlock<IFMBlock, 32>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else
                    assert(0);
            }
        };

        struct SelectIFMBlock{
            inline static void choose(int num_ifm, int num_ofm, int out_width, int out_height, convolve_maxpool_call_params const& call_params){
                if (num_ifm % 16 == 0)
                    SelectOFMBlock<16>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else if (num_ifm % 4 == 0)
                    SelectOFMBlock<4>::choose(num_ifm, num_ofm, out_width, out_height, call_params);
                else
                    assert(0);
            }
        };

        inline static void choose(
            convolution_pooling_i16 *primitive,
            const nn::workload_data<int16_t> *input_buffer,
            const nn::workload_data<int16_t> *weights_buffer,
            const nn::workload_data<int32_t> *bias_buffer,
            nn::workload_data<int16_t> *output_buffer)
        {
            const auto OFMOutBlock = output_buffer->parent->lengths.t[NN_DATA_COORD_p];
            const auto num_output_feature_maps = output_buffer->parent->lengths.t[NN_DATA_COORD_z] * OFMOutBlock;
            const auto output_view_width = output_buffer->view_end.t[NN_DATA_COORD_x] - output_buffer->view_begin.t[NN_DATA_COORD_x] + 1;
            const auto output_view_height = output_buffer->view_end.t[NN_DATA_COORD_y] - output_buffer->view_begin.t[NN_DATA_COORD_y] + 1;
            SelectIFMBlock::choose(input_buffer->parent->lengths.t[NN_DATA_COORD_p], num_output_feature_maps, output_view_width, output_view_height, std::forward_as_tuple(primitive, input_buffer, weights_buffer, bias_buffer, output_buffer));
        }
    };

    void convolution_pooling_i16::run_convolve_maxpool2x2_fixedpoint(
        const nn::workload_data<int16_t> *input_buffer,
        const nn::workload_data<int16_t> *weights_buffer,
        const nn::workload_data<int32_t> *bias_buffer,
        nn::workload_data<int16_t> *output_buffer)
    {
        const auto acc_shift = activation.fractions.accumulator - activation.fractions.output;

        if (bias_buffer != nullptr &&
            bias_buffer->parent != nullptr &&
            bias_buffer->parent->data_buffer != nullptr)
        {
            // Invoke convolution.
            if (activation.basic_arguments.function == NN_ACTIVATION_FUNCTION_NONE)
            {
                if (acc_shift >= 0)
                    convolution_avx2_int16_fixedpoint_selector<Store16_woActiv_shiftR, AddBias32>::choose(this, input_buffer, weights_buffer, bias_buffer, output_buffer);
                else
                    convolution_avx2_int16_fixedpoint_selector<Store16_woActiv_shiftL, AddBias32>::choose(this, input_buffer, weights_buffer, bias_buffer, output_buffer);
            }
            else if (activation.basic_arguments.function == NN_ACTIVATION_FUNCTION_RELU)
            {
                if (acc_shift >= 0)
                    convolution_avx2_int16_fixedpoint_selector<Store16_wRELu_shiftR, AddBias32>::choose(this, input_buffer, weights_buffer, bias_buffer, output_buffer);
                else
                    convolution_avx2_int16_fixedpoint_selector<Store16_wRELu_shiftL, AddBias32>::choose(this, input_buffer, weights_buffer, bias_buffer, output_buffer);
            }
        }
        else
        {
            // Invoke convolution.
            if (activation.basic_arguments.function == NN_ACTIVATION_FUNCTION_NONE)
            {
                if (acc_shift >= 0)
                    convolution_avx2_int16_fixedpoint_selector<Store16_woActiv_shiftR, AddNoBias>::choose(this, input_buffer, weights_buffer, bias_buffer, output_buffer);
                else
                    convolution_avx2_int16_fixedpoint_selector<Store16_woActiv_shiftL, AddNoBias>::choose(this, input_buffer, weights_buffer, bias_buffer, output_buffer);
            }
            else if (activation.basic_arguments.function == NN_ACTIVATION_FUNCTION_RELU)
            {
                if (acc_shift >= 0)
                    convolution_avx2_int16_fixedpoint_selector<Store16_wRELu_shiftR, AddNoBias>::choose(this, input_buffer, weights_buffer, bias_buffer, output_buffer);
                else
                    convolution_avx2_int16_fixedpoint_selector<Store16_wRELu_shiftL, AddNoBias>::choose(this, input_buffer, weights_buffer, bias_buffer, output_buffer);
            }
        }
    }

    /*
    * Inputs layout required:
    *      nn_workload_data_layout_t in_view_layout =
    *      {
    *          { 0, 0, 0, 0, 0, 0 }, // tile_log2
    *          { 0, 0, 0, 0, 0, 0 }, // alignment
    *          { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
    *          NN_DATATYPE_INT16
    *      };
    *
    *      nn_workload_data_coords_t input_view_coords =
    *      {
    *          NumBatch,
    *          Width,
    *          Height,
    *          NumInputs/FMBlock,
    *          FMBlock,
    *          1
    *      };
    *
    * Output is configured similarly, with outputs either Int16 or Int32
    *
    */
    void unpack_convolve_pooling_fixedpoint_callback_handle(void *void_handle) {
        convolution_pooling_i16::request_handle* handle = reinterpret_cast<convolution_pooling_i16::request_handle*>(void_handle);
        handle->primitive->run_convolve_maxpool2x2_fixedpoint(handle->input_view, handle->weights_view, handle->biases_view, handle->output_view);
    }

    void run_multithreaded_convolve_pooling_fixedpoint_work_item(nn_workload_item *const work_item, nn_device_internal* device)
    {
        nn::workload_data<int16_t>* input_view = reinterpret_cast<nn::workload_data<int16_t> *>(work_item->input[0].get_data_view());
        nn::workload_data<int16_t>* output_view = reinterpret_cast<nn::workload_data<int16_t> *>(work_item->output[0]);

        if (static_cast<convolution_pooling_i16 *>(work_item->primitive)->device->thread_pool.get_num_threads() > 1)
        {
            static_cast<convolution_pooling_i16 *>(work_item->primitive)
                ->forward(
                input_view,
                reinterpret_cast<nn::workload_data<int16_t> *>(work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2_fixedpoint.weights),
                reinterpret_cast<nn::workload_data<int32_t> *>(work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2_fixedpoint.biases),
                output_view);
        }
        else
        {
            static_cast<convolution_pooling_i16 *>(work_item->primitive)
                ->run_convolve_maxpool2x2_fixedpoint(
                input_view,
                reinterpret_cast<nn::workload_data<int16_t> *>(work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2_fixedpoint.weights),
                reinterpret_cast<nn::workload_data<int32_t> *>(work_item->arguments.forward_convolution_pooling_max_2x2_stride_2x2_fixedpoint.biases),
                output_view);
        }
    }

    size_t convolution_pooling_i16::get_required_input_w() {
        return ((output_size_x - 1) * pool_stride + pool_size - 1) * stride_x + kernel_w;
    }

    size_t convolution_pooling_i16::get_required_input_h() {
        return ((output_size_y - 1) * pool_stride + pool_size - 1) * stride_y + kernel_h;
    }
} // namepace cpu16

nn_primitive_handle_t NN_API_CALL_CONVENTION nn_primitives_convolution_pooling_i16_create_0(
    nn_device_t *device,    /* IDLF device handle */
    size_t kernel_w,        /* convolution kernel width */
    size_t kernel_h,        /* convolution kernel height */
    size_t num_input,       /* number of input feature maps */
    size_t num_output,      /* number of output feature maps */
    size_t output_w,        /* output width */
    size_t output_h,        /* output height */
    size_t center_offset_x, /* horizontal offset of kernel's center point w/ relation to top left corner */
    size_t center_offset_y, /* vertical offset of kernel's center point w/ relation to top left corner */
    size_t stride_x,        /* convolution horizontal stride */
    size_t stride_y,        /* convolution vertical stride */
    const nn_argument_activation_t *activation, /* struct parameterizing optional activation function */
    size_t batch_size,                          /* size of input batch */
    size_t pooling_kernel_w,                    /* width of pooling kernel */
    size_t pooling_kernel_h,                    /* height of pooling kernel */
    size_t pooling_stride_x,                    /* horizontal pooling stride */
    size_t pooling_stride_y,                    /* vertical pooling stride */
    const nn_primitives_convolution_pooling_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */)
{
    assert(pooling_kernel_w == 2);
    assert(pooling_kernel_h == 2);
    assert(pooling_stride_x == 2);
    assert(pooling_stride_y == 2);
    SET_STATUS(NN_API_STATUS_OK);

    nn_primitives_convolution_pooling_hints_t hints_ = {};
    hints_.fixed_point_fraction_bits.accumulator = 16;
    hints_.fixed_point_fraction_bits.output = 8;

    if (hints != nullptr)
        hints_ = *hints;

    nn_argument_activation_fixedpoint_t activation_fixedpoint;
    activation_fixedpoint.basic_arguments = *activation;
    activation_fixedpoint.fractions.accumulator = hints_.fixed_point_fraction_bits.accumulator;
    activation_fixedpoint.fractions.output = hints_.fixed_point_fraction_bits.output;

    return new int16_fixedpoint::convolution_pooling_i16(
        kernel_w,
        kernel_h,
        num_input,
        num_output,
        output_w,
        output_h,
        center_offset_x,
        center_offset_y,
        stride_x,
        stride_y,
        activation_fixedpoint,
        batch_size,
        hints_.output_padding.left,
        hints_.output_padding.right,
        hints_.output_padding.top,
        hints_.output_padding.bottom,
        reinterpret_cast<nn_device_internal *>(device));
}