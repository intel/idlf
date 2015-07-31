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

#include "naive_implementations.h"

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation.cpp #1
//cpu_layer_normalization_response_across_maps_fixedpoint.cpp
//cpu_layer_pooling_int16_fixedpoint_avx2.cpp
//cpu_layer_pooling_int16_fixedpoint_avx2_compilation.cpp
//cpu_layer_convolution_int16_fixedpoint_avx2.cpp
//cpu_layer_convolution_int16_fixedpoint_avx2_compilation.cpp
//cpu_layer_convolution_pooling_int16_fixedpoint_avx2.cpp
//cpu_layer_convolution_pooling_int16_fixedpoint_avx2_compilation.cpp
void ult_nn_merge_convolution_fixedpoint_naive_set_input_value(
    int16_t* input_ref,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t input_column,
    uint_least32_t input_row,
    uint_least32_t input_map,
    int16_t value,
    uint_least32_t& offset)
{
    offset = input_column + input_row*input_feature_map_width + input_map*input_feature_map_width*input_feature_map_height;

    input_ref[offset] = value;
}

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation.cpp #2
//cpu_layer_convolution_int16_fixedpoint_avx2.cpp
//cpu_layer_convolution_int16_fixedpoint_avx2_compilation.cpp
//cpu_layer_convolution_pooling_int16_fixedpoint_avx2.cpp
//cpu_layer_convolution_pooling_int16_fixedpoint_avx2_compilation.cpp
void ult_nn_merge_convolution_fixedpoint_naive_set_kernel_value(
    int16_t* kernel_ref,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t num_input_feature_maps,
    uint_least32_t kernel_column,
    uint_least32_t kernel_row,
    uint_least32_t kernel_input_map,
    uint_least32_t kernel_output_map,
    int16_t value,
    uint_least32_t& offset)
{
    offset = kernel_column + kernel_row*kernel_width + kernel_input_map*kernel_width*kernel_height + kernel_output_map*kernel_width*kernel_height*num_input_feature_maps;
    kernel_ref[offset] = value;
}

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation.cpp #3
void ult_nn_merge_convolution_fixedpoint_both_initialize_matrices(
    int16_t* input,
    int16_t* output,
    int32_t* biases,
    int16_t* kernel,
    int16_t* input_ref,
    int16_t* output_ref,
    int32_t* biases_ref,
    int16_t* kernel_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t center_x,
    uint_least32_t center_y
    )
{
    uint_least32_t input_size = input_feature_map_width * input_feature_map_height * num_input_feature_maps * sizeof(int16_t);
    int16_t * inputT = (int16_t*)_mm_malloc(input_size, 4096);
    uint_least32_t kernel_size = num_input_feature_maps * num_output_feature_maps * kernel_width * kernel_height * sizeof(int16_t);
    int16_t * weightT = (int16_t*)_mm_malloc(kernel_size, 4096);

    uint32_t IFMBlock = 8;
    if (num_input_feature_maps == 4) IFMBlock = 4;
    uint32_t OFMBlock = 32;
    uint32_t OFMpBlock = 2;
    if (num_output_feature_maps == 3072 && num_input_feature_maps == 1024)
    {
        IFMBlock = 8;
        OFMBlock = 384;
    }

    for (uint_least32_t input_map = 0; input_map < num_input_feature_maps; input_map++)
    {
        uint_least32_t element = 0;
        int16_t value = input_map * 0x0100;
        for (uint_least32_t row = 0; row < input_feature_map_height; row++)
        {
            for (uint_least32_t column = 0; column < input_feature_map_width; column++)
            {
                uint_least32_t offset;
                //ult_nn_convolution_optimized_set_input_value(inputT, input_feature_map_width, num_input_feature_maps, column, row, input_map, value, offset);
                ult_nn_merge_convolution_fixedpoint_naive_set_input_value(inputT, input_feature_map_width, input_feature_map_height, column, row, input_map, value, offset);
                ult_nn_merge_convolution_fixedpoint_naive_set_input_value(input_ref, input_feature_map_width, input_feature_map_height, column, row, input_map, value, offset);
                value++;
            }
        }
    }

    int16_t value = 0;
    for (uint_least32_t outmapa = 0; outmapa < num_output_feature_maps; outmapa++)
    {
        for (uint_least32_t input_map = 0; input_map < num_input_feature_maps; input_map++)
        {
            for (uint_least32_t row = 0; row < kernel_height; row++)
            {
                for (uint_least32_t column = 0; column < kernel_width; column++)
                {
                    uint_least32_t offset;
                    //ult_nn_convolution_optimized_set_kernel_value
                    ult_nn_merge_convolution_fixedpoint_naive_set_kernel_value(kernel, kernel_width, kernel_height, num_input_feature_maps, column, row, input_map, outmapa, value, offset);
                    ult_nn_merge_convolution_fixedpoint_naive_set_kernel_value(kernel_ref, kernel_width, kernel_height, num_input_feature_maps, column, row, input_map, outmapa, value, offset);
                    value++;
                }
            }
        }
    }

    for (uint_least32_t outmapa = 0; outmapa < num_output_feature_maps; outmapa++)
    {
        for (uint_least32_t row = 0; row < output_feature_map_height + 2 * center_y; row++)
        {
            for (uint_least32_t column = 0; column < output_feature_map_width + 2 * center_x; column++)
            {
                uint32_t index = column + row * (output_feature_map_width + 2 * center_x) + outmapa * (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y);
                output[index] = 0;
                output_ref[index] = 0;
            }
        }
    }


    for (uint_least32_t outmapa = 0; outmapa < num_output_feature_maps; outmapa++)
    {
        biases[outmapa] = outmapa;
        biases_ref[outmapa] = outmapa;

    }

    //prepare right input layout for zxy
    for (size_t y = 0; y < input_feature_map_height; y++)
    {
        for (size_t x = 0; x < input_feature_map_width; x++)
        {
            for (size_t z = 0; z < num_input_feature_maps; z++)
            {
                input[z + x * num_input_feature_maps + y * num_input_feature_maps * input_feature_map_width]
                    = inputT[z * input_feature_map_width * input_feature_map_height + y * input_feature_map_height + x];
            }
        }
    }
    _mm_free(inputT);
}

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation.cpp #4
bool ult_nn_merge_convolution_fixedpoint_check_outputs(
    nn::data<int16_t, 3>* output,
    int16_t* output_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t center_x,
    uint_least32_t center_y,
    int8_t merge_axis
    )
{
    size_t num_output_feature_maps_merge = merge_axis == 2 ? NUM_MERGED_CONVOLUTIONS * num_output_feature_maps : num_output_feature_maps;
    size_t output_feature_map_height_merge = merge_axis == 1 ? NUM_MERGED_CONVOLUTIONS * output_feature_map_height : output_feature_map_height;
    size_t output_feature_map_width_merge = merge_axis == 0 ? NUM_MERGED_CONVOLUTIONS * output_feature_map_width : output_feature_map_width;

    // zxy -> xyz
    uint_least32_t output_size = (output_feature_map_width_merge + 2 * center_x) * (output_feature_map_height_merge + 2 * center_y) * num_output_feature_maps_merge * sizeof(int16_t);
    int16_t* outputT = (int16_t*)_mm_malloc(output_size, 64);
    int16_t* outputT2 = (int16_t*)_mm_malloc(output_size, 64);
    int16_t * outputOpt = (int16_t *)output->buffer;
    uint32_t OFMOutBlock = 8;

    for (size_t y = 0; y < output_feature_map_height_merge; y++)
    {
        for (size_t x = 0; x < output_feature_map_width_merge; x++)
        {
            for (size_t z = 0; z < num_output_feature_maps_merge; z++)
            {
                outputT[z * output_feature_map_width_merge * output_feature_map_height_merge + y * output_feature_map_width_merge + x]
                    = outputOpt[z + x * num_output_feature_maps_merge + y * num_output_feature_maps_merge * output_feature_map_width_merge];
            }
        }
    }

    // xyz -> pxyznq  - TODO: replace these 2 conversions with 1
    for (uint32_t i = 0; i < num_output_feature_maps_merge / OFMOutBlock; i++)
    for (uint32_t j = 0; j < output_feature_map_width_merge * output_feature_map_height_merge; j++)
    for (uint32_t n = 0; n < OFMOutBlock; n++)
        outputT2[n + j * OFMOutBlock + i * output_feature_map_width_merge * output_feature_map_height_merge * OFMOutBlock]
        = outputT[n * output_feature_map_width_merge * output_feature_map_height_merge + j + i * output_feature_map_width_merge * output_feature_map_height_merge * OFMOutBlock];

    outputOpt = outputT2;

    bool passed = true;
    int32_t count = 0;
    int16_t *comparePtr;
    if (merge_axis == 0)
        comparePtr = outputOpt + output_feature_map_width * OFMOutBlock;
    if (merge_axis == 1)
        comparePtr = outputOpt + output_feature_map_height * output_feature_map_width * OFMOutBlock;
    if (merge_axis == 2)
        comparePtr = outputOpt + num_output_feature_maps * output_feature_map_height * output_feature_map_width;

    for (uint32_t z = 0; z < num_output_feature_maps / OFMOutBlock; z++)
    {
        for (uint32_t y = 0; y < (output_feature_map_height + 2 * center_y); y++)
        {
            for (uint32_t x = 0; x < (output_feature_map_width + 2 * center_x); x++)
            {
                for (uint32_t n = 0; n < OFMOutBlock; n++)
                {
                    if (comparePtr[n + x * OFMOutBlock] != outputOpt[n + x * OFMOutBlock])
                        passed = false;
                }
            }

            if (merge_axis == 0)
            {
                outputOpt += output_feature_map_width * OFMOutBlock * NUM_MERGED_CONVOLUTIONS;
                comparePtr += output_feature_map_width * OFMOutBlock * NUM_MERGED_CONVOLUTIONS;
            }
        }

        if (merge_axis == 1)
        {
            outputOpt += output_feature_map_height * output_feature_map_width * OFMOutBlock * NUM_MERGED_CONVOLUTIONS;
            comparePtr += output_feature_map_height * output_feature_map_width * OFMOutBlock * NUM_MERGED_CONVOLUTIONS;
        }
    }

    return passed;
}

//cpu_layer_merge_convolution_int16_fixedpoint_avx2_compilation.cpp #5
void ult_nn_convolve_fp_naive(
    int16_t* input_ref,
    int16_t* output_ref,
    int32_t* biases_ref,
    int16_t* kernel_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    uint_least32_t center_x,
    uint_least32_t center_y,
    NN_ACTIVATION_FUNCTION activation)
{
    bool BiasEn = 1;

    int32_t * output_ref_temp = (int32_t*)_mm_malloc((output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y) * num_output_feature_maps * sizeof(int32_t), 64);
    memset(output_ref_temp, 0, (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y) * num_output_feature_maps * sizeof(int32_t));

    for (unsigned int ofmItr = 0; ofmItr < num_output_feature_maps; ofmItr++)
    { // For each output feature map
        for (unsigned int ifmItr = 0; ifmItr < num_input_feature_maps; ifmItr++)
        { // Go over all input feature maps

            for (unsigned int hItr = 0; hItr < (input_feature_map_height - kernel_height + 1); hItr += kernel_stride_y)
            { // For each input feature map, go over all locations where the kernel-sized stencil would fit - in both dimensions, y...
                for (unsigned int wItr = 0; wItr < (input_feature_map_width - kernel_width + 1); wItr += kernel_stride_x)
                { // and x...
                    for (unsigned int kH = 0; kH < kernel_height; kH++)
                    {
                        for (unsigned int kW = 0; kW < kernel_width; kW++)
                        { // For each stencil placement, compute 2D convolution at the placement
                            short kernel_pixel = kernel_ref[ofmItr * num_input_feature_maps * kernel_height * kernel_width + ifmItr * kernel_width * kernel_height + kernel_width*kH + kW];
                            short ifm_pixel = input_ref[(ifmItr * input_feature_map_width * input_feature_map_height + (input_feature_map_width*hItr + wItr) + kH*input_feature_map_width + kW)];
                            output_ref_temp[
                                (ofmItr * (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y)
                                    + (center_y + hItr / kernel_stride_y) * (output_feature_map_width + 2 * center_x)
                                    + (center_x + wItr / kernel_stride_x))]
                                    += ifm_pixel*kernel_pixel;
                        }
                    }
                    // Also add bias, but only once for each output feature map - when going over input feature map 0
                    output_ref_temp[(ofmItr * (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y)
                        + (center_y + hItr / kernel_stride_y) * (output_feature_map_width + 2 * center_x)
                        + (center_x + wItr / kernel_stride_x))]
                        += (((ifmItr == 0) && BiasEn) ? 1 : 0) * biases_ref[ofmItr];
                }
            }
        }
    }

    const auto acc_shift = accumulator_fraction - output_fraction;

    for (uint32_t ofmItr = 0; ofmItr < num_output_feature_maps; ++ofmItr)
    { // For each output feature map
        for (uint32_t hItr = 0; hItr < (output_feature_map_height + 2 * center_y); ++hItr)
        {
            for (uint32_t wItr = 0; wItr < (output_feature_map_width + 2 * center_x); ++wItr)
            {
                auto indx = wItr + hItr * (output_feature_map_width + 2 * center_x) + ofmItr * (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y);

                switch (activation)
                {
                case NN_ACTIVATION_FUNCTION_RELU:
                    output_ref_temp[indx] = std::max(0, output_ref_temp[indx]);
                    break;
                case NN_ACTIVATION_FUNCTION_NONE:
                    break;
                default:
                    break;
                }

                if (acc_shift > 0)
                {
                    output_ref_temp[indx] = output_ref_temp[indx] >> acc_shift;
                }
                else
                {
                    output_ref_temp[indx] = output_ref_temp[indx] << -acc_shift;
                }

                output_ref_temp[indx] = std::min(output_ref_temp[indx], 32767);
                output_ref_temp[indx] = std::max(output_ref_temp[indx], -32768);
                output_ref[indx] = (int16_t)(output_ref_temp[indx]);

            }
        }
    }
}

//cpu_layer_normalization.cpp
bool compare_results(nn_workload_item *&work_item, nn::data<float, 4> &output_ref) {
    for (uint32_t batch = 0; batch < output_ref.size[3]; ++batch) {
        for (uint32_t output_element_y = 0; output_element_y < output_ref.size[2]; ++output_element_y) {
            for (uint32_t output_element_x = 0; output_element_x < output_ref.size[1]; ++output_element_x) {
                for (uint32_t output_element_z = 0; output_element_z < output_ref.size[0]; ++output_element_z) {
                    float value = nn_workload_data_get<float>(
                        work_item->output[0], batch, output_element_x, output_element_y, output_element_z, 0, 0);
                    float value_ref = output_ref.at(output_element_z, output_element_x, output_element_y, batch);

                    float diff = fabs(value_ref - value);

                    if (value_ref == 0.0f || value == 0.0f || diff < FLT_MIN)
                    {
                        if (diff > FLT_MIN)
                        {
                            return false;
                        }
                    }
                    else
                    {
                        if (diff / value_ref > 5.2e-04F)
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }

    return true;
}

//cpu_layer_normalization.cpp
template <class T_primitive>
bool run_naive_layer_normalization(
    float alpha, float beta, uint32_t k, uint32_t n, const nn::data<float, 4> &input, nn::data<float, 4> &output) {
    assert(0);
    return false;
}

/* data ordering is Z,X,Y,N */
template <>
bool run_naive_layer_normalization<layer::normalization_elementwise_linear_f32>(
    float alpha, float beta, uint32_t k, uint32_t n, const nn::data<float, 4> &input, nn::data<float, 4> &output) {
    float coeff_a = alpha;
    float coeff_b = beta;
    for (uint32_t batch = 0; batch < input.size[3]; ++batch)
    {
        for (uint32_t output_element_y = 0; output_element_y < input.size[2]; ++output_element_y)
        {
            for (uint32_t output_element_x = 0; output_element_x < input.size[1]; ++output_element_x)
            {
                for (uint32_t output_element_z = 0; output_element_z < input.size[0]; ++output_element_z)
                {
                    float value = input.at(output_element_z, output_element_x, output_element_y, batch);

                    value = value * coeff_a + coeff_b;

                    output.at(output_element_z, output_element_x, output_element_y, batch) = value;
                }
            }
        }
    }
    return true;
}

/* data ordering is Z,X,Y,N */
template <>
bool run_naive_layer_normalization<layer::normalization_response_across_maps_f32>(
    float alpha, float beta, uint32_t k, uint32_t n, const nn::data<float, 4> &input, nn::data<float, 4> &output) {
    float coeff_a = alpha;
    float coeff_b = beta;
    uint32_t coeff_n = n;
    uint32_t coeff_k = k;

    for (uint32_t batch = 0; batch < input.size[3]; ++batch)
    {
        for (uint32_t output_element_y = 0; output_element_y < input.size[2]; ++output_element_y)
        {
            for (uint32_t output_element_x = 0; output_element_x < input.size[1]; ++output_element_x)
            {
                for (uint32_t output_element_z = 0; output_element_z < input.size[0]; ++output_element_z)
                {
                    float acc = 0.0f;
                    float raw_input = input.at(output_element_z, output_element_x, output_element_y, batch);

                    for (int32_t input_element = (int32_t)output_element_z - (int32_t)(coeff_n / 2);
                        input_element <= (int32_t)output_element_z + (int32_t)(coeff_n / 2);
                        ++input_element) {
                        float value = 0.0f;
                        if (input_element >= 0 && input_element < input.size[0])
                            value = input.at(input_element, output_element_x, output_element_y, batch);

                        acc += value*value;
                    }

                    acc = acc * coeff_a + (int32_t)coeff_k;
                    acc = std::pow(acc, -coeff_b);

                    acc = raw_input * acc;

                    output.at(output_element_z, output_element_x, output_element_y, batch) = acc;
                }
            }
        }
    }

    return true;
}

//cpu_layer_normalization_response_across_maps_fixedpoint.cpp
void ult_nn_lrn_fp_naive(
    int16_t * input_ref,
    int16_t * output_ref,
    int32_t num_feature_maps,
    int32_t feature_map_width,
    int32_t feature_map_height,
    int32_t batch_size,
    int32_t input_fraction,
    int32_t output_fraction,
    float coeff_alpha,
    float coeff_beta,
    float coeff_k
    )
{
    const int32_t coeff_N = 5;
    float scale_in = 1.0f / (float)(1 << input_fraction);

    auto MapSizeXY = feature_map_width * feature_map_height;
    auto MapSizeXYZ = feature_map_width * feature_map_height * num_feature_maps;

    for (auto itrBatch = 0; itrBatch < batch_size; ++itrBatch)
    for (auto itrMapY = 0; itrMapY < feature_map_height; ++itrMapY)
    for (auto itrMapX = 0; itrMapX < feature_map_width; ++itrMapX)
    for (auto itrMapZ = 0; itrMapZ < num_feature_maps; ++itrMapZ)
    {
        int32_t sum = 0;
        int16_t v_int = 0;
        float v_float = 0;
        int16_t v_int0 = input_ref[itrMapX + itrMapY * feature_map_width + (itrMapZ + 0) * MapSizeXY + itrBatch * MapSizeXYZ];
        for (int32_t itrN = -coeff_N / 2; itrN < coeff_N / 2 + 1; ++itrN)
        {

            if ((itrMapZ + itrN < 0) || (itrMapZ + itrN >= num_feature_maps))
                v_int = 0;
            else
                v_int = input_ref[itrMapX + itrMapY * feature_map_width + (itrMapZ + itrN) * MapSizeXY + itrBatch * MapSizeXYZ];

            sum += v_int * v_int;
        }
        v_float = (float)sum * scale_in * scale_in;
        v_float = (v_float * coeff_alpha + coeff_k);
        v_float = pow(v_float, coeff_beta);

        v_float = (float)v_int0 * scale_in / v_float;

        auto float2int16 = [&output_fraction](float in) {
            auto scaled = in * (1 << output_fraction);
            if (scaled < INT16_MIN)
                scaled = INT16_MIN;
            if (scaled > INT16_MAX)
                scaled = INT16_MAX;
            return static_cast<std::int16_t>(round(scaled));
        };
        output_ref[itrMapX + itrMapY * feature_map_width + (itrMapZ + 0) * MapSizeXY + itrBatch * MapSizeXYZ] = float2int16(v_float);
    }
}

//cpu_layer_normalization_response_across_maps_fixedpoint_compilation.cpp
// Naive normalization_lrn
void ult_nn_lrn_fp_comp_naive(
    int16_t * input_ref,
    int16_t * output_ref,
    int32_t num_feature_maps,
    int32_t feature_map_width,
    int32_t feature_map_height,
    int32_t batch_size,
    int32_t input_fraction,
    int32_t output_fraction,
    float coeff_alpha,
    float coeff_beta,
    float coeff_k
    )
{
    const int32_t coeff_N = 5;
    float scale_in = 1.0f / (float)(1 << input_fraction);

    auto MapSizeXY = feature_map_width * feature_map_height;
    auto MapSizeXYZ = feature_map_width * feature_map_height * num_feature_maps;

    for (auto itrBatch = 0; itrBatch < batch_size; ++itrBatch)
    for (auto itrMapY = 0; itrMapY < feature_map_height; ++itrMapY)
    for (auto itrMapX = 0; itrMapX < feature_map_width; ++itrMapX)
    for (auto itrMapZ = 0; itrMapZ < num_feature_maps; ++itrMapZ)
    {
        float sum = 0;
        int16_t v_int = 0;
        float v_float = 0;
        int16_t v_int0 = input_ref[itrMapX + itrMapY * feature_map_width + (itrMapZ + 0) * MapSizeXY + itrBatch * MapSizeXYZ];
        for (int32_t itrN = -coeff_N / 2; itrN < coeff_N / 2 + 1; ++itrN)
        {

            if ((itrMapZ + itrN < 0) || (itrMapZ + itrN >= num_feature_maps))
                v_int = 0;
            else
                v_int = input_ref[itrMapX + itrMapY * feature_map_width + (itrMapZ + itrN) * MapSizeXY + itrBatch * MapSizeXYZ];

            sum += v_int * v_int;
        }
        v_float = (float)sum * scale_in * scale_in;
        v_float = (v_float * coeff_alpha + coeff_k);
        v_float = pow(v_float, coeff_beta);

        v_float = (float)v_int0 * scale_in / v_float;

        auto float2int16 = [&output_fraction](float in) {
            auto scaled = in * (1 << output_fraction);
            if (scaled < INT16_MIN)
                scaled = INT16_MIN;
            if (scaled > INT16_MAX)
                scaled = INT16_MAX;
            return static_cast<std::int16_t>(round(scaled));
        };
        output_ref[itrMapX + itrMapY * feature_map_width + (itrMapZ + 0) * MapSizeXY + itrBatch * MapSizeXYZ] = float2int16(v_float);
    }
}

//cpu_layer_pooling_avx2.cpp
void run_reference(nn::workload_data<float> *input,
    nn::data<float, 4> &output,
    size_t pool_size_x,
    size_t pool_size_y,
    size_t pool_stride_x,
    size_t pool_stride_y) {
    for (uint32_t batch = 0; batch < output.size[3]; ++batch)
    {
        for (uint32_t output_element_y = 0; output_element_y < output.size[2]; ++output_element_y)
        {
            for (uint32_t output_element_x = 0; output_element_x < output.size[1]; ++output_element_x)
            {
                for (uint32_t output_element_z = 0; output_element_z < output.size[0]; ++output_element_z)
                {
                    bool first_value = true;
                    float acc = 0.0f;

                    for (uint32_t input_element_y = 0; input_element_y < pool_size_y; ++input_element_y)
                    {
                        for (uint32_t input_element_x = 0; input_element_x < pool_size_x; ++input_element_x)
                        {
                            float value = nn_workload_data_get<float>(
                                input,
                                batch,
                                output_element_x * pool_stride_x + input_element_x,
                                output_element_y * pool_stride_y + input_element_y,
                                output_element_z,
                                0,
                                0);

                            acc = (first_value) ? value : std::max(acc, value);
                            first_value = false;
                        }
                    }

                    output.at(output_element_z, output_element_x, output_element_y, batch) = acc;
                }
            }
        }
    }
}

//cpu_layer_pooling_int16_fixedpoint_avx2.cpp
/*static*/ void ult_nn_maxpooling_naive(
    int16_t* input_ref,
    int16_t* output_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t output_feature_map_width_int,
    uint_least32_t output_feature_map_height_int,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t pool_width,
    uint_least32_t pool_height,
    uint_least32_t pool_stride_x,
    uint_least32_t pool_stride_y,
    uint_least32_t center_x,
    uint_least32_t center_y)
{
    uint_least32_t output_int_size = output_feature_map_width_int * output_feature_map_height_int * num_output_feature_maps * sizeof(int32_t);
    int32_t * output_int = (int32_t*)_mm_malloc(output_int_size, 64);
    memset(output_int, 0, output_int_size);
    uint_least32_t output_ref_size = (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y) * num_output_feature_maps * sizeof(int16_t);
    memset(output_ref, 0, output_ref_size);

    for (unsigned int ofmItr = 0; ofmItr < num_output_feature_maps; ofmItr++)
    {
        for (uint32_t y = 0; y < output_feature_map_height; y++)
        {
            for (uint32_t x = 0; x < output_feature_map_width; x++)
            {
                int coord = ofmItr * output_feature_map_height_int * output_feature_map_width_int + y * pool_stride_y * output_feature_map_height_int + x * pool_stride_x;
                int32_t max_t = input_ref[coord];
                for (uint32_t maxY = 0; maxY < pool_height; maxY++)
                {
                    for (uint32_t maxX = 0; maxX < pool_width; maxX++)
                    {
                        int coord2 = ofmItr * output_feature_map_height_int * output_feature_map_width_int + (y * pool_stride_y + maxY) * output_feature_map_height_int + x * pool_stride_x + maxX;
                        int32_t next_val = input_ref[coord2];
                        max_t = std::max(max_t, next_val);
                    }
                }

                int coord3 = ofmItr * (output_feature_map_height + 2 * center_y) * (output_feature_map_width + 2 * center_x) + (y + center_y) * (output_feature_map_width + 2 * center_x) + x + center_x;
                output_ref[coord3] = max_t;
            }
        }
    }

    _mm_free(output_int);
}

//cpu_layer_softmax.cpp
void cpu_layer_softmax(
    nn_workload_item* &work_item,
    bool is_ref,
    nn_device_t *device)
{
    for (uint32_t batch = 0;
        batch < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_n];
        ++batch)
    {
        float sum = 0.0f;
        for (uint32_t output_element = 0;
            output_element < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_x];
            ++output_element)
        {
            float value = exp(nn_workload_data_get<float>(work_item->input[0].get_data_view(), batch, output_element, 0, 0, 0, 0));
            sum += value;

            nn_workload_data_get<float>(work_item->output[0], batch, output_element, 0, 0, 0, 0) = value;
        }

        sum = 1.0f / sum;

        for (uint32_t output_element = 0;
            output_element < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_x];
            ++output_element)
        {
            float value = nn_workload_data_get<float>(work_item->output[0], batch, output_element, 0, 0, 0, 0);
            value *= sum;

            nn_workload_data_get<float>(work_item->output[0], batch, output_element, 0, 0, 0, 0) = value;
        }

    }
}

//cpu_fully_connected_int16_avx2.cpp
static int16_t& get_input(int16_t * in, int32_t NumBatch, int32_t batch, int32_t idx)//needed for ult_nn_fc_naive to work
{
    return in[idx / 2 * NumBatch * 2 + batch * 2 + idx % 2];
}

static int32_t& get_output(int32_t * out, int32_t NumBatch, int32_t batch, int32_t idx)//needed for ult_nn_fc_naive to work
{
    return out[idx / 2 * NumBatch * 2 + batch * 2 + idx % 2];
}

static int16_t& get_weight(int16_t * weights, int32_t numInputNeurons, int32_t in, int32_t out)//needed for ult_nn_fc_naive to work
{
    return weights[out / 8 * 8 * numInputNeurons + in / 2 * 2 * 8 + (out % 8) * 2 + in % 2];
}

/*static*/ void ult_nn_fc_naive(
    int16_t* input,
    int32_t* output,
    int32_t* biases,
    int16_t* kernel,
    uint_least32_t num_input_neurons,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size,
    NN_ACTIVATION_FUNCTION activation)
{
    bool BiasEn = 1;

    for (int batchItr = 0; batchItr < batch_size; ++batchItr)
    {
        for (unsigned int outputNeuronsItr = 0; outputNeuronsItr < num_output_neurons; outputNeuronsItr++)
        {
            if (BiasEn)
                get_output(output, batch_size, batchItr, outputNeuronsItr) = biases[outputNeuronsItr];
            else
                get_output(output, batch_size, batchItr, outputNeuronsItr) = 0;

            for (unsigned int inputNeuronsItr = 0; inputNeuronsItr < num_input_neurons; inputNeuronsItr++)
            {
                get_output(output, batch_size, batchItr, outputNeuronsItr) +=
                    get_input(input, batch_size, batchItr, inputNeuronsItr) * get_weight(kernel, num_input_neurons, inputNeuronsItr, outputNeuronsItr);
            }
            switch (activation)
            {
            case NN_ACTIVATION_FUNCTION_RELU:
                get_output(output, batch_size, batchItr, outputNeuronsItr) = std::max(0, get_output(output, batch_size, batchItr, outputNeuronsItr));
                break;
            case NN_ACTIVATION_FUNCTION_NONE:
                break;
            default:
                break;
            }
        }
    }
}

//cpu_layer_convolution_avx2 #1
//cpu_layer_convolution_pooling_avx2.cpp #1
float ult_nn_convolution_naive_get_output_value(
    float* output_ref,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    uint_least32_t batch,
    uint_least32_t& offset)
{
    offset = batch*output_feature_map_width*output_feature_map_height*num_output_feature_maps + output_column + output_row*output_feature_map_width + output_map*output_feature_map_width*output_feature_map_height;
    return output_ref[offset];
}

//cpu_layer_convolution_avx2 #2
//cpu_layer_convolution_pooling_avx2.cpp #2
void ult_nn_convolution_naive_set_output_value(
    float* output_ref,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    uint_least32_t batch,
    float value,
    uint_least32_t& offset)
{
    offset = batch*output_feature_map_width*output_feature_map_height*num_output_feature_maps + output_column + output_row*output_feature_map_width + output_map*output_feature_map_width*output_feature_map_height;
    output_ref[offset] = value;
}

//cpu_layer_convolution_avx2 #3
//cpu_layer_convolution_pooling_avx2.cpp #3
void ult_nn_convolution_naive_set_input_value(
    float* input_ref,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t num_input_feature_maps,
    uint_least32_t input_column,
    uint_least32_t input_row,
    uint_least32_t input_map,
    uint_least32_t batch,
    float value,
    uint_least32_t& offset)
{
    offset = batch*num_input_feature_maps*input_feature_map_width*input_feature_map_height + input_column + input_row*input_feature_map_width + input_map*input_feature_map_width*input_feature_map_height;
    input_ref[offset] = value;
}

//cpu_layer_convolution_avx2 #4
//cpu_layer_convolution_pooling_avx2.cpp #4
void ult_nn_convolution_naive_set_kernel_value(
    float* kernel_ref,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t num_input_feature_maps,
    uint_least32_t kernel_column,
    uint_least32_t kernel_row,
    uint_least32_t kernel_input_map,
    uint_least32_t kernel_output_map,
    float value,
    uint_least32_t& offset)
{
    offset = kernel_column + kernel_row*kernel_width + kernel_input_map*kernel_width*kernel_height + kernel_output_map*kernel_width*kernel_height*num_input_feature_maps;
    kernel_ref[offset] = value;
}

//cpu_layer_convolution_avx2 #5
//cpu_layer_convolution_pooling_avx2.cpp #5
void ult_nn_convolve_naive(
    float* input_ref,
    float* output_ref,
    float* biases_ref,
    float* kernel_ref,
    uint_least32_t batch_size,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    NN_ACTIVATION_FUNCTION activation)
{
    for (uint_least32_t batch = 0; batch < batch_size; batch++)
    {
        for (uint_least32_t output_feature_map = 0; output_feature_map < num_output_feature_maps; output_feature_map += 8)
        {
            for (uint_least32_t input_row = 0, output_row = 0; output_row < output_feature_map_height; input_row += kernel_stride_y, output_row++)
            {
                for (uint_least32_t input_column = 0, output_column = 0; output_column < output_feature_map_width; input_column += kernel_stride_x, output_column++)
                {
                    const uint_least32_t out_ofss = output_feature_map_width*output_feature_map_height;
                    const uint_least32_t out_base = batch*output_feature_map_width*output_feature_map_height*num_output_feature_maps + output_column + output_row*output_feature_map_width + output_feature_map*output_feature_map_width*output_feature_map_height;
                    float accumulator0 = 0.0f;
                    float accumulator1 = 0.0f;
                    float accumulator2 = 0.0f;
                    float accumulator3 = 0.0f;
                    float accumulator4 = 0.0f;
                    float accumulator5 = 0.0f;
                    float accumulator6 = 0.0f;
                    float accumulator7 = 0.0f;

                    for (uint_least32_t input_feature_map = 0; input_feature_map < num_input_feature_maps; input_feature_map++)
                    {
                        for (uint_least32_t kernel_row = 0; kernel_row < kernel_height; kernel_row++)
                        {
                            const uint_least32_t kern_ofss = kernel_width*kernel_height*num_input_feature_maps;
                            uint_least32_t kern_base = kernel_row*kernel_width + input_feature_map*kernel_width*kernel_height + output_feature_map*kernel_width*kernel_height*num_input_feature_maps;

                            for (uint_least32_t kernel_column = 0; kernel_column < kernel_width; kernel_column++)
                            {
                                float weight0 = kernel_ref[kern_base + 0 * kern_ofss];
                                float weight1 = kernel_ref[kern_base + 1 * kern_ofss];
                                float weight2 = kernel_ref[kern_base + 2 * kern_ofss];
                                float weight3 = kernel_ref[kern_base + 3 * kern_ofss];
                                float weight4 = kernel_ref[kern_base + 4 * kern_ofss];
                                float weight5 = kernel_ref[kern_base + 5 * kern_ofss];
                                float weight6 = kernel_ref[kern_base + 6 * kern_ofss];
                                float weight7 = kernel_ref[kern_base + 7 * kern_ofss];

                                ++kern_base;

                                float input = input_ref[batch*num_input_feature_maps*input_feature_map_width*input_feature_map_height + kernel_column + kernel_row*input_feature_map_width + input_column + input_row*input_feature_map_width + input_feature_map*input_feature_map_width*input_feature_map_height];

                                accumulator0 += weight0 * input;
                                accumulator1 += weight1 * input;
                                accumulator2 += weight2 * input;
                                accumulator3 += weight3 * input;
                                accumulator4 += weight4 * input;
                                accumulator5 += weight5 * input;
                                accumulator6 += weight6 * input;
                                accumulator7 += weight7 * input;
                            }
                        }
                    }

                    switch (activation)
                    {
                    case NN_ACTIVATION_FUNCTION_RELU:
                        accumulator0 = std::max(0.0f, accumulator0 + biases_ref[output_feature_map + 0]);
                        accumulator1 = std::max(0.0f, accumulator1 + biases_ref[output_feature_map + 1]);
                        accumulator2 = std::max(0.0f, accumulator2 + biases_ref[output_feature_map + 2]);
                        accumulator3 = std::max(0.0f, accumulator3 + biases_ref[output_feature_map + 3]);
                        accumulator4 = std::max(0.0f, accumulator4 + biases_ref[output_feature_map + 4]);
                        accumulator5 = std::max(0.0f, accumulator5 + biases_ref[output_feature_map + 5]);
                        accumulator6 = std::max(0.0f, accumulator6 + biases_ref[output_feature_map + 6]);
                        accumulator7 = std::max(0.0f, accumulator7 + biases_ref[output_feature_map + 7]);
                        break;

                    case NN_ACTIVATION_FUNCTION_NONE:
                        accumulator0 = accumulator0 + biases_ref[output_feature_map + 0];
                        accumulator1 = accumulator1 + biases_ref[output_feature_map + 1];
                        accumulator2 = accumulator2 + biases_ref[output_feature_map + 2];
                        accumulator3 = accumulator3 + biases_ref[output_feature_map + 3];
                        accumulator4 = accumulator4 + biases_ref[output_feature_map + 4];
                        accumulator5 = accumulator5 + biases_ref[output_feature_map + 5];
                        accumulator6 = accumulator6 + biases_ref[output_feature_map + 6];
                        accumulator7 = accumulator7 + biases_ref[output_feature_map + 7];
                        break;

                    default:
                        break;
                    }

                    output_ref[out_base + 0 * out_ofss] = accumulator0;
                    output_ref[out_base + 1 * out_ofss] = accumulator1;
                    output_ref[out_base + 2 * out_ofss] = accumulator2;
                    output_ref[out_base + 3 * out_ofss] = accumulator3;
                    output_ref[out_base + 4 * out_ofss] = accumulator4;
                    output_ref[out_base + 5 * out_ofss] = accumulator5;
                    output_ref[out_base + 6 * out_ofss] = accumulator6;
                    output_ref[out_base + 7 * out_ofss] = accumulator7;
                }
            }
        }
    }
}

//cpu_layer_convolution_pooling_avx2.cpp #6
void ult_nn_maxpooling_naive_convolution_pooling(
    float* input_ref,
    float* output_ref,
    uint_least32_t batch_size,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t pool_width,
    uint_least32_t pool_height,
    uint_least32_t pool_stride_x,
    uint_least32_t pool_stride_y,
    NN_ACTIVATION_FUNCTION activation)
{
    for (uint_least32_t batch = 0; batch < batch_size; batch++)
    {
        for (uint_least32_t output_feature_map = 0; output_feature_map < num_output_feature_maps; output_feature_map++)
        {
            for (uint_least32_t input_row = 0, output_row = 0; output_row < output_feature_map_height; input_row += pool_stride_y, output_row++)
            {
                for (uint_least32_t input_column = 0, output_column = 0; output_column < output_feature_map_width; input_column += pool_stride_x, output_column++)
                {
                    const uint_least32_t out_base = batch*output_feature_map_width*output_feature_map_height*num_output_feature_maps + output_column + output_row*output_feature_map_width + output_feature_map*output_feature_map_width*output_feature_map_height;
                    float accumulator0 = output_ref[out_base];

                    for (uint_least32_t kernel_row = 0; kernel_row < pool_height; kernel_row++)
                    {
                        for (uint_least32_t kernel_column = 0; kernel_column < pool_width; kernel_column++)
                        {
                            float input = input_ref[batch*num_output_feature_maps*input_feature_map_width*input_feature_map_height + kernel_column + kernel_row*input_feature_map_width + input_column + input_row*input_feature_map_width + output_feature_map*input_feature_map_width*input_feature_map_height];

                            accumulator0 = std::max(accumulator0, input);
                        }
                    }

                    output_ref[out_base] = accumulator0;
                }
            }
        }
    }
}

//cpu_layer_convolution_int16_fixedpoint_avx2.cpp #1
int16_t ult_nn_convolution_fp_naive_get_output_value(
    int16_t* output_ref,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    uint_least32_t& offset)
{
    offset = output_column + output_row*output_feature_map_width + output_map*output_feature_map_width*output_feature_map_height;
    return output_ref[offset];
}

//cpu_layer_convolution_int16_fixedpoint_avx2.cpp #2
/*static*/ void ult_nn_convolution_fp_naive_set_output_value(
    int16_t* output_ref,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_column,
    uint_least32_t output_row,
    uint_least32_t output_map,
    int16_t value,
    uint_least32_t& offset)
{
    offset = output_column + output_row*output_feature_map_width + output_map*output_feature_map_width*output_feature_map_height;
    output_ref[offset] = value;
}

//cpu_layer_convolution_pooling_int16_fixedpoint_avx2.cpp
//cpu_layer_convolution_pooling_int16_fixedpoint_avx2_compilation.cpp
void ult_nn_maxpooling_naive_pooling_int16_fixedpoint(
    int16_t* input_ref,
    int16_t* output_ref,
    int32_t* biases_ref,
    int16_t* kernel_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t output_feature_map_width_int,
    uint_least32_t output_feature_map_height_int,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    uint_least32_t pool_width,
    uint_least32_t pool_height,
    uint_least32_t pool_stride_x,
    uint_least32_t pool_stride_y,
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    uint_least32_t center_x,
    uint_least32_t center_y,
    NN_ACTIVATION_FUNCTION activation)
{
    bool BiasEn = 1;

    uint_least32_t output_int_size = output_feature_map_width_int * output_feature_map_height_int * num_output_feature_maps * sizeof(int32_t);
    int32_t * output_int = (int32_t*)_mm_malloc(output_int_size, 64);
    memset(output_int, 0, output_int_size);
    uint_least32_t output_ref_size = (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y) * num_output_feature_maps * sizeof(int16_t);
    memset(output_ref, 0, output_ref_size);

    for (unsigned int ofmItr = 0; ofmItr < num_output_feature_maps; ofmItr++)
    { // For each output feature map
        for (unsigned int ifmItr = 0; ifmItr < num_input_feature_maps; ifmItr++)
        { // Go over all input feature maps
            for (unsigned int hItr = 0; hItr < (input_feature_map_height - kernel_height + 1); hItr += kernel_stride_y)
            { // For each input feature map, go over all locations where the kernel-sized stencil would fit - in both dimensions, y...
                for (unsigned int wItr = 0; wItr < (input_feature_map_width - kernel_width + 1); wItr += kernel_stride_x)
                { // and x...
                    for (unsigned int kH = 0; kH < kernel_height; kH++)
                    {
                        for (unsigned int kW = 0; kW < kernel_width; kW++)
                        { // For each stencil placement, compute 2D convolution at the placement
                            short kernel_pixel = kernel_ref[ofmItr * num_input_feature_maps * kernel_height * kernel_width + ifmItr * kernel_width * kernel_height + kernel_width*kH + kW];
                            short ifm_pixel = input_ref[(ifmItr * input_feature_map_width * input_feature_map_height + (input_feature_map_width*hItr + wItr) + kH*input_feature_map_width + kW)];
                            output_int[(ofmItr * output_feature_map_width_int * output_feature_map_height_int + (hItr / kernel_stride_y) * output_feature_map_height_int + (wItr / kernel_stride_x))] += ifm_pixel*kernel_pixel;
                        }
                    }
                    // Also add bias, but only once for each output feature map - when going over input feature map 0
                    output_int[(ofmItr * output_feature_map_width_int * output_feature_map_height_int + (hItr / kernel_stride_y) * output_feature_map_width_int + (wItr / kernel_stride_x))] += ((ifmItr == 0 && BiasEn) ? 1 : 0) * biases_ref[ofmItr];
                }
            }
        }
    }

    const auto acc_shift = accumulator_fraction - output_fraction;

    for (unsigned int ofmItr = 0; ofmItr < num_output_feature_maps; ofmItr++)
    for (uint32_t y = 0; y < output_feature_map_height; y++)
    for (uint32_t x = 0; x < output_feature_map_width; x++)
    {
        int32_t max_t = output_int[ofmItr * output_feature_map_height_int * output_feature_map_width_int + y * pool_stride_y * output_feature_map_height_int + x * pool_stride_x];
        for (uint32_t maxY = 0; maxY < pool_stride_y; maxY++)
        for (uint32_t maxX = 0; maxX < pool_stride_x; maxX++)
        {
            int32_t max_t1 = output_int[ofmItr * output_feature_map_height_int * output_feature_map_width_int + (y * pool_stride_y + maxY) * output_feature_map_height_int + x * pool_stride_x + maxX];
            max_t = std::max(max_t, max_t1);
        }

        switch (activation)
        {
        case NN_ACTIVATION_FUNCTION_RELU:
            max_t = std::max(0, max_t);
            break;
        case NN_ACTIVATION_FUNCTION_NONE:
            break;
        default:
            break;
        }

        if (acc_shift > 0)
        {
            max_t = max_t >> acc_shift;
        }
        else
        {
            max_t = max_t << -acc_shift;
        }

        max_t = std::min(max_t, 32767);
        max_t = std::max(max_t, -32768);
        output_ref[ofmItr * (output_feature_map_height + 2 * center_y) * (output_feature_map_width + 2 * center_x) + (y + center_y) * (output_feature_map_width + 2 * center_x) + x + center_x] = max_t;
    }

    _mm_free(output_int);

}

//cpu_layer_fullyconnected.cpp
void cpu_layer_fullyconnected(
    nn_workload_item* &work_item,
    NN_ACTIVATION_FUNCTION activation_function)
    {
    auto &arguments = work_item->parameters;
    for (uint32_t batch = 0;
        batch < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_n];
        ++batch)
    {
        for (uint32_t output_element = 0;
            output_element < work_item->output[0]->parent->lengths.t[NN_DATA_COORD_x];
            ++output_element)
        {
            float accumulator = 0.0f;
            for (uint32_t input_element = 0;
                input_element < work_item->input[0].get_data_view()->parent->lengths.t[NN_DATA_COORD_x];
                ++input_element)
            {
                accumulator +=
                    nn_workload_data_get<float>(work_item->input[0].get_data_view(), batch, input_element, 0, 0, 0, 0) *
                    nn_workload_data_get<float>(arguments[0], 0, input_element, output_element, 0, 0, 0);
            }

            accumulator += nn_workload_data_get<float>(arguments[1], 0, output_element, 0, 0, 0, 0);

            if (activation_function == NN_ACTIVATION_FUNCTION_RELU)
            {
                accumulator = std::max(0.0f, accumulator);
            }

            nn_workload_data_get<float>(work_item->output[0], batch, output_element, 0, 0, 0, 0) = accumulator;
        }
    }
}