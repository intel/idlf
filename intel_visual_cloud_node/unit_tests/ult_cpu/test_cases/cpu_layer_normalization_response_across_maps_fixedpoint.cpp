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

#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include "gtest/gtest.h"

#include "../../devices/common/nn_workload_data.h"
#include "../../devices/device_cpu/core/fixedpoint/layer_normalization_response_across_maps_int16_avx2.h"
#include "../../devices/device_cpu/api_internal/nn_device_interface_0_internal.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

static void ult_lrn_buffers_alloc(
    int16_t* &input,
    int16_t* &output,
    int16_t* &input_ref,
    int16_t* &output_ref,
    int32_t num_feature_maps,
    int32_t feature_map_width,
    int32_t feature_map_height,
    int32_t batch_size
    )
{
    uint32_t size = batch_size * num_feature_maps * feature_map_width * feature_map_height * sizeof(int16_t);

    input_ref = (int16_t*)_mm_malloc(size, 64);
    output_ref = (int16_t*)_mm_malloc(size, 64);

    input = (int16_t*)_mm_malloc(size, 64);
    output = (int16_t*)_mm_malloc(size, 64);
}

static void ult_nn_lrn_fp_naive_set_input_value(
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


// Initialize both buffers.
static void ult_lrn_buffers_initialize(
    int16_t* input,
    int16_t* output,
    int16_t* input_ref,
    int16_t* output_ref,
    int32_t num_feature_maps,
    int32_t feature_map_width,
    int32_t feature_map_height,
    int32_t batch_size
    )
{
    uint32_t IFMBlock = 8;

    for (uint_least32_t batch = 0; batch < batch_size; batch++)
    for (uint_least32_t input_map = 0; input_map < num_feature_maps; input_map++)
    {
        uint_least32_t element = 0;
        int16_t value = input_map * 0x0100;
        for (uint_least32_t row = 0; row < feature_map_height; row++)
        {
            for (uint_least32_t column = 0; column < feature_map_width; column++)
            {
                uint32_t index = column + row * feature_map_width + input_map * feature_map_width* feature_map_height + batch * feature_map_width* feature_map_height* num_feature_maps;
                input_ref[index] = (int16_t)((value + batch) % 256);
                value++;
            }
        }
    }
    for (uint_least32_t batch = 0; batch < batch_size; batch++)
    for (uint_least32_t outmapa = 0; outmapa < num_feature_maps; outmapa++)
    {
        for (uint_least32_t row = 0; row < feature_map_height; row++)
        {
            for (uint_least32_t column = 0; column < feature_map_width; column++)
            {
                uint32_t index = column + row * (feature_map_width)+outmapa * (feature_map_width)* (feature_map_height)+batch * (feature_map_width)* (feature_map_height)* num_feature_maps;
                output[index] = 0;
                output_ref[index] = 0;
            }
        }
    }

    //prepare right input layout for naive implementation
    for (uint32_t batch = 0; batch < batch_size; batch++)
    for (uint32_t i = 0; i < num_feature_maps / IFMBlock; i++)
    for (uint32_t j = 0; j < feature_map_width * feature_map_height; j++)
    for (uint32_t n = 0; n < IFMBlock; n++)
        input[n + j * IFMBlock + i * feature_map_width * feature_map_height * IFMBlock + batch * feature_map_width * feature_map_height * num_feature_maps]
        = input_ref[n * feature_map_width * feature_map_height + j + i * feature_map_width * feature_map_height * IFMBlock + batch * feature_map_width * feature_map_height * num_feature_maps];
}

// Naive convolution
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
    //float scale_in = 1.0f / (float)(1 << input_fraction);

    //for (auto itrBatch = 0; itrBatch < batch_size; ++itrBatch)
    //for (auto itrMapY = 0; itrMapY < feature_map_height; ++itrMapY)
    //for (auto itrMapX = 0; itrMapX < feature_map_width; ++itrMapX)
    //for (auto itrMapZ = 0; itrMapZ < num_feature_maps; ++itrMapZ)
    //{
    //    int32_t sum = 0;
    //    int16_t v_int = 0;
    //    float v_float = 0;
    //    int16_t v_int0 = input_ref[itrMapX + itrMapY * feature_map_width + (itrMapZ + 0) * feature_map_width * feature_map_height];
    //    for (int32_t itrN = -coeff_N / 2; itrN < coeff_N / 2 + 1; ++itrN)
    //    {

    //        if ((itrMapZ + itrN < 0) || (itrMapZ + itrN >= num_feature_maps))
    //            v_int = 0;
    //        else
    //            v_int = input_ref[itrMapX + itrMapY * feature_map_width + (itrMapZ + itrN) * feature_map_width * feature_map_height];

    //        sum += v_int * v_int;
    //    }
    //    v_float = (float)sum * scale_in * scale_in;
    //    v_float = (v_float * coeff_alpha + coeff_k);
    //    v_float = pow(v_float, coeff_beta);

    //    v_float = (float)v_int0 * scale_in / v_float;

    //    auto float2int16 = [&output_fraction](float in) {
    //        auto scaled = in * (1 << output_fraction);
    //        if (scaled < INT16_MIN)
    //            scaled = INT16_MIN;
    //        if (scaled > INT16_MAX)
    //            scaled = INT16_MAX;
    //        return static_cast<std::int16_t>(round(scaled));
    //    };
    //    output_ref[itrMapX + itrMapY * feature_map_width + (itrMapZ + 0) * feature_map_width * feature_map_height] = float2int16(v_float);
    //}
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


static void ult_nn_lrn_fp_initialize_work_item(
    nn_workload_item* &work_item,
    nn_workload_item* &input_item,
    int16_t * input,
    int16_t * output,
    uint32_t num_feature_maps,
    uint32_t feature_map_width,
    uint32_t feature_map_height,
    uint32_t batch_size,
    uint32_t input_fraction,
    uint32_t output_fraction,
    float coeff_alpha,
    float coeff_beta,
    float coeff_k)
{
    uint32_t IFMBlock = 8;
    nn_workload_data_coords_t in_out_coords =
    {
        batch_size,
        feature_map_width,
        feature_map_height,
        num_feature_maps / IFMBlock,
        IFMBlock,
        1
    };

    nn_workload_data_layout_t in_out_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_INT16
    };

    nn::nn_workload_data_t<int16_t> *output_data = new nn::nn_workload_data_t<int16_t>(in_out_coords, in_out_layout);

    work_item = new nn_workload_item();
    work_item->type = NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN;

    nn_arguments_normalization_response_across_maps_forward_i16qn_t &arguments = work_item->arguments.normalization_response_across_maps_forward_i16qn;
    arguments.alpha = coeff_alpha;
    arguments.beta = coeff_beta;
    arguments.k = coeff_k;
    arguments.n = 5u;
    arguments.fractions.input = input_fraction;
    arguments.fractions.output = output_fraction;
    work_item->output = output_data;

    //work_item->output = new nn::nn_workload_data_t<int16_t>(*output_data, nn_view_begin, nn_view_end);
    work_item->output = new nn::nn_workload_data_t<int16_t>(in_out_coords, in_out_layout);

    memcpy(work_item->output->parent->data_buffer, output, work_item->output->parent->buffer_size);

    input_item = new nn_workload_item();
    input_item->type = NN_WORK_ITEM_TYPE_INPUT;
    nn::nn_workload_data_t<int16_t> *input_data = new nn::nn_workload_data_t<int16_t>(in_out_coords, in_out_layout);

    memcpy(input_data->parent->data_buffer, input, input_data->parent->buffer_size);
    input_item->output = input_data;

    work_item->input.push_back(input_item);

}

static bool ult_nn_lrn_fp_interface_run(nn_workload_item* &work_item)
{
    bool retvalue = true;

    //device_int16::run_singlethreaded_convolve_fixedpoint_work_item(work_item);
    int16_fixedpoint::wrapper_lrn_fixedpoint_work_item(work_item);

    return retvalue;
}


bool ult_nn_lrn_fp_check_outputs(
    nn_workload_data_t* output,
    int16_t* output_ref,
    uint_least32_t num_feature_maps,
    uint_least32_t feature_map_width,
    uint_least32_t feature_map_height,
    uint_least32_t batch
    )
{
    uint32_t OFMOutBlock = 8;
    int16_t * outputOpt = (int16_t *)output->parent->data_buffer;

    uint_least32_t output_size = feature_map_width * feature_map_height * num_feature_maps * batch * sizeof(int16_t);
    int16_t* outputT = (int16_t*)_mm_malloc(output_size, 64);
    for (uint32_t b = 0; b < batch; b++)
    for (uint32_t i = 0; i < num_feature_maps / OFMOutBlock; i++)
    for (uint32_t j = 0; j < feature_map_width * feature_map_height; j++)
    for (uint32_t n = 0; n < OFMOutBlock; n++)
    {
        outputT[n + j * OFMOutBlock + i * feature_map_width * feature_map_height * OFMOutBlock + b * feature_map_width * feature_map_height * num_feature_maps]
            = output_ref[n * feature_map_width * feature_map_height + j + i * feature_map_width * feature_map_height * OFMOutBlock + b * feature_map_width * feature_map_height * num_feature_maps];
    }

    bool passed = true;
    for (uint_least32_t i = 0; i < (output_size / sizeof(int16_t)) && passed; i++)
    if ((outputT[i] <  outputOpt[i] - 3) ||
        (outputT[i] >  outputOpt[i] + 3))

        passed = false;

    _mm_free(outputT);

    return passed;
}

static void ult_nn_lrn_fp_deinitialize_work_item(nn_workload_item* &work_item)
{

    work_item->input.clear();
    delete work_item->output;

    delete work_item;

    work_item = nullptr;
}

static void ult_nn_lrn_fp_both_dealloc(
    int16_t* &input,
    int16_t* &output,
    int16_t* &input_ref,
    int16_t* &output_ref)
{
    if (input != 0)
    {
        _mm_free(input);
        input = 0;
    }

    if (output != 0)
    {
        _mm_free(output);
        output = 0;
    }

    if (input_ref != 0)
    {
        _mm_free(input_ref);
        input_ref = 0;
    }

    if (output_ref != 0)
    {
        _mm_free(output_ref);
        output_ref = 0;
    }


}


static bool ult_perform_test(
    uint32_t    batch_size,
    uint32_t    feature_map_width,
    uint32_t    feature_map_height,
    uint32_t    num_feature_maps,
    float       coeff_alpha,
    float       coeff_beta,
    float       coeff_k,
    uint32_t    input_fraction,
    uint32_t    output_fraction,
    NN_NORMALIZATION_MODE mode
    )
{
    bool return_value = true;
    bool passed = true;

    nn_workload_item* work_item = nullptr;
    nn_workload_item* input_item = nullptr;

    int16_t* input = 0;
    int16_t* output = 0;

    int16_t* input_ref = 0;
    int16_t* output_ref = 0;

    // Allocate naive and optimized buffers.
    ult_lrn_buffers_alloc(
        input,
        output,
        input_ref,
        output_ref,
        num_feature_maps,
        feature_map_width,
        feature_map_height,
        batch_size
        );

    // Initialize both buffers.
    ult_lrn_buffers_initialize(
        input,
        output,
        input_ref,
        output_ref,
        num_feature_maps,
        feature_map_width,
        feature_map_height,
        batch_size
        );

    // Naive convolution
    ult_nn_lrn_fp_naive(
        input_ref,
        output_ref,
        num_feature_maps,
        feature_map_width,
        feature_map_height,
        batch_size,
        input_fraction,
        output_fraction,
        coeff_alpha,
        coeff_beta,
        coeff_k
        );

    {
        // Perform data copy to interface test.
        ult_nn_lrn_fp_initialize_work_item(
            work_item,
            input_item,
            input,
            output,
            num_feature_maps,
            feature_map_width,
            feature_map_height,
            batch_size,
            input_fraction,
            output_fraction,
            coeff_alpha,
            coeff_beta,
            coeff_k
            );

        //    //Optimized convolution.
        passed = ult_nn_lrn_fp_interface_run(work_item);
    }

    if (passed)
    {
        //Basic check between optimized and naive versions.
        passed = ult_nn_lrn_fp_check_outputs(
            work_item->output,
            output_ref,
            num_feature_maps,
            feature_map_width,
            feature_map_height,
            batch_size);

    }

    //// Cleanup.
    ult_nn_lrn_fp_deinitialize_work_item(work_item);

    ult_nn_lrn_fp_both_dealloc(
        input,
        output,
        input_ref,
        output_ref);

    return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.

TEST(cpu_normalization_artificial_linear_latency, cpu_normalization_lnr_base)
{
    // krizhevsky
    EXPECT_EQ(true, ult_perform_test(
        1,                                 //batch_size
        13,                                //feature_map_width
        13,                                //feature_map_height
        16,                                //num_feature_maps
        0.0001,                            //coeff_alpha
        0.75,                              //coeff_beta
        2,                                 //coeff_k
        8,                                 //input_fraction
        8,                                 //output_fraction
        NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS));                             //mode

    //EXPECT_EQ(true, ult_perform_test(
    //    1,                                 //batch_size
    //    16,                                //feature_map_width
    //    16,                                //feature_map_height
    //    16,                                //num_feature_maps
    //    0.0001,                             //coeff_alpha
    //    0.75,                              //coeff_beta
    //    2,                                 //coeff_k
    //    8,                                 //input_fraction
    //    8,                                 //output_fraction
    //    NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS));                             //mode

    //EXPECT_EQ(true, ult_perform_test(
    //    1,                                 //batch_size
    //    16,                                //feature_map_width
    //    16,                                //feature_map_height
    //    16,                                //num_feature_maps
    //    0.001,                             //coeff_alpha
    //    0.75,                              //coeff_beta
    //    2,                                 //coeff_k
    //    8,                                 //input_fraction
    //    8,                                 //output_fraction
    //    NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS));                             //mode
}


