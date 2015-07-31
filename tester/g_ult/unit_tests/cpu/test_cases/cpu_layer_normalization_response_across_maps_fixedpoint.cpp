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

#include "tester/g_ult/unit_tests/cpu/naive_implementations.h"
#include "device/cpu/core/fixedpoint/layer_normalization_response_across_maps_int16_avx2.h"

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
    uint32_t IFMBlock = 16;

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
    uint32_t IFMBlock = 16;
    nn_workload_data_coords_t in_out_coords =
    {
        batch_size,
        feature_map_width,
        feature_map_height,
        num_feature_maps / IFMBlock,
        IFMBlock,
        1
    };

    nn_workload_data_layout_t in_out_layout = nn::workload_data<int16_t>::layout.pxyznq;

    nn::workload_data<int16_t> *output_data = new nn::workload_data<int16_t>(in_out_coords, in_out_layout);

    work_item = new nn_workload_item();
    work_item->type = NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN;

    nn_arguments_normalization_response_across_maps_forward_i16qn_t &arguments = work_item->arguments.normalization_response_across_maps_forward_i16qn;
    arguments.alpha = coeff_alpha;
    arguments.beta = coeff_beta;
    arguments.k = coeff_k;
    arguments.n = 5u;
    arguments.fractions.input = input_fraction;
    arguments.fractions.output = output_fraction;
    work_item->output.push_back(output_data);

    //work_item->output = new nn::workload_data<int16_t>(*output_data, nn_view_begin, nn_view_end);
    work_item->output.push_back(new nn::workload_data<int16_t>(in_out_coords, in_out_layout));

    memcpy(work_item->output[0]->parent->data_buffer, output, work_item->output[0]->parent->buffer_size);

    input_item = new nn_workload_item();
    input_item->type = NN_WORK_ITEM_TYPE_INPUT;
    nn::workload_data<int16_t> *input_data = new nn::workload_data<int16_t>(in_out_coords, in_out_layout);

    memcpy(input_data->parent->data_buffer, input, input_data->parent->buffer_size);
    input_item->output.push_back(input_data);

    work_item->input.push_back({ input_item, 0 });

    // Create primitive
    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;

    nn_device_load(&device_description);
    nn_device_interface_open(0, &device_interface_0);

    work_item->primitive = new int16_fixedpoint::normalization_response_across_maps_i16(
        coeff_k,
        5u,
        coeff_alpha,
        coeff_beta,
        static_cast<float>(1 << input_fraction),
        static_cast<float>(1 << output_fraction),
        feature_map_width,
        feature_map_height,
        num_feature_maps,
        batch_size,
        0,
        0,
        0,
        0,
        reinterpret_cast<nn_device_internal*>(device_interface_0.device));

    // Create primitive output
    work_item->output[0] = static_cast<int16_fixedpoint::normalization_response_across_maps_i16 *>(work_item->primitive)->create_outputs(false)[0];
}

static bool ult_nn_lrn_fp_interface_run(nn_workload_item* &work_item)
{
    bool retvalue = true;

    //device_int16::run_singlethreaded_convolve_fixedpoint_work_item(work_item);
    int16_fixedpoint::wrapper_lrn_fixedpoint_work_item(work_item, nullptr);

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
    uint32_t OFMOutBlock = 16;
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
    delete work_item->output[0];

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
            work_item->output[0],
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


