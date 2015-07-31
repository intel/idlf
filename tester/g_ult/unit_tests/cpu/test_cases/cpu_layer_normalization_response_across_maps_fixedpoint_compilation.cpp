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

namespace {
    void test_setup(nn_device_description_t &device_description, nn_device_interface_0_t &device_interface_0) {
        // load device & validate it has 0 as a startin interface version
        nn_device_load(&device_description);
        EXPECT_EQ(device_description.version_first, 0);
        // open interface 0
        EXPECT_EQ(0, nn_device_interface_open(0, &device_interface_0));
    }

    void test_teardown(nn_device_description_t &device_description, nn_device_interface_0_t &device_interface_0) {
        // close interface 0
        EXPECT_EQ(0, nn_device_interface_close(&device_interface_0));
        // unload device
        EXPECT_EQ(0, nn_device_unload());
    }
} //namespace

static void ult_lrn_comp_buffers_alloc(
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
static void ult_lrn_comp_buffers_initialize(
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
    uint_least32_t input_size = feature_map_width * feature_map_height * num_feature_maps * sizeof(int16_t);
    int16_t * inputT = (int16_t*)_mm_malloc(input_size, 4096);

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
                input_ref[index] = (int16_t)((value + batch) % 25600);
                inputT[index] = (int16_t)((value + batch) % 25600);
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

    //prepare zxy format
    for (size_t y = 0; y < feature_map_height; y++)
    {
        for (size_t x = 0; x < feature_map_width; x++)
        {
            for (size_t z = 0; z < num_feature_maps; z++)
            {
                input[z + x * num_feature_maps + y * num_feature_maps * feature_map_width]
                    = inputT[z * feature_map_width * feature_map_height + y * feature_map_height + x];
            }
        }
    }

    _mm_free(inputT);
}

bool ult_nn_lrn_fp_check_outputs(
    nn::data<int16_t, 3>* output,
    int16_t* output_ref,
    uint_least32_t num_feature_maps,
    uint_least32_t feature_map_width,
    uint_least32_t feature_map_height,
    uint_least32_t batch
    )
{
    int16_t * outputOpt = (int16_t *)output->buffer;
    uint32_t OFMOutBlock = 16;

    int16_t *zxyref = new int16_t[num_feature_maps * feature_map_height * feature_map_width];

    for (size_t y = 0; y < feature_map_height; y++)
    {
        for (size_t x = 0; x < feature_map_width; x++)
        {
            for (size_t z = 0; z < num_feature_maps; z++)
            {
                auto out_ref = output_ref[z * feature_map_width * feature_map_height + y * feature_map_width + x];
                zxyref[z + num_feature_maps * (x + feature_map_width * (y))] = out_ref;
            }
        }
    }

    bool passed = true;
    for (size_t y = 0; y < feature_map_height; y++)
    {
        for (size_t x = 0; x < feature_map_width; x++)
        {
            for (size_t z = 0; z < num_feature_maps; z++)
            {
                auto out_ref = output_ref[z * feature_map_width * feature_map_height + y * feature_map_width + x];
                auto out = outputOpt[z + num_feature_maps * (x + feature_map_width * (y))];

                if ((out < out_ref - 3) || (out > out_ref + 3)) {
                    passed = false;
                    break;
                }
            }
        }
    }

    return passed;
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


static void fill_workflow(
    nn_workflow_t **workflow,
    nn_device_interface_0_t *di,
    nn_workflow_item_t **input,
    nn_workflow_item_t **output,
    nn_workflow_item_t **normalization,
    uint32_t num_feature_maps,
    uint32_t feature_map_width,
    uint32_t feature_map_height,
    uint32_t batch_size,
    uint32_t input_fraction,
    uint32_t output_fraction,
    float coeff_alpha,
    float coeff_beta,
    float coeff_k
    )
{
    // workflow creation
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_create_function(workflow, 1, 1));

    // creating workflow items: input & output
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(input, 0, nullptr, 1));
    (*input)->type = NN_WORK_ITEM_TYPE_INPUT;
    (*input)->arguments.input.index = 0;
    (*input)->output_format[0].format = NN_DATA_FORMAT_3D;
    (*input)->output_format[0].format_3d = nn_output_format_3d{ { feature_map_width, feature_map_height, num_feature_maps } };

    nn_workflow_use_descriptor_t desc0 = { *input, 0 };
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(normalization, 1, &desc0, 1));
    (*normalization)->type = NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN;
    auto &arguments = (*normalization)->arguments.normalization_response_across_maps_forward_i16qn;
    arguments.alpha = coeff_alpha;
    arguments.beta = coeff_beta;
    arguments.k = coeff_k;
    arguments.n = 5u;
    arguments.fractions.input = input_fraction;
    arguments.fractions.output = output_fraction;

    nn_workflow_use_descriptor_t desc1 = { *normalization, 0 };
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(output, 1, &desc1, 1));
    (*output)->type = NN_WORK_ITEM_TYPE_OUTPUT;
    (*output)->arguments.output.index = 0;

    if (feature_map_width == 1 && feature_map_height == 1){
        (*normalization)->output_format[0].format = NN_DATA_FORMAT_1D;
        (*normalization)->output_format[0].format_1d = nn_output_format_1d{ { num_feature_maps } };
        (*output)->output_format[0].format = NN_DATA_FORMAT_1D;
        (*output)->output_format[0].format_1d = (*normalization)->output_format[0].format_1d;
    }
    else{
        (*normalization)->output_format[0].format = NN_DATA_FORMAT_3D;
        (*normalization)->output_format[0].format_3d = nn_output_format_3d{ { feature_map_width, feature_map_height, num_feature_maps } };
        (*output)->output_format[0].format = NN_DATA_FORMAT_3D;
        (*output)->output_format[0].format_3d = (*normalization)->output_format[0].format_3d;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
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

    int16_t* input = 0;
    int16_t* output = 0;

    int16_t* input_ref = 0;
    int16_t* output_ref = 0;

    // Allocate naive and optimized buffers.
    ult_lrn_comp_buffers_alloc(
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
    ult_lrn_comp_buffers_initialize(
        input,
        output,
        input_ref,
        output_ref,
        num_feature_maps,
        feature_map_width,
        feature_map_height,
        batch_size
        );

    // Naive normalization_lrn
    ult_nn_lrn_fp_comp_naive(
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

    nn_workflow_t *workflow = nullptr;
    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    test_setup(device_description, device_interface_0);

    // shorter name for function calls
    nn_device_interface_0_t &di = device_interface_0;
    nn_workflow_item_t
        *workflow_input = nullptr
        , *workflow_output = nullptr
        , *workflow_normalization = nullptr;

    fill_workflow(
        &workflow,
        &di,
        &workflow_input,
        &workflow_output,
        &workflow_normalization,
        num_feature_maps,
        feature_map_width,
        feature_map_height,
        batch_size,
        input_fraction,
        output_fraction,
        coeff_alpha,
        coeff_beta,
        coeff_k);

    // attaching input/output to workflow
    workflow->input[0] = workflow_input;
    workflow->output[0] = workflow_output;

    nn::data<int16_t, 3>* input_datas[] = { new nn::data<int16_t, 3>((int16_t *)input, num_feature_maps, feature_map_width, feature_map_height) };
    nn::data<int16_t, 3>* output_datas[] = { new nn::data<int16_t, 3>((int16_t *)output, num_feature_maps, feature_map_width, feature_map_height) };

    // compile workflow
    NN_API_STATUS status;
    nn_workload_t *workload;
    NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_I16_ZXY;
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_compile_function(&workload, di.device, workflow, &io_format, &io_format, 1));

    EXPECT_EQ(NN_API_STATUS_OK, di.workload_execute_function(workload, (void **)input_datas, (void **)output_datas, &status));


    // delete workload
    EXPECT_EQ(NN_API_STATUS_OK, di.workload_delete_function(workload));

    // delete workflow items
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_output));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_normalization));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_input));

    // delete workflow
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));

    test_teardown(device_description, device_interface_0);

    //Basic check between optimized and naive versions.
    passed = ult_nn_lrn_fp_check_outputs(
        output_datas[0],
        output_ref,
        num_feature_maps,
        feature_map_width,
        feature_map_height,
        batch_size);

    ult_nn_lrn_fp_both_dealloc(
        input,
        output,
        input_ref,
        output_ref);

    return passed;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(cpu_int16_normalization_lrn_fixedpoint_compilation, cpu_normalization_lrn_Overfeat)
{
    EXPECT_EQ(true, ult_perform_test(
        1,                                 //batch_size
        13,                                //feature_map_width
        13,                                //feature_map_height
        256,                               //num_feature_maps
        0.0001,                            //coeff_alpha
        0.75,                              //coeff_beta
        2,                                 //coeff_k
        8,                                 //input_fraction
        8,                                 //output_fraction
        NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS));//mode

    //// krizhevsky
    //EXPECT_EQ(true, ult_perform_test(
    //    1,                                 //batch_size
    //    13,                                //feature_map_width
    //    13,                                //feature_map_height
    //    256,                               //num_feature_maps
    //    0.0001,                            //coeff_alpha
    //    0.75,                              //coeff_beta
    //    2,                                 //coeff_k
    //    8,                                 //input_fraction
    //    8,                                 //output_fraction
    //    NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS));//mode
}
