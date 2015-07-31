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

const uint32_t C_simd_width = sizeof(__m256) / sizeof(int32_t);

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

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_merge_convolution_fixedpoint_both_alloc(
    int16_t* &input,
    int16_t* &output,
    int32_t* &biases,
    int16_t* &kernel,
    int16_t* &input_ref,
    int16_t* &output_ref,
    int32_t* &biases_ref,
    int16_t* &kernel_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_width,
    uint_least32_t output_height,
    uint_least32_t input_width,
    uint_least32_t input_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t center_x,
    uint_least32_t center_y
    )
{
    uint_least32_t input_size = input_width * input_height * num_input_feature_maps * sizeof(int16_t);
    uint_least32_t output_size = (output_width + 2 * center_x) * (output_height + 2 * center_y) * num_output_feature_maps * NUM_MERGED_CONVOLUTIONS * sizeof(int16_t);
    uint_least32_t bias_size = num_output_feature_maps * sizeof(int32_t);
    uint_least32_t kernel_size = num_input_feature_maps * num_output_feature_maps * kernel_width * kernel_height * sizeof(int16_t);

    input_ref = (int16_t*)_mm_malloc(input_size, 4096);
    output_ref = (int16_t*)_mm_malloc(output_size, 4096);
    biases_ref = (int32_t*)_mm_malloc(bias_size, 4096);
    kernel_ref = (int16_t*)_mm_malloc(kernel_size, 4096);

    input_size = input_width * input_height * num_input_feature_maps * sizeof(int16_t);
    // output_size = (output_width + 2 * center_x) * (output_height + 2 * center_y) * num_output_feature_maps * sizeof(int16_t);
    kernel_size = num_input_feature_maps * num_output_feature_maps * kernel_width * kernel_height * sizeof(int32_t);

    input = (int16_t*)_mm_malloc(input_size, 4096);
    output = (int16_t*)_mm_malloc(output_size, 4096);
    biases = (int32_t*)_mm_malloc(bias_size, 4096);
    kernel = (int16_t*)_mm_malloc(kernel_size, 4096);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_merge_convolution_fixedpoint_both_dealloc(
    int16_t* &input,
    int16_t* &output,
    int32_t* &biases,
    int16_t* &kernel,
    int16_t* &input_ref,
    int16_t* &output_ref,
    int32_t* &biases_ref,
    int16_t* &kernel_ref)
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

    if (biases != 0)
    {
        _mm_free(biases);
        biases = 0;
    }

    if (kernel != 0)
    {
        _mm_free(kernel);
        kernel = 0;
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

    if (biases_ref != 0)
    {
        _mm_free(biases_ref);
        biases_ref = 0;
    }

    if (kernel_ref != 0)
    {
        _mm_free(kernel_ref);
        kernel_ref = 0;
    }
}

static void fill_workflow(
    nn_workflow_t **workflow,
    nn_device_interface_0_t *di,
    nn_workflow_item_t **input,
    nn_workflow_item_t **output,
    nn_workflow_item_t **merge,
    nn_workflow_item_t **convolution,
    nn_workflow_item_t **pooling,
    nn_workflow_item_t **last_convolution,
    const std::int32_t* bias,
    const std::int16_t* weights,
    uint_least32_t num_input_feature_maps,
    uint_least32_t num_output_feature_maps,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    uint_least32_t center_x,
    uint_least32_t center_y,
    uint8_t merge_axis,
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    NN_ACTIVATION_FUNCTION activation
    )
{
    nn::data<int32_t, 1> *bias_data = new nn::data<int32_t, 1>((int32_t *)bias, num_output_feature_maps);
    nn::data<int16_t, 4> *weight_data = new nn::data<int16_t, 4>((int16_t *)weights, kernel_width, kernel_height, num_input_feature_maps, num_output_feature_maps);

    // workflow creation
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_create_function(workflow, 1, 1));

    // creating workflow items: input & output
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(input, 0, nullptr, 1));
    (*input)->type = NN_WORK_ITEM_TYPE_INPUT;
    (*input)->arguments.input.index = 0;
    (*input)->output_format[0].format = NN_DATA_FORMAT_3D;
    (*input)->output_format[0].format_3d = nn_output_format_3d{ { input_feature_map_width, input_feature_map_height, num_input_feature_maps } };

    nn_workflow_use_descriptor_t desc0 = { *input, 0 };
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(&convolution[0], 1, &desc0, 1));
    convolution[0]->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
    auto &arguments1 = convolution[0]->arguments.forward_convolution_int16_fixedpoint;
    arguments1.stride[0] = kernel_stride_x;
    arguments1.stride[1] = kernel_stride_y;
    arguments1.padding = NN_PADDING_MODE_DATA_OR_ZERO;
    arguments1.center_offset[0] = center_x;
    arguments1.center_offset[1] = center_y;
    arguments1.activation.basic_arguments.function = activation;
    arguments1.activation.fractions.accumulator = accumulator_fraction;
    arguments1.activation.fractions.output = output_fraction;
    arguments1.biases = bias_data;
    arguments1.weights = weight_data;

    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(&convolution[1], 1, &desc0, 1));
    convolution[1]->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
    auto &arguments2 = convolution[1]->arguments.forward_convolution_int16_fixedpoint;
    arguments2.stride[0] = kernel_stride_x;
    arguments2.stride[1] = kernel_stride_y;
    arguments2.padding = NN_PADDING_MODE_DATA_OR_ZERO;
    arguments2.center_offset[0] = center_x;
    arguments2.center_offset[1] = center_y;
    arguments2.activation.basic_arguments.function = activation;
    arguments2.activation.fractions.accumulator = accumulator_fraction;
    arguments2.activation.fractions.output = output_fraction;
    arguments2.biases = bias_data;
    arguments2.weights = weight_data;

    uint32_t out_width = (input_feature_map_width - kernel_width) / kernel_stride_x + 1;
    uint32_t out_height = (input_feature_map_height - kernel_height) / kernel_stride_y + 1;
    uint32_t out_feature_maps;

    if (out_width == 1 && out_height == 1) {
        convolution[0]->output_format[0].format = NN_DATA_FORMAT_1D;
        convolution[0]->output_format[0].format_1d = nn_output_format_1d{ { num_output_feature_maps } };
        convolution[1]->output_format[0].format = NN_DATA_FORMAT_1D;
        convolution[1]->output_format[0].format_1d = nn_output_format_1d{ { num_output_feature_maps } };
    }
    else
    {
        convolution[0]->output_format[0].format = NN_DATA_FORMAT_3D;
        convolution[0]->output_format[0].format_3d = nn_output_format_3d{ { out_width, out_height, num_output_feature_maps } };
        convolution[1]->output_format[0].format = NN_DATA_FORMAT_3D;
        convolution[1]->output_format[0].format_3d = nn_output_format_3d{ { out_width, out_height, num_output_feature_maps } };
    }

    out_width = merge_axis == 0 ? out_width * NUM_MERGED_CONVOLUTIONS : out_width;
    out_height = merge_axis == 1 ? out_height * NUM_MERGED_CONVOLUTIONS : out_height;
    out_feature_maps = merge_axis == 2 ? num_output_feature_maps * NUM_MERGED_CONVOLUTIONS : num_output_feature_maps;

    nn_workflow_use_descriptor_t desc1[] = { { convolution[0], 0 }, { convolution[1], 0 } };
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(merge, 2, desc1, 1));
    (*merge)->type = NN_WORK_ITEM_TYPE_MERGE;
    (*merge)->arguments.forward_merge.axis = merge_axis;
    (*merge)->output_format[0].format = NN_DATA_FORMAT_3D;
    (*merge)->output_format[0].format_3d = nn_output_format_3d{ { out_width, out_height, out_feature_maps } };

    const size_t pool_stride_y = 2;
    const size_t pool_stride_x = 2;
    const size_t pool_size_x = 3;
    const size_t pool_size_y = 3;

    uint32_t pooling_out_width = out_width / pool_stride_x;
    uint32_t pooling_out_height = out_height / pool_stride_y;

    nn_workflow_use_descriptor_t desc2 = { *merge, 0 };
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(pooling, 1, &desc2, 1));
    (*pooling)->type = NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT;
    auto &arguments = (*pooling)->arguments.forward_pooling_fixedpoint;
    arguments.pool_stride[0] = pool_stride_x;
    arguments.pool_stride[1] = pool_stride_y;
    arguments.pool_size[0] = pool_size_x;
    arguments.pool_size[1] = pool_size_y;

    (*pooling)->output_format[0].format = NN_DATA_FORMAT_3D;
    (*pooling)->output_format[0].format_3d = nn_output_format_3d{ { pooling_out_width, pooling_out_height, out_feature_maps } };

    nn_workflow_use_descriptor_t desc3 = { *pooling, 0 };
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(output, 1, &desc3, 1));
    (*output)->type = NN_WORK_ITEM_TYPE_OUTPUT;
    (*output)->arguments.output.index = 0;

    if (pooling_out_width == 1 && pooling_out_height == 1) {
        (*output)->output_format[0].format = NN_DATA_FORMAT_1D;
        (*output)->output_format[0].format_1d = (*pooling)->output_format[0].format_1d;
    }
    else
    {
        (*output)->output_format[0].format = NN_DATA_FORMAT_3D;
        (*output)->output_format[0].format_3d = (*pooling)->output_format[0].format_3d;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static bool ult_perform_test(
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t kernel_width,
    uint_least32_t kernel_height,
    uint_least32_t kernel_stride_x,
    uint_least32_t kernel_stride_y,
    uint8_t merge_axis,
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    uint_least32_t center_x,
    uint_least32_t center_y,
    NN_ACTIVATION_FUNCTION activation)
{
    nn_workload_item* work_item = nullptr;
    nn_workload_item* input_item = nullptr;

    bool passed = false;

    int16_t* input = 0;
    int16_t* output = 0;
    int32_t* biases = 0;
    int16_t* kernel = 0;

    int16_t* input_ref = 0;
    int16_t* output_ref = 0;
    int32_t* biases_ref = 0;
    int16_t* kernel_ref = 0;

    uint32_t NoWItems = 1;

    uint_least32_t output_feature_map_width = (input_feature_map_width - kernel_width) / kernel_stride_x + 1;
    uint_least32_t output_feature_map_height = (input_feature_map_height - kernel_height) / kernel_stride_y + 1;

    num_output_feature_maps += (C_simd_width - (num_output_feature_maps % C_simd_width)) % C_simd_width;

    // Allocate naive and optimized buffers.
    ult_nn_merge_convolution_fixedpoint_both_alloc(
        input,
        output,
        biases,
        kernel,
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        input_feature_map_width,
        input_feature_map_height,
        kernel_width,
        kernel_height,
        center_x,
        center_y
        );

    // Initialize both buffers.
    ult_nn_merge_convolution_fixedpoint_both_initialize_matrices(
        input,
        output,
        biases,
        kernel,
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        input_feature_map_width,
        input_feature_map_height,
        kernel_width,
        kernel_height,
        center_x,
        center_y
        );

    // Naive convolution
    ult_nn_convolve_fp_naive(
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        input_feature_map_width,
        input_feature_map_height,
        kernel_width,
        kernel_height,
        kernel_stride_x,
        kernel_stride_y,
        accumulator_fraction,
        output_fraction,
        center_x,
        center_y,
        activation);

    nn_workflow_t *workflow = nullptr;
    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    test_setup(device_description, device_interface_0);

    // shorter name for function calls
    nn_device_interface_0_t &di = device_interface_0;
    nn_workflow_item_t
        *workflow_input = nullptr,
        *workflow_output = nullptr,
        *workflow_merge = nullptr,
        *workflow_pooling = nullptr,
        *workflow_last_convolution = nullptr;
    nn_workflow_item_t *workflow_convolution[2];
    workflow_convolution[0] = nullptr;
    workflow_convolution[1] = nullptr;

    fill_workflow(
        &workflow,
        &di,
        &workflow_input,
        &workflow_output,
        &workflow_merge,
        workflow_convolution,
        &workflow_pooling,
        &workflow_last_convolution,
        biases,
        kernel,
        num_input_feature_maps,
        num_output_feature_maps,
        input_feature_map_width,
        input_feature_map_height,
        kernel_width,
        kernel_height,
        kernel_stride_x,
        kernel_stride_y,
        center_x,
        center_y,
        merge_axis,
        accumulator_fraction,
        output_fraction,
        activation);

    // attaching input/output to workflow
    workflow->input[0] = workflow_input;
    workflow->output[0] = workflow_output;

    nn::data<int16_t, 3>* input_datas[] = { new nn::data<int16_t, 3>((int16_t *)input, num_input_feature_maps, input_feature_map_width, input_feature_map_height) };
    nn::data<int16_t, 3>* output_datas[] = { new nn::data<int16_t, 3>((int16_t *)output, workflow_pooling->output_format[0].format_3d.size[2], workflow_pooling->output_format[0].format_3d.size[0], workflow_pooling->output_format[0].format_3d.size[1]) };

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
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_last_convolution));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_pooling));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_convolution[0]));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_convolution[1]));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_merge));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_input));

    // delete workflow
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));

    test_teardown(device_description, device_interface_0);

    //Basic check between optimized and naive versions.
    passed = ult_nn_merge_convolution_fixedpoint_check_outputs(
        output_datas[0],
        output_ref,
        num_output_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        center_x,
        center_y,
        merge_axis);

    ult_nn_merge_convolution_fixedpoint_both_dealloc(
        input,
        output,
        biases,
        kernel,
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref);

    return passed;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(cpu_int16_merge_convolution_fixedpoint_compilation, cpu_merge_convolution_Overfeat)
{
    //krizhevsky
    EXPECT_EQ(true, ult_perform_test(128, 128, 27, 27, 3, 3, 1, 1, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_NONE));
    EXPECT_EQ(true, ult_perform_test(128, 128, 13, 13, 3, 3, 1, 1, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_NONE));

    //EXPECT_EQ(true, ult_perform_test(128, 128, 29, 29, 3, 3, 1, 1, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_NONE));
    //EXPECT_EQ(true, ult_perform_test(32, 32, 4, 4, 3, 3, 1, 1, 0, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU));
    //EXPECT_EQ(true, ult_perform_test(32, 32, 15, 15, 3, 3, 1, 1, 1, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_NONE));
    //EXPECT_EQ(true, ult_perform_test(64, 64, 16, 16, 3, 3, 1, 1, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU));
    //EXPECT_EQ(true, ult_perform_test(384, 32, 16, 16, 5, 5, 1, 1, 0, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_NONE));
    //EXPECT_EQ(true, ult_perform_test(384, 32, 5, 5, 5, 5, 1, 1, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_NONE));

    // Overfeat
    // Stage 2
    //EXPECT_EQ(true, ult_perform_test(256, 96, 28, 28, 5, 5, 1, 1, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU));

    // Stage 3
    //EXPECT_EQ(true, ult_perform_test(512, 256, 14, 14, 3, 3, 1, 1, 1, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU));

    // Stage 4
    //EXPECT_EQ(true, ult_perform_test(1024, 512, 14, 14, 3, 3, 1, 1, 0, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU));

    // Stage 6
    //EXPECT_EQ(true, ult_perform_test(3072, 1024, 6, 6, 6, 6, 1, 1, 1, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU)); // NOT WORKING
}
