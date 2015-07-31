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

// disabled until weight format will be in "plain format"
// merge to nn::data-releated changes was not done - both steps should be done in one step

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
static void ult_nn_convolution_fixedpoint_comp_both_initialize_matrices(
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

    uint32_t IFMBlock = 16;
    if (num_input_feature_maps == 4) IFMBlock = 4;
    uint32_t OFMBlock = 32;
    uint32_t OFMpBlock = 2;
    if (num_output_feature_maps == 3072 && num_input_feature_maps == 1024)
    {
        IFMBlock = 16;
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
                //ult_nn_convolution_fixedpoint_comp_optimized_set_input_value(inputT, input_feature_map_width, num_input_feature_maps, column, row, input_map, value, offset);
                ult_nn_merge_convolution_fixedpoint_naive_set_input_value(inputT, input_feature_map_width, input_feature_map_height, column, row, input_map, value, offset);
                ult_nn_merge_convolution_fixedpoint_naive_set_input_value(input_ref, input_feature_map_width, input_feature_map_height, column, row, input_map, value, offset);
                value++;
            }
        }
    }

    for (uint_least32_t outmapa = 0; outmapa < num_output_feature_maps; outmapa++)
    {
        for (uint_least32_t input_map = 0; input_map < num_input_feature_maps; input_map++)
        {
            uint_least32_t element = 0;
            int16_t value = input_map * 0x0100 + outmapa * 0x2000;
            for (uint_least32_t row = 0; row < kernel_height; row++)
            {
                for (uint_least32_t column = 0; column < kernel_width; column++)
                {
                    element++;
                    uint_least32_t offset;
                    //ult_nn_convolution_fixedpoint_comp_optimized_set_kernel_value
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

    //prepare right input layout for naive implementation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
static bool ult_nn_convolution_fixedpoint_comp_check_outputs(
    nn::data<int16_t, 3>* output,
    int16_t* output_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t center_x,
    uint_least32_t center_y
    )
{
    // zxy -> xyz
    uint_least32_t output_size = (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y) * num_output_feature_maps * sizeof(int16_t);
    int16_t* outputT = (int16_t*)_mm_malloc(output_size, 64);
    int16_t * outputOpt = (int16_t *)output->buffer;
    uint32_t OFMOutBlock = 16;

    for (size_t y = 0; y < output_feature_map_height; y++)
    {
        for (size_t x = 0; x < output_feature_map_width; x++)
        {
            for (size_t z = 0; z < num_output_feature_maps; z++)
            {
                outputT[z * output_feature_map_width * output_feature_map_height + y * output_feature_map_height + x]
                    = outputOpt[z + x * num_output_feature_maps + y * num_output_feature_maps * output_feature_map_width];
            }
        }
    }

    bool passed = true;
    for (uint_least32_t i = 0; i < (output_size / sizeof(int16_t)) && passed; i++)
    if (output_ref[i] != outputT[i])
        passed = false;

    _mm_free(outputT);

    return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_convolution_fixedpoint_comp_both_alloc(
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
    uint_least32_t output_size = (output_width + 2 * center_x) * (output_height + 2 * center_y) * num_output_feature_maps * sizeof(int16_t);
    uint_least32_t bias_size = num_output_feature_maps * sizeof(int32_t);
    uint_least32_t kernel_size = num_input_feature_maps * num_output_feature_maps * kernel_width * kernel_height * sizeof(int16_t);

    input_ref = (int16_t*)_mm_malloc(input_size, 4096);
    output_ref = (int16_t*)_mm_malloc(output_size, 4096);
    biases_ref = (int32_t*)_mm_malloc(bias_size, 4096);
    kernel_ref = (int16_t*)_mm_malloc(kernel_size, 4096);

    input_size = input_width * input_height * num_input_feature_maps * sizeof(int16_t);
    output_size = (output_width + 2 * center_x) * (output_height + 2 * center_y) * num_output_feature_maps * sizeof(int16_t);
    kernel_size = num_input_feature_maps * num_output_feature_maps * kernel_width * kernel_height * sizeof(int32_t);

    input = (int16_t*)_mm_malloc(input_size, 4096);
    output = (int16_t*)_mm_malloc(output_size, 4096);
    biases = (int32_t*)_mm_malloc(bias_size, 4096);
    kernel = (int16_t*)_mm_malloc(kernel_size, 4096);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_convolution_fixedpoint_comp_both_dealloc(
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
    nn_workflow_item_t **convolution,
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
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    NN_ACTIVATION_FUNCTION activation
    )
{
    nn_workload_data_layout_t bias_layout = nn::workload_data<int32_t>::layout.xnyzpq;

    nn_workload_data_layout_t weight_layout = nn::workload_data<int16_t>::layout.nxyzpq;

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
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(convolution, 1, &desc0, 1));
    (*convolution)->type = NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT;
    auto &arguments = (*convolution)->arguments.forward_convolution_pooling_fixedpoint;
    arguments.stride[0] = kernel_stride_x;
    arguments.stride[1] = kernel_stride_y;
    arguments.padding = NN_PADDING_MODE_DATA_OR_ZERO;
    arguments.center_offset[0] = center_x;
    arguments.center_offset[1] = center_y;
    arguments.activation.basic_arguments.function = activation;
    arguments.activation.fractions.accumulator = accumulator_fraction;
    arguments.activation.fractions.output = output_fraction;
    arguments.biases = bias_data;
    arguments.weights = weight_data;

    (*convolution)->output_format[0].format = NN_DATA_FORMAT_3D;
    (*convolution)->output_format[0].format_3d = nn_output_format_3d{ { ((input_feature_map_width - kernel_width) / kernel_stride_x + 1) / 2, ((input_feature_map_height - kernel_height) / kernel_stride_y + 1) / 2, num_output_feature_maps } };

    //memcpy(arguments.biases->parent->data_buffer, bias, arguments.biases->parent->buffer_size);
    //memcpy(arguments.weights->parent->data_buffer, weights, arguments.weights->parent->buffer_size);

    nn_workflow_use_descriptor_t desc1 = { *convolution, 0 };
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(output, 1, &desc1, 1));
    (*output)->type = NN_WORK_ITEM_TYPE_OUTPUT;
    (*output)->arguments.output.index = 0;
    (*output)->output_format[0].format = NN_DATA_FORMAT_3D;
    (*output)->output_format[0].format_3d = (*convolution)->output_format[0].format_3d;
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
    uint_least32_t pool_stride_x,
    uint_least32_t pool_stride_y,
    uint_least32_t pool_size_x,
    uint_least32_t pool_size_y,
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    uint_least32_t center_x,
    uint_least32_t center_y,
    NN_ACTIVATION_FUNCTION activation,
    NN_POOLING_MODE mode)
{
    uint32_t IFMBlock = 16;
    uint32_t OFMOutBlock = 16;

    nn_workload_item* work_item = nullptr;
    nn_workload_item* work_items[12];

    nn_workload_item* input_item = nullptr;
    nn_workload_item* input_items[12];

    std::fill_n(work_items, 12, nullptr);

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

    uint_least32_t output_feature_map_width = (((input_feature_map_width - kernel_width) / kernel_stride_x + 1) - pool_size_x) / pool_stride_x + 1;
    uint_least32_t output_feature_map_height = (((input_feature_map_height - kernel_height) / kernel_stride_y + 1) - pool_size_y) / pool_stride_y + 1;

    uint_least32_t output_feature_map_width_int = (input_feature_map_width - kernel_width) / kernel_stride_x + 1;
    uint_least32_t output_feature_map_height_int = (input_feature_map_height - kernel_height) / kernel_stride_y + 1;

    num_output_feature_maps += (C_simd_width - (num_output_feature_maps % C_simd_width)) % C_simd_width;

    // Allocate naive and optimized buffers.
    ult_nn_convolution_fixedpoint_comp_both_alloc(
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
        center_y);

    // Initialize both buffers.
    ult_nn_convolution_fixedpoint_comp_both_initialize_matrices(
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
        center_y);

    // Naive maxpooling.
    ult_nn_maxpooling_naive_pooling_int16_fixedpoint(
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        output_feature_map_width_int,
        output_feature_map_height_int,
        input_feature_map_width,
        input_feature_map_height,
        kernel_width,
        kernel_height,
        kernel_stride_x,
        kernel_stride_y,
        pool_size_x,
        pool_size_y,
        pool_stride_x,
        pool_stride_y,
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
        *workflow_input = nullptr
        , *workflow_output = nullptr
        , *workflow_convolution = nullptr;

    fill_workflow(
        &workflow,
        &di,
        &workflow_input,
        &workflow_output,
        &workflow_convolution,
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
        accumulator_fraction,
        output_fraction,
        activation);

    // attaching input/output to workflow
    workflow->input[0] = workflow_input;
    workflow->output[0] = workflow_output;

    nn::data<int16_t, 3>* input_datas[] = { new nn::data<int16_t, 3>((int16_t *)input, num_input_feature_maps, input_feature_map_width, input_feature_map_height ) };
    nn::data<int16_t, 3>* output_datas[] = { new nn::data<int16_t, 3>((int16_t *)output, num_output_feature_maps, output_feature_map_width, output_feature_map_height) };
    if (num_input_feature_maps == 4)
        IFMBlock = 4;


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
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_convolution));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_input));

    // delete workflow
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));

    test_teardown(device_description, device_interface_0);


    // Basic check between optimized and naive versions.
    passed = ult_nn_convolution_fixedpoint_comp_check_outputs(
        output_datas[0],
        output_ref,
        num_output_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        center_x,
        center_y);

    ult_nn_convolution_fixedpoint_comp_both_dealloc(
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
// Tests.
TEST(cpu_int16_convolution_maxpooling2x2_fixedpoint_compilation, cpu_convolution_maxpooling2x2_stride1)
{
    //EXPECT_EQ(true, ult_perform_test(32, 32, 4, 4, 3, 3, 1, 1, 2, 2, 2, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_NONE, NN_POOLING_MODE_MAX));
    //EXPECT_EQ(true, ult_perform_test(32, 32, 15, 15, 3, 3, 1, 1, 2, 2, 2, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU, NN_POOLING_MODE_MAX));
    EXPECT_EQ(true, ult_perform_test(32, 64, 16, 16, 3, 3, 1, 1, 2, 2, 2, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_NONE, NN_POOLING_MODE_MAX));
    EXPECT_EQ(true, ult_perform_test(32, 32, 16, 16, 5, 5, 1, 1, 2, 2, 2, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU, NN_POOLING_MODE_MAX));

    //// Overfeat
    //// Stage: 1
    //EXPECT_EQ(true, ult_perform_test(96, 4, 231, 231, 11, 11, 4, 4, 2, 2, 2, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU, NN_POOLING_MODE_MAX));
    //EXPECT_EQ(true, ult_perform_test(96, 4, 231, 231, 11, 11, 4, 4, 2, 2, 2, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_NONE, NN_POOLING_MODE_MAX));

    //// Stage: 2
    //EXPECT_EQ(true, ult_perform_test(256, 96, 28, 28, 5, 5, 1, 1, 2, 2, 2, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU, NN_POOLING_MODE_MAX));

    //// Stage: 5
    //EXPECT_EQ(true, ult_perform_test(1024, 1024, 14, 14, 3, 3, 1, 1, 2, 2, 2, 2, 16, 0, 0, 0, NN_ACTIVATION_FUNCTION_RELU, NN_POOLING_MODE_MAX)); // too long
}