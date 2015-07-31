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

#include "gtest/gtest.h"

#include "device/api/nn_device_api.h"
#include "device/common/nn_workload_data.h"
#include "device/api/nn_device_interface_0.h"

#include <random>
#include <cstdint>
#include <vector>

///////////////////////////////////////////////////////////////////////////////////////////////////

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


TEST(api_workloads, workflow_create_delete)
{
    // test configuration
    const uint32_t inputs_min = 1u;
    const uint32_t inputs_max = 10u;
    const uint32_t outputs_min = 1u;
    const uint32_t outputs_max = 10;
    const auto number_of_runs = 100;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    test_setup(device_description, device_interface_0);

    // shorter name for function calls
    nn_device_interface_0_t &di = device_interface_0;

    // create workflow with various number of inputs & outputs
    {
        std::default_random_engine random_engine(1);
        std::uniform_int_distribution<uint32_t>  input_distribution(inputs_min, inputs_max);
        std::uniform_int_distribution<uint32_t> output_distribution(outputs_min, outputs_max);
        for (auto run = 0u; run<number_of_runs; ++run) {
            nn_workflow_t *workflow = nullptr;
            const auto  input_count = input_distribution(random_engine);
            const auto output_count = output_distribution(random_engine);
            // creation
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&workflow, input_count, output_count));
            // deletion
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));
        }
    }

    test_teardown(device_description, device_interface_0);
}


TEST(api_workloads, workflow_item_create_delete)
{
    // test configuration
    const uint32_t inputs_min = 1u;
    const uint32_t inputs_max = 10u;
    const auto number_of_runs = 1000;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    test_setup(device_description, device_interface_0);

    // shorter name for function calls
    nn_device_interface_0_t &di = device_interface_0;

    // create workflow item with various number of inputs
    {
        std::default_random_engine random_engine(1);
        std::uniform_int_distribution<uint32_t>  input_distribution(inputs_min, inputs_max);
        for (auto run = 0u; run<number_of_runs; ++run) {
            nn_workflow_item_t *workflow_item = nullptr;
            const auto  input_count = input_distribution(random_engine);
            // creation of input nodes
            nn_workflow_use_descriptor_t *input = new nn_workflow_use_descriptor_t[input_count];
            for (auto index = 0u; index<input_count; ++index) {
                nn_workflow_item_t *temp = nullptr;
                EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&temp, 0, nullptr, 1));
                input[index] = { temp, 0 };
            }
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&workflow_item, input_count, input, 1));
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_item));
            for (auto index = 0u; index<input_count; ++index) {
                EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input[index].item));
            }
        }
    }

    test_teardown(device_description, device_interface_0);
}

TEST(api_workloads, workflow_trivial)
{
    // test configuration
    const auto number_of_runs = 100;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    test_setup(device_description, device_interface_0);

    // shorter name for function calls
    nn_device_interface_0_t &di = device_interface_0;

    // create workflow with 1 input connected directly to 1 output
    {
        for (auto run = 0u; run<number_of_runs; ++run) {
            nn_workflow_t *workflow = nullptr;

            // workflow creation
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&workflow, 1, 1));

            // creating workflow items: input & output
            nn_workflow_item_t  *input = nullptr
                , *output = nullptr;

            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&input, 0, nullptr, 1));
            input->type = NN_WORK_ITEM_TYPE_INPUT;
            input->arguments.input.index = 0;
            input->output_format[0].format = NN_DATA_FORMAT_1D;
            input->output_format[0].format_1d =  { { 1 } };

            nn_workflow_use_descriptor_t desc = { input, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&output, 1, &desc, 1));
            output->type = NN_WORK_ITEM_TYPE_OUTPUT;
            output->arguments.output.index = 0;
            output->output_format[0].format = NN_DATA_FORMAT_1D;
            output->output_format[0].format_1d = { { 1 } };

            // attaching input/output to workflow
            workflow->input[0] = input;
            workflow->output[0] = output;

            // delete workflow items
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(output));
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input));

            // delete workflow
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));
        }
    }

    test_teardown(device_description, device_interface_0);
}

TEST(api_workloads, workflow_trivial_compilation)
{
    // test configuration
    const auto number_of_runs = 100;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    test_setup(device_description, device_interface_0);

    // shorter name for function calls
    nn_device_interface_0_t &di = device_interface_0;

    // create & compile workflow with 1 input connected directly to 1 output
    {
        for (auto run = 0u; run<number_of_runs; ++run) {
            nn_workflow_t *workflow = nullptr;

            // workflow creation
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&workflow, 1, 1));

            // creating workflow items: input & output
            nn_workflow_item_t  *input = nullptr
                , *output = nullptr;

            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&input, 0, nullptr, 1));
            input->type = NN_WORK_ITEM_TYPE_INPUT;
            input->arguments.input.index = 0;
            input->output_format[0].format = NN_DATA_FORMAT_1D;
            input->output_format[0].format_1d = { { 1 } };
            
            nn_workflow_use_descriptor_t desc = { input, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&output, 1, &desc, 1));
            output->type = NN_WORK_ITEM_TYPE_OUTPUT;
            output->arguments.output.index = 0;
            output->output_format[0].format = NN_DATA_FORMAT_1D;
            output->output_format[0].format_1d = { { 1 } };

            // attaching input/output to workflow
            workflow->input[0] = input;
            workflow->output[0] = output;

            // compile workflow
            nn_workload_t *workload;
            NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_F32_1D;
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_compile_function(&workload, di.device, workflow, &io_format, &io_format, 1));

            // delete workload
            EXPECT_EQ(NN_API_STATUS_OK, di.workload_delete_function(workload));

            // delete workflow items
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(output));
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input));

            // delete workflow
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));
        }
    }

    test_teardown(device_description, device_interface_0);
}

TEST(api_workloads, workflow_in_pooling_out)
{
    // test configuration
    const auto number_of_runs = 100;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    test_setup(device_description, device_interface_0);

    // shorter name for function calls
    nn_device_interface_0_t &di = device_interface_0;

    // create workflow with 1 input connected directly to 1 output
    {
        for (auto run = 0u; run<number_of_runs; ++run) {
            nn_workflow_t *workflow = nullptr;

            // workflow creation
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&workflow, 1, 1));

            // creating workflow items: input & output
            nn_workflow_item_t  *input = nullptr
                , *pooling = nullptr
                , *output = nullptr;


            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&input, 0, nullptr, 1));
            input->type = NN_WORK_ITEM_TYPE_INPUT;
            input->arguments.input.index = 0;
            input->output_format[0].format = NN_DATA_FORMAT_2D;
            input->output_format[0].format_2d = { { 2, 2 } };

            nn_workflow_use_descriptor_t desc0 = { input, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&pooling, 1, &desc0, 1));
            pooling->type = NN_WORK_ITEM_TYPE_POOLING;
            pooling->arguments.forward_pooling = nn_arguments_forward_pooling_t{
                {1, 1},             /* stride during filtering operation */
                {2, 2},             /* pooling area size */
                NN_POOLING_MODE_MAX /* pooling mode */
            };
            pooling->output_format[0].format = NN_DATA_FORMAT_2D;
            pooling->output_format[0].format_2d = { { 1, 1 } };


            nn_workflow_use_descriptor_t desc1 = { pooling, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&output, 1, &desc1, 1));
            output->type = NN_WORK_ITEM_TYPE_OUTPUT;
            output->arguments.output.index = 0;
            output->output_format[0].format = NN_DATA_FORMAT_2D;
            output->output_format[0].format_2d = { { 1, 1 } };

            // attaching input/output to workflow
            workflow->input[0] = input;
            workflow->output[0] = output;

            // delete workflow items
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(output));
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(pooling));
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input));

            // delete workflow
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));
        }
    }

    test_teardown(device_description, device_interface_0);
}

TEST(api_workloads, workflow_in_pooling_out_compilation)
{
    // test configuration
    const auto number_of_runs = 100;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;
    test_setup(device_description, device_interface_0);

    // shorter name for function calls
    nn_device_interface_0_t &di = device_interface_0;

    // create workflow with 1 input, pooling and 1 output
    {
        for (auto run = 0u; run<number_of_runs; ++run) {
            nn_workflow_t *workflow = nullptr;

            // workflow creation
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&workflow, 1, 1));

            // creating workflow items: input & output
            nn_workflow_item_t  *input = nullptr
                , *pooling = nullptr
                , *output = nullptr;
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&input, 0, nullptr, 1));
            input->type = NN_WORK_ITEM_TYPE_INPUT;
            input->arguments.input.index = 0;
            input->output_format[0].format = NN_DATA_FORMAT_2D;
            input->output_format[0].format_2d = nn_output_format_2d{ { 32, 32 } };

            nn_workflow_use_descriptor_t desc0 = { input, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&pooling, 1, &desc0, 1));
            pooling->type = NN_WORK_ITEM_TYPE_POOLING;
            pooling->arguments.forward_pooling = nn_arguments_forward_pooling_t{
                {1, 1},             /* stride during filtering operation */
                {2, 2},             /* pooling area size */
                NN_POOLING_MODE_MAX /* pooling mode */
            };
            pooling->output_format[0].format = NN_DATA_FORMAT_2D;
            pooling->output_format[0].format_2d = nn_output_format_2d{ { 16, 16 } };

            nn_workflow_use_descriptor_t desc1 = { pooling, 0 };
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&output, 1, &desc1, 1));
            output->type = NN_WORK_ITEM_TYPE_OUTPUT;
            output->arguments.output.index = 0;
            output->output_format[0].format = NN_DATA_FORMAT_2D;
            output->output_format[0].format_2d = nn_output_format_2d{ { 16, 16 } };

            // attaching input/output to workflow
            workflow->input[0] = input;
            workflow->output[0] = output;

            // compile workflow
            nn_workload_t *workload;
            NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_F32_2D;
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_compile_function(&workload, di.device, workflow, &io_format, &io_format, 1));

            // delete workload
            EXPECT_EQ(NN_API_STATUS_OK, di.workload_delete_function(workload));

            // delete workflow items
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(output));
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(pooling));
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input));

            // delete workflow
            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));
        }
    }

    test_teardown(device_description, device_interface_0);
}

//TEST(api_workloads, workflow_in_convolve_int16_out_compilation)
//{
//    // test configuration
//    const auto number_of_runs = 100;
//
//    nn_device_description_t device_description;
//    nn_device_interface_0_t device_interface_0;
//    test_setup(device_description, device_interface_0);
//
//    // shorter name for function calls
//    nn_device_interface_0_t &di = device_interface_0;
//
//    nn_workload_data_layout_t bias_layout = {
//        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
//        { 0, 0, 0, 0, 0, 0 }, // alignment
//        { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
//        NN_DATATYPE_INT32
//    };
//
//    nn_workload_data_layout_t weight_layout = {
//        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
//        { 0, 0, 0, 0, 0, 0 }, // alignment
//        { NN_DATA_COORD_y, NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_q }, // ordering
//        NN_DATATYPE_INT16
//    };
//
//    nn_workload_data_coords_t bias_coords = { 1, 1, 1, 32, 1, 1 };
//
//    nn_workload_data_coords_t weight_coords = {
//        1,
//        3, 3,       //kernel_width, kernel_height
//        3,          //num_input_feature_maps
//        32,         //num_output_feature_maps
//        1,
//    };
//
//    nn::workload_data<int32_t> *bias_data = new nn::workload_data<int32_t>(bias_coords, bias_layout);
//    nn::workload_data<int16_t> *weight_data = new nn::workload_data<int16_t>(weight_coords, weight_layout);
//
//    // create workflow with 1 input connected directly to 1 output
//    {
//        for (auto run = 0u; run<number_of_runs; ++run) {
//            nn_workflow_t *workflow = nullptr;
//
//            // workflow creation
//            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_create_function(&workflow, 1, 1));
//
//            // creating workflow items: input & output
//            nn_workflow_item_t  *input = nullptr
//                , *pooling = nullptr
//                , *pooling2 = nullptr
//                , *output = nullptr;
//            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&input, 0, nullptr));
//            input->type = NN_WORK_ITEM_TYPE_INPUT;
//            input->arguments.input.index = 0;
//            input->output_format[0].format = NN_DATA_FORMAT_3D;
//            input->output_format[0].format_3d = nn_output_format_3d{ { 32, 32, 3} };
//
//            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&pooling, 1, &input));
//            pooling->type = NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT;
//            auto &arguments = pooling->arguments.forward_convolution_int16_fixedpoint;
//            arguments.stride[0] = 1;
//            arguments.stride[1] = 1;
//            arguments.padding = NN_PADDING_MODE_ZERO;
//            arguments.center_offset[0] = 0;
//            arguments.center_offset[1] = 0;
//            arguments.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;
//            arguments.activation.fractions.accumulator = 16;
//            arguments.activation.fractions.output = 8;
//            arguments.biases = bias_data;
//            arguments.weights = weight_data;
//
//            pooling->output_format[0].format = NN_DATA_FORMAT_3D;
//            pooling->output_format[0].format_3d = {{32 - 3 + 1, 32 - 3 + 1}};
//
//            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&pooling2, 1, &pooling));
//            pooling2->type = NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT;
//            nn_arguments_forward_merged_convolution_pooling_max_2x2_stride_2x2_fixedpoint_t &arguments2 = pooling2->arguments.forward_convolution_pooling_fixedpoint;
//            arguments2.stride[0] = 4;
//            arguments2.stride[1] = 4;
//            arguments2.padding = NN_PADDING_MODE_ZERO;
//            arguments2.activation.basic_arguments.function = NN_ACTIVATION_FUNCTION_RELU;
//            arguments2.activation.fractions.accumulator = 16;
//            arguments2.activation.fractions.output = 0;
//            arguments2.biases = bias_data;
//            arguments2.weights = weight_data;
//
//            pooling2->output_format[0].format = NN_DATA_FORMAT_2D;
//            pooling2->output_format[0].format_2d = nn_output_format_2d{ { 16, 16 } };
//
//            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_create_function(&output, 1, &pooling2));
//            output->type = NN_WORK_ITEM_TYPE_OUTPUT;
//            output->arguments.output.index = 0;
//            output->output_format[0].format = NN_DATA_FORMAT_2D;
//            output->output_format[0].format_2d = nn_output_format_2d{ { 16, 16 } };
//
//            // attaching input/output to workflow
//            workflow->input[0] = input;
//            workflow->output[0] = output;
//
//            // compile workflow
//            nn_workload_t *workload;
//            NN_DATA_TYPE io_format = NN_DATA_TYPE_DATA_T;
//            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_compile_function(&workload, di.device, workflow, &io_format, &io_format, 1));
//
//            // delete workload
//            EXPECT_EQ(NN_API_STATUS_OK, di.workload_delete_function(workload));
//
//            // delete workflow items
//            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(output));
//            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(pooling));
//            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(input));
//
//            // delete workflow
//            EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));
//        }
//    }
//
//    test_teardown(device_description, device_interface_0);
//}