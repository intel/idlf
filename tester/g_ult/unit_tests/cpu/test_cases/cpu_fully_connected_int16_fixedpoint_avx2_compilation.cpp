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

#if 0
// disabled until weight format will be in "plain format"
// merge to nn::data-releated changes was not done - both steps should be done in one step

#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include "gtest/gtest.h"

#include "device/api/nn_device_api.h"
#include "device/common/nn_workload_data.h"
#include "device/cpu/core/fixedpoint/layer_fully_connected_int16_fixedpoint_avx2.h"
#include "device/api/nn_device_interface_0.h"
#include "device/cpu/api_internal/nn_device_interface_0_internal.h"

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
// Helper classess and functions.

template <typename T_output_type> struct work_item_helper;

template <> struct work_item_helper<int16_t> {
    static inline nn_arguments_fully_connected_forward_i16qn_i16qn_t &get_arguments(nn_workflow_item *const &work_item) {
        return work_item->arguments.fully_connected_forward_i16qn_i16qn;
    }

    static const NN_WORK_ITEM_TYPE work_item_type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN;
};

template <> struct work_item_helper<int32_t> {
    static inline nn_arguments_fully_connected_forward_i16qn_i32qn_t &get_arguments(nn_workflow_item *const &work_item) {
        return work_item->arguments.fully_connected_forward_i16qn_i32qn;
    }

    static const NN_WORK_ITEM_TYPE work_item_type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN;
};

//////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T_output_type>
static void ult_nn_fc_deinitialize_work_item(nn_workload_item* &work_item){
    auto &arguments = work_item_helper<T_output_type>::get_arguments(work_item);

    delete arguments.biases;
    delete arguments.weights;

    for(auto& output : work_item->output)
    {
        delete output;
        output = nullptr;
    }

    delete work_item;
    work_item = nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static int16_t &get_input(int16_t *in, int32_t NumBatch, int32_t batch, int32_t idx) {
    return in[idx / 2 * NumBatch * 2 + batch * 2 + idx % 2];
}

template <typename T_output_type>
static T_output_type &get_output(T_output_type *out, int32_t NumBatch, int32_t batch, int32_t idx) {
    return out[idx / 2 * NumBatch * 2 + batch * 2 + idx % 2];
}

static int16_t &get_weight(int16_t *weights, int32_t numInputNeurons, int32_t in, int32_t out) {
    //return weights[out / 8 * 8 * numInputNeurons + in / 2 * 2 * 8 + (out % 8) * 2 + in % 2];
    return weights[out*numInputNeurons + in];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T_output_type>
static void ult_nn_fc_fixedpoint_comp_both_initialize(
    int16_t* input,
    T_output_type* output,
    int32_t* biases,
    int16_t* kernel,
    int16_t* input_ref,
    T_output_type* output_ref,
    int32_t* biases_ref,
    int16_t* kernel_ref,
    uint_least32_t num_input_neurons,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size)
{

    //init the weights
    for (int in = 0; in < num_input_neurons; ++in)
    {
        for (int out = 0; out < num_output_neurons; ++out)
        {
            // if(out!=1)
            //get_weight(kernel, num_input_neurons, in, out) = (in + out) & 0x0f;
            get_weight(kernel, num_input_neurons, in, out) = (in + 0x100 * out);
            //else
            //    get_weight(kernel, num_input_neurons, in, out) = 0;
        }
    }

    for (int i = 0; i < num_output_neurons; i++)
    {
        biases[i] = (i % 10) << 16;
        biases_ref[i] = (i % 10) << 16;
    }

    //// init input neurons
    //    for (int in = 0; in < num_input_neurons; ++in)
    //        for (int batch = 0; batch < batch_size; ++batch) 
    //        {
    //            if (1)//(in+batch) % 2)
    //                get_input(input, batch_size, batch, in) = (batch + in) & 0x0f;  //((in)&0x0f) + (batch << 4);
    //            else
    //                get_input(input, batch_size, batch, in) = -((batch + in) & 0x0f);  //((in)&0x0f) + (batch << 4);
    //        }

    for (int batch = 0; batch < batch_size; ++batch)
    for (auto in = 0; in < num_input_neurons; ++in)
    {
        get_input(input, batch_size, batch, in) = input[batch * num_input_neurons + in] = batch + in;
    }

    //for (int out = 0; out < num_output_neurons; ++out)
    //for (auto in = 0; in < num_input_neurons; ++in)
    //{
    //    kernel[out * num_input_neurons + in] = (in + 0x100 * out);
    //}

    // init output neurons
    for (int batch = 0; batch < batch_size; ++batch)
    for (int out = 0; out < num_output_neurons; ++out) {
        output_ref[out + batch * num_output_neurons] = 0;
        output[out + batch * num_output_neurons] = 0;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T_output_type>
bool ult_nn_fc_fixedpoint_comp_check_outputs(
    nn::data<T_output_type, 1>* output,
    T_output_type* output_ref,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size)
{

    T_output_type * outputOpt = (T_output_type *)output->buffer;
    bool passed = true;
    for (uint_least32_t i = 0; i < (num_output_neurons * batch_size) && passed; i++)
    if (std::labs(output_ref[i] - outputOpt[i])>2)
        passed = false;

    return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T_output_type>
static void ult_nn_fc_fixedpoint_comp_both_alloc(
    int16_t* &input,
    T_output_type* &output,
    int32_t* &biases,
    int16_t* &kernel,
    int16_t* &input_ref,
    T_output_type* &output_ref,
    int32_t* &biases_ref,
    int16_t* &kernel_ref,
    uint_least32_t num_input_neurons,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size)
{
    uint_least32_t input_size = num_input_neurons * batch_size * sizeof(int16_t);
    uint_least32_t output_size = num_output_neurons * batch_size * sizeof(T_output_type);
    uint_least32_t kernel_size = num_input_neurons * num_output_neurons * sizeof(int16_t);
    uint_least32_t bias_size = num_output_neurons * sizeof(int32_t);

    input_ref = (int16_t*)_mm_malloc(input_size, 64);
    output_ref = (T_output_type*)_mm_malloc(output_size, 64);
    biases_ref = (int32_t*)_mm_malloc(bias_size, 64);
    kernel_ref = (int16_t*)_mm_malloc(kernel_size, 64);

    input = (int16_t*)_mm_malloc(input_size, 64);
    output = (T_output_type*)_mm_malloc(output_size, 64);
    biases = (int32_t*)_mm_malloc(bias_size, 64);
    kernel = (int16_t*)_mm_malloc(kernel_size, 64);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T_output_type>
static void ult_nn_fc_fixedpoint_comp_both_dealloc(
    int16_t* &input,
    T_output_type* &output,
    int32_t* &biases,
    int16_t* &kernel,
    int16_t* &input_ref,
    T_output_type* &output_ref,
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

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T_output_type>
static void ult_nn_fc_fixedpoint_comp_naive(
    int16_t* input,
    T_output_type* output,
    int32_t* biases,
    int16_t* kernel,
    uint_least32_t num_input_neurons,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size,
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    NN_ACTIVATION_FUNCTION activation)
{
    bool BiasEn = 1;

    uint_least32_t output_size = num_output_neurons * batch_size * sizeof(int32_t);
    int32_t * output32 = (int32_t*)_mm_malloc(output_size, 64);

    const auto acc_shift = accumulator_fraction - output_fraction;

    for (int batchItr = 0; batchItr < batch_size; ++batchItr)
    {
        for (unsigned int outputNeuronsItr = 0; outputNeuronsItr < num_output_neurons; outputNeuronsItr++)
        {
            if (BiasEn)
                get_output(output32, batch_size, batchItr, outputNeuronsItr) = biases[outputNeuronsItr];
            else
                get_output(output32, batch_size, batchItr, outputNeuronsItr) = 0;

            for (unsigned int inputNeuronsItr = 0; inputNeuronsItr < num_input_neurons; inputNeuronsItr++)
            {
                get_output(output32, batch_size, batchItr, outputNeuronsItr) +=
                    get_input(input, batch_size, batchItr, inputNeuronsItr) * get_weight(kernel, num_input_neurons, inputNeuronsItr, outputNeuronsItr);
            }

            int32_t value = get_output(output32, batch_size, batchItr, outputNeuronsItr);
            float value_f;

            switch (activation)
            {
            case NN_ACTIVATION_FUNCTION_RELU:
                value = std::max(0, value);
                break;
            case NN_ACTIVATION_FUNCTION_LOGISTIC:
                value_f = ((float)value / (1 << accumulator_fraction));
                value_f = 1.0 / (1.0 + std::exp(-(double)value_f));
                value = (int32_t)(value_f * (1 << output_fraction));
                break;
            case NN_ACTIVATION_FUNCTION_NONE:
                break;
            default:
                break;
            }

            if (activation == NN_ACTIVATION_FUNCTION_NONE || activation == NN_ACTIVATION_FUNCTION_RELU)
            {
                if (acc_shift > 0)
                {
                    value = value >> acc_shift;
                }
                else
                {
                    value = value << -acc_shift;
                }
            }

            //value = std::min(value, 32767);
            //value = std::max(value, -32768);
            get_output(output, batch_size, batchItr, outputNeuronsItr) = (T_output_type)(value);

        }
    }
    _mm_free(output32);
}

template<typename T_output_type>
static void fill_workflow(
    nn_workflow_t **workflow,
    nn_device_interface_0_t *di,
    nn_workflow_item_t **input,
    nn_workflow_item_t **output,
    nn_workflow_item_t **fullyconnected_workflow,
    const std::int32_t* bias,
    const std::int16_t* weights,
    uint_least32_t num_input_neurons,
    uint_least32_t num_output_neurons,
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    NN_ACTIVATION_FUNCTION activation
    )
{
    nn_workload_data_layout_t bias_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_x, NN_DATA_COORD_n, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_INT32
    };

    nn_workload_data_layout_t weight_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_INT16
    };

    // TODO: remove hardcoded values
    nn_workload_data_coords_t bias_coords = {
        1,
        num_output_neurons,       //num_output_neurons
        1,
        1,
        1,
        1,
    };

    nn_workload_data_coords_t weight_coords =
    {
        1,
        num_input_neurons,
        num_output_neurons,
        1,
        1,
        1
    };

    auto bias_data = new nn::data<int32_t, 1>((int32_t *)bias, num_output_neurons);
    auto weight_data = new nn::data<int16_t, 2>((int16_t *)weights, num_input_neurons, num_output_neurons);

    /*nn::workload_data<int32_t> *bias_data = new nn::workload_data<int32_t>((void*)bias, bias_coords, bias_layout);
    nn::workload_data<int16_t> *weight_data = new nn::workload_data<int16_t>((void*)weights, weight_coords, weight_layout);*/

    // workflow creation
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_create_function(workflow, 1, 1));

    // creating workflow items: input & output
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(input, 0, nullptr));
    (*input)->type = NN_WORK_ITEM_TYPE_INPUT;
    (*input)->arguments.input.index = 0;
    (*input)->output_format[0].format = NN_DATA_FORMAT_1D;
    (*input)->output_format[0].format_1d = nn_output_format_1d{ { num_input_neurons } };

    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(fullyconnected_workflow, 1, input));
    (*fullyconnected_workflow)->type = work_item_helper<T_output_type>::work_item_type;
    auto &arguments = work_item_helper<T_output_type>::get_arguments(*fullyconnected_workflow);
    arguments.activation.basic_arguments.function = activation;
    arguments.activation.fractions.accumulator = accumulator_fraction;
    arguments.activation.fractions.output = output_fraction;
    arguments.biases = bias_data;
    arguments.weights = weight_data;

    (*fullyconnected_workflow)->output_format[0].format = NN_DATA_FORMAT_1D;
    (*fullyconnected_workflow)->output_format[0].format_1d = nn_output_format_1d{ { num_output_neurons } };

    /*memcpy(arguments.biases->parent->data_buffer, bias, arguments.biases->parent->buffer_size);
    memcpy(arguments.weights->parent->data_buffer, weights, arguments.weights->parent->buffer_size);*/

    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(output, 1, fullyconnected_workflow));
    (*output)->type = NN_WORK_ITEM_TYPE_OUTPUT;
    (*output)->arguments.output.index = 0;
    (*output)->output_format[0].format = NN_DATA_FORMAT_1D;
    (*output)->output_format[0].format_1d = (*fullyconnected_workflow)->output_format[0].format_1d;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T_output_type>
bool ult_fc_int16_fixedpoint_perform_test(
    uint_least32_t num_input_neurons,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size,
    bool check_out_views,
    uint8_t accumulator_fraction,
    uint8_t output_fraction,
    NN_ACTIVATION_FUNCTION activation)
{
    bool passed = false;

    int16_t* input = 0;
    T_output_type* output = 0;
    int32_t* biases = 0;
    int16_t* kernel = 0;

    int16_t* input_ref = 0;
    T_output_type* output_ref = 0;
    int32_t* biases_ref = 0;
    int16_t* kernel_ref = 0;

    num_output_neurons += (C_simd_width - (num_output_neurons % C_simd_width)) % C_simd_width;

    // Allocate naive and optimized buffers.
    ult_nn_fc_fixedpoint_comp_both_alloc(
        input,
        output,
        biases,
        kernel,
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref,
        num_input_neurons,
        num_output_neurons,
        batch_size);

    // Initialize both buffers.    
    ult_nn_fc_fixedpoint_comp_both_initialize(
        input,
        output,
        biases,
        kernel,
        input_ref,
        output_ref,
        biases_ref,
        kernel_ref,
        num_input_neurons,
        num_output_neurons,
        batch_size);

    // Naive 
    ult_nn_fc_fixedpoint_comp_naive(
        input,
        output_ref,
        biases,
        kernel,
        num_input_neurons,
        num_output_neurons,
        batch_size,
        accumulator_fraction,
        output_fraction,
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
        , *workflow_fullyconnected = nullptr;

    fill_workflow<T_output_type>(
        &workflow,
        &di,
        &workflow_input,
        &workflow_output,
        &workflow_fullyconnected,
        biases,
        kernel,
        num_input_neurons,
        num_output_neurons,
        accumulator_fraction,
        output_fraction,
        activation);

    // attaching input/output to workflow
    workflow->input[0] = workflow_input;
    workflow->output[0] = workflow_output;

    nn_workload_data_layout_t input_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_INT16
    };

    nn_workload_data_layout_t output_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
        std::is_same<T_output_type, std::int16_t>::value ? NN_DATATYPE_INT16 : NN_DATATYPE_INT32
    };

    nn_workload_data_coords_t input_coords =
    {
        batch_size,
        1,
        1,
        num_input_neurons / 2,
        2,
        1
    };

    nn_workload_data_coords_t output_coords =
    {
        batch_size,
        1,
        1,
        num_output_neurons / 2,
        2,
        1
    };

    //nn::workload_data<std::int16_t>* input_datas[] = { new nn::workload_data<std::int16_t>(input, input_coords, input_layout) };
    //nn::workload_data<T_output_type>* output_datas[] = { new nn::workload_data<T_output_type>(output, output_coords, output_layout) };

    nn::data<int16_t, 1>* input_datas[] = { new nn::data<int16_t, 1>((int16_t *)input, num_input_neurons) };
    nn::data<T_output_type, 1>* output_datas[] = { new nn::data<T_output_type, 1>((T_output_type *)output, num_output_neurons) };

    // compile workflow
    NN_API_STATUS status;
    nn_workload_t *workload;
    NN_WORKLOAD_DATA_TYPE io_format = NN_WORKLOAD_DATA_TYPE_I16_1D;
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_compile_function(&workload, di.device, workflow, &io_format, &io_format, batch_size));

    EXPECT_EQ(NN_API_STATUS_OK, di.workload_execute_function(workload, (void **)input_datas, (void **)output_datas, &status));


    // delete workload
    EXPECT_EQ(NN_API_STATUS_OK, di.workload_delete_function(workload));

    // delete workflow items
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_output));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_fullyconnected));
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_item_delete_function(workflow_input));

    // delete workflow
    EXPECT_EQ(NN_API_STATUS_OK, di.workflow_delete_function(workflow));

    test_teardown(device_description, device_interface_0);

    //Basic check between optimized and naive versions.
    passed = ult_nn_fc_fixedpoint_comp_check_outputs(output_datas[0], output_ref, num_output_neurons, batch_size);

    ult_nn_fc_fixedpoint_comp_both_dealloc(
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

TEST(cpu_fully_connected_int16_fixedpoint_compilation, cpu16_fully_connected_test)
{
    EXPECT_EQ(true, ult_fc_int16_fixedpoint_perform_test<std::int16_t>(32, 32, 1, false, 16, 0, NN_ACTIVATION_FUNCTION_RELU));
    //EXPECT_EQ(true, ult_fc_int16_fixedpoint_perform_test<std::int16_t>(64, 64, 1, false, 16, 0, NN_ACTIVATION_FUNCTION_RELU));
    //EXPECT_EQ(true, ult_fc_int16_fixedpoint_perform_test<std::int32_t>(128, 128, 1, false, 19, 11, NN_ACTIVATION_FUNCTION_NONE));

    // Overfeat
    // Stage: 7
    //EXPECT_EQ(true, ult_fc_int16_fixedpoint_perform_test<std::int32_t>(3072, 4096, 1, false, 19, 11, NN_ACTIVATION_FUNCTION_NONE));

    // Stage: 8
    //EXPECT_EQ(true, ult_fc_int16_fixedpoint_perform_test<std::int32_t>(4096, 1000, 1, false, 19, 11, NN_ACTIVATION_FUNCTION_NONE));
}
#endif