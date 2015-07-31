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

const uint32_t C_simd_width = sizeof(__m256)/sizeof(int32_t);


///////////////////////////////////////////////////////////////////////////////////////////////////
// Helper classess and functions.

static void ult_nn_fc_initialize_work_item(
    nn_workload_item* &work_item,
    nn_workload_item* &input_item,
    int16_t *const input,
    int32_t*const biases,
    int32_t *const output,
    int16_t *const kernel,
    uint32_t num_input_neurons,
    uint32_t num_output_neurons,
    uint32_t batch_size,
    NN_ACTIVATION_FUNCTION activation)
{

    nn_workload_data_layout_t in_layout = nn::workload_data<int16_t>::layout.xzynpq;

    nn_workload_data_layout_t out_layout = nn::workload_data<int32_t>::layout.xzynpq;

    nn_workload_data_layout_t weight_layout = nn::workload_data<int16_t>::layout.xzynpq;

    nn_workload_data_coords_t input_coords = 
    { 
        1,
        2,
        num_input_neurons / 2,
        batch_size,
        1,
        1
    };

    nn_workload_data_coords_t output_coords =
    {
        1,
        2,
        num_output_neurons / 2,
        batch_size,
        1,
        1
    };

    nn_workload_data_coords_t bias_coords =
    {
        1,
        1,
        num_output_neurons,
        1,
        1,
        1
    };

    nn_workload_data_coords_t weight_coords = 
    { 
        1,
        2,
        num_input_neurons / 2,
        C_simd_width,                        //8
        num_output_neurons / C_simd_width,
        1
    };

    nn::workload_data<int32_t> *output_data = new nn::workload_data<int32_t>(output_coords, out_layout);
    nn::workload_data<int32_t> *bias_data = new nn::workload_data<int32_t>(bias_coords, out_layout);
    nn::workload_data<int16_t> *weight_data = new nn::workload_data<int16_t>(weight_coords, weight_layout);

    work_item = new nn_workload_item();
    work_item->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;
    
    auto &arguments = work_item->arguments.fully_connected_forward_i16qn_i16qn;
    arguments.activation.basic_arguments.function = activation;
    work_item->output.push_back(output_data);
    arguments.biases = bias_data;
    arguments.weights = weight_data;

    memcpy(work_item->output[0]->parent->data_buffer, output, work_item->output[0]->parent->buffer_size);
    memcpy(arguments.biases->parent->data_buffer, biases, arguments.biases->parent->buffer_size);
    memcpy(arguments.weights->parent->data_buffer, kernel, arguments.weights->parent->buffer_size);

    input_item = new nn_workload_item();
    input_item->type = NN_WORK_ITEM_TYPE_INPUT;

    nn::workload_data<int16_t> *input_data = new nn::workload_data<int16_t>(input_coords, in_layout);
    memcpy(input_data->parent->data_buffer, input, input_data->parent->buffer_size);
    input_item->output.push_back(input_data);

    work_item->input.push_back({ input_item, 0 });
}

static void ult_nn_fc_initialize_work_items(
    nn_workload_item* work_items[],
    nn_workload_item* &work_item,
    nn_workload_item* input_items[],
    nn_workload_item* &input_item,
    int16_t *const input,
    int32_t *const biases,
    int32_t *const output,
    int16_t *const kernel,
    uint32_t num_output_feature_maps,
    uint32_t num_input_feature_maps,
    uint32_t output_feature_map_width,
    uint32_t output_feature_map_height,
    uint32_t input_feature_map_width,
    uint32_t input_feature_map_height,
    uint32_t kernel_width,
    uint32_t kernel_height,
    uint32_t kernel_stride_x,
    uint32_t kernel_stride_y,
    NN_ACTIVATION_FUNCTION activation )
{
    nn_workload_data_layout_t inp_layout = nn::workload_data<int16_t>::layout.zpxynq;

    nn_workload_data_layout_t out_layout = nn::workload_data<int32_t>::layout.zpxynq;

    nn_workload_data_layout_t weight_layout = nn::workload_data<int16_t>::layout.pzqxyn;

    nn_workload_data_coords_t input_coords =
    {
        1,
        input_feature_map_width,
        input_feature_map_height,
        num_input_feature_maps,
        1,
        1
    };

    nn_workload_data_coords_t output_coords =
    {
        1,
        output_feature_map_width,
        output_feature_map_height,
        num_output_feature_maps,
        1,
        1
    };

    nn_workload_data_coords_t bias_coords =
    {
        1,
        1,
        1,
        num_output_feature_maps,
        1,
        1
    };

    nn_workload_data_coords_t weight_coords =
    {
        2,
        32,
        num_input_feature_maps / 2,
        kernel_width,
        kernel_height,
        num_output_feature_maps / 32
    };

    nn::workload_data<int16_t> *input_data = new nn::workload_data<int16_t>(input_coords, inp_layout);
    nn::workload_data<int32_t> *output_data = new nn::workload_data<int32_t>(output_coords, out_layout);
    nn::workload_data<int32_t> *bias_data = new nn::workload_data<int32_t>(bias_coords, out_layout);
    nn::workload_data<int16_t> *weight_data = new nn::workload_data<int16_t>(weight_coords, weight_layout);

    memcpy(output_data->parent->data_buffer, output, output_data->parent->buffer_size);
    memcpy(bias_data->parent->data_buffer, biases, bias_data->parent->buffer_size);
    memcpy(weight_data->parent->data_buffer, kernel, weight_data->parent->buffer_size);

    work_item = new nn_workload_item();
    work_item->type = NN_WORK_ITEM_TYPE_FULLY_CONNECTED;

    input_item = new nn_workload_item();
    input_item->type = NN_WORK_ITEM_TYPE_INPUT;

    memcpy(input_data->parent->data_buffer, input, input_data->parent->buffer_size);
    input_item->output.push_back(input_data);

    work_item->input.push_back({ input_item, 0 });
}

//////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_fc_deinitialize_work_item(nn_workload_item* &work_item)
{
    if (work_item->type == NN_WORK_ITEM_TYPE_FULLY_CONNECTED)
    {
        delete work_item->parameters[0];
        delete work_item->parameters[1];
    }

    work_item->input.clear();
    delete work_item->output[0];

    delete work_item;

    work_item = nullptr;
}

static void ult_nn_fc_deinitialize_work_items(nn_workload_item* work_items[])
{
    for (unsigned int item = 0; item < 8; item++)
    {
        work_items[item]->input.clear();
        delete work_items[item]->output[0];

        delete work_items[item];

        work_items[item] = nullptr;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
static bool ult_nn_fc_interface_run(nn_workload_item* work_item)
{
    bool retvalue = true;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;

    nn_device_load(&device_description);
    nn_device_interface_open(0, &device_interface_0);

    int16_fixedpoint::run_multithreaded_fully_connected_fixedpoint_work_item(work_item, reinterpret_cast<nn_device_internal*>(device_interface_0.device));

    nn_device_interface_close(&device_interface_0);
    nn_device_unload();

    return retvalue;
}

bool ult_nn_fc_interface_run(nn_workload_item* work_items[])
{
    bool retvalue = true;

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;

    nn_device_load(&device_description);
    nn_device_interface_open(0, &device_interface_0);

    for (unsigned int item = 0; item < 8 && retvalue; item++)
    {
        int16_fixedpoint::run_multithreaded_fully_connected_fixedpoint_work_item(work_items[item], reinterpret_cast<nn_device_internal*>(device_interface_0.device));
    }

    nn_device_interface_close(&device_interface_0);
    nn_device_unload();


    return retvalue;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static int16_t& get_input(int16_t * in, int32_t NumBatch, int32_t batch, int32_t idx) 
{
  return in[idx / 2 * NumBatch * 2 + batch * 2 + idx % 2];
}

static int32_t& get_output(int32_t * out, int32_t NumBatch, int32_t batch, int32_t idx) 
{
  return out[idx / 2 * NumBatch * 2 + batch * 2 + idx % 2];
}

// static int32_t& get_reference(int32_t batch, int32_t idx) 
// {
  // return Reference[idx / 2 * NumBatch * 2 + batch * 2 + idx % 2];
// }

static int16_t& get_weight(int16_t * weights, int32_t numInputNeurons, int32_t in, int32_t out) 
{
  return weights[out/8*8 * numInputNeurons + in/2*2*8 + (out%8)*2 + in%2];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_fc_both_initialize(
    int16_t* input,
    int32_t* output,
    int32_t* biases,
    int16_t* kernel,
    int16_t* input_ref,
    int32_t* output_ref,
    int32_t* biases_ref,
    int16_t* kernel_ref,
    uint_least32_t num_input_neurons,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size)
{

    //init the weights
    for (int in = 0; in < num_input_neurons; ++in)
    {
        for(int out = 0; out < num_output_neurons; ++out)
        {
            //if(out!=1)
                get_weight(kernel, num_input_neurons, in, out) = (in + 0x100 * out);
            //else
              //  get_weight(kernel, num_input_neurons, in, out) = 0;
        }
    }

    for (int i = 0; i < num_output_neurons; i++)
    {
        biases[i] = i%10;
        biases_ref[i] = i%10;
    }	
    
// init input neurons
    for (int in = 0; in < num_input_neurons; ++in)
        for (int batch = 0; batch < batch_size; ++batch) 
        {
            if (1)//(in+batch) % 2)
                get_input(input, batch_size, batch, in) = (batch + in) & 0x0f;  //((in)&0x0f) + (batch << 4);
            else
                get_input(input, batch_size, batch, in) = -((batch + in) & 0x0f);  //((in)&0x0f) + (batch << 4);
        }

    for (int batch = 0; batch < batch_size; ++batch)
    for (auto in = 0; in < num_input_neurons; ++in)
    {
        input[batch * num_input_neurons + in] = batch + in;
    }

    for (int out = 0; out < num_output_neurons; ++out)
    for (auto in = 0; in < num_input_neurons; ++in)
    {
        kernel_ref[out * num_input_neurons + in] = (in + 0x100 * out);
    }

    // init output neurons
        for (int batch = 0; batch < batch_size; ++batch)
        for (int out = 0; out < num_output_neurons; ++out)
        {
            output_ref[out + batch * num_output_neurons] = 0;
            output[out + batch * num_output_neurons] = 0;
        }

    
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool ult_nn_fc_check_outputs(
    nn_workload_data_t* output,
    int32_t* output_ref,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size)
{

    int32_t * outputOpt = (int32_t *)output->parent->data_buffer;
    bool passed = true;
    for (uint_least32_t i = 0; i < (num_output_neurons * batch_size) && passed; i++)
    if (output_ref[i] != outputOpt[i])
        passed = false;

    return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_fc_both_alloc(
    int16_t* &input,
    int32_t* &output,
    int32_t* &biases,
    int16_t* &kernel,
    int16_t* &input_ref,
    int32_t* &output_ref,
    int32_t* &biases_ref,
    int16_t* &kernel_ref,
    uint_least32_t num_input_neurons,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size)
{
    uint_least32_t input_size = num_input_neurons * batch_size * sizeof(int16_t);
    uint_least32_t output_size = num_output_neurons * batch_size * sizeof(int32_t);
    uint_least32_t kernel_size = num_input_neurons * num_output_neurons * sizeof(int16_t);
    uint_least32_t bias_size = num_output_neurons * sizeof(int32_t);
    
    input_ref = (int16_t*)_mm_malloc(input_size, 64);
    output_ref = (int32_t*)_mm_malloc(output_size, 64);
    biases_ref = (int32_t*)_mm_malloc(bias_size, 64);
    kernel_ref = (int16_t*)_mm_malloc(kernel_size, 64);

    input = (int16_t*)_mm_malloc(input_size, 64);
    output = (int32_t*)_mm_malloc(output_size, 64);
    biases = (int32_t*)_mm_malloc(bias_size, 64);
    kernel = (int16_t*)_mm_malloc(kernel_size, 64);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_fc_both_dealloc(
    int16_t* &input,
    int32_t* &output,
    int32_t* &biases,
    int16_t* &kernel,
    int16_t* &input_ref,
    int32_t* &output_ref,
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
bool ult_fc_perform_test(
    uint_least32_t num_input_neurons,
    uint_least32_t num_output_neurons,
    uint_least32_t batch_size,
    bool check_out_views,
    NN_ACTIVATION_FUNCTION activation )
{
    nn_workload_item* work_item = nullptr;
    nn_workload_item* work_items[8];
    nn_workload_item* input_item = nullptr;
    nn_workload_item* input_items[8];

    std::fill_n(work_items, 8, nullptr);

    bool passed = false;

    int16_t* input = 0;
    int32_t* output = 0;
    int32_t* biases = 0;
    int16_t* kernel = 0;

    int16_t* input_ref = 0;
    int32_t* output_ref = 0;
    int32_t* biases_ref = 0;
    int16_t* kernel_ref = 0;

    num_output_neurons += (C_simd_width - (num_output_neurons % C_simd_width)) % C_simd_width;

    // Allocate naive and optimized buffers.
    ult_nn_fc_both_alloc(
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
    ult_nn_fc_both_initialize(
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
    ult_nn_fc_naive(
        input,
        output_ref,
        biases,
        kernel,
        num_input_neurons,
        num_output_neurons,
        batch_size,
        activation);

     if (check_out_views)
     {
        // // Perform data copy to interface test.
        // ult_nn_fc_initialize_work_items(
            // work_items,
            // work_item,
            // input,
            // biases,
            // output,
            // kernel,
            // num_input_neurons,
            // num_output_neurons,
            // batch_size,			
            // activation);

        // // Optimized convolution.
        // passed = ult_nn_fc_interface_run(work_items);
    }
    else
    {
        // Perform data copy to interface test.
        ult_nn_fc_initialize_work_item(
            work_item,
            input_item,
            input,
            biases,
            output,
            kernel,
            num_input_neurons,
            num_output_neurons,
            batch_size,			
            activation);

        //Optimized convolution.
        passed = ult_nn_fc_interface_run(work_item);
    }


    if (passed)
    {
         //Basic check between optimized and naive versions.
        passed = ult_nn_fc_check_outputs(work_item->output[0],
            output_ref, num_output_neurons, batch_size);
    }

    // Cleanup.
    ult_nn_fc_deinitialize_work_item(work_item);

    if (check_out_views)
    {
        //ult_nn_fc_deinitialize_work_items(work_items);
    }

    ult_nn_fc_both_dealloc(
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

TEST(cpu_int16_fully_connected, cpu16_fully_connected_test)
{
    // TODO coordinate mismatch
    //EXPECT_EQ(true, ult_fc_perform_test(64, 64, 8, false, NN_ACTIVATION_FUNCTION_NONE));
    //EXPECT_EQ(true, ult_fc_perform_test(64, 96, 1, false, NN_ACTIVATION_FUNCTION_NONE));
    //EXPECT_EQ(true, ult_fc_perform_test(1024, 1024, 1, false, NN_ACTIVATION_FUNCTION_NONE));
    //EXPECT_EQ(true, ult_fc_perform_test(1024, 1024, 8, false, NN_ACTIVATION_FUNCTION_NONE));
    //EXPECT_EQ(true, ult_fc_perform_test(1024, 1024, 1, false, NN_ACTIVATION_FUNCTION_RELU));
    //EXPECT_EQ(true, ult_fc_perform_test(1024, 1024, 8, false, NN_ACTIVATION_FUNCTION_RELU));
    //EXPECT_EQ(true, ult_fc_perform_test(8192, 8192, 1, false, NN_ACTIVATION_FUNCTION_NONE));
    //EXPECT_EQ(true, ult_fc_perform_test(8192, 8192, 8, false, NN_ACTIVATION_FUNCTION_RELU));
}
