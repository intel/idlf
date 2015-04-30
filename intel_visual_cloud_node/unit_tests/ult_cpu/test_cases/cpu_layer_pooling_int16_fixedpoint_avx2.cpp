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

#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include "gtest/gtest.h"


#include "../../devices/common/nn_workload_data.h"
#include "../../devices/device_cpu/core/fixedpoint/layer_pooling_int16_fixedpoint_avx2.h"
#include "../../devices/api/nn_device_interface_0.h"
#include "../../devices/device_cpu/api_internal/nn_device_interface_0_internal.h"

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

static void ult_nn_pooling_fixedpoint_initialize_work_item(
    nn_workload_item* &work_item,
    nn_workload_item* &input_item,
    int16_t *const input,
    int16_t *const output,
    uint32_t num_output_feature_maps,
    uint32_t num_input_feature_maps,
    uint32_t output_feature_map_width,
    uint32_t output_feature_map_height,
    uint32_t input_feature_map_width,
    uint32_t input_feature_map_height,
    uint_least32_t pool_size_x,
    uint_least32_t pool_size_y,
    uint_least32_t pool_stride_x,
    uint_least32_t pool_stride_y,
    uint_least32_t center_x,
    uint_least32_t center_y)
{
    uint32_t IFMBlock = 8;
    uint32_t OFMBlock = 32;
    uint32_t OFMpBlock = 2;
    uint32_t OFMOutBlock = 8;

    if (num_input_feature_maps == 4) IFMBlock = 4;

    nn_workload_data_layout_t in_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_INT16
    };
    nn_workload_data_layout_t out_layout = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, // ordering
        NN_DATATYPE_INT16
    };

    nn_workload_data_coords_t input_coords =
    {
        1, // batch size
        input_feature_map_width,
        input_feature_map_height,
        num_input_feature_maps / IFMBlock,
        IFMBlock,
        1,
    };

    nn_workload_data_coords_t output_coords =
    {
        1, // batch size
        output_feature_map_width + 2 * center_x,
        output_feature_map_height + 2 * center_y,
        num_output_feature_maps / OFMOutBlock,
        OFMOutBlock,
        1,
    };

    nn_workload_data_coords_t nn_view_begin =
    {
        0,
        center_x,
        center_y,
        0,
        0,
        0
    };

    nn_workload_data_coords_t nn_view_end =
    {
        0,
        output_feature_map_width + center_x - 1,
        output_feature_map_height + center_y - 1,
        num_output_feature_maps / OFMOutBlock - 1,
        OFMOutBlock - 1,
        0
    };

    nn::nn_workload_data_t<int16_t> *output_data = new nn::nn_workload_data_t<int16_t>(output_coords, out_layout);

    work_item = new nn_workload_item();
    work_item->type = NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT;

    auto &arguments = work_item->arguments.forward_pooling_fixedpoint;
    arguments.pool_stride[0] = pool_stride_x;
    arguments.pool_stride[1] = pool_stride_y;
    arguments.pool_size[0] = pool_size_x;
    arguments.pool_size[1] = pool_size_y;
    work_item->output = output_data;

    work_item->output = new nn::nn_workload_data_t<int16_t>(*output_data, nn_view_begin, nn_view_end);
    memcpy(work_item->output->parent->data_buffer, output, work_item->output->parent->buffer_size);

    input_item = new nn_workload_item();
    input_item->type = NN_WORK_ITEM_TYPE_INPUT;

    nn::nn_workload_data_t<int16_t> *input_data = new nn::nn_workload_data_t<int16_t>(input_coords, in_layout);
    memcpy(input_data->parent->data_buffer, input, input_data->parent->buffer_size);
    input_item->output = input_data;

    work_item->input.push_back(input_item);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
    static void ult_nn_pooling_fixedpoint_deinitialize_work_item(nn_workload_item* &work_item)
{
    auto &arguments = work_item->arguments.forward_convolution_fixedpoint;

    work_item->input.clear();
    delete work_item->output;

    delete work_item;

    work_item = nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_pooling_fixedpoint_naive_set_input_value(
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

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_pooling_fixedpoint_both_initialize_matrices(
    int16_t* input,
    int16_t* output,
    int16_t* input_ref,
    int16_t* output_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t center_x,
    uint_least32_t center_y
    )
{
    uint_least32_t input_size = input_feature_map_width * input_feature_map_height * num_input_feature_maps * sizeof(int16_t);
    int16_t * inputT = (int16_t*)_mm_malloc(input_size, 4096);

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
                //ult_nn_pooling_fixedpoint_optimized_set_input_value(inputT, input_feature_map_width, num_input_feature_maps, column, row, input_map, value, offset);
                ult_nn_pooling_fixedpoint_naive_set_input_value(inputT, input_feature_map_width, input_feature_map_height, column, row, input_map, value, offset);
                ult_nn_pooling_fixedpoint_naive_set_input_value(input_ref, input_feature_map_width, input_feature_map_height, column, row, input_map, value, offset);
                value++;
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

    //prepare right input layout for naive implementation
    for (uint32_t i = 0; i < num_input_feature_maps / IFMBlock; i++)
    for (uint32_t j = 0; j < input_feature_map_width * input_feature_map_height; j++)
    for (uint32_t n = 0; n < IFMBlock; n++)
        input[n + j * IFMBlock + i * input_feature_map_width * input_feature_map_height * IFMBlock]
            = inputT[n * input_feature_map_width * input_feature_map_height + j + i * input_feature_map_width * input_feature_map_height * IFMBlock];


    _mm_free(inputT);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static bool ult_nn_pooling_fixedpoint_check_outputs(
    nn_workload_data_t* output,
    int16_t* output_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t output_feature_map_width,
    uint_least32_t output_feature_map_height,
    uint_least32_t center_x,
    uint_least32_t center_y
    )
{
    int16_t * outputOpt = (int16_t *)output->parent->data_buffer;
    uint32_t OFMOutBlock = 8;

    uint_least32_t output_size = (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y) * num_output_feature_maps * sizeof(int16_t);
    int16_t* outputT = (int16_t*)_mm_malloc(output_size, 64);
    for (uint32_t i = 0; i < num_output_feature_maps / OFMOutBlock; i++)
    for (uint32_t j = 0; j < (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y); j++)
    for (uint32_t n = 0; n < OFMOutBlock; n++)
    {
        outputT[n + j * OFMOutBlock + i * (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y) * OFMOutBlock]
            = output_ref[n * (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y) + j + i * (output_feature_map_width + 2 * center_x) * (output_feature_map_height + 2 * center_y) * OFMOutBlock];
    }

    bool passed = true;
    for (uint_least32_t i = 0; i < (output_size / sizeof(int16_t)) && passed; i++)
    if (outputT[i] != outputOpt[i])
        passed = false;

    _mm_free(outputT);

    return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_pooling_fixedpoint_both_alloc(
    int16_t* &input,
    int16_t* &output,
    int16_t* &input_ref,
    int16_t* &output_ref,
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t output_width,
    uint_least32_t output_height,
    uint_least32_t input_width,
    uint_least32_t input_height,
    uint_least32_t center_x,
    uint_least32_t center_y
    )
{
    uint_least32_t input_size = input_width * input_height * num_input_feature_maps * sizeof(int16_t);
    uint_least32_t output_size = (output_width + 2 * center_x) * (output_height + 2 * center_y) * num_output_feature_maps * sizeof(int16_t);

    input_ref = (int16_t*)_mm_malloc(input_size, 4096);
    output_ref = (int16_t*)_mm_malloc(output_size, 4096);

    input_size = input_width * input_height * num_input_feature_maps * sizeof(int16_t);
    output_size = (output_width + 2 * center_x) * (output_height + 2 * center_y) * num_output_feature_maps * sizeof(int16_t);

    input = (int16_t*)_mm_malloc(input_size, 4096);
    output = (int16_t*)_mm_malloc(output_size, 4096);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_pooling_fixedpoint_both_dealloc(
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

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ult_nn_maxpooling_naive(
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

static void fill_workflow(
    nn_workflow_t **workflow,
    nn_device_interface_0_t *di,
    nn_workflow_item_t **input,
    nn_workflow_item_t **output,
    nn_workflow_item_t **pooling,
    uint_least32_t num_input_feature_maps,
    uint_least32_t num_output_feature_maps,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t pool_size_x,
    uint_least32_t pool_size_y,
    uint_least32_t pool_stride_x,
    uint_least32_t pool_stride_y
    )
{

    // workflow creation
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_create_function(workflow, 1, 1));

    // creating workflow items: input & output
    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(input, 0, nullptr));
    (*input)->type = NN_WORK_ITEM_TYPE_INPUT;
    (*input)->arguments.input.index = 0;
    (*input)->output_format.format = NN_DATA_FORMAT_3D;
    (*input)->output_format.format_3d = nn_output_format_3d{ { input_feature_map_width, input_feature_map_height, num_input_feature_maps } };

    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(pooling, 1, input));
    (*pooling)->type = NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT;
    auto &arguments = (*pooling)->arguments.forward_pooling_fixedpoint;
    arguments.pool_stride[0] = pool_stride_x;
    arguments.pool_stride[1] = pool_stride_y;
    arguments.pool_size[0] = pool_size_x;
    arguments.pool_size[1] = pool_size_y;

    (*pooling)->output_format.format = NN_DATA_FORMAT_3D;
    (*pooling)->output_format.format_3d = nn_output_format_3d{ { input_feature_map_width / pool_stride_x, input_feature_map_height / pool_stride_x, num_output_feature_maps } };

    EXPECT_EQ(NN_API_STATUS_OK, di->workflow_item_create_function(output, 1, pooling));
    (*output)->type = NN_WORK_ITEM_TYPE_OUTPUT;
    (*output)->arguments.output.index = 0;
    (*output)->output_format.format = NN_DATA_FORMAT_3D;
    (*output)->output_format.format_3d = (*pooling)->output_format.format_3d;
}

static void create_input_item_subview(
    nn_workload_item* &sub_work_item,
    nn_workload_item* master_work_item,
    uint32_t slave_id)
{
    nn_workload_data_coords_t in_out_coords =
    {
        0,
        0,
        0,
        (slave_id) ? master_work_item->output->parent->lengths.t[NN_DATA_COORD_z] / 2 : 0,
        0,
        0
    };

    nn_workload_data_coords_t in_out_end_coords =
    {
        master_work_item->output->parent->lengths.t[NN_DATA_COORD_n] - 1,
        master_work_item->output->parent->lengths.t[NN_DATA_COORD_x] - 1,
        master_work_item->output->parent->lengths.t[NN_DATA_COORD_y] - 1,
        (slave_id) ? master_work_item->output->parent->lengths.t[NN_DATA_COORD_z] - 1 : master_work_item->output->parent->lengths.t[NN_DATA_COORD_z] / 2 - 1,
        master_work_item->output->parent->lengths.t[NN_DATA_COORD_p] - 1,
        0
    };

    sub_work_item = new nn_workload_item();
    sub_work_item->type = master_work_item->type;

    sub_work_item->output =
        new nn::nn_workload_data_t<int16_t>(*(reinterpret_cast<nn::nn_workload_data_t<int16_t>*>(master_work_item->output)), in_out_coords, in_out_end_coords);
}

static void create_work_item_subview(
    nn_workload_item* &sub_work_item,
    nn_workload_item* master_work_item,
    nn_workload_item* input_work_item,
    uint32_t slave_id)
{
    auto &master_arguments = master_work_item->arguments.forward_pooling_fixedpoint;

    nn_workload_data_coords_t in_out_coords =
    {
        0,
        0,
        0,
        (slave_id) ? master_work_item->output->parent->lengths.t[NN_DATA_COORD_z] / 2 : 0,
        0,
        0
    };

    nn_workload_data_coords_t in_out_end_coords =
    {
        master_work_item->output->parent->lengths.t[NN_DATA_COORD_n] - 1,
        master_work_item->output->parent->lengths.t[NN_DATA_COORD_x] - 1,
        master_work_item->output->parent->lengths.t[NN_DATA_COORD_y] - 1,
        (slave_id) ? master_work_item->output->parent->lengths.t[NN_DATA_COORD_z] - 1 : master_work_item->output->parent->lengths.t[NN_DATA_COORD_z] / 2 - 1,
        master_work_item->output->parent->lengths.t[NN_DATA_COORD_p] - 1,
        0
    };

    sub_work_item = new nn_workload_item();
    sub_work_item->type = master_work_item->type;

    sub_work_item->input.push_back(input_work_item);

    sub_work_item->arguments.forward_pooling_fixedpoint = master_arguments;

    sub_work_item->output =
        new nn::nn_workload_data_t<int16_t>(*(reinterpret_cast<nn::nn_workload_data_t<int16_t>*>(master_work_item->output)), in_out_coords, in_out_end_coords);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static bool ult_perform_test(
    uint_least32_t num_output_feature_maps,
    uint_least32_t num_input_feature_maps,
    uint_least32_t input_feature_map_width,
    uint_least32_t input_feature_map_height,
    uint_least32_t pool_stride_x,
    uint_least32_t pool_stride_y,
    uint_least32_t pool_size_x,
    uint_least32_t pool_size_y,
    uint_least32_t center_x,
    uint_least32_t center_y,
    NN_POOLING_MODE mode,
    bool check_views)
{
    nn_workload_item* work_item = nullptr;
    nn_workload_item* input_item = nullptr;

    bool passed = false;

    int16_t* input = 0;
    int16_t* output = 0;

    int16_t* input_ref = 0;
    int16_t* output_ref = 0;

    uint32_t NoWItems = 1;

    uint_least32_t output_feature_map_width = (input_feature_map_width - pool_size_x) / pool_stride_x + 1;
    uint_least32_t output_feature_map_height = (input_feature_map_height - pool_size_y) / pool_stride_y + 1;

    uint_least32_t output_feature_map_width_int = input_feature_map_width;
    uint_least32_t output_feature_map_height_int = input_feature_map_height;

    num_output_feature_maps += (C_simd_width - (num_output_feature_maps % C_simd_width)) % C_simd_width;

    // Allocate naive and optimized buffers.
    ult_nn_pooling_fixedpoint_both_alloc(
        input,
        output,
        input_ref,
        output_ref,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        input_feature_map_width,
        input_feature_map_height,
        center_x,
        center_y);

    // Initialize both buffers.
    ult_nn_pooling_fixedpoint_both_initialize_matrices(
        input,
        output,
        input_ref,
        output_ref,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        input_feature_map_width,
        input_feature_map_height,
        center_x,
        center_y);

    // Naive maxpooling.
    ult_nn_maxpooling_naive(
        input_ref,
        output_ref,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        output_feature_map_width_int,
        output_feature_map_height_int,
        input_feature_map_width,
        input_feature_map_height,
        pool_size_x,
        pool_size_y,
        pool_stride_x,
        pool_stride_y,
        center_x,
        center_y);

    ult_nn_pooling_fixedpoint_initialize_work_item(
        work_item,
        input_item,
        input,
        output,
        num_output_feature_maps,
        num_input_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        input_feature_map_width,
        input_feature_map_height,
        pool_size_x,
        pool_size_y,
        pool_stride_x,
        pool_stride_y,
        center_x,
        center_y);

    nn_device_description_t device_description;
    nn_device_interface_0_t device_interface_0;

    test_setup(device_description, device_interface_0);
    if (check_views)
    {
        nn_workload_item* sub_input_item_0 = nullptr;
        nn_workload_item* sub_input_item_1 = nullptr;
        create_input_item_subview(sub_input_item_0, input_item, 0);
        create_input_item_subview(sub_input_item_1, input_item, 1);

        nn_workload_item* sub_work_item_0 = nullptr;
        nn_workload_item* sub_work_item_1 = nullptr;
        create_work_item_subview(sub_work_item_0, work_item, sub_input_item_0, 0);
        create_work_item_subview(sub_work_item_1, work_item, sub_input_item_1, 1);

        int16_fixedpoint::run_pooling_work_item(sub_work_item_0);
        int16_fixedpoint::run_pooling_work_item(sub_work_item_1);

        ult_nn_pooling_fixedpoint_deinitialize_work_item(sub_work_item_0);
        ult_nn_pooling_fixedpoint_deinitialize_work_item(sub_work_item_1);
    }
    else
    {
        int16_fixedpoint::run_pooling_work_item(work_item);
    }

    test_teardown(device_description, device_interface_0);

    // Basic check between optimized and naive versions.
    passed = ult_nn_pooling_fixedpoint_check_outputs(
        work_item->output,
        output_ref,
        num_output_feature_maps,
        output_feature_map_width,
        output_feature_map_height,
        center_x,
        center_y);

    // Cleanup.
    ult_nn_pooling_fixedpoint_deinitialize_work_item(work_item);

    ult_nn_pooling_fixedpoint_both_dealloc(
        input,
        output,
        input_ref,
        output_ref);

    return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Tests.
TEST(cpu_int16_maxpooling_fixedpoint, cpu_maxpooling_stride1)
{
    // krizhevsky
    EXPECT_EQ(true, ult_perform_test(96, 96, 55, 55, 2, 2, 3, 3, 0, 0, NN_POOLING_MODE_MAX, true));

    // stride 2x2 pool 3x3
    EXPECT_EQ(true, ult_perform_test(16, 16, 5, 5, 2, 2, 3, 3, 0, 0, NN_POOLING_MODE_MAX, true));
    EXPECT_EQ(true, ult_perform_test(16, 16, 3, 3, 2, 2, 3, 3, 0, 0, NN_POOLING_MODE_MAX, true));
    EXPECT_EQ(true, ult_perform_test(96, 96, 3, 3, 2, 2, 3, 3, 0, 0, NN_POOLING_MODE_MAX, true));
}