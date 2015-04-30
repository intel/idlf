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

extern "C" {
#include "nn_workload_data.h"
}

// default ordering - used for test where just any ordering would work fine
const nn_workload_data_coords_t default_ordering = { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q };

typedef struct
{
    uint32_t test_id;
    nn_workload_data_coords_t lengths;
    nn_workload_data_layout_t layout;
} test_conf_t;

static test_conf_t test_conf[] = {
    {
        0, //test_id
        { 4, 4, 4, 4, 4, 4 }, // lengths
        {
            { 0, 0, 0, 0, 0, 0 }, // tile_log2
            { 0, 0, 0, 0, 0, 0 }, // alignment
            default_ordering, // ordering
            NN_DATATYPE_FLOAT
        }
    }
};

TEST(nn_workload_data_c_api, basic_float_test)
{
    nn_workload_data_t *data;
    test_conf_t* tc = &test_conf[0];

    data = nn_workload_data_create(&tc->lengths, &tc->layout);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data->parent->data_buffer, nullptr);
    EXPECT_EQ(data->parent->buffer_size, 4 * 4 * 4 * 4 * 4 * 4 * sizeof(float));

    float v = 0.0f;

    for (uint32_t q = 0; q < tc->lengths.t[5]; q++)
    for (uint32_t p = 0; p < tc->lengths.t[4]; p++)
    for (uint32_t z = 0; z < tc->lengths.t[3]; z++)
    for (uint32_t y = 0; y < tc->lengths.t[2]; y++)
    for (uint32_t x = 0; x < tc->lengths.t[1]; x++)
    for (uint32_t n = 0; n < tc->lengths.t[0]; n++, v++)
        nn_workload_data_set_float32(data, v, n, x, y, z, p, q);

    // check data integrity 
    v = 0.0f;
    for (uint32_t q = 0; q < tc->lengths.t[5]; q++)
    for (uint32_t p = 0; p < tc->lengths.t[4]; p++)
    for (uint32_t z = 0; z < tc->lengths.t[3]; z++)
    for (uint32_t y = 0; y < tc->lengths.t[2]; y++)
    for (uint32_t x = 0; x < tc->lengths.t[1]; x++)
    for (uint32_t n = 0; n < tc->lengths.t[0]; n++, v++)
        EXPECT_EQ(v, nn_workload_data_get_float32(data, n, x, y, z, p, q));

    EXPECT_EQ(NN_DATA_STATUS_OK, nn_workload_data_delete(data));
}

TEST(nn_workload_data_c_api, create_view_test)
{
    nn_workload_data_t *data;
    nn_workload_data_t *view1, *view2, *view3, *view4;
    
    nn_workload_data_coords_t lengths = { 8, 8, 8, 8, 8, 8 };
    nn_workload_data_layout_t layout =
    {
        { 1, 1, 1, 1, 1, 1 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        default_ordering, // ordering
        NN_DATATYPE_INT16
    };

    nn_workload_data_coords_t view1_begin(2, 2, 2, 2, 2, 2);
    nn_workload_data_coords_t view1_end(5, 5, 5, 5, 5, 5);

    nn_workload_data_coords_t view2_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t view2_end(1, 1, 1, 1, 1, 1);

    nn_workload_data_coords_t view3_begin(2, 2, 2, 2, 2, 2);
    nn_workload_data_coords_t view3_end(3, 3, 3, 3, 3, 3);

    nn_workload_data_coords_t view4_begin(1, 1, 1, 1, 1, 1);
    nn_workload_data_coords_t view4_end(2, 2, 2, 2, 2, 2);

    data = nn_workload_data_create(&lengths, &layout);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data->parent->data_buffer, nullptr);
    EXPECT_EQ(data->parent->buffer_size, 8 * 8 * 8 * 8 * 8 * 8 * sizeof(int16_t));
    EXPECT_EQ(1, data->parent->reference_count);

    view1 = nn_workload_data_create_view(data, &view1_begin, &view1_end);
    ASSERT_NE(view1, nullptr);
    ASSERT_EQ(data->parent, view1->parent);
    EXPECT_EQ(2, data->parent->reference_count);

    view2 = nn_workload_data_create_view(view1, &view2_begin, &view2_end);
    ASSERT_NE(view2, nullptr);
    ASSERT_EQ(data->parent, view2->parent);
    EXPECT_EQ(3, data->parent->reference_count);

    view3 = nn_workload_data_create_view(view1, &view3_begin, &view3_end);
    ASSERT_NE(view3, nullptr);
    ASSERT_EQ(data->parent, view3->parent);
    EXPECT_EQ(4, data->parent->reference_count);

    // This view is requested not granular to tiles - shall not succeed
    view4 = nn_workload_data_create_view(view1, &view4_begin, &view4_end);
    EXPECT_EQ(view4, nullptr);
    EXPECT_EQ(4, data->parent->reference_count);

    int16_t v1 = -1234, v2 = 2468;
    nn_workload_data_set_int16(data, v1, 2, 3, 2, 3, 2, 3);
    nn_workload_data_set_int16(data, v2, 4, 5, 4, 5, 4, 5);

    EXPECT_EQ(v1, nn_workload_data_get_int16(data, 2, 3, 2, 3, 2, 3));
    EXPECT_EQ(v1, nn_workload_data_get_int16(view1, 0, 1, 0, 1, 0, 1));
    EXPECT_EQ(v1, nn_workload_data_get_int16(view2, 0, 1, 0, 1, 0, 1));

    EXPECT_EQ(v2, nn_workload_data_get_int16(data, 4, 5, 4, 5, 4, 5));
    EXPECT_EQ(v2, nn_workload_data_get_int16(view1, 2, 3, 2, 3, 2, 3));
    EXPECT_EQ(v2, nn_workload_data_get_int16(view3, 0, 1, 0, 1, 0, 1));

    // It's OK to delete data before deleting view1, view2 and view3
    EXPECT_EQ(4, data->parent->reference_count);
    EXPECT_EQ(NN_DATA_STATUS_OK, nn_workload_data_delete(data));
    EXPECT_EQ(3, view1->parent->reference_count);
    EXPECT_EQ(NN_DATA_STATUS_OK, nn_workload_data_delete(view1));
    EXPECT_EQ(2, view2->parent->reference_count);
    EXPECT_EQ(NN_DATA_STATUS_OK, nn_workload_data_delete(view2));
    EXPECT_EQ(1, view3->parent->reference_count);
    EXPECT_EQ(NN_DATA_STATUS_OK, nn_workload_data_delete(view3));
}

/*
    Create nn_workload_data_t that uses preallocated data buffer.
*/
TEST(nn_workload_data_c_api, create_from_existing_buffer_test)
{
    const uint32_t LENGTH = 4;
    const uint32_t buffer_size = LENGTH * LENGTH * LENGTH * LENGTH * LENGTH * LENGTH;
    int16_t buffer[buffer_size];

    // Write some data to the buffer
    int16_t v = 0;
    for (auto &b : buffer)
    {
        b = v++;
    }

    // Structures describing buffer's size and layout
    nn_workload_data_coords_t lengths = { LENGTH, LENGTH, LENGTH, LENGTH, LENGTH, LENGTH };
    nn_workload_data_layout_t layout =
    {
        { 0, 0, 0, 0, 0, 0 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        default_ordering, // ordering
        NN_DATATYPE_INT16
    };

    // Create nn_workload_data that will use existing buffer
    nn_workload_data_t *data = nn_workload_data_create_from_buffer(buffer, &lengths, &layout);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->parent->data_buffer, buffer);
    EXPECT_EQ(data->parent->buffer_size, buffer_size * sizeof(int16_t));
    EXPECT_EQ(data->parent->reference_count, 1);

    // Validate data corectness
    v = 0;
    for (uint32_t q = 0; q < LENGTH; q++)
    for (uint32_t p = 0; p < LENGTH; p++)
    for (uint32_t z = 0; z < LENGTH; z++)
    for (uint32_t y = 0; y < LENGTH; y++)
    for (uint32_t x = 0; x < LENGTH; x++)
    for (uint32_t n = 0; n < LENGTH; n++, v++)
        EXPECT_EQ(v, nn_workload_data_get_int16(data, n, x, y, z, p, q));

    EXPECT_EQ(NN_DATA_STATUS_OK, nn_workload_data_delete(data));
}