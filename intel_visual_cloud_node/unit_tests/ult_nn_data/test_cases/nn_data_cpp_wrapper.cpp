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

#include "nn_workload_data_cpp_wrapper.h"

// default ordering - used for tests where just any ordering would work fine
const nn_workload_data_coords_t default_ordering = { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q };
typedef struct
{
    uint32_t test_id;
    nn_workload_data_coords_t lengths;
    nn_workload_data_layout_t layout;
} test_conf_t;

static test_conf_t test_conf[] = {
    {
        /* basic test */
        0, //test_id
        { 4, 4, 4, 4, 4, 4 }, // lengths
        {
            { 0, 0, 0, 0, 0, 0 }, // tile_log2
            { 0, 0, 0, 0, 0, 0 }, // alignment
            default_ordering,     // ordering
            NN_DATATYPE_FLOAT
        }
    },
    {
        /* basic test with tiles */
        1,                    //test_id
        { 4, 4, 4, 4, 4, 4 }, // lengths
        {
            { 1, 1, 1, 1, 1, 1 }, // tile_log2
            { 0, 0, 0, 0, 0, 0 }, // alignment
            default_ordering,     // ordering
            NN_DATATYPE_FLOAT
        }
    },
    {
        /* tiles sizes aren't equal in each direction */
        2,
        { 16, 16, 16, 16, 16, 16 }, // lengths
        {
            { 1, 2, 0, 1, 0, 2 },       // tile_log2
            { 0, 0, 0, 0, 0, 0 },       // alignment
            { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q },       // ordering
            NN_DATATYPE_FLOAT
        }
    },
    {
        3,
        { 3, 24, 15, 16, 5, 5 },   // lengths
        {
            { 1, 2, 0, 1, 0, 2 },       // tile_log2
            { 0, 0, 0, 0, 0, 0 },       // alignment
            { NN_DATA_COORD_n, NN_DATA_COORD_y, NN_DATA_COORD_q, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_x },       // ordering
            NN_DATATYPE_FLOAT
        }
    },
    {
        4, //alignment test
        { 4, 4, 4, 4, 4, 4 }, // lengths
        {
            { 0, 0, 0, 0, 0, 0 }, // tile_log2
            { 0, 5, 8, 0, 0, 0 }, //
            default_ordering, // ordering
            NN_DATATYPE_FLOAT
        }
    },
};

void nn_workload_data_test(test_conf_t& tc)
{
    nn::nn_workload_data_t<float> nndata(tc.lengths, tc.layout);

    ASSERT_NE(nndata.parent->data_buffer, nullptr);

    ASSERT_EQ(nndata.get_length(NN_DATA_COORD_n), tc.lengths.t[NN_DATA_COORD_n]);
    ASSERT_EQ(nndata.get_length(NN_DATA_COORD_x), tc.lengths.t[NN_DATA_COORD_x]);
    ASSERT_EQ(nndata.get_length(NN_DATA_COORD_y), tc.lengths.t[NN_DATA_COORD_y]);
    ASSERT_EQ(nndata.get_length(NN_DATA_COORD_z), tc.lengths.t[NN_DATA_COORD_z]);
    ASSERT_EQ(nndata.get_length(NN_DATA_COORD_p), tc.lengths.t[NN_DATA_COORD_p]);
    ASSERT_EQ(nndata.get_length(NN_DATA_COORD_q), tc.lengths.t[NN_DATA_COORD_q]);

    ASSERT_EQ(nndata.get_tile_length(NN_DATA_COORD_n), 1 << tc.layout.tile_lengths_log2.t[NN_DATA_COORD_n]);
    ASSERT_EQ(nndata.get_tile_length(NN_DATA_COORD_x), 1 << tc.layout.tile_lengths_log2.t[NN_DATA_COORD_x]);
    ASSERT_EQ(nndata.get_tile_length(NN_DATA_COORD_y), 1 << tc.layout.tile_lengths_log2.t[NN_DATA_COORD_y]);
    ASSERT_EQ(nndata.get_tile_length(NN_DATA_COORD_z), 1 << tc.layout.tile_lengths_log2.t[NN_DATA_COORD_z]);
    ASSERT_EQ(nndata.get_tile_length(NN_DATA_COORD_p), 1 << tc.layout.tile_lengths_log2.t[NN_DATA_COORD_p]);
    ASSERT_EQ(nndata.get_tile_length(NN_DATA_COORD_q), 1 << tc.layout.tile_lengths_log2.t[NN_DATA_COORD_q]);

    float v = 0.0f;

    for (uint32_t q = 0; q < tc.lengths.t[NN_DATA_COORD_q]; q++)
    for (uint32_t p = 0; p < tc.lengths.t[NN_DATA_COORD_p]; p++)
    for (uint32_t z = 0; z < tc.lengths.t[NN_DATA_COORD_z]; z++)
    for (uint32_t y = 0; y < tc.lengths.t[NN_DATA_COORD_y]; y++)
    for (uint32_t x = 0; x < tc.lengths.t[NN_DATA_COORD_x]; x++)
    for (uint32_t n = 0; n < tc.lengths.t[NN_DATA_COORD_n]; n++, v++)
        nndata(n, x, y, z, p, q) = v;

    // check data integrity 
    v = 0.0f;
    for (uint32_t q = 0; q < tc.lengths.t[NN_DATA_COORD_q]; q++)
    for (uint32_t p = 0; p < tc.lengths.t[NN_DATA_COORD_p]; p++)
    for (uint32_t z = 0; z < tc.lengths.t[NN_DATA_COORD_z]; z++)
    for (uint32_t y = 0; y < tc.lengths.t[NN_DATA_COORD_y]; y++)
    for (uint32_t x = 0; x < tc.lengths.t[NN_DATA_COORD_x]; x++)
    for (uint32_t n = 0; n < tc.lengths.t[NN_DATA_COORD_n]; n++, v++)
        EXPECT_EQ(v, nndata(n, x, y, z, p, q)());
}

TEST(nn_workload_data_cpp_wrapper, basic_float_test)
{
    for (auto& test : test_conf)
        nn_workload_data_test(test);
}

template <typename T>
void validate_view(nn::nn_workload_data_t<T>& nndata, nn::nn_workload_data_t<T>& nnview, nn_workload_data_coords_t& view_begin, nn_workload_data_coords_t& view_end)
{
    T v = 0;

    for (uint32_t q = view_begin.t[NN_DATA_COORD_q]; q <= view_end.t[NN_DATA_COORD_q]; q++)
    for (uint32_t p = view_begin.t[NN_DATA_COORD_p]; p <= view_end.t[NN_DATA_COORD_p]; p++)
    for (uint32_t z = view_begin.t[NN_DATA_COORD_z]; z <= view_end.t[NN_DATA_COORD_z]; z++)
    for (uint32_t y = view_begin.t[NN_DATA_COORD_y]; y <= view_end.t[NN_DATA_COORD_y]; y++)
    for (uint32_t x = view_begin.t[NN_DATA_COORD_x]; x <= view_end.t[NN_DATA_COORD_x]; x++)
    for (uint32_t n = view_begin.t[NN_DATA_COORD_n]; n <= view_end.t[NN_DATA_COORD_n]; n++, v++)
        nndata(n, x, y, z, p, q) = v;

    v = 0;
    for (uint32_t q = view_begin.t[NN_DATA_COORD_q]; q <= view_end.t[NN_DATA_COORD_q]; q++)
    for (uint32_t p = view_begin.t[NN_DATA_COORD_p]; p <= view_end.t[NN_DATA_COORD_p]; p++)
    for (uint32_t z = view_begin.t[NN_DATA_COORD_z]; z <= view_end.t[NN_DATA_COORD_z]; z++)
    for (uint32_t y = view_begin.t[NN_DATA_COORD_y]; y <= view_end.t[NN_DATA_COORD_y]; y++)
    for (uint32_t x = view_begin.t[NN_DATA_COORD_x]; x <= view_end.t[NN_DATA_COORD_x]; x++)
    for (uint32_t n = view_begin.t[NN_DATA_COORD_n]; n <= view_end.t[NN_DATA_COORD_n]; n++, v++)
        EXPECT_EQ(v, nndata(n, x, y, z, p, q)());

    v = 0;
    for (uint32_t q = 0; q <= (view_end.t[NN_DATA_COORD_q] - view_begin.t[NN_DATA_COORD_q]); q++)
    for (uint32_t p = 0; p <= (view_end.t[NN_DATA_COORD_p] - view_begin.t[NN_DATA_COORD_p]); p++)
    for (uint32_t z = 0; z <= (view_end.t[NN_DATA_COORD_z] - view_begin.t[NN_DATA_COORD_z]); z++)
    for (uint32_t y = 0; y <= (view_end.t[NN_DATA_COORD_y] - view_begin.t[NN_DATA_COORD_y]); y++)
    for (uint32_t x = 0; x <= (view_end.t[NN_DATA_COORD_x] - view_begin.t[NN_DATA_COORD_x]); x++)
    for (uint32_t n = 0; n <= (view_end.t[NN_DATA_COORD_n] - view_begin.t[NN_DATA_COORD_n]); n++, v++)
        EXPECT_EQ(v, (nnview)(n, x, y, z, p, q)());

}

TEST(nn_workload_data_cpp_wrapper, basic_view_test)
{
    nn_workload_data_coords_t lengths = { 4, 4, 4, 4, 4, 4 };
    nn_workload_data_layout_t layout =
    {
        { 0, 0, 0, 0, 0, 0 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        default_ordering, // ordering
        NN_DATATYPE_FLOAT
    };

    nn_workload_data_coords_t view_begin(1, 1, 1, 1, 1, 1);
    nn_workload_data_coords_t view_end(2, 2, 2, 2, 2, 2);
    
    nn::nn_workload_data_t<float> nndata(lengths, layout);
    
    nn::nn_workload_data_t<float> nnview(nndata, view_begin, view_end);

    ASSERT_NE(nndata.parent->data_buffer, nullptr);
   
    // Two views that have the same parent
    ASSERT_EQ(nnview.parent, nndata.parent);
    
    validate_view(nndata, nnview, view_begin, view_end);

}   

TEST(nn_workload_data_cpp_wrapper, tiling_view_test)
{
    nn_workload_data_coords_t lengths = { 8, 8, 8, 8, 8, 8 };
    nn_workload_data_layout_t layout =
    {
        { 1, 1, 1, 1, 1, 1 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        default_ordering, // ordering
        NN_DATATYPE_INT16
    };

    nn_workload_data_coords_t view_begin(2, 2, 2, 2, 2, 2);
    nn_workload_data_coords_t view_end(5, 5, 5, 5, 5, 5);

    nn::nn_workload_data_t<int16_t> nndata(lengths, layout);

    nn::nn_workload_data_t<int16_t> nnview(nndata, view_begin, view_end);

    ASSERT_NE(nndata.parent->data_buffer, nullptr);

    // Two views that have the same parent
    ASSERT_EQ(nnview.parent, nndata.parent);

    validate_view(nndata, nnview, view_begin, view_end);

    // View created from a view
    nn_workload_data_coords_t view2_begin(0, 0, 0, 0, 0, 0);
    nn_workload_data_coords_t view2_end(1, 1, 1, 1, 1, 1);
    nn_workload_data_coords_t view3_begin(2, 2, 2, 2, 2, 2);
    nn_workload_data_coords_t view3_end(3, 3, 3, 3, 3, 3);
    nn::nn_workload_data_t<int16_t> nnview2(nnview, view2_begin, view2_end);
    nn::nn_workload_data_t<int16_t> nnview3(nnview, view3_begin, view3_end);

    EXPECT_EQ(4, nndata.parent->reference_count);

    // make sure the views give access to the correct data
    validate_view(nnview, nnview2, view2_begin, view2_end);
    validate_view(nnview, nnview3, view3_begin, view3_end);

    // Invalid view requested - causes an exception
    nn_workload_data_coords_t view4_begin(1, 1, 1, 1, 1, 1);
    nn_workload_data_coords_t view4_end(2, 2, 2, 2, 2, 2);
    EXPECT_THROW(nn::nn_workload_data_t<int16_t> nnview3(nnview, view4_begin, view4_end), std::bad_alloc);
}

/*
    Test that exception is thrown when a user provides
    incorrect parameters for creating nn_workload_data.
*/
TEST(nn_workload_data_cpp_wrapper, incorrect_parameters_test)
{
    nn_workload_data_coords_t lengths = { 8, 8, 1, 8, 1, 8 };
    nn_workload_data_coords_t incorrect_lengths = { 8, 8, 0, 8, 1, 8 };
    nn_workload_data_layout_t layout =
    {
        { 1, 1, 1, 1, 1, 1 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        default_ordering, // ordering
        NN_DATATYPE_INT16
    };
    nn_workload_data_layout_t incorrect_layout_1 =
    {
        { 1, 1, 1, 1, 1, 1 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_x, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering - incorrect
        NN_DATATYPE_INT16
    };
    nn_workload_data_layout_t incorrect_layout_2 =
    {
        { 1, 1, 1, 1, 1, 1 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { NN_DATA_COORD_x, NN_DATA_COORD_q, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, // ordering - incorrect
        NN_DATATYPE_INT16
    };

    EXPECT_NO_THROW(nn::nn_workload_data_t<int16_t> nndata(lengths, layout));
    EXPECT_THROW(nn::nn_workload_data_t<int16_t> nndata(incorrect_lengths, layout), std::bad_alloc);
    EXPECT_THROW(nn::nn_workload_data_t<int16_t> nndata(lengths, incorrect_layout_1), std::bad_alloc);
    EXPECT_THROW(nn::nn_workload_data_t<int16_t> nndata(lengths, incorrect_layout_2), std::bad_alloc);
}

/* helper macros for ordering and tiling tests */
#define VAL_N_SHIFT 0
#define VAL_X_SHIFT 2
#define VAL_Y_SHIFT 4
#define VAL_Z_SHIFT 6
#define VAL_P_SHIFT 8
#define VAL_Q_SHIFT 10

/* Create value from coordinates */
#define VAL_FROM_COORD(n,x,y,z,p,q) ((n<<VAL_N_SHIFT) | (x<<VAL_X_SHIFT) | (y<<VAL_Y_SHIFT) | (z<<VAL_Z_SHIFT) | (p<<VAL_P_SHIFT) | (q<<VAL_Q_SHIFT))

/* Create value from tile coordinates */
#define VAL_FROM_TILE(n,x,y,z,p,q, tile_len_log2) VAL_FROM_COORD(n >> tile_len_log2[NN_DATA_COORD_n], \
    x >> tile_len_log2[NN_DATA_COORD_x], \
    y >> tile_len_log2[NN_DATA_COORD_y], \
    z >> tile_len_log2[NN_DATA_COORD_z], \
    p >> tile_len_log2[NN_DATA_COORD_p], \
    q >> tile_len_log2[NN_DATA_COORD_q])

/*
    Fill out the data with values based on coordinates,
    so that we will be able to infer the coordinates from values.
*/
template<typename T>
void fill_data(nn::nn_workload_data_t<T>& nndata)
{
    for (uint32_t q = 0; q < nndata.get_length(NN_DATA_COORD_q); q++)
    for (uint32_t p = 0; p < nndata.get_length(NN_DATA_COORD_p); p++)
    for (uint32_t z = 0; z < nndata.get_length(NN_DATA_COORD_z); z++)
    for (uint32_t y = 0; y < nndata.get_length(NN_DATA_COORD_y); y++)
    for (uint32_t x = 0; x < nndata.get_length(NN_DATA_COORD_x); x++)
    for (uint32_t n = 0; n < nndata.get_length(NN_DATA_COORD_n); n++)
        nndata(n, x, y, z, p, q) = VAL_FROM_COORD(n, x, y, z, p, q);

    // validate that nothing was overwritten
    for (uint32_t q = 0; q < nndata.get_length(NN_DATA_COORD_q); q++)
    for (uint32_t p = 0; p < nndata.get_length(NN_DATA_COORD_p); p++)
    for (uint32_t z = 0; z < nndata.get_length(NN_DATA_COORD_z); z++)
    for (uint32_t y = 0; y < nndata.get_length(NN_DATA_COORD_y); y++)
    for (uint32_t x = 0; x < nndata.get_length(NN_DATA_COORD_x); x++)
    for (uint32_t n = 0; n < nndata.get_length(NN_DATA_COORD_n); n++)
        EXPECT_EQ(VAL_FROM_COORD(n, x, y, z, p, q), nndata(n, x, y, z, p, q)());
}

/*
Each tile is filled out with a value that identifies that tile.
The value is const accross a tile but unique for each tile.
*/
template<typename T>
void fill_data_with_tile_id(nn::nn_workload_data_t<T>& nndata)
{
    for (uint32_t q = 0; q < nndata.get_length(NN_DATA_COORD_q); q++)
    for (uint32_t p = 0; p < nndata.get_length(NN_DATA_COORD_p); p++)
    for (uint32_t z = 0; z < nndata.get_length(NN_DATA_COORD_z); z++)
    for (uint32_t y = 0; y < nndata.get_length(NN_DATA_COORD_y); y++)
    for (uint32_t x = 0; x < nndata.get_length(NN_DATA_COORD_x); x++)
    for (uint32_t n = 0; n < nndata.get_length(NN_DATA_COORD_n); n++)
        nndata(n, x, y, z, p, q) = VAL_FROM_TILE(n, x, y, z, p, q, nndata.parent->layout.tile_lengths_log2.t);

    // validate that nothing was overwritten
    for (uint32_t q = 0; q < nndata.get_length(NN_DATA_COORD_q); q++)
    for (uint32_t p = 0; p < nndata.get_length(NN_DATA_COORD_p); p++)
    for (uint32_t z = 0; z < nndata.get_length(NN_DATA_COORD_z); z++)
    for (uint32_t y = 0; y < nndata.get_length(NN_DATA_COORD_y); y++)
    for (uint32_t x = 0; x < nndata.get_length(NN_DATA_COORD_x); x++)
    for (uint32_t n = 0; n < nndata.get_length(NN_DATA_COORD_n); n++)
        EXPECT_EQ(VAL_FROM_TILE(n, x, y, z, p, q, nndata.parent->layout.tile_lengths_log2.t), nndata(n, x, y, z, p, q)());
}


/*
    Validate that data is placed according to requested ordering.
*/
TEST(nn_workload_data_cpp_wrapper, ordering_test)
{
    nn_workload_data_coords_t lengths = { 3, 4, 3, 4, 3, 4 };
    nn_workload_data_layout_t layout=
        {
            { 0, 0, 0, 0, 0, 0 }, // tile_log2
            { 0, 0, 0, 0, 0, 0 }, // alignment
            { 0, 0, 0, 0, 0, 0 }, // ordering - will be set later for each test case
            NN_DATATYPE_INT16
        };

    // we'll run test for three sample orderings
    nn_workload_data_coords_t ordering_nxyzpq = { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q };
    nn_workload_data_coords_t ordering_xyzpqn = { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n };
    nn_workload_data_coords_t ordering_qnxzyp = { NN_DATA_COORD_q, NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_p };

    {
        layout.ordering = ordering_nxyzpq;
        nn::nn_workload_data_t<int16_t> nndata(lengths, layout);

        fill_data(nndata);

        int16_t* buffer = (int16_t*)nndata.parent->data_buffer;

        EXPECT_EQ(1 << VAL_X_SHIFT, buffer[lengths.t[NN_DATA_COORD_n]]);

        EXPECT_EQ((1 << VAL_Y_SHIFT), buffer[lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x]]);
        EXPECT_EQ((2 << VAL_Y_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x]]);

        EXPECT_EQ((1 << VAL_Z_SHIFT), buffer[lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y]]);
        EXPECT_EQ((1 << VAL_P_SHIFT), buffer[lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z]]);
        EXPECT_EQ((1 << VAL_Q_SHIFT), buffer[lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_p]]);
        EXPECT_EQ((3 << VAL_Q_SHIFT), buffer[3 * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_p]]);
    }

    {
        layout.ordering = ordering_xyzpqn;
        nn::nn_workload_data_t<int16_t> nndata(lengths, layout);

        fill_data(nndata);

        int16_t* buffer = (int16_t*)nndata.parent->data_buffer;

        EXPECT_EQ(1 << VAL_Y_SHIFT, buffer[lengths.t[NN_DATA_COORD_x]]);

        EXPECT_EQ((1 << VAL_Z_SHIFT), buffer[lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y]]);
        EXPECT_EQ((3 << VAL_Z_SHIFT), buffer[3 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y]]);

        EXPECT_EQ((2 << VAL_P_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z]]);
        EXPECT_EQ((3 << VAL_Q_SHIFT), buffer[3 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_p]]);
        EXPECT_EQ((1 << VAL_N_SHIFT), buffer[1 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_p] * lengths.t[NN_DATA_COORD_q]]);
        EXPECT_EQ((2 << VAL_N_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_p] * lengths.t[NN_DATA_COORD_q]]);
    }

    {
        layout.ordering = ordering_qnxzyp;
        nn::nn_workload_data_t<int16_t> nndata(lengths, layout);

        fill_data(nndata);

        int16_t* buffer = (int16_t*)nndata.parent->data_buffer;
        EXPECT_EQ((1 << VAL_N_SHIFT), buffer[lengths.t[NN_DATA_COORD_q]]);

        EXPECT_EQ((1 << VAL_X_SHIFT), buffer[    lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n]]);
        EXPECT_EQ((2 << VAL_X_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n]]);                                                                                                      
        EXPECT_EQ((2 << VAL_Z_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x]]);
        EXPECT_EQ((2 << VAL_Y_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_z]]);
        EXPECT_EQ((1 << VAL_P_SHIFT), buffer[1 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_y]]);
        EXPECT_EQ((2 << VAL_P_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_y]]);
    }
}

/*
    Validate that data is placed according to requested ordering.
    This test validates ordering inside a tile.
*/
TEST(nn_workload_data_cpp_wrapper, ordering_inside_tile_test)
{
    nn_workload_data_coords_t lengths = { 4, 4, 4, 4, 4, 4 };
    nn_workload_data_coords_t ordering_qnxzyp = { NN_DATA_COORD_q, NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_p };
    
    // layout - create one tile that contains entire data structure
    nn_workload_data_layout_t layout =
    {
        { 2, 2, 2, 2, 2, 2 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        ordering_qnxzyp,      // ordering
        NN_DATATYPE_INT16
    };

    nn::nn_workload_data_t<int16_t> nndata(lengths, layout);

    fill_data(nndata);

    int16_t* buffer = (int16_t*)nndata.parent->data_buffer;
    EXPECT_EQ((1 << VAL_N_SHIFT), buffer[lengths.t[NN_DATA_COORD_q]]);

    EXPECT_EQ((1 << VAL_X_SHIFT), buffer[lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n]]);
    EXPECT_EQ((2 << VAL_X_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n]]);
    EXPECT_EQ((2 << VAL_Z_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x]]);
    EXPECT_EQ((2 << VAL_Y_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_z]]);
    EXPECT_EQ((1 << VAL_P_SHIFT), buffer[1 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_y]]);
    EXPECT_EQ((2 << VAL_P_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_y]]);
}


/* In this test case data lengths are multiple of tile lengths,
   so there are no "gaps" in the tiles */
TEST(nn_workload_data_cpp_wrapper, tiling_test)
{
    nn_workload_data_coords_t lengths = { 3, 4, 6, 8, 8, 8 };
    nn_workload_data_layout_t layout =
    {
        { 0, 1, 1, 2, 2, 2 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        default_ordering, // ordering
        NN_DATATYPE_INT16
    };

    nn::nn_workload_data_t<int16_t> nndata(lengths, layout);
    
    /* Each tile is filled out with a value that identifies that tile.
       The value is const accross a tile but unique for each tile. */
    fill_data_with_tile_id(nndata);

    auto tile_size = 1;
    for (auto l : layout.tile_lengths_log2.t)
        tile_size <<= l;

    int16_t* buffer = (int16_t*)nndata.parent->data_buffer;

    /* Verify that the tiles are filled in expected way */
    for (auto tile = 0; tile < (nndata.parent->buffer_size / (tile_size*sizeof(int16_t))); tile++)
    {
        for (auto i = 0; i < tile_size; i++)
        {
            EXPECT_EQ(buffer[tile_size * tile], buffer[tile_size * tile + i]);
        }
    }
}

/* In this test case data lengths are not multiple of tile lengths,
    so there are "gaps" in the buffer */
TEST(nn_workload_data_cpp_wrapper, tiling_test2)
{
    nn_workload_data_coords_t lengths = { 3, 4, 5, 6, 7, 8 };
    nn_workload_data_layout_t layout =
    {
        { 0, 1, 2, 3, 2, 3 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        default_ordering, // ordering
        NN_DATATYPE_INT16
    };

    nn::nn_workload_data_t<int16_t> nndata(lengths, layout);

    /* This will allow to detect memory locations that are not used ("gaps") */
    const int MEMSET_MAGIC = 0xAB;
    const int16_t INT16_MAGIC = (int16_t) (MEMSET_MAGIC << 8) | MEMSET_MAGIC;
    memset(nndata.parent->data_buffer, MEMSET_MAGIC, nndata.parent->buffer_size);

    /* Each tile is filled out with a value that identifies that tile.
       The value is const accross a tile but unique for each tile */
    fill_data_with_tile_id(nndata);

    auto tiles_count = 1;
    for (auto i = 0; i < NN_DIMENSION_COUNT; i++)
    {
        auto tile_lenght = 1 << layout.tile_lengths_log2.t[i];
        tiles_count *= (lengths.t[i] + tile_lenght - 1) / tile_lenght;
    }
    auto tile_size = nndata.parent->buffer_size / tiles_count / sizeof(int16_t);
    int16_t* buffer = (int16_t*)nndata.parent->data_buffer;

    /* Verify that the tiles are filled in expected way */
    for (auto tile = 0; tile < tiles_count; tile++)
    {
        for (auto i = 0; i < tile_size; i++)
        {
            EXPECT_EQ(true, (buffer[tile_size * tile] == buffer[tile_size * tile + i]) || (INT16_MAGIC == buffer[tile_size * tile + i]));
        }
    }
}

/*
    Tile alignment test.
*/
TEST(nn_workload_data_cpp_wrapper, tile_alignment_test)
{
    nn_workload_data_coords_t lengths = { 3, 4, 4, 4, 4, 4 };
    const uint16_t alignment = 6; // 2^6 = 64
    nn_workload_data_layout_t layout =
    {
        { 0, 1, 1, 1, 0, 0 }, // tile_log2
        { alignment, 0, 0, 0, 0, 0 }, // alignment
        default_ordering, // ordering
        NN_DATATYPE_INT16
    };

    nn::nn_workload_data_t<int16_t> nndata(lengths, layout);

    /* This will allow to detect memory locations that are not used ("gaps") */
    const int MEMSET_MAGIC = 0xAB;
    const int16_t INT16_MAGIC = (int16_t)(MEMSET_MAGIC << 8) | MEMSET_MAGIC;
    memset(nndata.parent->data_buffer, MEMSET_MAGIC, nndata.parent->buffer_size);

    /* Each tile is filled out with a value that identifies that tile.
    The value is const accross a tile but unique for each tile. */
    fill_data_with_tile_id(nndata);

    int16_t* buffer = (int16_t*)nndata.parent->data_buffer;
    int16_t value = buffer[0];
    auto tile_num = 1;
    
    /* Verify that the tiles are filled in expected way */
    for (auto i = 0; i < nndata.parent->buffer_size / sizeof(int16_t); i++)
    {
        if ((value != buffer[i]) && (INT16_MAGIC != buffer[i]))
        {
            // each tile should be aligned
            EXPECT_EQ(i, tile_num * ((1 << alignment)/sizeof(int16_t)));
            tile_num++;
            value = buffer[i];
        }
    }
}

/*
    Alignment test without tiling
*/
TEST(nn_workload_data_cpp_wrapper, alignment_test)
{
    nn_workload_data_coords_t lengths = { 3, 4, 4, 4, 4, 4 };
    nn_workload_data_layout_t layout =
    {
        { 0, 0, 0, 0, 0, 0 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        default_ordering, // ordering
        NN_DATATYPE_INT16
    };

    // we'll run test for two examples
    nn_workload_data_coords_t alignment_n8 = { 3, 0, 0, 0, 0, 0 };
    nn_workload_data_coords_t alignment_n8_x64 = { 3, 6, 0, 0, 0, 0 };

    {
        layout.alignment_log2 = alignment_n8;
        nn::nn_workload_data_t<int16_t> nndata(lengths, layout);

        memset(nndata.parent->data_buffer, 0, nndata.parent->buffer_size);
        fill_data(nndata);

        int16_t* buffer = (int16_t*)nndata.parent->data_buffer;

        // without multiplication by 1, VC++ shows warnings...
        EXPECT_EQ(1 << VAL_N_SHIFT, buffer[1 * (1<<layout.alignment_log2.t[NN_DATA_COORD_n])/sizeof(int16_t)]);
        EXPECT_EQ(1 << VAL_X_SHIFT, buffer[1 * lengths.t[NN_DATA_COORD_n] * (1 << layout.alignment_log2.t[NN_DATA_COORD_n]) / sizeof(int16_t)]);
        EXPECT_EQ(1 << VAL_Y_SHIFT, buffer[1 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_n] * (1 << layout.alignment_log2.t[NN_DATA_COORD_n]) / sizeof(int16_t)]);
        EXPECT_EQ(2 << VAL_Y_SHIFT, buffer[2 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_n] * (1 << layout.alignment_log2.t[NN_DATA_COORD_n]) / sizeof(int16_t)]);
        EXPECT_EQ(2 << VAL_Z_SHIFT, buffer[2 * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_n] * (1 << layout.alignment_log2.t[NN_DATA_COORD_n]) / sizeof(int16_t)]);
    }

    {
        layout.alignment_log2 = alignment_n8_x64;
        nn::nn_workload_data_t<int16_t> nndata(lengths, layout);

        memset(nndata.parent->data_buffer, 0, nndata.parent->buffer_size);
        fill_data(nndata);

        int16_t* buffer = (int16_t*)nndata.parent->data_buffer;
        
        // without multiplication by 1, VC++ shows warnings...
        EXPECT_EQ(1 << VAL_N_SHIFT, buffer[1 * (1 << layout.alignment_log2.t[NN_DATA_COORD_n]) / sizeof(int16_t)]);
        EXPECT_EQ(1 << VAL_X_SHIFT, buffer[1 * (1 << layout.alignment_log2.t[NN_DATA_COORD_x]) / sizeof(int16_t)]);
        EXPECT_EQ(1 << VAL_Y_SHIFT, buffer[1 * lengths.t[NN_DATA_COORD_x] * (1 << layout.alignment_log2.t[NN_DATA_COORD_x]) / sizeof(int16_t)]);
        EXPECT_EQ(2 << VAL_Y_SHIFT, buffer[2 * lengths.t[NN_DATA_COORD_x] * (1 << layout.alignment_log2.t[NN_DATA_COORD_x]) / sizeof(int16_t)]);
        EXPECT_EQ(2 << VAL_Z_SHIFT, buffer[2 * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_x] * (1 << layout.alignment_log2.t[NN_DATA_COORD_x]) / sizeof(int16_t)]);
    }
}


/*
    Test creating nn_workload_data_t that uses preallocated data buffer.
*/
TEST(nn_workload_data_cpp_wrapper, create_from_existing_buffer_test)
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
    nn::nn_workload_data_t<int16_t> nndata(buffer, lengths, layout);
    
    EXPECT_EQ(nndata.parent->data_buffer, buffer);
    EXPECT_EQ(nndata.parent->buffer_size, buffer_size * sizeof(int16_t));

    // Validate data corectness
    v = 0;
    for (uint32_t q = 0; q < LENGTH; q++)
    for (uint32_t p = 0; p < LENGTH; p++)
    for (uint32_t z = 0; z < LENGTH; z++)
    for (uint32_t y = 0; y < LENGTH; y++)
    for (uint32_t x = 0; x < LENGTH; x++)
    for (uint32_t n = 0; n < LENGTH; n++, v++)
        EXPECT_EQ(v, nndata(n, x, y, z, p, q)());

}

/*
    A test case for nn_workload_data copy functionality.
    Validate that data is correctly copied according to the new layout.
*/
TEST(nn_workload_data_cpp_wrapper, copy_test)
{
    nn_workload_data_coords_t lengths = { 3, 4, 3, 4, 3, 4 };
    nn_workload_data_layout_t layout =
    {
        { 0, 0, 0, 0, 0, 0 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { 0, 0, 0, 0, 0, 0 }, // ordering - will be set later for each test case
        NN_DATATYPE_INT16
    };

    nn_workload_data_coords_t ordering_nxyzpq = { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q };
    nn_workload_data_coords_t ordering_xyzpqn = { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n };
    nn_workload_data_coords_t ordering_qnxzyp = { NN_DATA_COORD_q, NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_p };

    layout.ordering = ordering_nxyzpq;
    nn::nn_workload_data_t<int16_t> nndata(lengths, layout);

    fill_data(nndata);

    {
        layout.ordering = ordering_xyzpqn;
        // Create new nn_workload_data_t based on data copied from another nn_workload_data_t
        nn::nn_workload_data_t<int16_t> nndata_1(nndata, layout);

        int16_t* buffer = (int16_t*)nndata_1.parent->data_buffer;

        EXPECT_EQ(1 << VAL_Y_SHIFT, buffer[lengths.t[NN_DATA_COORD_x]]);

        EXPECT_EQ((1 << VAL_Z_SHIFT), buffer[lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y]]);
        EXPECT_EQ((3 << VAL_Z_SHIFT), buffer[3 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y]]);

        EXPECT_EQ((2 << VAL_P_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z]]);
        EXPECT_EQ((3 << VAL_Q_SHIFT), buffer[3 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_p]]);
        EXPECT_EQ((1 << VAL_N_SHIFT), buffer[1 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_p] * lengths.t[NN_DATA_COORD_q]]);
        EXPECT_EQ((2 << VAL_N_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_y] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_p] * lengths.t[NN_DATA_COORD_q]]);
    }

    {
        layout.ordering = ordering_qnxzyp;
        nn::nn_workload_data_t<int16_t> nndata_1(lengths, layout);

        // Copy from another nn_workload_data_t
        nndata_1.copy(nndata);

        int16_t* buffer = (int16_t*)nndata_1.parent->data_buffer;
        EXPECT_EQ((1 << VAL_N_SHIFT), buffer[lengths.t[NN_DATA_COORD_q]]);

        EXPECT_EQ((1 << VAL_X_SHIFT), buffer[lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n]]);
        EXPECT_EQ((2 << VAL_X_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n]]);
        EXPECT_EQ((2 << VAL_Z_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x]]);
        EXPECT_EQ((2 << VAL_Y_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_z]]);
        EXPECT_EQ((1 << VAL_P_SHIFT), buffer[1 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_y]]);
        EXPECT_EQ((2 << VAL_P_SHIFT), buffer[2 * lengths.t[NN_DATA_COORD_q] * lengths.t[NN_DATA_COORD_n] * lengths.t[NN_DATA_COORD_x] * lengths.t[NN_DATA_COORD_z] * lengths.t[NN_DATA_COORD_y]]);
    }
}

/*
    A test case for nn_workload_data copy functionality.
    Validate that data is correctly copied according to the new layout.
*/
TEST(nn_workload_data_cpp_wrapper, copy_float_test)
{
    nn_workload_data_coords_t lengths = { 3, 4, 3, 4, 3, 4 };
    nn_workload_data_layout_t layout =
    {
        { 0, 0, 0, 0, 0, 0 }, // tile_log2
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { 0, 0, 0, 0, 0, 0 }, // ordering - will be set later for each test case
        NN_DATATYPE_FLOAT
    };

    nn_workload_data_coords_t ordering_nxyzpq = { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q };
    nn_workload_data_coords_t ordering_xyzpqn = { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n };
    nn_workload_data_coords_t ordering_qnxzyp = { NN_DATA_COORD_q, NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_p };

    layout.ordering = ordering_nxyzpq;
    nn::nn_workload_data_t<float> nndata(lengths, layout);

    // Fill the buffer with some data
    float v = 0.0;
    for (uint32_t q = 0; q < lengths.t[NN_DATA_COORD_q]; q++)
    for (uint32_t p = 0; p < lengths.t[NN_DATA_COORD_p]; p++)
    for (uint32_t z = 0; z < lengths.t[NN_DATA_COORD_z]; z++)
    for (uint32_t y = 0; y < lengths.t[NN_DATA_COORD_y]; y++)
    for (uint32_t x = 0; x < lengths.t[NN_DATA_COORD_x]; x++)
    for (uint32_t n = 0; n < lengths.t[NN_DATA_COORD_n]; n++, v++)
        nndata(n, x, y, z, p, q) = v;

    {
        layout.ordering = ordering_xyzpqn;
        // Create new nn_workload_data_t based on data copied from another nn_workload_data_t
        nn::nn_workload_data_t<float> nndata_1(nndata, layout);

        // check data integrity 
        v = 0.0f;
        for (uint32_t q = 0; q < lengths.t[NN_DATA_COORD_q]; q++)
        for (uint32_t p = 0; p < lengths.t[NN_DATA_COORD_p]; p++)
        for (uint32_t z = 0; z < lengths.t[NN_DATA_COORD_z]; z++)
        for (uint32_t y = 0; y < lengths.t[NN_DATA_COORD_y]; y++)
        for (uint32_t x = 0; x < lengths.t[NN_DATA_COORD_x]; x++)
        for (uint32_t n = 0; n < lengths.t[NN_DATA_COORD_n]; n++, v++)
            EXPECT_EQ(v, nndata(n, x, y, z, p, q)());
    }

    {
        layout.ordering = ordering_qnxzyp;
        nn::nn_workload_data_t<float> nndata_1(lengths, layout);

        // Copy from another nn_workload_data_t
        nndata_1.copy(nndata);
        
        // check data integrity 
        v = 0.0f;
        for (uint32_t q = 0; q < lengths.t[NN_DATA_COORD_q]; q++)
        for (uint32_t p = 0; p < lengths.t[NN_DATA_COORD_p]; p++)
        for (uint32_t z = 0; z < lengths.t[NN_DATA_COORD_z]; z++)
        for (uint32_t y = 0; y < lengths.t[NN_DATA_COORD_y]; y++)
        for (uint32_t x = 0; x < lengths.t[NN_DATA_COORD_x]; x++)
        for (uint32_t n = 0; n < lengths.t[NN_DATA_COORD_n]; n++, v++)
            EXPECT_EQ(v, nndata(n, x, y, z, p, q)());
    }
}
