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

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "nn_workload_data.h"
#include "nn_allocate.h"

/* For functions parameters validation */
#define NN_COORD_SUM (NN_DATA_COORD_n + NN_DATA_COORD_x + NN_DATA_COORD_y + NN_DATA_COORD_z + NN_DATA_COORD_p + NN_DATA_COORD_q)

nn_workload_data_core_t::nn_workload_data_core_t(uint32_t data_type_size,
                                                 const nn_workload_data_coords_t lengths,
                                                 const nn_workload_data_layout_t layout,
                                                 void *buffer,
                                                 bool allow_empty_data,
                                                 bool allocate_delta = false)
    : data_type_size(data_type_size),
      lengths(lengths),
      layout(layout),
      delta_buffer(nullptr),
      tag(NN_WORKLOAD_DATA_TAG_UNKNOWN)
{
    uint32_t data_count = 1;
    uint32_t *size = new uint32_t[dimension];
    strides = new uint32_t[dimension];

    use_client_buffer = false;

    for (size_t i = 0; i < dimension; i++) {

        size[i] = lengths.t[i];
        data_count *= size[i];
    }

    // calculate stride for each dimension
    for (size_t i = 0; i < dimension; i++)
    {
        uint32_t dim = layout.ordering.t[i];
        uint32_t stride = 1;

        if (i>0)
        {
            uint32_t previous_dim = layout.ordering.t[i - 1];
            stride = strides[previous_dim] * size[previous_dim];
        }

        strides[dim] = stride;
    }

    buffer_size = data_count * data_type_size;

    if (buffer != nullptr)
    {
        data_buffer = buffer;
        use_client_buffer = 1;
    } else if (allow_empty_data){
        data_buffer = nullptr;
        use_client_buffer = 1;
    } else {
        data_buffer = nn_allocate_aligned(buffer_size);

        if (data_buffer == nullptr) {
            assert(0);
            throw std::bad_alloc();
        }
    }

    if(allocate_delta)
    {
        delta_buffer = nn_allocate_aligned(buffer_size);

        if(delta_buffer == nullptr) {
            assert(0);
            if(use_client_buffer == 0) {
                nn_free_aligned(data_buffer);
            }
            throw std::bad_alloc();
        }
    }

    delete[] size;
}

nn_workload_data_core_t::~nn_workload_data_core_t() {
    if (use_client_buffer == 0) {

        if(delta_buffer != data_buffer)
            nn_free_aligned(delta_buffer);
        nn_free_aligned(data_buffer);
    }

    delete[] strides;
}

void nn_workload_data_core_t::allocate_delta_buffer(){
    if(delta_buffer == nullptr)
    {
        delta_buffer = nn_allocate_aligned(buffer_size);

        if(delta_buffer == nullptr) {
            throw std::bad_alloc();
        }
    }
    else
    {
        // We should never get here
        assert(0);
    }

    memset(delta_buffer, 0, nn_aligned_size(buffer_size));
}

namespace
{
template <typename T>
void check_not_null(T* ptr)
{
    if (ptr == nullptr)
        throw std::runtime_error("unexpected null pointer passed");
}

/*
    Allocates and fills out nn_workload_data->parent structure.
    Does not allocate nn_workload_data nor nn_workload_data->parent->data_buffer
*/
static void nn_workload_data_parent_create(nn_workload_data_t *nn_workload_data,
                                           const nn_workload_data_coords_t *lenghts,
                                           const nn_workload_data_layout_t *layout,
                                           void *buffer,
                                           bool  allow_empty_data,
                                           bool  allocate_delta) {
    uint32_t ordering_coord_sum = 0;

    check_not_null(nn_workload_data);
    check_not_null(lenghts);
    check_not_null(layout);

    assert(lenghts->dimension==layout->ordering.dimension);

    for (uint32_t i = 0; i < static_cast<uint32_t>(lenghts->dimension); i++)
    {
        if ((lenghts->t[i] == 0) || (layout->ordering.t[i] > NN_DATA_COORD_MAX))
            throw std::runtime_error("invalid memory layout for dimension " + std::to_string(i));
        ordering_coord_sum += layout->ordering.t[i];
    }

    /* validate correct ordering input */
    if (ordering_coord_sum != NN_COORD_SUM)
        throw std::runtime_error("invalid memory layout -> doesn't provide full ordering");

    uint32_t data_type_size;
    if (layout->data_type == NN_DATATYPE_FLOAT) {
        data_type_size = sizeof(float);
    } else if (layout->data_type == NN_DATATYPE_INT16) {
        data_type_size = sizeof(short);
    } else if (layout->data_type == NN_DATATYPE_INT32) {
        data_type_size = sizeof(int);
    } else {
        throw std::runtime_error("invalid memory layout -> invalid data type");
    }

    // nn_workload_data is just a view, parent is the actual data
    nn_workload_data->parent = std::make_shared<nn_workload_data_core_t>(data_type_size, *lenghts, *layout, buffer, allow_empty_data, allocate_delta);
}

} //namespace

/*
    Creates nn_workload_data structure using workload_data allocated by a caller.
    Optionally also data buffer may provided by a caller.

    nn_workload_data - pointer to the structure allocated by a caller
    allow_empty_data       - if set, there will be no data allocated internally nor caller data will be used,
                       this option can be used when user require only data structure definition, without
                       actual data inside
    buffer           - pointer to the data buffer. If NULL, it will be allocated by this function.
    lenghts          - size of data structure in each dimension.
    layout           - contains information such as ordering and data type.
*/
void nn_workload_data_placement_create(nn_workload_data_t *nn_workload_data,
                                       void *buffer,
                                       const nn_workload_data_coords_t *lenghts,
                                       const nn_workload_data_layout_t *layout,
                                       bool allow_empty_data,
                                       bool allocate_delta) {
    check_not_null(nn_workload_data);
    check_not_null(lenghts);
    check_not_null(layout);

    if(lenghts->dimension != layout->ordering.dimension)
        throw std::runtime_error("lengths size is different than layout");

    std::unique_ptr<nn_workload_data_coords_t> temp(nn_workload_data_coords_t::create(lenghts->dimension));

    nn_workload_data->view_begin = *temp;
    nn_workload_data->view_end = *lenghts;

    for (int i = 0; i < nn_workload_data->view_end.dimension; i++)
        --nn_workload_data->view_end.t[i];

    nn_workload_data_parent_create(nn_workload_data, lenghts, layout, buffer, allow_empty_data, allocate_delta);
}

/*
    Creates view that references signal from another workload_data. Uses workload_data allocated by a caller.

    Resulting view has the same layout as original.
    If view cannot be created (outside image) - 0 is returned.
*/
void nn_workload_data_placement_create_view(nn_workload_data_t *nn_workload_data,
                                            const nn_workload_data_t *nn_source,
                                            const nn_workload_data_coords_t *coords_begin,
                                            const nn_workload_data_coords_t *coords_end) {
    uint32_t i;
    check_not_null(nn_workload_data);
    check_not_null(nn_source);
    check_not_null(coords_begin);
    check_not_null(coords_end);

    for (i = 0; i < nn_source->parent->dimension; i++)
    {
        auto view_begin = nn_source->view_begin.t[i] + coords_begin->t[i];
        auto view_end = nn_source->view_begin.t[i] + coords_end->t[i];
        auto current_end = nn_source->view_end.t[i];

        auto check_in_current = [=](uint32_t arg, std::string what) {
                if (arg > current_end)
                    throw std::runtime_error("view " + what + " for dimension " + std::to_string(i)
                        + " is outside the current buffer: " + std::to_string(arg) + " > "
                        + std::to_string(current_end));
            };

        check_in_current(view_begin, "begin");
        check_in_current(view_end, "end");
    }

    nn_workload_data->parent = nn_source->parent;
    nn_workload_data->view_begin = nn_source->view_begin;
    nn_workload_data->view_end = nn_source->view_end;

    for (i = 0; i < nn_source->parent->dimension; i++)
    {
        nn_workload_data->view_begin.t[i] = nn_source->view_begin.t[i] + coords_begin->t[i];
        nn_workload_data->view_end.t[i] = nn_source->view_begin.t[i] + coords_end->t[i];
    }
}


/*
    Creates view that references signal from another workload_data (i.e. another view).

    Resulting view has the same layout as original.
    If view cannot be created (outside image) - 0 is returned.
*/
nn_workload_data_t * nn_workload_data_create_view(const nn_workload_data_t* source,
                                                 const nn_workload_data_coords_t* coords_begin,
                                                 const nn_workload_data_coords_t* coords_end) {
    nn_workload_data_t* nn_workload_data = NULL;

    assert(source != NULL);
    assert(coords_begin != NULL);
    assert(coords_end != NULL);

    if (source == NULL || coords_begin == NULL || coords_end == NULL)
    {
        return NULL;
    }

    nn_workload_data = new nn_workload_data_t;
    if (nn_workload_data == NULL)
        return NULL;

    try {
        nn_workload_data_placement_create_view(nn_workload_data, source, coords_begin, coords_end);
    } catch (...) {
        delete nn_workload_data;
        return NULL;
    }
    return nn_workload_data;
}

/*
    Calculates index that is used to retrieve a value from a data buffer.
*/
uint32_t calculate_idx(const nn_workload_data_t* data, uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q)
{
    uint32_t coordinates[6] = { n, x, y, z, p, q };
    uint32_t index = 0;
    uint32_t i;

    auto view_end_array = &data->view_end.t[0];
    auto view_begin_array = &data->view_begin.t[0];
    auto parent_strides_array = &data->parent->strides[0];

    assert(n <= (view_end_array[NN_DATA_COORD_n] - view_begin_array[NN_DATA_COORD_n]));
    assert(x <= (view_end_array[NN_DATA_COORD_x] - view_begin_array[NN_DATA_COORD_x]));
    assert(y <= (view_end_array[NN_DATA_COORD_y] - view_begin_array[NN_DATA_COORD_y]));
    assert(z <= (view_end_array[NN_DATA_COORD_z] - view_begin_array[NN_DATA_COORD_z]));
    assert(p <= (view_end_array[NN_DATA_COORD_p] - view_begin_array[NN_DATA_COORD_p]));
    assert(q <= (view_end_array[NN_DATA_COORD_q] - view_begin_array[NN_DATA_COORD_q]));

    for (i = 0; i < data->parent->dimension; i++)
    {
        uint32_t coordinate;

        coordinate = (coordinates[i] + view_begin_array[i]);
        index += coordinate * parent_strides_array[i];
    }

    return index;
}

/*
    Copy data from source to destination.
    Data ordering and may differ between source and destination,
    but lenghts in corresponding dimensions must be equal.

    Assumption is that source and destination have different layouts,
    so we can't just use memcpy()
*/

template<typename T, bool copy_delta>
NN_DATA_STATUS nn_workload_copy(nn_workload_data_t* destination, const nn_workload_data_t* source)
{
    uint32_t n, x, y, z, p, q;

    // Get sizes of both views (source & destination)
    unsigned int source_sizes[NN_DATA_COORD_MAX + 1];
    unsigned int destination_sizes[NN_DATA_COORD_MAX + 1];

    assert(source != NULL);
    assert(destination != NULL);

    if (source == NULL || destination == NULL)
    {
        return NN_DATA_STATUS_ERROR_INVALID_POINTER;
    }

    source_sizes[NN_DATA_COORD_n] = source->view_end.t[NN_DATA_COORD_n] - source->view_begin.t[NN_DATA_COORD_n] + 1;
    source_sizes[NN_DATA_COORD_x] = source->view_end.t[NN_DATA_COORD_x] - source->view_begin.t[NN_DATA_COORD_x] + 1 ;
    source_sizes[NN_DATA_COORD_y] = source->view_end.t[NN_DATA_COORD_y] - source->view_begin.t[NN_DATA_COORD_y] + 1 ;
    source_sizes[NN_DATA_COORD_z] = source->view_end.t[NN_DATA_COORD_z] - source->view_begin.t[NN_DATA_COORD_z] + 1 ;
    source_sizes[NN_DATA_COORD_p] = source->view_end.t[NN_DATA_COORD_p] - source->view_begin.t[NN_DATA_COORD_p] + 1 ;
    source_sizes[NN_DATA_COORD_q] = source->view_end.t[NN_DATA_COORD_q] - source->view_begin.t[NN_DATA_COORD_q] + 1 ;

    destination_sizes[NN_DATA_COORD_n] = destination->view_end.t[NN_DATA_COORD_n] - destination->view_begin.t[NN_DATA_COORD_n] + 1;
    destination_sizes[NN_DATA_COORD_x] = destination->view_end.t[NN_DATA_COORD_x] - destination->view_begin.t[NN_DATA_COORD_x] + 1 ;
    destination_sizes[NN_DATA_COORD_y] = destination->view_end.t[NN_DATA_COORD_y] - destination->view_begin.t[NN_DATA_COORD_y] + 1 ;
    destination_sizes[NN_DATA_COORD_z] = destination->view_end.t[NN_DATA_COORD_z] - destination->view_begin.t[NN_DATA_COORD_z] + 1 ;
    destination_sizes[NN_DATA_COORD_p] = destination->view_end.t[NN_DATA_COORD_p] - destination->view_begin.t[NN_DATA_COORD_p] + 1 ;
    destination_sizes[NN_DATA_COORD_q] = destination->view_end.t[NN_DATA_COORD_q] - destination->view_begin.t[NN_DATA_COORD_q] + 1 ;

    if( (source_sizes[NN_DATA_COORD_n] != destination_sizes[NN_DATA_COORD_n]) ||
        (source_sizes[NN_DATA_COORD_x] != destination_sizes[NN_DATA_COORD_x]) ||
        (source_sizes[NN_DATA_COORD_y] != destination_sizes[NN_DATA_COORD_y]) ||
        (source_sizes[NN_DATA_COORD_z] != destination_sizes[NN_DATA_COORD_z]) ||
        (source_sizes[NN_DATA_COORD_p] != destination_sizes[NN_DATA_COORD_p]) ||
        (source_sizes[NN_DATA_COORD_q] != destination_sizes[NN_DATA_COORD_q])
        ) throw std::runtime_error("Error: Can not copy data, dimensions must be equal.");

    // Views sizes need to match
    if (memcmp(source_sizes, destination_sizes, sizeof(unsigned int) * (NN_DATA_COORD_MAX + 1)) != 0 )
    {
        // Cannot copy if lenghts don't match
        return NN_DATA_STATUS_ERROR_INVALID_MEMORY_LAYOUT;
    }

    // Try fast path for buffers with same layout and dimensionality.
    if(source->parent->layout == destination->parent->layout &&
       source->parent->dimension == destination->parent->dimension)
    {
        bool no_view = true;

        // Get number of dimensions to check.
        const uint32_t num_dimensions = source->parent->dimension;

        // Check if there are no views on any used dimensions.
        for(uint32_t dimension = 0; dimension < num_dimensions; ++dimension)
        {
            // Get coordinate for this dimension.
            const auto coordinate = source->parent->layout.ordering.t[dimension];

            // Check buffers view on this coordinate.
            if(source_sizes[coordinate] != source->parent->lengths.t[coordinate] ||
               destination_sizes[coordinate] != destination->parent->lengths.t[coordinate])
            {
                // View detected.
                no_view = false;
                break;
            }
        }

        if(no_view)
        {
            memcpy( destination->parent->data_buffer, source->parent->data_buffer, destination->parent->buffer_size );
            return NN_DATA_STATUS_OK;
        }
    }

    for (q = 0; q < source_sizes[NN_DATA_COORD_q]; q++)
    for (p = 0; p < source_sizes[NN_DATA_COORD_p]; p++)
    for (z = 0; z < source_sizes[NN_DATA_COORD_z]; z++)
    for (y = 0; y < source_sizes[NN_DATA_COORD_y]; y++)
    for (x = 0; x < source_sizes[NN_DATA_COORD_x]; x++)
    for (n = 0; n < source_sizes[NN_DATA_COORD_n]; n++)
    if(copy_delta)
        nn_workload_data_get_delta<T>(destination, n, x, y, z, p, q) = nn_workload_data_get_delta<T>(source, n, x, y, z, p, q);
    else
        nn_workload_data_get<T>(destination, n, x, y, z, p, q) = nn_workload_data_get<T>(source, n, x, y, z, p, q);

    destination->parent->tag = source->parent->tag;

    return NN_DATA_STATUS_OK;
}


NN_DATA_STATUS nn_workload_copy(nn_workload_data_t* destination, const nn_workload_data_t* source, bool copy_delta)
{
    auto type = source->parent->layout.data_type;

    assert(type == destination->parent->layout.data_type);

    // TODO: this seems wrong, refactoring needed :)
    switch(type){
        case NN_DATATYPE_FLOAT:
            if(copy_delta)
                return nn_workload_copy<float, true>(destination, source);
            else
                return nn_workload_copy<float, false >(destination, source);
        case NN_DATATYPE_INT32:
            if(copy_delta)
                return nn_workload_copy<int32_t, true>(destination, source);
            else
                return nn_workload_copy<int32_t, false >(destination, source);
        case NN_DATATYPE_INT16:
            if(copy_delta)
                return nn_workload_copy<int16_t, true>(destination, source);
            else
                return nn_workload_copy<int16_t, false >(destination, source);
        default:
            assert(0);
            return NN_DATA_STATUS_ERROR_DATA_NOT_CONSISTENT;
    }
}

nn_workload_data_layout_t nn::layout_t<nn::layout_nxyzpq_f32>::layout     = { { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_xyznpq_f32>::layout     = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_nzxypq_f32>::layout     = { { NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_pnzxyq_f32>::layout     = { { NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_pxyznq_f32>::layout     = { { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_pxqzyn_f32>::layout     = { { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_q, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_n }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_pzxyqn_f32>::layout     = { { NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_xyzpnq_f32>::layout     = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_xyzpqn_f32>::layout     = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_yxzpqn_f32>::layout     = { { NN_DATA_COORD_y, NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_zxynpq_f32>::layout     = { { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_pzqxyn_f32>::layout     = { { NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_q, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<nn::layout_zxyn_f32>::layout       = nn::layout_t<nn::layout_zxynpq_f32>::layout;
nn_workload_data_layout_t nn::layout_t<nn::layout_nx_f32>::layout         = nn::layout_t<nn::layout_nxyzpq_f32>::layout;
nn_workload_data_layout_t nn::layout_t<nn::layout_zblockxyzn_f32>::layout = nn::layout_t<nn::layout_pxyznq_f32>::layout;
nn_workload_data_layout_t nn::layout_t<nn::layout_oblockixyo_f32>::layout = nn::layout_t<nn::layout_pzxyqn_f32>::layout;
nn_workload_data_layout_t nn::layout_t<nn::layout_oblockioxy_f32>::layout = nn::layout_t<nn::layout_pzqxyn_f32>::layout;
nn_workload_data_layout_t nn::layout_t<nn::layout_nblockzxyn_f32>::layout = nn::layout_t<nn::layout_pzqxyn_f32>::layout;
nn_workload_data_layout_t nn::layout_t<nn::layout_o_f32>::layout          = nn::layout_t<nn::layout_nxyzpq_f32>::layout;
nn_workload_data_layout_t nn::layout_t<nn::layout_oi_f32>::layout         = nn::layout_t<nn::layout_yxzpqn_f32>::layout;
nn_workload_data_layout_t nn::layout_t<nn::layout_oblockio_f32>::layout   = nn::layout_t<nn::layout_pzxyqn_f32>::layout;
nn_workload_data_layout_t nn::layout_t<nn::layout_zxy_f32>::layout        = nn::layout_t<nn::layout_zxynpq_f32>::layout;

nn_workload_data_layout_t nn::layout_t<nn::layout_nxyzpq_i16>::layout     = { { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<nn::layout_pnzxyq_i16>::layout     = { { NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<nn::layout_pxyznq_i16>::layout     = { { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<nn::layout_pzqxyn_i16>::layout     = { { NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_q, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<nn::layout_xyzpqn_i16>::layout     = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<nn::layout_xzynpq_i16>::layout     = { { NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<nn::layout_ypznxq_i16>::layout     = { { NN_DATA_COORD_y, NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<nn::layout_zpxynq_i16>::layout     = { { NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<nn::layout_zxynpq_i16>::layout     = { { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };

nn_workload_data_layout_t nn::layout_t<nn::layout_nxyzpq_i32>::layout     = { { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<nn::layout_pnzxyq_i32>::layout     = { { NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<nn::layout_xnyzpq_i32>::layout     = { { NN_DATA_COORD_x, NN_DATA_COORD_n, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<nn::layout_xyzpqn_i32>::layout     = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<nn::layout_xzynpq_i32>::layout     = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<nn::layout_zpxynq_i32>::layout     = { { NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<nn::layout_zxynpq_i32>::layout     = { { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };
