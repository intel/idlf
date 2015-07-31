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

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "nn_workload_data.h"

#define BUFFER_ALIGNMENT 4096
#define BUFFER_SIZE_ALIGNEMENT 4096
/* For functions parameters validation */
#define NN_COORD_SUM (NN_DATA_COORD_n + NN_DATA_COORD_x + NN_DATA_COORD_y + NN_DATA_COORD_z + NN_DATA_COORD_p + NN_DATA_COORD_q)


template <typename T1, typename T2> T1 nn_align(T1 a, T2 b){
    return (a + b - 1) / b * b;
}

nn_workload_data_core_t::nn_workload_data_core_t(uint32_t data_type_size,
                                                 const nn_workload_data_coords_t lengths,
                                                 const nn_workload_data_layout_t layout,
                                                 void *buffer,
                                                 bool empty_data,
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
    } else if (empty_data){
        data_buffer = nullptr;
        use_client_buffer = 1;
    } else {
#if defined(__linux__) || defined(__CYGWIN__)
        if (0 !=
            posix_memalign(&data_buffer, BUFFER_ALIGNMENT, nn_align(buffer_size, BUFFER_SIZE_ALIGNEMENT))) {
            data_buffer = nullptr;
        }
#else
        data_buffer =
            (void *)_aligned_malloc(nn_align(buffer_size, BUFFER_SIZE_ALIGNEMENT), BUFFER_ALIGNMENT);
#endif // defined(__linux__) || defined(__CYGWIN__)

        if (data_buffer == nullptr) {
            assert(0);
            throw std::bad_alloc();
        }
    }

    if(allocate_delta)
    {
#if defined(__linux__) || defined(__CYGWIN__)
        if (0 !=
            posix_memalign(&delta_buffer, BUFFER_ALIGNMENT, nn_align(buffer_size, BUFFER_SIZE_ALIGNEMENT))) {
            delta_buffer = nullptr;
        }
#else
        delta_buffer =
            (void *)_aligned_malloc(nn_align(buffer_size, BUFFER_SIZE_ALIGNEMENT), BUFFER_ALIGNMENT);
#endif // defined(__linux__) || defined(__CYGWIN__)

        if(delta_buffer == nullptr) {
            assert(0);
            if(use_client_buffer == 0) {
#if defined(__linux__) || defined(__CYGWIN__)
                free(data_buffer);
#else
                _aligned_free(data_buffer);
#endif //__linux__
            }
            throw std::bad_alloc();
        }
    }
}

nn_workload_data_core_t::~nn_workload_data_core_t() {
    if (use_client_buffer == 0) {

#if defined(__linux__) || defined(__CYGWIN__)
        if(delta_buffer != data_buffer && delta_buffer != nullptr)
            free(delta_buffer);
        free(data_buffer);

#else
        if(delta_buffer != data_buffer && delta_buffer != nullptr)
            _aligned_free(delta_buffer);
        _aligned_free(data_buffer);
#endif //__linux__
    }
}

void nn_workload_data_core_t::allocate_delta_buffer(){
    if(delta_buffer == nullptr)
    {
#if defined(__linux__) || defined(__CYGWIN__)
        if (0 !=
            posix_memalign(&delta_buffer, BUFFER_ALIGNMENT, nn_align(buffer_size, BUFFER_SIZE_ALIGNEMENT))) {
            delta_buffer = nullptr;
        }
#else
        delta_buffer =
            (void *)_aligned_malloc(nn_align(buffer_size, BUFFER_SIZE_ALIGNEMENT), BUFFER_ALIGNMENT);
#endif // defined(__linux__) || defined(__CYGWIN__)

        if(delta_buffer == nullptr) {
            throw std::bad_alloc();
        }
    }
    else
    {
        // We should never get here
        assert(0);
    }

    memset(delta_buffer, 0, nn_align(buffer_size, BUFFER_SIZE_ALIGNEMENT));
}

/*
    Allocates and fills out nn_workload_data->parent structure.
    Does not allocate nn_workload_data nor nn_workload_data->parent->data_buffer
*/
static NN_DATA_STATUS nn_workload_data_parent_create(nn_workload_data_t *nn_workload_data,
                                                     const nn_workload_data_coords_t *lenghts,
                                                     const nn_workload_data_layout_t *layout,
                                                     void *buffer,
                                                     bool  empty_data,
                                                     bool  allocate_delta) {
    uint32_t ordering_coord_sum = 0;
    uint32_t i;

    assert(nn_workload_data != NULL);
    assert(lenghts != NULL);
    assert(layout != NULL);

    if (nn_workload_data == NULL || lenghts == NULL || layout == NULL)
    {
        return NN_DATA_STATUS_ERROR_INVALID_POINTER;
    }

    assert(lenghts->dimension==layout->ordering.dimension);

    for (i = 0; i < lenghts->dimension; i++)
    {
        if ((lenghts->t[i] == 0) || (layout->ordering.t[i] > NN_DATA_COORD_MAX))
        {
            return NN_DATA_STATUS_ERROR_INVALID_MEMORY_LAYOUT;
        }

        ordering_coord_sum += layout->ordering.t[i];
    }

    /* validate correct ordering input */
    if (ordering_coord_sum != NN_COORD_SUM)
    {
        return NN_DATA_STATUS_ERROR_INVALID_MEMORY_LAYOUT;
    }

    uint32_t data_type_size;
    if (layout->data_type == NN_DATATYPE_FLOAT) {
        data_type_size = sizeof(float);
    } else if (layout->data_type == NN_DATATYPE_INT16) {
        data_type_size = sizeof(short);
    } else if (layout->data_type == NN_DATATYPE_INT32) {
        data_type_size = sizeof(int);
    } else {
        return NN_DATA_STATUS_ERROR_INVALID_MEMORY_LAYOUT;
    }

    // nn_workload_data is just a view, parent is the actual data
    nn_workload_data->parent = std::make_shared<nn_workload_data_core_t>(data_type_size, *lenghts, *layout, buffer, empty_data, allocate_delta);

    return NN_DATA_STATUS_OK;
}

/*
    Creates nn_workload_data structure using workload_data allocated by a caller.
    Optionally also data buffer may provided by a caller.

    nn_workload_data - pointer to the structure allocated by a caller
    empty_data       - if set, there will be no data allocated internally nor caller data will be used,
                       this option can be used when user require only data structure definition, without
                       actual data inside
    buffer           - pointer to the data buffer. If NULL, it will be allocated by this function.
    lenghts          - size of data structure in each dimension.
    layout           - contains information such as ordering and data type.
*/
NN_DATA_STATUS nn_workload_data_placement_create(nn_workload_data_t *nn_workload_data,
                                                 void *buffer,
                                                 const nn_workload_data_coords_t *lenghts,
                                                 const nn_workload_data_layout_t *layout,
                                                 bool empty_data,
                                                 bool allocate_delta) {
    NN_DATA_STATUS status;
    uint32_t i;

    assert(nn_workload_data != NULL);
    assert(lenghts != NULL);
    assert(layout != NULL);

    if (nn_workload_data == NULL || lenghts == NULL || layout == NULL)
    {
        return NN_DATA_STATUS_ERROR_INVALID_POINTER;
    }

    assert(lenghts->dimension == layout->ordering.dimension);

    for (i = 0; i < lenghts->dimension; i++)
        nn_workload_data->view_end.t[i] = lenghts->t[i] - 1;

    status = nn_workload_data_parent_create(nn_workload_data, lenghts, layout, buffer, empty_data, allocate_delta);
    if (status != NN_DATA_STATUS_OK)
    {
        return status;
    }

    return NN_DATA_STATUS_OK;
}

/*
    Creates view that references signal from another workload_data. Uses workload_data allocated by a caller.

    Resulting view has the same layout as original.
    If view cannot be created (outside image) - 0 is returned.
*/
NN_DATA_STATUS nn_workload_data_placement_create_view(nn_workload_data_t *nn_workload_data,
                                                      const nn_workload_data_t *nn_source,
                                                      const nn_workload_data_coords_t *coords_begin,
                                                      const nn_workload_data_coords_t *coords_end) {
    uint32_t i;

    assert(nn_workload_data != NULL);
    assert(nn_source != NULL);
    assert(coords_begin != NULL);
    assert(coords_end != NULL);

    if (nn_workload_data == NULL || nn_source == NULL || coords_begin == NULL || coords_end == NULL)
    {
        return NN_DATA_STATUS_ERROR_INVALID_POINTER;
    }

    for (i = 0; i < nn_source->parent->dimension; i++)
    {

        // check if the view is within the image
        if (coords_begin->t[i] > nn_source->view_end.t[i])
            return NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;

        if (coords_end->t[i] > nn_source->view_end.t[i])
            return NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;
    }

    nn_workload_data->parent = nn_source->parent;

    for (i = 0; i < nn_source->parent->dimension; i++)
    {
        nn_workload_data->view_begin.t[i] = nn_source->view_begin.t[i] + coords_begin->t[i];
        nn_workload_data->view_end.t[i] = nn_source->view_begin.t[i] + coords_end->t[i];
    }

    return NN_DATA_STATUS_OK;
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

    if (0 != nn_workload_data_placement_create_view(
        nn_workload_data,
        source,
        coords_begin,
        coords_end
        ))
    {
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
    nn_workload_data_coords_t coordinates = { n, x, y, z, p, q };
    uint32_t index = 0;
    uint32_t i;

    assert(n <= (data->view_end.t[NN_DATA_COORD_n] - data->view_begin.t[NN_DATA_COORD_n]));
    assert(x <= (data->view_end.t[NN_DATA_COORD_x] - data->view_begin.t[NN_DATA_COORD_x]));
    assert(y <= (data->view_end.t[NN_DATA_COORD_y] - data->view_begin.t[NN_DATA_COORD_y]));
    assert(z <= (data->view_end.t[NN_DATA_COORD_z] - data->view_begin.t[NN_DATA_COORD_z]));
    assert(p <= (data->view_end.t[NN_DATA_COORD_p] - data->view_begin.t[NN_DATA_COORD_p]));
    assert(q <= (data->view_end.t[NN_DATA_COORD_q] - data->view_begin.t[NN_DATA_COORD_q]));

    for (i = 0; i < data->parent->dimension; i++)
    {
        uint32_t coordinate;

        coordinate = (coordinates.t[i] + data->view_begin.t[i]);
        index += coordinate * data->parent->strides[i];
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

nn_workload_data_layout_t nn::layout_t<float>::nxyzpq = { { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<float>::xyznpq = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<float>::nzxypq = { { NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<float>::pnzxyq = { { NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<float>::pxyznq = { { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<float>::pxqzyn = { { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_q, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_n }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<float>::pzxyqn = { { NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<float>::xyzpnq = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<float>::xyzpqn = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<float>::yxzpqn = { { NN_DATA_COORD_y, NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_FLOAT };
nn_workload_data_layout_t nn::layout_t<float>::zxynpq = { { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_FLOAT };

nn_workload_data_layout_t nn::layout_t<int16_t>::nxyzpq = { { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<int16_t>::pnzxyq = { { NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<int16_t>::pxyznq = { { NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<int16_t>::pzqxyn = { { NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_q, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<int16_t>::xyzpqn = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<int16_t>::xzynpq = { { NN_DATA_COORD_x, NN_DATA_COORD_z, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<int16_t>::ypznxq = { { NN_DATA_COORD_y, NN_DATA_COORD_p, NN_DATA_COORD_z, NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<int16_t>::zpxynq = { { NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };
nn_workload_data_layout_t nn::layout_t<int16_t>::zxynpq = { { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT16 };

nn_workload_data_layout_t nn::layout_t<int32_t>::nxyzpq = { { NN_DATA_COORD_n, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<int32_t>::pnzxyq = { { NN_DATA_COORD_p, NN_DATA_COORD_n, NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<int32_t>::xnyzpq = { { NN_DATA_COORD_x, NN_DATA_COORD_n, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<int32_t>::xyzpqn = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<int32_t>::xzynpq = { { NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_q, NN_DATA_COORD_n }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<int32_t>::zpxynq = { { NN_DATA_COORD_z, NN_DATA_COORD_p, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };
nn_workload_data_layout_t nn::layout_t<int32_t>::zxynpq = { { NN_DATA_COORD_z, NN_DATA_COORD_x, NN_DATA_COORD_y, NN_DATA_COORD_n, NN_DATA_COORD_p, NN_DATA_COORD_q }, NN_DATATYPE_INT32 };