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
#define NN_ALIGN( a, b )                 ( ( ( ( a ) % ( b ) ) != 0 ) ?  ( ( a ) - ( ( a ) % ( b ) ) + ( b ) ) : ( a ) )

/*
    Creates new nn_workload_data_t according to specification.

    lenghts - size of data structure in each dimension
    layout  - contains information such as tiling in all dimensions, alignment coordinate,
              alignment size and data type. Tiling and alignment values are specified in integer log2(size).
*/
nn_workload_data_t *nn_workload_data_create(const nn_workload_data_coords_t *lenghts,
                                            const nn_workload_data_layout_t *layout) {
    nn_workload_data_t* nn_workload_data;

    assert(lenghts != NULL);
    assert(layout != NULL);

    if (lenghts == NULL || layout == NULL)
    {
        return NULL;
    }

    // allocate the view
    nn_workload_data = new nn_workload_data_t;
    
    if (nn_workload_data == NULL)
    {
        return NULL;
    }

    if (0 != nn_workload_data_placement_create(nn_workload_data, NULL, lenghts, layout))
    {
        delete nn_workload_data;
        return NULL;
    }

   return nn_workload_data;
}

nn_workload_data_core_t::nn_workload_data_core_t(uint32_t data_type_size,
                                                 const nn_workload_data_coords_t lengths,
                                                 const nn_workload_data_layout_t layout,
                                                 void *buffer)
    : data_type_size(data_type_size), lengths(lengths), layout(layout) {
    uint32_t tile_nelements = 1;
    uint32_t tiles_count = 1;
    uint32_t tiles_in_dimension[NN_DIMENSION_COUNT];
    uint32_t total_padding_size = 0; /* additional allocation size due to alignment */

    use_client_buffer = false;

    for (size_t i = 0; i < NN_DIMENSION_COUNT; i++) {
        uint32_t tile_length = 1 << layout.tile_lengths_log2.t[i];

        tile_nelements <<= layout.tile_lengths_log2.t[i];

        tiles_in_dimension[i] = (lengths.t[i] + tile_length - 1) / tile_length;
        tiles_count *= tiles_in_dimension[i];
    }

    tile_size = tile_nelements;

    // calculate stride for each dimension
    for (size_t i = 0; i < NN_DIMENSION_COUNT; i++)
    {
        uint32_t dim = layout.ordering.t[i];
        uint32_t tile_stride = 1;

        tile_idx_mask[dim] = (1 << layout.tile_lengths_log2.t[dim]) - 1;
        tile_idx_shift[dim] = 0;
        if (i>0)
        {
            uint32_t previous_dim = layout.ordering.t[i - 1];
            tile_stride = tile_strides[previous_dim] * tiles_in_dimension[previous_dim];

            tile_idx_shift[dim] = tile_idx_shift[previous_dim] + layout.tile_lengths_log2.t[previous_dim];

        }
        pdep_mask[dim] = tile_idx_mask[dim] << tile_idx_shift[dim];

        if (layout.alignment_log2.t[dim] > 0)
        {
            if (tile_nelements == 1)
            {
                uint32_t j;
                uint32_t ntiles_in_higher_dim = 1;
                // no tiling i.e. one element per tile
                uint32_t unaligned_stride = tile_stride;
                uint32_t align = (1 << layout.alignment_log2.t[dim]) / data_type_size;

                tile_stride = ((tile_stride + align - 1) / align) * align;

                // number of tiles in this and higher dimensions
                for (j = i; j < NN_DIMENSION_COUNT; j++)
                    ntiles_in_higher_dim *= tiles_in_dimension[layout.ordering.t[j]];

                // additional allocation size caused by alignment
                total_padding_size += (tile_stride - unaligned_stride) * ntiles_in_higher_dim;
            }
            else
            {
                // align entire tile
                uint32_t unaligned_size = tile_size;
                uint32_t align = (1 << layout.alignment_log2.t[dim]) / data_type_size;

                tile_size = ((tile_stride + align - 1) / align) * align;
                // additional allocation size caused by alignment
                total_padding_size += (tile_size - unaligned_size) * tiles_count;
            }
        }

        tile_strides[dim] = tile_stride;
    }

    buffer_size = (tiles_count * tile_nelements + total_padding_size) * data_type_size;

    if (buffer != NULL) {
        data_buffer = buffer;
        use_client_buffer = 1;
    } else {
#if defined(__linux__) || defined(__CYGWIN__)
        if (0 !=
            posix_memalign(&data_buffer, BUFFER_ALIGNMENT, NN_ALIGN(buffer_size, BUFFER_SIZE_ALIGNEMENT))) {
            data_buffer = NULL;
        }
#else
        data_buffer =
            (void *)_aligned_malloc(NN_ALIGN(buffer_size, BUFFER_SIZE_ALIGNEMENT), BUFFER_ALIGNMENT);
#endif // defined(__linux__) || defined(__CYGWIN__)

        if (data_buffer == NULL) {
            assert(0);
            throw std::bad_alloc();
        }
    }
}

nn_workload_data_core_t::~nn_workload_data_core_t() {
    if (use_client_buffer == 0) {
#if defined(__linux__) || defined(__CYGWIN__)
        free(data_buffer);
#else
        _aligned_free(data_buffer);
#endif //__linux__
    }
}

/*
    Allocates and fills out nn_workload_data->parent structure.
    Does not allocate nn_workload_data nor nn_workload_data->parent->data_buffer
*/
static NN_DATA_STATUS nn_workload_data_parent_create(nn_workload_data_t *nn_workload_data,
                                                     const nn_workload_data_coords_t *lenghts,
                                                     const nn_workload_data_layout_t *layout,
                                                     void *buffer) {
    uint32_t ordering_coord_sum = 0;
    uint32_t i;

    assert(nn_workload_data != NULL);
    assert(lenghts != NULL);
    assert(layout != NULL);

    if (nn_workload_data == NULL || lenghts == NULL || layout == NULL)
    {
        return NN_DATA_STATUS_ERROR_INVALID_POINTER;
    }

    for (i = 0; i < NN_DIMENSION_COUNT; i++)
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
    nn_workload_data->parent = std::make_shared<nn_workload_data_core_t>(data_type_size, *lenghts, *layout, buffer);

    return NN_DATA_STATUS_OK;
}

/*
    Creates nn_workload_data structure using nn_workload_data_t allocated by a caller.
    Optionally also data buffer may provided by a caller.

    nn_workload_data - pointer to the structure allocated by a caller
    buffer  - pointer to the data buffer. If NULL, it will be allocated by this function.
    lenghts - size of data structure in each dimension.
    layout  - contains information such as tiling in all dimensions, alignment coordinate,
              alignment size and data type. Tiling and alignment values are specified in integer log2(size).
*/
NN_DATA_STATUS nn_workload_data_placement_create(nn_workload_data_t *nn_workload_data,
                                                 void *buffer,
                                                 const nn_workload_data_coords_t *lenghts,
                                                 const nn_workload_data_layout_t *layout) {
    NN_DATA_STATUS status;
    uint32_t i;

    assert(nn_workload_data != NULL);
    assert(lenghts != NULL);
    assert(layout != NULL);

    if (nn_workload_data == NULL || lenghts == NULL || layout == NULL)
    {
        return NN_DATA_STATUS_ERROR_INVALID_POINTER;
    }

    memset(&nn_workload_data->view_begin, 0, sizeof(nn_workload_data->view_begin));

    for (i = 0; i < NN_DIMENSION_COUNT; i++)
        nn_workload_data->view_end.t[i] = lenghts->t[i] - 1;

    status = nn_workload_data_parent_create(nn_workload_data, lenghts, layout, buffer);
    if (status != NN_DATA_STATUS_OK)
    {
        return status;
    }

    return NN_DATA_STATUS_OK;
}

/*
    Creates view that references signal from another nn_workload_data_t. Uses nn_workload_data_t allocated by a caller.

    Resulting view has the same layout as original.
    If view cannot be created (outside image, position not granular to tile) - 0 is returned.
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

    for (i = 0; i < NN_DIMENSION_COUNT; i++)
    {
        // validate that coords_begin and coords_end are granular to tile 
        if (coords_begin->t[i] & ((1 << nn_source->parent->layout.tile_lengths_log2.t[i]) - 1))
            return NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;
        
        if ((coords_end->t[i] + 1) & ((1 << nn_source->parent->layout.tile_lengths_log2.t[i]) - 1))
            return NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;

        // check if the view is within the image
        if (coords_begin->t[i] > nn_source->view_end.t[i])
            return NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;
        
        if (coords_end->t[i] > nn_source->view_end.t[i])
            return NN_DATA_STATUS_ERROR_INVALID_PARAMETERS;
    }

    nn_workload_data->parent = nn_source->parent;

    for (i = 0; i < NN_DIMENSION_COUNT; i++)
    {
        nn_workload_data->view_begin.t[i] = nn_source->view_begin.t[i] + coords_begin->t[i];
        nn_workload_data->view_end.t[i] = nn_source->view_begin.t[i] + coords_end->t[i];
    }

    return NN_DATA_STATUS_OK;
}


/*
    Creates view that references signal from another nn_workload_data_t (i.e. another view).

    Resulting view has the same layout as original.
    If view cannot be created (outside image, position not granular to tile) - 0 is returned.
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
    Releases memory pointed by nn_workload_data.
    If nn_workload_data is the last existing view that is using nn_workload_data->parent, then nn_workload_data->parent is also realeased.
    
    If nn_workload_data is the last existing view that is using nn_workload_data->parent, 
    and data buffer (nn_workload_data->parent->data_buffer) has not been provided by a user
    then the data buffer is also released.
*/
NN_DATA_STATUS nn_workload_data_delete(nn_workload_data_t *nn_workload_data)
{
    assert(nn_workload_data != NULL);

    if (nn_workload_data == NULL)
    {
        return NN_DATA_STATUS_ERROR_INVALID_POINTER;
    }

    delete nn_workload_data;
    return NN_DATA_STATUS_OK;
}

/*
    Calculates index that is used to retrieve a value from a data buffer.
*/
uint32_t calculate_idx(const nn_workload_data_t* data, uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q)
{
    nn_workload_data_coords_t coordinates = { n, x, y, z, p, q };
    uint32_t index = 0, tile_index = 0;
    uint32_t i;

    assert(n <= (data->view_end.t[NN_DATA_COORD_n] - data->view_begin.t[NN_DATA_COORD_n]));
    assert(x <= (data->view_end.t[NN_DATA_COORD_x] - data->view_begin.t[NN_DATA_COORD_x]));
    assert(y <= (data->view_end.t[NN_DATA_COORD_y] - data->view_begin.t[NN_DATA_COORD_y]));
    assert(z <= (data->view_end.t[NN_DATA_COORD_z] - data->view_begin.t[NN_DATA_COORD_z]));
    assert(p <= (data->view_end.t[NN_DATA_COORD_p] - data->view_begin.t[NN_DATA_COORD_p]));
    assert(q <= (data->view_end.t[NN_DATA_COORD_q] - data->view_begin.t[NN_DATA_COORD_q]));

    for (i = 0; i < NN_DIMENSION_COUNT; i++)
    {
        uint32_t tile_coordinate;
#ifdef USE_PDEP
        // PDEP version (index within a tile) - not supported before Haswell
        index |= _pdep_u32(lenghts->t[i], data->m_pdep_mask[i]);
#else
        //index within a tile
        index |= ((coordinates.t[i] + data->view_begin.t[i])& data->parent->tile_idx_mask[i]) << data->parent->tile_idx_shift[i];
#endif

        // index of a tile
        tile_coordinate = (coordinates.t[i] + data->view_begin.t[i]) >> data->parent->layout.tile_lengths_log2.t[i];
        tile_index += tile_coordinate * data->parent->tile_strides[i];
    }
    index += tile_index * data->parent->tile_size;

    return index;
}

/*
    Copy data from source to destination.
    Data ordering and/or tiling may differ between source and destination,
    but lenghts in corresponding dimensions must be equal.

    Assumption is that source and destination have different layouts,
    so we can't just use memcpy()
*/
NN_DATA_STATUS nn_workload_data_copy(nn_workload_data_t* destination, const nn_workload_data_t* source)
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
    source_sizes[NN_DATA_COORD_x] = source->view_end.t[NN_DATA_COORD_x] - source->view_begin.t[NN_DATA_COORD_x] +1 ; 
    source_sizes[NN_DATA_COORD_y] = source->view_end.t[NN_DATA_COORD_y] - source->view_begin.t[NN_DATA_COORD_y] +1 ; 
    source_sizes[NN_DATA_COORD_z] = source->view_end.t[NN_DATA_COORD_z] - source->view_begin.t[NN_DATA_COORD_z] +1 ; 
    source_sizes[NN_DATA_COORD_p] = source->view_end.t[NN_DATA_COORD_p] - source->view_begin.t[NN_DATA_COORD_p] +1 ; 
    source_sizes[NN_DATA_COORD_q] = source->view_end.t[NN_DATA_COORD_q] - source->view_begin.t[NN_DATA_COORD_q] +1 ; 

    destination_sizes[NN_DATA_COORD_n] = destination->view_end.t[NN_DATA_COORD_n] - destination->view_begin.t[NN_DATA_COORD_n] + 1;  
    destination_sizes[NN_DATA_COORD_x] = destination->view_end.t[NN_DATA_COORD_x] - destination->view_begin.t[NN_DATA_COORD_x] +1 ; 
    destination_sizes[NN_DATA_COORD_y] = destination->view_end.t[NN_DATA_COORD_y] - destination->view_begin.t[NN_DATA_COORD_y] +1 ; 
    destination_sizes[NN_DATA_COORD_z] = destination->view_end.t[NN_DATA_COORD_z] - destination->view_begin.t[NN_DATA_COORD_z] +1 ; 
    destination_sizes[NN_DATA_COORD_p] = destination->view_end.t[NN_DATA_COORD_p] - destination->view_begin.t[NN_DATA_COORD_p] +1 ; 
    destination_sizes[NN_DATA_COORD_q] = destination->view_end.t[NN_DATA_COORD_q] - destination->view_begin.t[NN_DATA_COORD_q] +1 ; 

    // Views sizes need to match
    if (memcmp(source_sizes, destination_sizes, sizeof(unsigned int) * (NN_DATA_COORD_MAX + 1)) != 0 )
    {
        // Cannot copy if lenghts don't match
        return NN_DATA_STATUS_ERROR_INVALID_MEMORY_LAYOUT;
    }

    if( !memcmp( source_sizes, source->parent->lengths.t, sizeof( source_sizes ) ) &&
        !memcmp( destination_sizes, destination->parent->lengths.t, sizeof( destination_sizes ) ) &&
        !memcmp( &source->parent->layout, &destination->parent->layout, sizeof( source->parent->layout ) )
        )
    {
        memcpy( destination->parent->data_buffer, source->parent->data_buffer, destination->parent->buffer_size );
        return NN_DATA_STATUS_OK;
    }

    if ((source->parent->layout.data_type == NN_DATATYPE_FLOAT) &&
        (destination->parent->layout.data_type == NN_DATATYPE_FLOAT))
    {
        for (q = 0; q < source_sizes[NN_DATA_COORD_q]; q++)
        for (p = 0; p < source_sizes[NN_DATA_COORD_p]; p++)
        for (z = 0; z < source_sizes[NN_DATA_COORD_z]; z++)
        for (y = 0; y < source_sizes[NN_DATA_COORD_y]; y++)
        for (x = 0; x < source_sizes[NN_DATA_COORD_x]; x++)
        for (n = 0; n < source_sizes[NN_DATA_COORD_n]; n++)
            nn_workload_data_get<float>(destination, n, x, y, z, p, q) = nn_workload_data_get<float>(source, n, x, y, z, p, q);
    }
    else if ((source->parent->layout.data_type == NN_DATATYPE_INT16) &&
             (destination->parent->layout.data_type == NN_DATATYPE_INT16))
    {
        for (q = 0; q < source_sizes[NN_DATA_COORD_q]; q++)
        for (p = 0; p < source_sizes[NN_DATA_COORD_p]; p++)
        for (z = 0; z < source_sizes[NN_DATA_COORD_z]; z++)
        for (y = 0; y < source_sizes[NN_DATA_COORD_y]; y++)
        for (x = 0; x < source_sizes[NN_DATA_COORD_x]; x++)
        for (n = 0; n < source_sizes[NN_DATA_COORD_n]; n++)
            nn_workload_data_get<int16_t>(destination, n, x, y, z, p, q) = nn_workload_data_get<int16_t>(source, n, x, y, z, p, q);
    }
    else if ((source->parent->layout.data_type == NN_DATATYPE_INT32) &&
        (destination->parent->layout.data_type == NN_DATATYPE_INT32))
    {
        for (q = 0; q < source_sizes[NN_DATA_COORD_q]; q++)
        for (p = 0; p < source_sizes[NN_DATA_COORD_p]; p++)
        for (z = 0; z < source_sizes[NN_DATA_COORD_z]; z++)
        for (y = 0; y < source_sizes[NN_DATA_COORD_y]; y++)
        for (x = 0; x < source_sizes[NN_DATA_COORD_x]; x++)
        for (n = 0; n < source_sizes[NN_DATA_COORD_n]; n++)
            nn_workload_data_get<int32_t>(destination, n, x, y, z, p, q) = nn_workload_data_get<int32_t>(source, n, x, y, z, p, q);
    }
    else
    {
        assert(0);
        return NN_DATA_STATUS_ERROR_DATA_NOT_CONSISTENT;
    }

    return NN_DATA_STATUS_OK;
}

/*
    Create nn_workload_data structure using existing data buffer which size and organization
    is described by lenghts and layaout parameters.
    Lifetime of the data buffer must be longer than lifetime of nn_workload_data_t.
*/
nn_workload_data_t *nn_workload_data_create_from_buffer(void *buffer,
                                                        const nn_workload_data_coords_t *lenghts,
                                                        const nn_workload_data_layout_t *layout) {
    nn_workload_data_t* nn_workload_data;

    assert(buffer != NULL);
    assert(lenghts != NULL);
    assert(layout != NULL);

    if (buffer == NULL || lenghts == NULL || layout == NULL)
    {
        return NULL;
    }

    // allocate the view
    nn_workload_data = new nn_workload_data_t;

    if (nn_workload_data == NULL)
    {
        return NULL;
    }

    if (NN_DATA_STATUS_OK != nn_workload_data_placement_create(nn_workload_data, buffer, lenghts, layout))
    {
        delete nn_workload_data;
        return NULL;
    }

    return nn_workload_data;
}
