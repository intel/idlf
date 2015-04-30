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

#pragma once

#include <string.h>
#include <assert.h>
#include <memory>
#include <cstdint>
#include <new>

typedef enum {
    NN_DATA_STATUS_OK = 0,

    /* errors are negative numbers */
    NN_DATA_STATUS_ERROR_OUT_OF_MEMORY = 0x80000000,
    NN_DATA_STATUS_ERROR_INVALID_POINTER,
    NN_DATA_STATUS_ERROR_INVALID_MEMORY_LAYOUT,
    NN_DATA_STATUS_ERROR_DATA_NOT_CONSISTENT,
    NN_DATA_STATUS_ERROR_INVALID_PARAMETERS,
    NN_DATA_STATUS_ERROR_OTHER,
    NN_DATA_STATUS_LAST = NN_DATA_STATUS_ERROR_OTHER
} NN_DATA_STATUS;


/* Supported data types */
typedef enum
{
    NN_DATATYPE_FLOAT,
    NN_DATATYPE_INT16,
    NN_DATATYPE_INT32
} nn_workload_data_type_t;

/* helper to get nn_workload_data_type_t from type */
template <typename T> struct type_to_datatype;

template <> struct type_to_datatype<float> : std::integral_constant<nn_workload_data_type_t, NN_DATATYPE_FLOAT> {};
template <> struct type_to_datatype<int16_t> : std::integral_constant<nn_workload_data_type_t, NN_DATATYPE_INT16> {};
template <> struct type_to_datatype<int32_t> : std::integral_constant<nn_workload_data_type_t, NN_DATATYPE_INT32> {};

typedef enum
{
    NN_DATA_COORD_n = 0, /* count of instances in a batch */
    NN_DATA_COORD_x = 1, /* width  of input/output or filter */
    NN_DATA_COORD_y = 2, /* height of input/output or filter */
    NN_DATA_COORD_z = 3, /* depth  of input/output or filter */
    NN_DATA_COORD_p = 4, /* width of array of independent width*height*depth filters for local connectivity layers */
    NN_DATA_COORD_q = 5, /* height of array of independent width*height*depth filters for local connectivity layers */
    NN_DATA_COORD_MAX = NN_DATA_COORD_q
} nn_workload_data_coord_index_t;

#define NN_DIMENSION_COUNT (NN_DATA_COORD_MAX + 1)

/*
  Convenient structure for passing data coordinates, data and tile lenghts, alignment coordinates etc.
*/
typedef struct nn_workload_data_coords_s {
    uint32_t t[NN_DIMENSION_COUNT];
    nn_workload_data_coords_s() {
        for (auto &element : t)
            element = 0;
    };

    nn_workload_data_coords_s(uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) {
        t[0] = n;
        t[1] = x;
        t[2] = y;
        t[3] = z;
        t[4] = p;
        t[5] = q;
    };

} nn_workload_data_coords_t;

/*
    Data structure layout.

    Contains information such as tiling in all dimensions, 
    alignment coordinates and data type. 
    Tiling and alignment values are specified in integer log2(size).
*/
typedef struct {
    nn_workload_data_coords_t tile_lengths_log2; /* Tile lengths specified in integer log2(length).
                                           Set all coordinates to 0 if tiling is not used */
    nn_workload_data_coords_t alignment_log2;    /* Alignment specified in integer log2(alignment) */
    nn_workload_data_coords_t ordering;          /* List of dimensions in increasing order of strides */
    nn_workload_data_type_t data_type;           /* Data type specified by nn_workload_data_type_t */
} nn_workload_data_layout_t;

/* Data structure that is referenced by at least one view */
struct nn_workload_data_core_t {

    nn_workload_data_core_t(uint32_t data_type_size,
                            nn_workload_data_coords_t lenghts,
                            nn_workload_data_layout_t layout,
                            void *buffer);

    ~nn_workload_data_core_t();

    nn_workload_data_coords_t lengths;  /* Data structure size in each dimension */
    nn_workload_data_layout_t layout;   /* Data structure layout */
    void *data_buffer;
    uint32_t buffer_size;      /* Size of allocated buffer */
    uint32_t data_type_size;   /* Size of one data item */

    uint32_t tile_strides[NN_DIMENSION_COUNT]; /* Determines how the tiles are ordered in the buffer */
    uint32_t tile_size;                        /* Size of one tile in the buffer  */

    uint32_t tile_idx_mask[NN_DIMENSION_COUNT];  /* Bit mask used for calculating data placement within a tile */
    uint32_t tile_idx_shift[NN_DIMENSION_COUNT]; /* Bit shift used for calculating data placement within a tile */
    uint32_t pdep_mask[NN_DIMENSION_COUNT];      /* Mask used as input for PDEP instruction for calculating data placement within a tile */

    bool use_client_buffer;
};


/* 
    nn_workload_data_t structure. 
    It consists of a view and underlying data structure.
    A view may give access to the entire data structure or limit the visibility
    to just a part of the data.
*/
struct nn_workload_data_t {
    nn_workload_data_coords_t view_begin; /* first point of view [min corner of hyper parallelogram] */
    nn_workload_data_coords_t view_end;   /* last point of view [max corner of hyper parallelogram] */
    std::shared_ptr<nn_workload_data_core_t> parent; /* points to the data visible through this nn_workload_data */
};

/*
    Creates new nn_workload_data_t according to specification.

    lenghts - size of data structure in each dimension
    layout  - contains information such as tiling in all dimensions, alignment coordinate, 
              alignment size and data type. Tiling and alignment values are specified in integer log2(size).
*/
nn_workload_data_t* nn_workload_data_create(
    const nn_workload_data_coords_t* lenghts,
    const nn_workload_data_layout_t* layout
    );

/*
    Creates nn_workload_data structure using nn_workload_data_t allocated by a caller.
    Optionally also data buffer may provided by a caller.

    data    - pointer to the structure allocated by a caller
    buffer  - pointer to the data buffer. If NULL, it will be allocated by this function.
    lenghts - size of data structure in each dimension.
    layout  - contains information such as tiling in all dimensions, alignment coordinate,
    alignment size and data type. Tiling and alignment values are specified in integer log2(size).
*/
NN_DATA_STATUS nn_workload_data_placement_create(
    nn_workload_data_t* data,
    void* buffer,
    const nn_workload_data_coords_t* lenghts,
    const nn_workload_data_layout_t* layout
    );

/* 
    Creates view that references signal from another nn_workload_data_t.
    
    Resulting view has the same layout as original.
    If view cannot be created (outside image, position not granular to tile) - 0 is returned.
*/
nn_workload_data_t *nn_workload_data_create_view(
    const nn_workload_data_t* source,
    const nn_workload_data_coords_t* coords_begin,
    const nn_workload_data_coords_t* coords_end
    );

/*
    Creates view that references signal from another nn_workload_data_t. Uses nn_workload_data_t allocated by a caller.

    Resulting view has the same layout as original.
    If view cannot be created (outside image, position not granular to tile) - 0 is returned.
*/
NN_DATA_STATUS nn_workload_data_placement_create_view(
    nn_workload_data_t* data,
    const nn_workload_data_t* source,
    const nn_workload_data_coords_t* coords_begin,
    const nn_workload_data_coords_t* coords_end
    );

/*
    Deletes nn_workload_data_t view and decrements data reference count.
    If the reference count is 0, the underlying data is also deleted.
*/
NN_DATA_STATUS nn_workload_data_delete(
    nn_workload_data_t *data
    );

/*
    Returns index of element in the data buffer.

    n  count of instances in a batch
    x  width  of input/output or filter
    y  height of input/output or filter
    z  depth  of input/output or filter
    p  width of array of independent width*height*depth filters for local connectivity layers 
    q  height of array of independent width*height*depth filters for local connectivity layers
*/
uint32_t calculate_idx(const nn_workload_data_t* data, uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q);

/*
    Returns reference to data at position specified by arguments.
    If there's a type mismatch or out of bound access is required run-time assert() will fail.

    n  count of instances in a batch
    x  width  of input/output or filter
    y  height of input/output or filter
    z  depth  of input/output or filter
    p  width of array of independent width*height*depth filters for local connectivity layers
    q  height of array of independent width*height*depth filters for local connectivity layers
*/
template <typename T>
static inline T &nn_workload_data_get(
    nn_workload_data_t *data, uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) {
    T* data_buffer = reinterpret_cast<T *>(data->parent->data_buffer);

    assert(data->parent->layout.data_type == type_to_datatype<T>::value);
    assert(data->parent->data_type_size == sizeof(T));

    return data_buffer[calculate_idx(data, n, x, y, z, p, q)];
};

/*
    Returns const reference to data at position specified by arguments.
    If there's a type mismatch or out of bound access is required run-time assert() will fail.

    n  count of instances in a batch
    x  width  of input/output or filter
    y  height of input/output or filter
    z  depth  of input/output or filter
    p  width of array of independent width*height*depth filters for local connectivity layers
    q  height of array of independent width*height*depth filters for local connectivity layers
*/
template <typename T>
static inline const T &nn_workload_data_get(
    const nn_workload_data_t *data, uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) {
    const T *data_buffer = reinterpret_cast<const T *>(data->parent->data_buffer);

    assert(data->parent->layout.data_type == type_to_datatype<T>::value);
    assert(data->parent->data_type_size == sizeof(T));

    return data_buffer[calculate_idx(data, n, x, y, z, p, q)];
};

/*
    Create nn_workload_data structure using existing data buffer which size and organization
    is described by lenghts and layout parameters.
    Lifetime of the data buffer must be longer than lifetime of nn_workload_data_t.
*/
nn_workload_data_t* nn_workload_data_create_from_buffer(
    void* buffer,
    nn_workload_data_coords_t* lenghts,
    nn_workload_data_layout_t* layout
    );

/*
    Copy data from source to destination.
    Data ordering and/or tiling may differ between source and destination,
    but lenghts in corresponding dimensions must be equal. TBD: less confusing description probably needed
*/
NN_DATA_STATUS nn_workload_data_copy(
    nn_workload_data_t* destination, const nn_workload_data_t* source
    );

namespace nn {
    template<typename T> class nn_workload_data_t;
    template<typename T> class nn_workload_data_item_t{
        ::nn_workload_data_t &nn_workload_data;
        nn_workload_data_coords_t nn_coords;
 
        nn_workload_data_item_t(::nn_workload_data_t& nn_workload_data, nn_workload_data_coords_t nn_coords) : nn_workload_data(nn_workload_data), nn_coords(nn_coords){};
    public:
        float operator() (){
            return nn_workload_data_get<T>(&nn_workload_data,
                                           nn_coords.t[NN_DATA_COORD_n],
                                           nn_coords.t[NN_DATA_COORD_x],
                                           nn_coords.t[NN_DATA_COORD_y],
                                           nn_coords.t[NN_DATA_COORD_z],
                                           nn_coords.t[NN_DATA_COORD_p],
                                           nn_coords.t[NN_DATA_COORD_q]);
        }
        nn_workload_data_item_t<T> &operator=(T v) {
            nn_workload_data_get<T>(&nn_workload_data,
                                    nn_coords.t[NN_DATA_COORD_n],
                                    nn_coords.t[NN_DATA_COORD_x],
                                    nn_coords.t[NN_DATA_COORD_y],
                                    nn_coords.t[NN_DATA_COORD_z],
                                    nn_coords.t[NN_DATA_COORD_p],
                                    nn_coords.t[NN_DATA_COORD_q]) = v;
            return *this;
        }
        nn_workload_data_item_t<T> &operator=(nn_workload_data_item_t<T> &source) {
            return (*this = source.operator T());
        }
        operator T() { return operator()(); }
        friend class nn_workload_data_t<T>;
    };

    template<typename T> class nn_workload_data_t : public ::nn_workload_data_t{

        static_assert(std::is_same<T, float>::value || std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value, "type not supported");

        public:
          nn_workload_data_t(const nn_workload_data_coords_t &nn_coords, const nn_workload_data_layout_t &layout) {
                if (NN_DATA_STATUS_OK != nn_workload_data_placement_create(this, nullptr, &nn_coords, &layout))
                    throw std::bad_alloc();
            };

            // Allocate buffer adding padding and set the view to point inside the paddings
            nn_workload_data_t(nn_workload_data_coords_t nn_coords,
                               const nn_workload_data_layout_t &layout,
                               uint32_t padding_left,
                               uint32_t padding_right,
                               uint32_t padding_top,
                               uint32_t padding_bottom) {
                nn_coords.t[NN_DATA_COORD_x] += padding_left + padding_right;
                nn_coords.t[NN_DATA_COORD_y] += padding_top + padding_bottom;
                if (NN_DATA_STATUS_OK != nn_workload_data_placement_create(this, nullptr, &nn_coords, &layout))
                    throw std::bad_alloc();

                memset(this->parent->data_buffer, 0, this->parent->buffer_size);

                this->view_begin.t[NN_DATA_COORD_x] += padding_left;
                this->view_end.t[NN_DATA_COORD_x] -= padding_right;
                this->view_begin.t[NN_DATA_COORD_y] += padding_top;
                this->view_end.t[NN_DATA_COORD_y] -= padding_bottom;
            };

            // Create using existing data buffer
            nn_workload_data_t(void *buffer,
                               const nn_workload_data_coords_t &nn_coords,
                               const nn_workload_data_layout_t &layout) {
                /* NOTE: don't change behavior, this method must disregard templated T parameter. Actual type is specified in layout struct */
                //TBD: I am not sure if throwing an exception in a constructor is OK
                if (NN_DATA_STATUS_OK != nn_workload_data_placement_create(this, buffer, &nn_coords, &layout))
                    throw std::bad_alloc();
            }

            // Allocate buffer and copy the data from another nn_workload_data_t with possibly different layout
            nn_workload_data_t(const nn_workload_data_t &source, const nn_workload_data_layout_t &new_layout) {
                if (NN_DATA_STATUS_OK != nn_workload_data_placement_create(this, nullptr, &source.parent->lengths, &new_layout))
                    throw std::bad_alloc();

                copy(source);
            }

            // Create a view
            nn_workload_data_t(const nn_workload_data_t &source,
                               const nn_workload_data_coords_t &coords_begin,
                               const nn_workload_data_coords_t &coords_end) {
                //TBD: I am not sure if throwing an exception in a constructor is OK
                if (NN_DATA_STATUS_OK != nn_workload_data_placement_create_view(this, &source, &coords_begin, &coords_end))
                    throw std::bad_alloc();
            }

            nn_workload_data_t(nn_workload_data_t& source) {
                nn_workload_data_coords_t begin{0,0,0,0,0,0};
                nn_workload_data_coords_t end =
                {
                    source.get_length(0) - 1,
                    source.get_length(1) - 1,
                    source.get_length(2) - 1,
                    source.get_length(3) - 1,
                    source.get_length(4) - 1,
                    source.get_length(5) - 1
                };
                if (NN_DATA_STATUS_OK != nn_workload_data_placement_create_view(this, &source, &begin, &end))
                    throw std::bad_alloc();
            }

            ~nn_workload_data_t(){};

            // TODO: remove this; use operator= instead
            void copy(const nn_workload_data_t &source) {
                operator=(source);
            }

            nn_workload_data_t &operator=(const nn_workload_data_t &source) { 
                if (NN_DATA_STATUS_OK != nn_workload_data_copy(this, &source)) throw std::bad_alloc();
                return *this;
            }

            nn_workload_data_item_t<T> operator()(uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) {
                return nn_workload_data_item_t<T>(*this, nn_workload_data_coords_t(n,x,y,z,p,q));
            }
        
        uint32_t get_length(uint32_t dimension) const { return (view_end.t[dimension] - view_begin.t[dimension] + 1); }
        nn_workload_data_coords_t get_length() const { 
            return nn_workload_data_coords_t(
                  view_end.t[0]-view_begin.t[0]+1
                , view_end.t[1]-view_begin.t[1]+1
                , view_end.t[2]-view_begin.t[2]+1
                , view_end.t[3]-view_begin.t[3]+1
                , view_end.t[4]-view_begin.t[4]+1
                , view_end.t[5]-view_begin.t[5]+1
                );
        }
        uint32_t get_tile_length(uint32_t dimension) { return (1 << parent->layout.tile_lengths_log2.t[dimension]); }
    };
}

