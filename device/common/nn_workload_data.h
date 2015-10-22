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
#include <vector>

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

typedef enum {
    NN_WORKLOAD_DATA_TAG_UNKNOWN,
    NN_WORKLOAD_DATA_TAG_ZXYN,       /* I/O 3D layout */
    NN_WORKLOAD_DATA_TAG_NX,         /* I/O 1D layout */
    NN_WORKLOAD_DATA_TAG_X2NX,
    NN_WORKLOAD_DATA_TAG_ZBLOCKXYZN,
    NN_WORKLOAD_DATA_TAG_XBLOCKNX,
    NN_WORKLOAD_DATA_TAG_OBLOCKIXYO, /* weights layout for convolution layer */
    NN_WORKLOAD_DATA_TAG_OBLOCKIOXY, /* weights layout for asmjit convolution layer */
    NN_WORKLOAD_DATA_TAG_NBLOCKZXYN, /* layout for asmjit convolution layer */
    NN_WORKLOAD_DATA_TAG_O,          /* bias layout */
    NN_WORKLOAD_DATA_TAG_OI,         /* weights layout for fully connected layer, no batching */
    NN_WORKLOAD_DATA_TAG_OBLOCKIO,   /* weights layout for fully connected layer, with batching */
    NN_WORKLOAD_DATA_TAG_I2O32IXYO,
    NN_WORKLOAD_DATA_TAG_I2O8IO,
    NN_WORKLOAD_DATA_TAG_ZXY
} nn_workload_data_tag_t;

/* Supported data types */
typedef enum
{
    NN_DATATYPE_FLOAT,
    NN_DATATYPE_INT16,
    NN_DATATYPE_INT32
} nn_workload_data_type_t;

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

/*
  Convenient structure for passing data coordinates, data lenghts etc.
*/

struct nn_workload_data_coords_t {
    int dimension;
    std::vector<uint32_t> t;

    static nn_workload_data_coords_t *create(const uint32_t arg) {
        nn_workload_data_coords_t *result = new nn_workload_data_coords_t();
        result->dimension = arg;
        result->t.resize(arg);
        return result;
    }

    nn_workload_data_coords_t() : dimension(0) {};

    template<typename... T> nn_workload_data_coords_t(uint32_t arg, T ...args) {
        dimension = sizeof...(args) + 1;
        t = { arg, uint32_t(args)...};
    }

    nn_workload_data_coords_t(const nn_workload_data_coords_t &arg) : dimension(arg.dimension), t(dimension) {
        for (int n = 0; n<dimension; ++n) t[n] = arg.t[n];
    }

    nn_workload_data_coords_t &operator=(const nn_workload_data_coords_t &source) {
        if (dimension != source.dimension) {

            t.resize(source.dimension);
            dimension = source.dimension;
        }
        for (int n = 0; n < this->dimension; ++n){
            this->t[n] = source.t[n];
        }
        return *this;
    }

    bool operator== (const nn_workload_data_coords_t& arg) const
    {
        for (int n = 0; n<dimension; ++n)
            if (t[n] != arg.t[n]) return false;
        return true;
    }

    bool operator!= (const nn_workload_data_coords_t& arg) const
    {
        return !operator==(arg);
    }

    size_t size()
    {
        size_t ret = 1;
        for (auto i = 0; i < dimension; ++i)
            ret *= t[i];
        return ret;
    }

    ~nn_workload_data_coords_t() {
    };
};

/*
    Data structure layout.

    Contains information such as ordering and data type.
*/
typedef struct nn_workload_data_layout_s{
    nn_workload_data_coords_t ordering;          /* List of dimensions in increasing order of strides */
    nn_workload_data_type_t data_type;           /* Data type specified by nn_workload_data_type_t */

    bool operator== (const nn_workload_data_layout_s& arg) const
    {
        return this->data_type == arg.data_type &&
               this->ordering == arg.ordering;
    }

    bool operator!= (const nn_workload_data_layout_s& arg) const
    {
        return !operator==(arg);
    }
} nn_workload_data_layout_t;

/* Data structure that is referenced by at least one view */
struct nn_workload_data_core_t {

    nn_workload_data_core_t(uint32_t data_type_size,
                            nn_workload_data_coords_t lenghts,
                            nn_workload_data_layout_t layout,
                            void *buffer,
                            bool  allow_empty_data,
                            bool  allocate_delta);

    ~nn_workload_data_core_t();

    uint16_t dimension = 6;
    nn_workload_data_coords_t lengths;  /* Data structure size in each dimension */
    nn_workload_data_layout_t layout;   /* Data structure layout */
    void *data_buffer;
    void *delta_buffer;
    uint32_t buffer_size;      /* Size of allocated buffer */
    uint32_t data_type_size;   /* Size of one data item */

    uint32_t *strides; /* Determines order in the buffer */

    bool use_client_buffer;

    nn_workload_data_tag_t tag;
    void allocate_delta_buffer();
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

inline auto get_length(const nn_workload_data_t& view, nn_workload_data_coord_index_t dim_index)
    -> uint32_t
{
    return view.view_end.t[dim_index] - view.view_begin.t[dim_index] + 1;
}

inline auto get_length(const nn_workload_data_t& view)
    -> nn_workload_data_coords_t
{
    return {
            get_length(view, NN_DATA_COORD_n),
            get_length(view, NN_DATA_COORD_x),
            get_length(view, NN_DATA_COORD_y),
            get_length(view, NN_DATA_COORD_z),
            get_length(view, NN_DATA_COORD_p),
            get_length(view, NN_DATA_COORD_q)
        };
}

namespace nn {
    /* helper to get nn_workload_data_type_t from type */
    template <typename T> struct type_to_datatype;

    template <> struct type_to_datatype<float> : std::integral_constant<nn_workload_data_type_t, NN_DATATYPE_FLOAT>{};
    template <> struct type_to_datatype<int16_t> : std::integral_constant<nn_workload_data_type_t, NN_DATATYPE_INT16>{};
    template <> struct type_to_datatype<int32_t> : std::integral_constant<nn_workload_data_type_t, NN_DATATYPE_INT32>{};
}

/*
    Creates nn_workload_data structure using nn_workload_data_t allocated by a caller.
    Optionally also data buffer may provided by a caller.

    data        - pointer to the structure allocated by a caller
    allow_empty_data  - if set, there will be no data allocated internally nor caller data will be used,
                  this option can be used when user require only data structure definition, without
                  actual data inside
    buffer      - pointer to the data buffer. If NULL, it will be allocated by this function.
    lenghts     - size of data structure in each dimension.
    layout      - contains information such as ordering and data type.
*/
void nn_workload_data_placement_create(
    nn_workload_data_t* data,
    void* buffer,
    const nn_workload_data_coords_t* lenghts,
    const nn_workload_data_layout_t* layout,
    bool allow_empty_data = false,
    bool allocate_delta = false);

/*
    Creates view that references signal from another nn_workload_data_t.

    Resulting view has the same layout as original.
    If view cannot be created (outside image) - 0 is returned.
*/
nn_workload_data_t *nn_workload_data_create_view(
    const nn_workload_data_t* source,
    const nn_workload_data_coords_t* coords_begin,
    const nn_workload_data_coords_t* coords_end
    );

/*
    Creates view that references signal from another nn_workload_data_t. Uses nn_workload_data_t allocated by a caller.

    Resulting view has the same layout as original.
    If view cannot be created (outside image) - 0 is returned.
*/
void nn_workload_data_placement_create_view(
    nn_workload_data_t* data,
    const nn_workload_data_t* source,
    const nn_workload_data_coords_t* coords_begin,
    const nn_workload_data_coords_t* coords_end
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

    assert(data->parent->layout.data_type == nn::type_to_datatype<T>::value);
    assert(data->parent->data_type_size == sizeof(T));

    return data_buffer[calculate_idx(data, n, x, y, z, p, q)];
};

template <typename T>
static inline T &nn_workload_data_get_delta(
    nn_workload_data_t *data, uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) {
    T* delta_buffer = reinterpret_cast<T *>(data->parent->delta_buffer);

    assert(delta_buffer != nullptr);
    assert(data->parent->layout.data_type == nn::type_to_datatype<T>::value);
    assert(data->parent->data_type_size == sizeof(T));

    return delta_buffer[calculate_idx(data, n, x, y, z, p, q)];
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

    assert(data->parent->layout.data_type == nn::type_to_datatype<T>::value);
    assert(data->parent->data_type_size == sizeof(T));

    return data_buffer[calculate_idx(data, n, x, y, z, p, q)];
};

template <typename T>
static inline const T &nn_workload_data_get_delta(
    const nn_workload_data_t *data, uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) {
    const T *delta_buffer = reinterpret_cast<const T *>(data->parent->delta_buffer);

    assert(delta_buffer != nullptr);
    assert(data->parent->layout.data_type == nn::type_to_datatype<T>::value);
    assert(data->parent->data_type_size == sizeof(T));

    return delta_buffer[calculate_idx(data, n, x, y, z, p, q)];
};

/*
    Copy data from source to destination.
    Data ordering and may differ between source and destination,
    but lenghts in corresponding dimensions must be equal. TBD: less confusing description probably needed
*/
NN_DATA_STATUS nn_workload_copy(
    nn_workload_data_t* destination, const nn_workload_data_t* source, bool delta_copy
    );


inline NN_DATA_STATUS nn_workload_data_copy(
    nn_workload_data_t* destination, const nn_workload_data_t* source
    )
{
    return nn_workload_copy(
        destination, source, false
        );
}

inline NN_DATA_STATUS nn_workload_delta_copy(
    nn_workload_data_t* destination, const nn_workload_data_t* source
    )
{
    return nn_workload_copy(
        destination, source, true
        );
}

namespace nn {
    template<typename T> class workload_data;

    class layout_unknown;
    class layout_f32;
    class layout_nxyzpq_f32;
    class layout_xyznpq_f32;
    class layout_nzxypq_f32;
    class layout_pnzxyq_f32;
    class layout_pxyznq_f32;
    class layout_pxqzyn_f32;
    class layout_pzxyqn_f32;
    class layout_xyzpnq_f32;
    class layout_xyzpqn_f32;
    class layout_yxzpqn_f32;
    class layout_zxynpq_f32;
    class layout_pzqxyn_f32;
    class layout_zxyn_f32;
    class layout_nx_f32;
    class layout_x2nx_f32;
    class layout_zblockxyzn_f32;
    class layout_oblockixyo_f32;
    class layout_oblockioxy_f32;
    class layout_nblockzxyn_f32;
    class layout_o_f32;
    class layout_oi_f32;
    class layout_oblockio_f32;
    class layout_zxy_f32;

    class layout_nxyzpq_i16;
    class layout_pnzxyq_i16;
    class layout_pxyznq_i16;
    class layout_pzqxyn_i16;
    class layout_xyzpqn_i16;
    class layout_xzynpq_i16;
    class layout_ypznxq_i16;
    class layout_zpxynq_i16;
    class layout_zxynpq_i16;

    class layout_nxyzpq_i32;
    class layout_pnzxyq_i32;
    class layout_xnyzpq_i32;
    class layout_xyzpqn_i32;
    class layout_xzynpq_i32;
    class layout_zpxynq_i32;
    class layout_zxynpq_i32;

    template<typename T> struct get_layout_type;

    template<> struct get_layout_type<int16_t>               { using type = int16_t; };
    template<> struct get_layout_type<int32_t>               { using type = int32_t; };

    template<> struct get_layout_type<layout_unknown>        { using type = std::nullptr_t; };
    template<> struct get_layout_type<layout_f32>            { using type = float; };
    template<> struct get_layout_type<layout_nxyzpq_f32>     { using type = float; };
    template<> struct get_layout_type<layout_xyznpq_f32>     { using type = float; };
    template<> struct get_layout_type<layout_nzxypq_f32>     { using type = float; };
    template<> struct get_layout_type<layout_pnzxyq_f32>     { using type = float; };
    template<> struct get_layout_type<layout_pxyznq_f32>     { using type = float; };
    template<> struct get_layout_type<layout_pxqzyn_f32>     { using type = float; };
    template<> struct get_layout_type<layout_pzxyqn_f32>     { using type = float; };
    template<> struct get_layout_type<layout_xyzpnq_f32>     { using type = float; };
    template<> struct get_layout_type<layout_xyzpqn_f32>     { using type = float; };
    template<> struct get_layout_type<layout_yxzpqn_f32>     { using type = float; };
    template<> struct get_layout_type<layout_zxynpq_f32>     { using type = float; };
    template<> struct get_layout_type<layout_pzqxyn_f32>     { using type = float; };
    template<> struct get_layout_type<layout_zxyn_f32>       { using type = float; };
    template<> struct get_layout_type<layout_nx_f32>         { using type = float; };
    template<> struct get_layout_type<layout_x2nx_f32>       { using type = float; };
    template<> struct get_layout_type<layout_zblockxyzn_f32> { using type = float; };
    template<> struct get_layout_type<layout_oblockixyo_f32> { using type = float; };
    template<> struct get_layout_type<layout_oblockioxy_f32> { using type = float; };
    template<> struct get_layout_type<layout_nblockzxyn_f32> { using type = float; };
    template<> struct get_layout_type<layout_o_f32>          { using type = float; };
    template<> struct get_layout_type<layout_oi_f32>         { using type = float; };
    template<> struct get_layout_type<layout_oblockio_f32>   { using type = float; };
    template<> struct get_layout_type<layout_zxy_f32>        { using type = float; };

    template<> struct get_layout_type<layout_nxyzpq_i16>     { using type = int16_t; };
    template<> struct get_layout_type<layout_pnzxyq_i16>     { using type = int16_t; };
    template<> struct get_layout_type<layout_pxyznq_i16>     { using type = int16_t; };
    template<> struct get_layout_type<layout_pzqxyn_i16>     { using type = int16_t; };
    template<> struct get_layout_type<layout_xyzpqn_i16>     { using type = int16_t; };
    template<> struct get_layout_type<layout_xzynpq_i16>     { using type = int16_t; };
    template<> struct get_layout_type<layout_ypznxq_i16>     { using type = int16_t; };
    template<> struct get_layout_type<layout_zpxynq_i16>     { using type = int16_t; };
    template<> struct get_layout_type<layout_zxynpq_i16>     { using type = int16_t; };

    template<> struct get_layout_type<layout_nxyzpq_i32>     { using type = int32_t; };
    template<> struct get_layout_type<layout_pnzxyq_i32>     { using type = int32_t; };
    template<> struct get_layout_type<layout_xnyzpq_i32>     { using type = int32_t; };
    template<> struct get_layout_type<layout_xyzpqn_i32>     { using type = int32_t; };
    template<> struct get_layout_type<layout_xzynpq_i32>     { using type = int32_t; };
    template<> struct get_layout_type<layout_zpxynq_i32>     { using type = int32_t; };
    template<> struct get_layout_type<layout_zxynpq_i32>     { using type = int32_t; };

    template <typename T> struct layout_t;

    template <> struct layout_t<layout_unknown> {};
    template <> struct layout_t<layout_f32> {};
    template <> struct layout_t<layout_nxyzpq_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_xyznpq_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_nzxypq_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_pnzxyq_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_pxyznq_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_pxqzyn_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_pzxyqn_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_xyzpnq_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_xyzpqn_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_yxzpqn_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_zxynpq_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_pzqxyn_f32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_zxyn_f32>       { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_nx_f32>         { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_x2nx_f32>       { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_zblockxyzn_f32> { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_oblockixyo_f32> { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_oblockioxy_f32> { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_nblockzxyn_f32> { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_o_f32>          { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_oi_f32>         { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_oblockio_f32>   { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_zxy_f32>        { static nn_workload_data_layout_t layout; };

    template <> struct layout_t<layout_nxyzpq_i16>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_pnzxyq_i16>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_pxyznq_i16>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_pzqxyn_i16>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_xyzpqn_i16>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_xzynpq_i16>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_ypznxq_i16>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_zpxynq_i16>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_zxynpq_i16>     { static nn_workload_data_layout_t layout; };

    template <> struct layout_t<layout_nxyzpq_i32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_pnzxyq_i32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_xnyzpq_i32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_xyzpqn_i32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_xzynpq_i32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_zpxynq_i32>     { static nn_workload_data_layout_t layout; };
    template <> struct layout_t<layout_zxynpq_i32>     { static nn_workload_data_layout_t layout; };

    template<typename T = layout_unknown> class workload_data : public ::nn_workload_data_t {

        static_assert(std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value

        || std::is_same<T, layout_unknown>::value
        || std::is_same<T, layout_f32>::value
        || std::is_same<T, layout_nxyzpq_f32>::value
        || std::is_same<T, layout_xyznpq_f32>::value
        || std::is_same<T, layout_nzxypq_f32>::value
        || std::is_same<T, layout_pnzxyq_f32>::value
        || std::is_same<T, layout_pxyznq_f32>::value
        || std::is_same<T, layout_pxqzyn_f32>::value
        || std::is_same<T, layout_pzxyqn_f32>::value
        || std::is_same<T, layout_xyzpnq_f32>::value
        || std::is_same<T, layout_xyzpqn_f32>::value
        || std::is_same<T, layout_yxzpqn_f32>::value
        || std::is_same<T, layout_zxynpq_f32>::value
        || std::is_same<T, layout_pzqxyn_f32>::value
        || std::is_same<T, layout_zxyn_f32>::value
        || std::is_same<T, layout_nx_f32>::value
        || std::is_same<T, layout_x2nx_f32>::value
        || std::is_same<T, layout_zblockxyzn_f32>::value
        || std::is_same<T, layout_oblockixyo_f32>::value
        || std::is_same<T, layout_oblockioxy_f32>::value
        || std::is_same<T, layout_nblockzxyn_f32>::value
        || std::is_same<T, layout_o_f32>::value
        || std::is_same<T, layout_oi_f32>::value
        || std::is_same<T, layout_oblockio_f32>::value
        || std::is_same<T, layout_zxy_f32>::value

        || std::is_same<T, layout_nxyzpq_i16>::value
        || std::is_same<T, layout_pnzxyq_i16>::value
        || std::is_same<T, layout_pxyznq_i16>::value
        || std::is_same<T, layout_pzqxyn_i16>::value
        || std::is_same<T, layout_xyzpqn_i16>::value
        || std::is_same<T, layout_xzynpq_i16>::value
        || std::is_same<T, layout_ypznxq_i16>::value
        || std::is_same<T, layout_zpxynq_i16>::value
        || std::is_same<T, layout_zxynpq_i16>::value

        || std::is_same<T, layout_nxyzpq_i32>::value
        || std::is_same<T, layout_pnzxyq_i32>::value
        || std::is_same<T, layout_xnyzpq_i32>::value
        || std::is_same<T, layout_xyzpqn_i32>::value
        || std::is_same<T, layout_xzynpq_i32>::value
        || std::is_same<T, layout_zpxynq_i32>::value
        || std::is_same<T, layout_zxynpq_i32>::value
        , "type not supported");

    public:
        static layout_t<T> layout;
        using item_type = typename get_layout_type<T>::type;

        workload_data(nn_workload_data_tag_t tag, const nn_workload_data_coords_t &nn_coords, const nn_workload_data_layout_t &layout, bool allow_empty_data = false, bool allocate_delta = false) {
            nn_workload_data_placement_create(this, nullptr, &nn_coords, &layout, allow_empty_data, allocate_delta);
            this->parent->tag = tag;
        }

        workload_data(const nn_workload_data_coords_t &nn_coords, const nn_workload_data_layout_t &layout, bool allow_empty_data = false, bool allocate_delta = false) {
            nn_workload_data_placement_create(this, nullptr, &nn_coords, &layout, allow_empty_data, allocate_delta);
            this->parent->tag = NN_WORKLOAD_DATA_TAG_UNKNOWN;
        }

        // Allocate buffer adding padding and set the view to point inside the paddings
        workload_data(nn_workload_data_tag_t tag,
            nn_workload_data_coords_t nn_coords,
            const nn_workload_data_layout_t &layout,
            uint32_t padding_left,
            uint32_t padding_right,
            uint32_t padding_top,
            uint32_t padding_bottom,
            bool allow_empty_data = false,
            bool allocate_delta = false) {
            nn_coords.t[NN_DATA_COORD_x] += padding_left + padding_right;
            nn_coords.t[NN_DATA_COORD_y] += padding_top + padding_bottom;
            nn_workload_data_placement_create(this, nullptr, &nn_coords, &layout, allow_empty_data, allocate_delta);

            if (!allow_empty_data)
                memset(this->parent->data_buffer, 0, this->parent->buffer_size);

            std::unique_ptr<nn_workload_data_coords_t> temp(nn_workload_data_coords_t::create(nn_coords.dimension));
            view_begin = *temp;
            view_end = nn_coords;

            for (uint32_t i = 0; i < static_cast<uint32_t>(view_end.dimension); i++)
                --view_end.t[i];

            this->view_begin.t[NN_DATA_COORD_x] += padding_left;
            this->view_end.t[NN_DATA_COORD_x] -= padding_right;
            this->view_begin.t[NN_DATA_COORD_y] += padding_top;
            this->view_end.t[NN_DATA_COORD_y] -= padding_bottom;

            this->parent->tag = tag;
        }

        // Calls constructor defined above, just cast size_t parameters to uint32_t
        workload_data(nn_workload_data_tag_t tag,
            nn_workload_data_coords_t nn_coords,
            const nn_workload_data_layout_t &layout,
            size_t padding_left,
            size_t padding_right,
            size_t padding_top,
            size_t padding_bottom,
            bool allow_empty_data = false,
            bool allocate_delta = false)
                : workload_data(
                    tag,
                    nn_coords,
                    layout,
                    static_cast<uint32_t>( padding_left ),
                    static_cast<uint32_t>( padding_right ),
                    static_cast<uint32_t>( padding_top ),
                    static_cast<uint32_t>( padding_bottom ),
                    allow_empty_data,
                    allocate_delta) {
        }

        // Create using existing data buffer
        workload_data(nn_workload_data_tag_t tag,
            void *buffer,
            const nn_workload_data_coords_t &nn_coords,
            const nn_workload_data_layout_t &layout) {
            /* NOTE: don't change behavior, this method must disregard templated T parameter. Actual type is specified in layout struct */
            //TBD: I am not sure if throwing an exception in a constructor is OK
            nn_workload_data_placement_create(this, buffer, &nn_coords, &layout);
            this->parent->tag = tag;
        }

        // Create using existing data buffer
        workload_data(void *buffer,
            const nn_workload_data_coords_t &nn_coords,
            const nn_workload_data_layout_t &layout) {
            /* NOTE: don't change behavior, this method must disregard templated T parameter. Actual type is specified in layout struct */
            //TBD: I am not sure if throwing an exception in a constructor is OK
            nn_workload_data_placement_create(this, buffer, &nn_coords, &layout, true);
            this->parent->tag = NN_WORKLOAD_DATA_TAG_UNKNOWN;
        }


        // Allocate buffer and copy the data from another nn_workload_data_t with possibly different layout
        workload_data(nn_workload_data_tag_t tag,
            const workload_data &source,
            const nn_workload_data_layout_t &new_layout) {
            nn_workload_data_placement_create(this, nullptr, &source.parent->lengths, &new_layout);
            copy(source);

            this->parent->tag = tag;
        }

        // Create a view
        workload_data(const workload_data &source,
            const nn_workload_data_coords_t &coords_begin,
            const nn_workload_data_coords_t &coords_end) {
            nn_workload_data_placement_create_view(this, &source, &coords_begin, &coords_end);
            this->parent->tag = source.parent->tag;
        }

        workload_data(workload_data& source) {
            nn_workload_data_coords_t begin{ 0, 0, 0, 0, 0, 0 };
            nn_workload_data_coords_t end =
            {
                source.get_length(0) - 1,
                source.get_length(1) - 1,
                source.get_length(2) - 1,
                source.get_length(3) - 1,
                source.get_length(4) - 1,
                source.get_length(5) - 1
            };
            nn_workload_data_placement_create_view(this, &source, &begin, &end);
            this->parent->tag = source->parent->tag;
        }

        ~workload_data(){};

        // TODO: remove this; use operator= instead
        void copy(const workload_data &source) {
            operator=(source);
        }

        workload_data &operator=(const workload_data &source) {
            if (NN_DATA_STATUS_OK != nn_workload_data_copy(this, &source)) throw std::bad_alloc();
            return *this;
        }

        item_type& at(uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) {
            item_type *data_buffer = reinterpret_cast<item_type *>(parent->data_buffer);
            assert(parent->layout.data_type == nn::type_to_datatype<item_type>::value);
            assert(parent->data_type_size == sizeof(item_type));
            return data_buffer[calculate_idx(this, n, x, y, z, p, q)];
        }

        item_type& operator ()(uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) {
            return at(n, x, y, z, p, q);
        }

        item_type at(uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) const {
            return const_cast<nn::workload_data<T> *>(this)->at(n, x, y, z, p, q);
        }

        item_type operator ()(uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) const {
            return at(n, x, y, z, p, q);
        }

        uint32_t get_length(uint32_t dimension) const { return (view_end.t[dimension] - view_begin.t[dimension] + 1); }
        nn_workload_data_coords_t get_length() const {
            return nn_workload_data_coords_t(
                view_end.t[0] - view_begin.t[0] + 1
                , view_end.t[1] - view_begin.t[1] + 1
                , view_end.t[2] - view_begin.t[2] + 1
                , view_end.t[3] - view_begin.t[3] + 1
                , view_end.t[4] - view_begin.t[4] + 1
                , view_end.t[5] - view_begin.t[5] + 1
                );
        }

        void allocate_delta_buffer() {
            parent->allocate_delta_buffer();
        }

        item_type delta_at(uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q) const
        {
            return nn_workload_data_get_delta<item_type>(this, n, x, y, z, p, q);
        }

        item_type& delta_at(uint32_t n, uint32_t x, uint32_t y, uint32_t z, uint32_t p, uint32_t q)
        {
            return nn_workload_data_get_delta<item_type>(this, n, x, y, z, p, q);
        }
    };


    static void update_data(nn::workload_data<nn::layout_f32> *forward_data,
        nn::workload_data<nn::layout_f32> *delta_data,
        float learning_rate) {
        // TODO validate buffer sizes.
        for (auto n = 0u; n < forward_data->parent->lengths.t[NN_DATA_COORD_n]; ++n)
            for (auto q = 0u; q < forward_data->parent->lengths.t[NN_DATA_COORD_q]; ++q)
            for (auto p = 0u; p < forward_data->parent->lengths.t[NN_DATA_COORD_p]; ++p)
            for (auto z = 0u; z < forward_data->parent->lengths.t[NN_DATA_COORD_z]; ++z)
            for (auto y = 0u; y < forward_data->parent->lengths.t[NN_DATA_COORD_y]; ++y)
            for (auto x = 0u; x < forward_data->parent->lengths.t[NN_DATA_COORD_x]; ++x)
                forward_data->at(n, x, y, z, p, q) -= delta_data->at(n, x, y, z, p, q) * learning_rate;
    }

    static void update_data(
        nn::workload_data<nn::layout_f32> *data,
        float learning_rate) {

        for(auto n = 0u; n < data->parent->lengths.t[NN_DATA_COORD_n]; ++n)
            for(auto q = 0u; q < data->parent->lengths.t[NN_DATA_COORD_q]; ++q)
            for(auto p = 0u; p < data->parent->lengths.t[NN_DATA_COORD_p]; ++p)
            for(auto z = 0u; z < data->parent->lengths.t[NN_DATA_COORD_z]; ++z)
            for(auto y = 0u; y < data->parent->lengths.t[NN_DATA_COORD_y]; ++y)
            for(auto x = 0u; x < data->parent->lengths.t[NN_DATA_COORD_x]; ++x)
                data->at(n, x, y, z, p, q) -= data->delta_at(n, x, y, z, p, q) * learning_rate;
    }

    // cast workload_data<#>* to workload_data<0>*
    // cast workload_data<0>* to workload_data<#>* validating layout
    template<typename T_new_layout = layout_unknown, typename T_old_layout = layout_unknown> inline nn::workload_data<T_new_layout> *workload_data_cast(nn::workload_data<T_old_layout> *arg) {
        assert(!!arg && (std::is_same<T_new_layout, layout_unknown>::value || (std::is_same<T_new_layout, layout_f32>::value && arg->parent->layout.data_type == NN_DATATYPE_FLOAT)));
        return reinterpret_cast<nn::workload_data<T_new_layout> *>(arg);
    }

    // cast const workload_data<#>* to const workload_data<0>*
    // cast const workload_data<0>* to const workload_data<#>* validating layout
    template<typename T_new_layout = layout_unknown, typename T_old_layout = layout_unknown> inline const nn::workload_data<T_new_layout> *workload_data_cast(const nn::workload_data<T_old_layout> *arg) {
        assert(!!arg && (std::is_same<T_new_layout, layout_unknown>::value || (std::is_same<T_new_layout, layout_f32>::value && arg->parent->layout.data_type == NN_DATATYPE_FLOAT)));
        return reinterpret_cast<const nn::workload_data<T_new_layout> *>(arg);
    }

    // cast workload_data<#>& to workload_data<0>&
    // cast workload_data<0>& to workload_data<#>& validating layout
    template<typename T_new_layout = layout_unknown, typename T_old_layout = layout_unknown> inline nn::workload_data<T_new_layout> &workload_data_cast(nn::workload_data<T_old_layout> &arg) {
        assert(!!arg && (std::is_same<T_new_layout, layout_unknown>::value || (std::is_same<T_new_layout, layout_f32>::value && arg.layout.data_type == NN_DATATYPE_FLOAT)));
        return *reinterpret_cast<nn::workload_data<T_new_layout> *>(&arg);
    }

    // cast const workload_data<#>& to const workload_data<0>&
    // cast const workload_data<0>& to const workload_data<#>& validating layout
    template<typename T_new_layout = layout_unknown, typename T_old_layout = layout_unknown> inline const nn::workload_data<T_new_layout> &workload_data_cast(const nn::workload_data<T_old_layout> &arg) {
        assert(!!arg && (std::is_same<T_new_layout, layout_unknown>::value || (std::is_same<T_new_layout, layout_f32>::value && arg.layout.data_type == NN_DATATYPE_FLOAT)));
        return *reinterpret_cast<const nn::workload_data<T_new_layout> *>(&arg);
    }

    // cast nn_workload_data_t* to workload_data<0>*
    // cast nn_workload_data_t* to workload_data<#>* validating layout
    template<typename T_layout = layout_unknown> inline nn::workload_data<T_layout> *workload_data_cast(nn_workload_data_t *arg) {
        assert(!!arg && (std::is_same<T_layout, layout_unknown>::value || (std::is_same<T_layout, layout_f32>::value && arg->parent->layout.data_type == NN_DATATYPE_FLOAT)));
        return reinterpret_cast<nn::workload_data<T_layout> *>(arg);
    }

    // cast const nn_workload_data_t* to const workload_data<0>*
    // cast const nn_workload_data_t* to const workload_data<#>* validating layout
    template<typename T_layout = layout_unknown> inline nn::workload_data<T_layout> const *workload_data_cast(const nn_workload_data_t *arg) {
        assert(!!arg && (std::is_same<T_layout, layout_unknown>::value || (std::is_same<T_layout, layout_f32>::value && arg->parent->layout.data_type == NN_DATATYPE_FLOAT)));
        return reinterpret_cast<const nn::workload_data<T_layout> *>(arg);
    }
}
