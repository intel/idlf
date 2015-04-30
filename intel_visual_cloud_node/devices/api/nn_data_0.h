/*
Copyright (c) 2015, Intel Corporation

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

#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>

#if defined _MSC_VER
#   pragma pack(push,1)
typedef struct 
#else
typedef struct __attribute__((packed))
#endif
nn_data {
#if defined __cplusplus
    nn_data() : buffer(nullptr), size(nullptr), dimension(0), sizeof_value(0) {};
#endif
    void           *const buffer;       /* buffer containig data */
    const size_t   *const size;         /* sizes of signal in each coordinate; unit is a value */
    const uint8_t         dimension;    /* dimensionality of data, as in http://en.wikipedia.org/wiki/Dimension */
    const uint8_t         sizeof_value; /* size of single value in buffer */
} nn_data_t;
#if defined _MSC_VER
#   pragma pack(pop)
#endif

/* begin workaround for MSVC not supporting inline in C */
#if !defined __cplusplus && defined _MSC_VER
#   define inline __inline
#endif

/* add va_copy for compilers with poor C99 compatiblilty */
#if !defined va_copy
#   if defined __va_copy
#       define va_copy(a,b) __va_copy(a,b)
#   else
#       define va_copy(a,b) ((a)=(b))
#   endif 
#endif

/* internal code; do not call directly */
static inline nn_data_t *internal_nn_data_create(nn_data_t *, size_t, void *, size_t, uint8_t, va_list);
static inline nn_data_t *internal_nn_data_create_ptr(nn_data_t *, size_t, void *, size_t, uint8_t, const size_t *);
static inline void *internal_nn_data_at(nn_data_t *, uint8_t, va_list);
static inline void *internal_nn_data_at_ptr(nn_data_t *, uint8_t, size_t *);



/* calculate size of buffer, sizes in arguments
   examples:
     nn_data_buffer_size(sizeof(float), 3, 2, 3, 4); // returns 96
       Buffer for 3-dimensional grid of floats with size [2,3,4] has size of 96 bytes.
     nn_data_buffer_size(sizeof(uint8_t), 2, 320, 240); // returns 76800
       Buffer for 2-dimensional grid of bytes with size [320,240] has size od 768000.
*/
static inline size_t nn_data_buffer_size(
    size_t  sizeof_value,   /* sizeof single value */
    uint8_t dimension,      /* dimensionality of data */
    va_list size_list       /* list of sizes - one per axis */
) {
    va_list list; va_copy(list, size_list);
    size_t buffer_size = sizeof_value;
    for(uint8_t at=0; at<dimension; ++at) buffer_size *= va_arg(list, size_t);
    return buffer_size;
}


/* calculate size of buffer, sizes in array 
   examples:
     size_t sizes[3] = {2, 3, 4};
     nn_data_buffer_size_ptr(sizeof(float), sizeof(sizes)/sizeof(sizes[0]), sizes);
       Buffer for 3-dimensional grid of floats with size [2,3,4] has size of 96 bytes.
     size_t sizes[2] = {320, 240};
     nn_data_buffer_size_ptr(sizeof(float), sizeof(sizes)/sizeof(sizes[0]), sizes);
       Buffer for 2-dimensional grid of bytes with size [320,240] has size od 768000.
*/
static inline size_t nn_data_buffer_size_ptr(
    size_t  sizeof_value,   /* sizeof single value */
    uint8_t dimension,      /* dimensionality of data */
    const size_t *size_ptr  /* array of sizes - one per axis */
) {
    assert(size_ptr);
    if(!size_ptr) return 0;
    size_t buffer_size = sizeof_value;
    for(uint8_t at=0; at<dimension; ++at) buffer_size *= size_ptr[at];
    return buffer_size;
}


/* create nn_data with shared buffer, sizes in arguments
   examples:
     float buffer[4][3][2] = {0};
     nn_data_t *data = nn_data_create_shared(buffer, sizeof(buffer[0][0][0]), 3, 2, 3, 4);
        Creates nn_data_t structure that describes existing 3-dimensional grid of floats with size [2,3,4].
     float buffer[240][320] = {0};
     nn_data_t *data = nn_data_create_shared(buffer, sizeof(buffer[0][0]), 2, 320, 240);
        Creates nn_data_t structure that describes existing 2-dimensional grid of floats with size [320, 240].
*/
static inline nn_data_t *nn_data_create_shared(
    void   *buffer,         /* buffer containing user data */
    size_t  sizeof_value,   /* sizeof single value */
    uint8_t dimension,      /* dimensionality of data */
    ...                     /* list of sizes - one per axis */
) {
    va_list list;
    va_start(list, dimension);
    nn_data_t *data = (nn_data_t *)malloc(sizeof(nn_data_t));
    nn_data_t *result = internal_nn_data_create(data, 0, buffer, sizeof_value, dimension, list);
    va_end(list);
    if(!result) free(data);
    return result;
}


/* create nn_data with shared buffer, sizes in array
   examples:
     float buffer[4][3][2] = {0};
     size_t sizes[3] = {2, 3, 4};
     nn_data_t *data = nn_data_create_shared(buffer, sizeof(buffer[0][0][0]), sizeof(sizes)/sizeof(sizes[0]), sizes);
        Creates nn_data_t structure that describes existing 3-dimensional grid of floats with size [2,3,4].
     float buffer[240][320] = {0};
     size_t sizes[2] = {320, 240};
     nn_data_t *data = nn_data_create_shared(buffer, sizeof(buffer[0][0][0]), sizeof(sizes)/sizeof(sizes[0]), sizes);
        Creates nn_data_t structure that describes existing 2-dimensional grid of floats with size [320, 240].
*/
static inline nn_data_t *nn_data_create_shared_ptr(
    void   *buffer,         /* buffer containing user data */
    size_t  sizeof_value,   /* sizeof single value */
    uint8_t dimension,      /* dimensionality of data */
    size_t *size_ptr        /* array of sizes - one per axis */
) {
    nn_data_t *data = (nn_data_t *)malloc(sizeof(nn_data_t));
    nn_data_t *result = internal_nn_data_create_ptr(data, 0, buffer, sizeof_value, dimension, size_ptr);
    if(!result) free(data);
    return result;
}


/* create nn_data, sizes in arguments
   examples:
     nn_data_t *data = nn_data_create(sizeof(float), 3, 2, 3, 4);
       Creates new nn_data_t container with 3-dimenasional grid of size [2,3,4].
     nn_data_t *data = nn_data_create(sizeof(uint8_t), 2, 320, 240);
       Creates new nn_data_t container with 2-dimenasional grid of size [320, 240].
*/
static inline nn_data_t *nn_data_create(
    size_t  sizeof_value,   /* sizeof single value */
    uint8_t dimension,      /* dimensionality of data */
    ...                     /* list of sizes - one per axis */
) {
    va_list list;
    nn_data_t *result;
    size_t buffer_size;
    nn_data_t *data = (nn_data_t *)malloc(sizeof(nn_data_t));
    va_start(list, dimension);
    buffer_size = nn_data_buffer_size(sizeof_value, dimension, list);
    result = buffer_size ? internal_nn_data_create(data, buffer_size, 0, sizeof_value, dimension, list) : 0; 
    va_end(list);
    if(!result) free(data);
    return result;
}


/* create nn_data, sizes in arguments
   examples:
     size_t sizes[3] = {2, 3, 4};
     nn_data_t *data = nn_data_create(sizeof(float), sizeof(sizes)/sizeof(sizes[0]), sizes);
       Creates new nn_data_t container with 3-dimenasional grid of size [2,3,4].
     size_t sizes[2] = {320, 240};
     nn_data_t *data = nn_data_create(sizeof(float), sizeof(sizes)/sizeof(sizes[0]), sizes);
       Creates new nn_data_t container with 2-dimenasional grid of size [320, 240].
*/
static inline nn_data_t *nn_data_create_ptr(
    size_t  sizeof_value,   /* sizeof single value */
    uint8_t dimension,      /* dimensionality of data */
    size_t *size_ptr        /* array of sizes - one per axis */
) {
    nn_data_t *result;
    size_t buffer_size;
    nn_data_t *data = (nn_data_t *)malloc(sizeof(nn_data_t));
    buffer_size = nn_data_buffer_size_ptr(sizeof_value, dimension, size_ptr);
    result = buffer_size ? internal_nn_data_create_ptr(data, buffer_size, 0, sizeof_value, dimension, size_ptr) : 0; 
    if(!result) free(data);
    return result;
}


/* delete nn_data
   example:
     nn_data_t *data = nn_data_create(sizeof(float), 3, 2, 3, 4);
     nn_data_delete(data);
*/
static inline void nn_data_delete(
    nn_data_t *data         /* pointer to nn_data to be deleted */
) {
    if(data) free((void *)data->size);
    free(data);
}


/* get address of value at coordinates in arguments
   nn_data_at(data, 3, 1, 2, 3);
     Returns pointer to data in 3-dimensional grid at position [1,2,3].
*/
static inline void *nn_data_at(
    nn_data_t  *data,       /* pointer to nn_data to be accessed */
    uint8_t     dimension,  /* dimensionality of coordinate list (for validation) */
    ...                     /* list of coordinates - one per axis */
) {
    if(    data==0                              /* invalid nn_data */
        || data->dimension!=dimension           /* dimensionality of input coordinates and nn_data is different */
    ) return 0;
    else {
        va_list list;
        void *result;
        va_start(list, dimension);
        result = internal_nn_data_at(data, dimension, list);
        va_end(list);
        return result;
    }
}


/* get address of value at coordinates in array
   size_t position[3] = {1, 2, 3};
   nn_data_at(data, 3, position);
     Returns pointer to data in 3-dimensional grid at position [1,2,3].
*/
static inline void *nn_data_at_ptr(
    nn_data_t  *data,       /* pointer to nn_data to be accessed */
    uint8_t     dimension,  /* dimensionality of coordinate list (for validation) */
    size_t     *at_ptr      /* list of coordinates - one per axis */
) {
    if(    !data                                /* invalid nn_data */
        || data->dimension!=dimension           /* dimensionality of input coordinates and nn_data is different */
        || !at_ptr                              /* invalid at */
    ) return 0;
    else return internal_nn_data_at_ptr(data, dimension, at_ptr);
}


/* internal code; do not call directly */
static inline nn_data_t *internal_nn_data_create(
    nn_data_t *data,        /* data buffer */
    size_t  alloc_size,     /* size of memory to be allocated */
    void   *buffer,         /* buffer pointer, 0 is not shared */
    size_t  sizeof_value,   /* sizeof single value */
    uint8_t dimension,      /* dimensionality of data */
    va_list size_list       /* list of sizes - one per axis */
) {
    /* validate scalar input parameters */
    if(    sizeof_value==0                      /* size of single value is zero */
        || sizeof_value>8                       /* size of single value is greater than 8 (size of IEEE double) */
        || (sizeof_value&(sizeof_value-1))!=0   /* size of single value is not a power of 2 */
        || dimension==0                         /* signal has dimension of zero */
        || (!buffer && alloc_size==0)           /* buffer not set and allocation size is 0 */
        || !data
      ) return 0;
    else {
        /* validate list of sizes */
        va_list list; va_copy(list, size_list);
        const size_t array_size = sizeof(size_t)*dimension;
        for(uint8_t at=0; at<dimension; ++at) if(va_arg(list, size_t)==0) return 0;
        alloc_size += array_size;
        /* allocate memory for structure, array of sizes & buffer of values */
        const size_t page_size = 4096;
        uint8_t *const raw_memory = (uint8_t *)malloc(alloc_size + (buffer ? 0 : page_size-1));
        if(!raw_memory) return 0;
        else {
            /* fill scalar fields of data structure */
            *(void **)   &(data->buffer)        = buffer ? buffer : (void *)(((uintptr_t)raw_memory + array_size + page_size-1)&(~(uintptr_t)(page_size-1)));
            *(uint8_t **)&(data->size)          = raw_memory;
            *(uint8_t *) &(data->dimension)     = dimension;
            *(uint8_t *) &(data->sizeof_value)  = (uint8_t)sizeof_value;
            /* fill array of dimensions */
            va_list list; va_copy(list, size_list);
            for(uint8_t at=0; at<dimension; ++at) {
                *(size_t *)(data->size+at) = va_arg(list, size_t);
            }
            return data;
        }
    }
}

/* internal code; do not call directly */
static inline nn_data_t *internal_nn_data_create_ptr(
    nn_data_t *data,        /* data buffer */
    size_t  alloc_size,     /* size of memory to be allocated */
    void   *buffer,         /* buffer pointer, 0 is not shared */
    size_t  sizeof_value,   /* sizeof single value */
    uint8_t dimension,      /* dimensionality of data */
    const size_t *size_ptr  /* array of sizes - one per axis */
) {
    /* validate scalar input parameters */
    if(    sizeof_value==0                      /* size of single value is zero */
        || sizeof_value>8                       /* size of single value is greater than 8 (size of IEEE double) */
        || (sizeof_value&(sizeof_value-1))!=0   /* size of single value is not a power of 2 */
        || dimension==0                         /* signal has dimension of zero */
        || (!buffer && alloc_size==0)           /* buffer not set and allocation size is 0 */
        || !data
        || !size_ptr
      ) return 0;
    else {
        /* validate list of sizes */
        const size_t array_size = sizeof(size_t)*dimension;
        for(uint8_t at=0; at<dimension; ++at) if(size_ptr[at]==0) return 0;
        alloc_size += array_size;
        /* allocate memory for structure, array of sizes & buffer of values */
        const size_t page_size = 4096;
        uint8_t *const raw_memory = (uint8_t *)malloc(alloc_size + (buffer ? 0 : page_size-1));
        if(!raw_memory) return 0;
        else {
            /* fill scalar fields of data structure */
            *(void **)   &(data->buffer)        = buffer ? buffer : (void *)(((uintptr_t)raw_memory + array_size + page_size-1)&(~(uintptr_t)(page_size-1)));
            *(uint8_t **)&(data->size)          = raw_memory;
            *(uint8_t *) &(data->dimension)     = dimension;
            *(uint8_t *) &(data->sizeof_value)  = (uint8_t)sizeof_value;
            /* fill array of dimensions */
            for(uint8_t at=0; at<dimension; ++at) {
                *(size_t *)(data->size+at) = size_ptr[at];
            }
            return data;
        }
    }
}

/* internal code; do not call directly, no parameter validation */
static inline void *internal_nn_data_at(
    nn_data_t *data,        /* data buffer */
    uint8_t dimension,      /* dimensionality of data */
    va_list at_list         /* list of cooridnates - one per axis */
) {
    va_list list; va_copy(list, at_list);
    size_t multiplier = data->sizeof_value;
    size_t position = 0;
    for(uint8_t at=0; at<dimension-1; ++at) {
        position += multiplier*va_arg(list, size_t);
        multiplier *= data->size[at];
    }
    position += multiplier*va_arg(list, size_t);
    return (uint8_t*)data->buffer + position;
}

/* internal code; do not call directly, no parameter validation */
static inline void *internal_nn_data_at_ptr(
    nn_data_t *data,        /* data buffer */
    uint8_t dimension,      /* dimensionality of data */
    size_t *at_ptr          /* array of coordinates - one per axis */
) {
    size_t multiplier = data->sizeof_value;
    size_t position = 0;
    for(uint8_t at=0; at<dimension-1; ++at) {
        position += multiplier*at_ptr[at];
        multiplier *= data->size[at];
    }
    position += multiplier*at_ptr[dimension-1];
    return (uint8_t*)data->buffer + position;
}

#if defined NN_DATA_VALUE_TYPE_PTR_CODE
#   error name clash: NN_DATA_VALUE_TYPE_PTR_CODE already defined
#endif

/* following macro allows one-line generation of function getting address for other types */
#define NN_DATA_VALUE_TYPE_AT_CODE(TYPE) \
static inline TYPE *nn_data_##TYPE##_at(nn_data_t *data, uint8_t dimension, ...) { \
    if(data==0 || data->dimension!=dimension || data->sizeof_value!=sizeof(TYPE)) return 0; \
    else { \
        va_list list; \
        TYPE *result; \
        va_start(list, dimension); \
        result = (TYPE *)internal_nn_data_at(data, dimension, list); \
        va_end(list); \
        return result; \
    } \
}

/* nn_data_[type]_ptr convenience functions 
   They allow
   "*nn_data_uint8_t_at(...) = 5 syntax", but are minimally slower than
   "*(uint8_t*)nn_data_at(...) = 5" because added type size validation.
*/
NN_DATA_VALUE_TYPE_AT_CODE(uint8_t);
NN_DATA_VALUE_TYPE_AT_CODE(int8_t);
NN_DATA_VALUE_TYPE_AT_CODE(uint16_t);
NN_DATA_VALUE_TYPE_AT_CODE(int16_t);
NN_DATA_VALUE_TYPE_AT_CODE(uint32_t);
NN_DATA_VALUE_TYPE_AT_CODE(int32_t);
NN_DATA_VALUE_TYPE_AT_CODE(float);

#undef NN_DATA_VALUE_TYPE_AT_CODE

/* end workaround for MSVC not supporting inline in C */
#if !defined __cplusplus && defined _MSC_VER
#   undef inline
#endif

#if defined __cplusplus
// C++ interface
#   include <new>
#   include <stdexcept>
#   include <utility>
#   include <cstring>
namespace nn {
    template<typename T_type, uint8_t T_dimension=0> class data : public nn_data {
        // compile-time validation if arguments can be fetched as size_t from variadic function call
        template<typename T_first, typename... T_rest> struct are_size_t_compatible{
            static const bool value = are_size_t_compatible<T_first>::value && are_size_t_compatible<T_rest...>::value;
        };
        template<typename T_first> struct are_size_t_compatible<T_first> {
            static const bool value = std::is_integral<T_first>::value;
        };

    public:
        // create nn:data with it's own buffer
        data(size_t size) {
            static_assert(T_dimension==0 || 1==T_dimension, "invalid number of dimensions");
            [](nn_data_t *data,        /* destination data structure */
               uint8_t dimension,      /* dimensionality of data */
               ...                     /* list of sizes - one per axis */
            ) -> void {
                va_list list;
                va_start(list, dimension);
                size_t buffer_size = nn_data_buffer_size(sizeof(T_type), dimension, list);
                nn_data_t *result = buffer_size ? internal_nn_data_create(data, buffer_size, 0, sizeof(T_type), dimension, list) : 0;
                va_end(list);
                if(!result) throw std::bad_alloc();
            } (this, 1, size);
        }

        // create nn:data with it's own buffer
        template<typename... T_sizes> data(size_t size, T_sizes... sizes) {
            static_assert(are_size_t_compatible<size_t, T_sizes...>::value, "one of arguments cannot be treated as size_t");
            static_assert(T_dimension==0 || 1+sizeof...(T_sizes)==T_dimension, "invalid number of dimensions");
            [](nn_data_t *data,        /* destination data structure */
               uint8_t dimension,      /* dimensionality of data */
               ...                     /* list of sizes - one per axis */
            ) -> void {
                va_list list;
                va_start(list, dimension);
                size_t buffer_size = nn_data_buffer_size(sizeof(T_type), dimension, list);
                nn_data_t *result = buffer_size ? internal_nn_data_create(data, buffer_size, 0, sizeof(T_type), dimension, list) : 0;
                va_end(list);
                if(!result) throw std::bad_alloc();
            } (this, sizeof...(T_sizes)+1, size, size_t(sizes)...);
        }

        // create nn:data with it's own buffer
        template<typename... T_sizes> data(const size_t *const size_ptr, uint8_t dimension) {
            assert(!T_dimension ? dimension>0 : T_dimension==dimension);
            size_t buffer_size = nn_data_buffer_size_ptr(sizeof(T_type), dimension, size_ptr);
            nn_data_t *result = buffer_size ? internal_nn_data_create_ptr(this, buffer_size, 0, sizeof(T_type), dimension, size_ptr) : 0;
            if(!result) throw std::bad_alloc();
        }

        // create nn:data with external buffer
        template<typename... T_sizes> data(T_type *buffer, size_t size, T_sizes... sizes) {
            static_assert(are_size_t_compatible<size_t, T_sizes...>::value, "one of arguments cannot be treated as size_t");
            static_assert(T_dimension==0 || sizeof...(T_sizes)+1==T_dimension, "invalid number of dimensions");
            [](nn_data_t *data,        /* destination data structure */
               void   *buffer,         /* buffer containing user data */
               uint8_t dimension,      /* dimensionality of data */
               ...                     /* list of sizes - one per axis */
            ) {
                va_list list;
                va_start(list, dimension);
                nn_data_t *result = internal_nn_data_create(data, 0, buffer, sizeof(T_type), dimension, list);
                va_end(list);
                if(!result) throw std::bad_alloc();
            }(this, buffer, sizeof...(T_sizes)+1, size, size_t(sizes)...);
        }

        // create nn:data with external buffer
        data(T_type *buffer, size_t *size_ptr, uint8_t dimension) {
            assert(!T_dimension ? dimension>0 : T_dimension==dimension);
            nn_data_t *result = internal_nn_data_create_ptr(this, 0, buffer, sizeof(T_type), dimension, size_ptr);
            if(!result) throw std::bad_alloc();
        }

        // delete nn:data
        ~data() {
            free((void*)(this->nn_data::size));
        }

        // copy operator
        template<uint8_t T_other_dimension> data &operator=(const data<T_type, T_other_dimension> &arg) {
            static_assert(T_dimension==T_other_dimension || T_dimension*T_other_dimension==0, "cannot change dimensionality");
            if(dimension!=arg.dimension) throw std::runtime_error();
            for(auto index=0u; index<dimension; ++index)
                if(nn_data_t::size[index]!=arg.nn_data_t::size[index])  throw std::runtime_error();
            size_t buffer_size = nn_data_buffer_size_ptr(sizeof(T_type), T_dimension, nn_data_t::size);
            std::memcpy(buffer, arg.buffer, buffer_size);
            return *this;
        }

        // copy constructor
        data(const data &arg) {
            size_t buffer_size = nn_data_buffer_size_ptr(sizeof(T_type), arg.dimension, arg.nn_data_t::size);
            nn_data_t *result = buffer_size ? internal_nn_data_create_ptr(this, buffer_size, 0, sizeof(T_type), arg.dimension, arg.nn_data_t::size) : 0;
            if(!result) throw std::bad_alloc();
            std::memcpy(buffer, arg.buffer, buffer_size);
        }

        // accessor to size
        struct {
            // return size in specified dimension
            size_t operator[](size_t index) const {
                auto base = get_base(); 
                if(index<base->dimension) return base->nn_data_t::size[index];
                else throw std::out_of_range("nn::data<T_type, T_dimension>::size operator[]");
            }
            // return size buffer
            explicit operator const size_t* const() const {
                auto base = get_base(); 
                return base->nn_data_t::size;
            }
        private:
            const data<T_type, T_dimension> *get_base() const {
                const uint8_t *ptr = reinterpret_cast<const uint8_t *>(this);
                ptr -= (size_t)&reinterpret_cast<const volatile char&>((((const nn::data<T_type,T_dimension> *)0)->size));
                return reinterpret_cast<const data<T_type, T_dimension> *>(ptr);
            }
        } size;

        // return count of elements in container
        size_t count() const {
            return nn_data_buffer_size_ptr(1, dimension, nn_data_t::size);
        }

        // return element at specific point
        template<typename... T_coords> auto at(T_coords... coords) -> T_type &{
            static_assert(are_size_t_compatible<T_coords...>::value, "one of arguments cannot be treated as size_t");
            static_assert(T_dimension==0 || sizeof...(T_coords)==T_dimension, "invalid number of dimensions");
            assert(sizeof...(T_coords)==dimension); // incorrect number of arguments
            if(void *result = nn_data_at(this, dimension, size_t(coords)...)) {
                return *reinterpret_cast<T_type *>(result);
            } else throw std::out_of_range("nn::data<T_type, T_dimension>::at");
        }
        template<typename... T_coords> auto at(T_coords&&... coords) const -> const T_type & {
            return const_cast<data<T_type,T_dimension> *>(this)->at(coords...);
        }

        // return element at specific point
        auto at(size_t *coords_ptr) -> T_type &{
            assert(coords_ptr); // incorrect cooridnate pointer
            if(void *result = nn_data_at_ptr(this, dimension, coords_ptr)) {
                return *reinterpret_cast<T_type *>(result);
            } else throw std::out_of_range("nn::data<T_type, T_dimension>::at");
        }
        auto at(size_t *coords_ptr) const -> const T_type & {
            return const_cast<data<T_type,T_dimension> *>(this)->at(coords_ptr);
        }

        // return element at specific point
        template<typename... T_coords> T_type &operator()(T_coords... coords) {
            return at(coords...);
        }

        typedef T_type type;
    };

    // cast data<T,#>* to data<T,0>*
    // cast data<T,0>* to data<T,#>* validating number of dimensions; data must be != nullptr
    template<typename T_type, uint8_t T_new_dimension, uint8_t T_old_dimension> inline data<T_type, T_new_dimension> *data_cast(data<T_type, T_old_dimension> *arg) {
        assert(!T_new_dimension || (!!arg && T_new_dimension==arg->dimension));
        return reinterpret_cast<data<T_type, T_new_dimension> *>(arg);
    }

    // cast const data<T,#>* to const data<T,0>*
    // cast const data<T,0>* to const data<T,#>* validating number of dimensions; data must be != nullptr
    template<typename T_type, uint8_t T_new_dimension, uint8_t T_old_dimension> inline const data<T_type, T_new_dimension> *data_cast(const data<T_type, T_old_dimension> *arg) {
        assert(!T_new_dimension || (!!arg && T_new_dimension==arg->dimension));
        return reinterpret_cast<const data<T_type, T_new_dimension> *>(arg);
    }

    // cast data<T,#>& to data<T,0>&
    // cast data<T,0>& to data<T,#>& validating number of dimensions; data must be != nullptr
    template<typename T_type, uint8_t T_new_dimension, uint8_t T_old_dimension> inline data<T_type, T_new_dimension> &data_cast(data<T_type, T_old_dimension> &arg) {
        assert(!T_new_dimension || T_new_dimension==arg.dimension);
        return *reinterpret_cast<data<T_type, T_new_dimension> *>(&arg);
    }

    // cast const data<T,#>& to const data<T,0>&
    // cast const data<T,0>& to const data<T,#>& validating number of dimensions; data must be != nullptr
    template<typename T_type, uint8_t T_new_dimension, uint8_t T_old_dimension> inline const data<T_type, T_new_dimension> &data_cast(const data<T_type, T_old_dimension> &arg) {
        assert(!T_new_dimension || T_new_dimension==arg.dimension);
        return *reinterpret_cast<const data<T_type, T_new_dimension> *>(&arg);
    }

    // cast nn_data_t* to data<T,0>* validating size of value; data must be != nullptr
    // cast nn_data_t* to data<T,#>* validating size of value and number of dimensions; data must be != nullptr
    template<typename T_type, uint8_t T_dimension> inline data<T_type, T_dimension> *data_cast(nn_data_t *arg) {
        assert(!!arg && arg->sizeof_value==sizeof(T_type) && (!T_dimension || arg->dimension==T_dimension));
        return reinterpret_cast<data<T_type, T_dimension> *>(arg);
    }

    // cast const nn_data_t* to const data<T,0>* validating size of value; data must be != nullptr
    // cast const nn_data_t* to const data<T,#>* validating size of value and number of dimensions; data must be != nullptr
    template<typename T_type, uint8_t T_dimension> inline const data<T_type, T_dimension> *data_cast(const nn_data_t *arg) {
        assert(!!arg && arg->sizeof_value==sizeof(T_type) && (!T_dimension || arg->dimension==T_dimension));
        return reinterpret_cast<const data<T_type, T_dimension> *>(arg);
    }
}
#endif
