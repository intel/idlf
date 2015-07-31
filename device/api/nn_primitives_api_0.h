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
#include "nn_types_0.h"

/*
This file contains API for interface version 0.

Basic types:

<nn_primitive_handle_t>
    Handle for primitive descriptor. Created with all parameters defining primitives operation, it is used to prepare
internal storage with correct data layouts and to perform optimized execution. Primitive handle holds complete execution
strategy.

<nn_data_t>
    Public data container. Defined with number of dimensions and size in each dimension. Can be used to describe
trivially laid out data. User creates this container to describe his own data for import into primitive's domain.

<nn_opaque_data_t>
    Internal storage structure. User manages internal storage's lifetime using create and delete methods. Create
functions are primitive specific to create storage with specific data layouts.

<nn_event_t>
    An operation handle - used to synchronize execution of asynchronous operations.


Flow of operation:
    1. Initialize a device;
    2. Obtain primitive handle;
    3. Encapsulate data with nn_data_t or create and fill new nn_data_t containers;
    4. Create internal storage using nn_data_t described buffers;
    5. Schedule execution using e.g. forward_async function;
    (Optional) 6. Schedule dependent primitive execution;
    7. Schedule copy output to obtain the outputs;
    8. Wait for operations to complete;

*/

/* enumeration of device types; will be extended in future */
typedef enum {
    NN_DEVICE_PRIMITIVES_TYPE_CPU = 0,
    NN_DEVICE_PRIMITIVES_TYPE_GPU,
    NN_DEVICE_PRIMITIVES_TYPE_LAST = NN_DEVICE_PRIMITIVES_TYPE_GPU
} NN_DEVICE_PRIMITIVES_TYPE;

/* enumeration of flags for opaque data container */
typedef enum {
    NN_OPAQUE_DATA_FLAGS_ALLOC_DELTA = 0x01,
} NN_OPAQUE_DATA_FLAGS;

/* description of device */
typedef struct {
    NN_DEVICE_PRIMITIVES_TYPE type;           /* device type */
    uint16_t        version_first;  /* first supported API version */
    uint16_t        version_last;   /*  last supported API version */
    const char     *name;           /* pointer to read-only memory with device name (single line) */
    const char     *description;    /* pointer to read-only memory with long device description (multi line)*/
} nn_device_primitives_description_t;

/* function type to obtain device description and supported versions */
typedef int32_t (NN_API_CALL_CONVENTION *nn_device_get_primitives_description_t)(
    nn_device_primitives_description_t *const description  /* pointer to description structure */
);

/* loads & initializes device
   If succeeds fills device description structure and returns non-negative value.
   Returns:
      0: success
     -1: load failed
     -2: invalid pointer */
NN_API_CALL int32_t NN_API_CALL_CONVENTION nn_device_get_primitives_description(
    nn_device_primitives_description_t *const description  /* pointer to description structure */
);

/* function type to obtain primitives struct pointer from shared library */
typedef int32_t (NN_API_CALL_CONVENTION *nn_device_get_primitives_t)(
    uint32_t version,               /* version of interface to create */
    void *const primitives          /* pointer to interface structure */
);

/* opens primitives
   Fills primitives structure with function pointers that application can use.
   Returns:
      0: success
     -1: interface open failed
     -2: invalid pointer
     -3: unsupported version (not in range returned from nn_device_load) */
NN_API_CALL int32_t NN_API_CALL_CONVENTION nn_device_get_primitives(
    uint32_t version,               /* version of interface to create */
    void *const primitives          /* pointer to interface structure */
);

/* Create device handle for primitives
    Returns: nn_device_t handle or NULL on failure
*/
typedef nn_device_t *(NN_API_CALL_CONVENTION *nn_primitives_create_device_with_thread_count_t)(
    size_t num_threads,     /* number of threads that the device should schedule work on, details are device specific */
    NN_API_STATUS *status   /* NN_API_STATUS_OK on success */
    );

/* Delete device and free its resources
    Returns: NN_API_STATUS_OK on success
*/
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_primitives_delete_device_t)(
    nn_device_t *device     /* the device handle */
    );

/* Delete internal storage and free its resources
    Returns: NN_API_STATUS_OK on success
*/
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_primitives_delete_opaque_data_t)(
    nn_opaque_data_t *opaque_data   /* internal storage handle */
    );

/* Delete primitive handle and free its resources
Returns: NN_API_STATUS_OK on success
*/
typedef NN_API_STATUS(NN_API_CALL_CONVENTION *nn_primitives_delete_primitive_t)(
    nn_primitive_handle_t primitive   /* primitive handle */
    );

/* Delete event and free its resources
    Returns: NN_API_STATUS_OK on success
*/
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_primitives_delete_event_t)(
    nn_event_t event        /* event handle */
    );

/* Wait until dependencies are ready
    Returns: NN_API_STATUS_OK on success
*/
typedef NN_API_STATUS(NN_API_CALL_CONVENTION *nn_primitives_wait_t)(
    size_t dependencies_count, /* size of dependencies array */
    nn_event_t *dependencies   /* dependencies which must be in completed state before function exits */
    );

/* Create internal data storage
    Returns: nn_opaque_data_t handle or NULL on failure
*/
typedef nn_opaque_data_t *(NN_API_CALL_CONVENTION *nn_primitives_create_opaque_data_t)(
    nn_primitive_handle_t handle, /* primitive handle */
    uint32_t flags,               /* optional flags - e.g. to request also allocation of buffer for deltas */
    NN_API_STATUS *status         /* set to NN_API_STATUS_OK on success */
    );

/* Create vector of internal data storage handles.
    Returns: void
*/
typedef void (NN_API_CALL_CONVENTION *nn_primitives_create_opaque_data_array_t)(
    nn_primitive_handle_t handle,      /* primitive handle */
    const size_t storage_count,        /* number of handles to create */
    nn_opaque_data_t *storage_array[], /* array of pointers to create, must hold storage_count pointers */
    uint32_t flags,                    /* optional flags - e.g. to request also allocation of buffer for deltas */
    NN_API_STATUS *status              /* set to NN_API_STATUS_OK on success */
    );

/* Create vector of internal data storage handles that represents views into one larger buffer. Views divide
   original storage in equal parts.
    Returns: nn_opaque_data_t* with handle to the large storage
*/
typedef nn_opaque_data_t *(NN_API_CALL_CONVENTION *nn_primitives_create_opaque_data_and_view_array_t)(
    nn_primitive_handle_t handle,   /* primitive handle */
    const size_t view_count,        /* number of handles to create */
    nn_opaque_data_t *view_array[], /* array of pointers to create, must hold storage_count pointers */
    NN_API_STATUS *status           /* set to NN_API_STATUS_OK on success */
    );

/* Create internal data storage and populate it with data from public data storage
    Returns: nn_opaque_data_t handle or NULL on failure
*/
typedef nn_opaque_data_t *(NN_API_CALL_CONVENTION *nn_primitives_create_opaque_data_from_data_t)(
    nn_primitive_handle_t handle, /* primitive handle */
    size_t data_index,            /* optional index in primitive storage lists */
    const nn_data_t *source,      /* source data in public data storage */
    NN_API_STATUS *status         /* set to NN_API_STATUS_OK on success */
    );

/* Create vector of internal data storage handles that represents views into an original larger buffer. Views divide
   original storage in equal parts.
    Returns: NN_API_STATUS_OK on success
*/
typedef NN_API_STATUS(NN_API_CALL_CONVENTION *nn_primitives_split_opaque_data_t)(
    const size_t part_count,        /* number of partitions to create */
    nn_opaque_data_t *part_array[], /* array of pointers to create, must hold partition_count pointers */
    nn_opaque_data_t *source        /* source data in public data storage */
    );

/* Create vector of internal data storage handles that represents views into an original larger buffer. Views divide
   original storage in equal parts.
    Returns: NN_API_STATUS_OK on success
*/
typedef NN_API_STATUS(NN_API_CALL_CONVENTION *nn_primitives_merge_opaque_data_t)(
    nn_opaque_data_t **destination,  /* source data in public data storage */
    size_t source_count,             /* number of partitions to merge */
    nn_opaque_data_t *source_array[] /* array of source handles, buffers must be compatible */
    );

/* Validates internal data storage compatibility for some usage
    Returns: non-zero if opaque_data is compatible
*/
typedef int(NN_API_CALL_CONVENTION *nn_primitives_validate_opaque_data_t)(
    nn_primitive_handle_t handle, /* primitive handle */
    size_t data_index,            /* optional index in primitive storage lists */
    nn_opaque_data_t *opaque_data /* internal data storage handle to validate */
    );

/* Asynchronously copy data from public data storage into internal data storage
Returns event handle to use for setting dependencies or to wait for task completion
*/
typedef nn_event_t(NN_API_CALL_CONVENTION *nn_primitives_copy_opaque_data_from_data_async_t)(
    nn_device_t *device,
    nn_opaque_data_t *destination, /* public data storage to copy data into */
    nn_data_t *source,             /* internal data storage to copy from */
    size_t dependency_count,       /* size of dependencies array */
    nn_event_t dependency_array[], /* array of nn_event_t objects for tasks that need to be
                                   completed before the copy is started */
    NN_API_STATUS *status          /* set to NN_API_STATUS_OK on scheduling success */
    );

/* Asynchronously copy data from internal data storage into a public data storage
    Returns event handle to use for setting dependencies or to wait for task completion
*/
typedef nn_event_t(NN_API_CALL_CONVENTION *nn_primitives_copy_data_from_opaque_data_async_t)(
    nn_device_t *device,
    nn_data_t *destination,        /* public data storage to copy data into */
    nn_opaque_data_t *source,      /* internal data storage to copy from */
    size_t dependency_count,       /* size of dependencies array */
    nn_event_t dependency_array[], /* array of nn_event_t objects for tasks that need to be
                                   completed before the copy is started */
    NN_API_STATUS *status          /* set to NN_API_STATUS_OK on scheduling success */
    );

/* Asynchronously update parameters based on deltas calculated during backward pass
    Returns event handle to use for setting dependencies or to wait for task completion
*/
typedef nn_event_t(NN_API_CALL_CONVENTION *nn_primitives_update_parameters_async_t)(
    nn_device_t *device,
    size_t parameters_count,              /* size of parameter array */
    nn_opaque_data_t *parameter_array[],  /* internal data storage with parameters to update */
    float learning_rate,                  /* learning rate */
    size_t dependency_count,              /* size of dependencies array */
    nn_event_t dependency_array[],        /* array of nn_event_t objects for tasks that need to be
                                             completed before the copy is started */
    NN_API_STATUS *status                 /* set to NN_API_STATUS_OK on scheduling success */
    );

/* Create internal data storage that is a view into an 3d window of original internal data storage
    Returns: nn_opaque_data_t handle or NULL on failure
*/
typedef nn_opaque_data_t *(NN_API_CALL_CONVENTION *nn_primitives_create_view_3d_t)(
    nn_opaque_data_t *source, /* source data in private data storage */
    size_t start_x,           /* first included X coordinate */
    size_t start_y,           /* first included Y coordinate */
    size_t start_z,           /* first included Z coordinate */
    size_t end_x,             /* last included X coordinate */
    size_t end_y,             /* last included Y coordinate */
    size_t end_z,             /* last included Z coordinate */
    NN_API_STATUS *status     /* set to NN_API_STATUS_OK on success */
    );

typedef nn_event_t(NN_API_CALL_CONVENTION *nn_primitives_execute_async_t)(
    nn_primitive_handle_t handle,              /* primitive handle */
    size_t input_count,                        /* */
    nn_opaque_data_t *const input_array[],     /* internal data storage with inputs */
    size_t parameter_count,                    /* */
    nn_opaque_data_t *const parameter_array[], /* internal data storage with weights */
    size_t output_count,                       /* */
    nn_opaque_data_t *output_array[],          /* internal data storage to store outputs */
    size_t dependency_count,                   /* size of dependencies array */
    nn_event_t dependency_array[],             /* array of nn_event_t objects for tasks that need to be
                                                 completed before the execution is started */
    NN_API_STATUS *status                      /* set to NN_API_STATUS_OK on scheduling success */
    );

typedef nn_event_t(NN_API_CALL_CONVENTION *nn_primitives_execute_backward_async_t)(
    nn_primitive_handle_t handle,              /* primitive handle */
    size_t input_count,                        /* */
    nn_opaque_data_t *input_array[],           /* internal data storage with inputs */
    size_t parameter_count,                    /* */
    nn_opaque_data_t *const parameter_array[], /* internal data storage with weights */
    size_t output_count,                       /* */
    nn_opaque_data_t *const output_array[],    /* internal data storage to store outputs */
    size_t dependency_count,                   /* size of dependencies array */
    nn_event_t dependency_array[],             /* array of nn_event_t objects for tasks that need to be
                                                 completed before the execution is started */
    NN_API_STATUS *status                      /* set to NN_API_STATUS_OK on scheduling success */
    );

typedef nn_event_t(NN_API_CALL_CONVENTION *nn_primitives_execute_backward_parameter_async_t)(
    nn_primitive_handle_t handle,              /* primitive handle */
    size_t data_index,                         /* index in primitive storage lists */
    size_t input_count,                        /* */
    nn_opaque_data_t *const input_array[],     /* internal data storage with inputs */
    size_t parameter_count,                    /* */
    nn_opaque_data_t *parameter_array[],       /* internal data storage with weights */
    size_t output_count,                       /* */
    nn_opaque_data_t *const output_array[],    /* internal data storage to store outputs */
    size_t dependency_count,                   /* size of dependencies array */
    nn_event_t dependency_array[],             /* array of nn_event_t objects for tasks that need to be
                                                 completed before the execution is started */
    NN_API_STATUS *status                      /* set to NN_API_STATUS_OK on scheduling success */
    );

typedef struct {
    struct {
        size_t left;
        size_t right;
        size_t top;
        size_t bottom;
    } output_padding;
    enum {
        NN_PRIMITIVES_CONVOLUTION_HINTS_OUTPUT_LAYOUT_3D,
        NN_PRIMITIVES_CONVOLUTION_HINTS_OUTPUT_LAYOUT_1D
    } output_layout;
    struct {
        size_t accumulator;
        size_t output;
    } fixed_point_fraction_bits;
} nn_primitives_convolution_hints_t;

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
    Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_convolution_create_t)(
    nn_device_t *device,    /* IDLF device handle */
    size_t kernel_w,        /* kernel width */
    size_t kernel_h,        /* kernel height */
    size_t num_input,       /* number of input feature maps */
    size_t num_output,      /* number of output feature maps */
    size_t output_w,        /* output width */
    size_t output_h,        /* output height */
    size_t center_offset_x, /* horizontal offset of kernel's center point w/ relation to top left corner */
    size_t center_offset_y, /* vertical offset of kernel's center point w/ relation to top left corner */
    size_t stride_x,        /* horizontal stride */
    size_t stride_y,        /* vertical stride */
    const nn_argument_activation_t *activation, /* struct parameterizing optional activation function */
    size_t batch_size,                          /* size of input batch */
    const nn_primitives_convolution_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

typedef struct {
    struct {
        size_t left;
        size_t right;
        size_t top;
        size_t bottom;
    } output_padding;
    enum {
        NN_PRIMITIVES_CONVOLUTION_HINTS_OUTPUT_LAYOUT_3D,
        NN_PRIMITIVES_CONVOLUTION_HINTS_OUTPUT_LAYOUT_1D
    } output_layout;
    struct {
        size_t input_output;
    } fixed_point_fraction_bits;
} nn_primitives_pooling_hints_t;

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
    Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_pooling_create_t)(
    nn_device_t *device,          /* IDLF device handle */
    NN_POOLING_MODE pooling_mode, /* pooling mode (e.g. MAX pooling) */
    size_t pool_size_x,           /* pooling kernel width */
    size_t pool_size_y,           /* pooling kernel height */
    size_t pool_stride_x,         /* pooling horizontal stride */
    size_t pool_stride_y,         /* pooling vertical stride */
    size_t num_feature_maps,      /* number of input/output feature maps */
    size_t output_w,              /* output width */
    size_t output_h,              /* output height */
    size_t batch_size,            /* size of input batch */
    const nn_primitives_pooling_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

typedef nn_primitives_convolution_hints_t nn_primitives_convolution_pooling_hints_t;

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
    Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_convolution_pooling_create_t)(
    nn_device_t *device,    /* IDLF device handle */
    size_t kernel_w,        /* convolution kernel width */
    size_t kernel_h,        /* convolution kernel height */
    size_t num_input,       /* number of input feature maps */
    size_t num_output,      /* number of output feature maps */
    size_t output_w,        /* output width */
    size_t output_h,        /* output height */
    size_t center_offset_x, /* horizontal offset of kernel's center point w/ relation to top left corner */
    size_t center_offset_y, /* vertical offset of kernel's center point w/ relation to top left corner */
    size_t stride_x,        /* convolution horizontal stride */
    size_t stride_y,        /* convolution vertical stride */
    const nn_argument_activation_t *activation, /* struct parameterizing optional activation function */
    size_t batch_size,                          /* size of input batch */
    size_t pooling_kernel_w,                    /* width of pooling kernel */
    size_t pooling_kernel_h,                    /* height of pooling kernel */
    size_t pooling_stride_x,                    /* horizontal pooling stride */
    size_t pooling_stride_y,                    /* vertical pooling stride */
    const nn_primitives_convolution_pooling_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

typedef struct {
    struct {
        size_t accumulator;
        size_t output;
    } fixed_point_fraction_bits;
} nn_primitives_fully_connected_hints_t;

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
    Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_fully_connected_create_t)(
    nn_device_t *device,                        /* IDLF device handle */
    size_t num_input,                           /* number of input feature maps */
    size_t num_output,                          /* number of output feature maps */
    const nn_argument_activation_t *activation, /* struct parameterizing optional activation function */
    size_t batch_size,                          /* size of input batch */
    const nn_primitives_fully_connected_hints_t *hints,
    NN_API_STATUS *status                       /* NN_API_STATUS_OK on success */
    );

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
    Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_convert_zxyn_nx_create_t)(
    nn_device_t *device,  /* IDLF device handle */
    size_t input_size_x,  /* input width */
    size_t input_size_y,  /* input height */
    size_t input_size_z,  /* number of input feature maps */
    size_t batch_size,    /* size of input batch */
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
    Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_convert_z_block_xyz_x2nx_create_t)(
    nn_device_t *device,  /* IDLF device handle */
    size_t input_size_x,  /* input width */
    size_t input_size_y,  /* input height */
    size_t input_size_z,  /* number of input feature maps */
    size_t batch_size,    /* size of input batch */
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
    Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_convert_z2nz_n8xn_create_t)(
    nn_device_t *device,  /* IDLF device handle */
    size_t input_size_x,  /* number of input feature maps */
    size_t batch_size,    /* size of input batch */
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );
/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
    Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_arithmetic_create_t)(
    nn_device_t *device,                        /* IDLF device handle */
    size_t image_size_x,                        /* image width */
    size_t image_size_y,                        /* image height */
    size_t image_size_z,                        /* number of feature maps */
    NN_ARITHMETIC_FUNCTION arithmetic_function, /* type of arithmetic operation */
    size_t batch_size,                          /* size of input batch */
    NN_API_STATUS *status                       /* NN_API_STATUS_OK on success */
    );

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
    Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_normalization_elementwise_linear_create_t)(
    nn_device_t *device,  /* IDLF device handle */
    float alpha,          /* multiplier */
    float beta,           /* offset */
    size_t image_size_x,  /* image width */
    size_t image_size_y,  /* image height */
    size_t image_size_z,  /* number of feature maps */
    size_t batch_size,    /* size of input batch */
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

typedef nn_primitives_convolution_hints_t nn_primitives_normalization_response_across_maps_hints_t;

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
    Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_normalization_response_across_maps_create_t)(
    nn_device_t *device,  /* IDLF device handle */
    float alpha,          /* sum scale */
    float beta,           /* sum power */
    uint32_t k,           /* square sum weight */
    uint32_t n,           /* size of moving window on the feature maps */
    size_t image_size_x,  /* image width */
    size_t image_size_y,  /* image height */
    size_t image_size_z,  /* number of feature maps */
    size_t batch_size,    /* size of input batch */
    const nn_primitives_normalization_response_across_maps_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

typedef nn_primitives_convolution_hints_t nn_primitives_softmax_hints_t;

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_softmax_create_t)(
    nn_device_t *device,  /* IDLF device handle */
    size_t num_features,  /* number of feature maps */
    size_t batch_size,    /* size of input batch */
    const nn_primitives_softmax_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

typedef nn_primitives_convolution_hints_t nn_primitives_convert_float_to_i16_hints_t;

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_convert_float_to_i16_create_t)(
    nn_device_t *device,  /* IDLF device handle */
    size_t image_size_x,  /* image width */
    size_t image_size_y,  /* image height */
    size_t image_size_z,  /* number of feature maps */
    size_t batch_size,    /* size of input batch */
    const nn_primitives_convert_float_to_i16_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

typedef nn_primitives_convolution_hints_t nn_primitives_relu_hints_t;

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_relu_create_t)(
    nn_device_t *device,  /* IDLF device handle */
    size_t image_size_x,  /* image width */
    size_t image_size_y,  /* image height */
    size_t image_size_z,  /* number of feature maps */
    size_t batch_size,    /* size of input batch */
    const nn_primitives_relu_hints_t *hints,
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_loss_function_create_t)(
    nn_device_t *device,  /* IDLF device handle */
    NN_LOSS_FUNCTION function, 
    size_t image_size_x,  /* image width */
    size_t image_size_y,  /* image height */
    size_t image_size_z,  /* number of feature maps */
    size_t batch_size,    /* size of input batch */
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

/* Creates primitive handle, decides on evaluation strategy, internal data layouts, etc.
Returns: primitive handle or NULL on failure
*/
typedef nn_primitive_handle_t(NN_API_CALL_CONVENTION *nn_primitives_dropout_create_t)(
    nn_device_t *device,  /* IDLF device handle */
    size_t image_size_x,  /* image width */
    size_t image_size_y,  /* image height */
    size_t image_size_z,  /* number of feature maps */
    size_t batch_size,    /* size of input batch */
    float drop_rate,      /* drop rate */
    NN_API_STATUS *status /* NN_API_STATUS_OK on success */
    );

typedef struct {
    /* Creates nn_device_t handle which can be then used to instantiate primitives */
    nn_primitives_create_device_with_thread_count_t create_device_with_thread_count;

    /* Deletes nn_device_t to free resources */
    nn_primitives_delete_device_t delete_device;

    /* Delete internal storage handle and free its resources */
    nn_primitives_delete_opaque_data_t delete_opaque_data;

    /* Delete primitive handle and free its resources */
    nn_primitives_delete_primitive_t delete_primitive;

    /* Delete event handle and free its resources */
    nn_primitives_delete_event_t delete_event;

    /* Wait until dependencies are ready */
    nn_primitives_wait_t wait;


    /* storage allocation methods ************************************************************************************/

    /* allocate input buffer for a primitive */
    nn_primitives_create_opaque_data_array_t create_inputs;

    /* allocate output buffer for a primitive */
    nn_primitives_create_opaque_data_array_t create_outputs;

    /* allocate vector of buffers for primitive parameters, see primitive description for details */
    nn_primitives_create_opaque_data_array_t create_parameters;


    /* storage transfer methods **************************************************************************************/

    /* Copy data to internal storage */
    nn_primitives_copy_opaque_data_from_data_async_t copy_to_opaque_async;

    /* Copy data from internal storage */
    nn_primitives_copy_data_from_opaque_data_async_t copy_from_opaque_async;

    /* Copy deltas to internal storage */
    nn_primitives_copy_opaque_data_from_data_async_t copy_delta_to_opaque_async;

    /* Copy deltas from internal storage */
    nn_primitives_copy_data_from_opaque_data_async_t copy_delta_from_opaque_async;

    /* Update parametrs */
    nn_primitives_update_parameters_async_t update_parameters_async;

    /* storage manipulation methods **********************************************************************************/

    /* create opaque handle directly from public data store (requires exact memory layout) */
    nn_primitives_create_opaque_data_from_data_t map_input;

    /* validate input buffer is correct for the primitive */
    nn_primitives_validate_opaque_data_t validate_input;

    /* can be used to transform internal output storage of another primitive to use as input */
    nn_primitives_create_view_3d_t create_view_3d;

    /* splits internal container along Z (features) axis and creates N views into its parts */
    nn_primitives_split_opaque_data_t split_z;

    /* creates new storage that merges supplied storage along Z (features) axis and creates a super-view over them */
    nn_primitives_merge_opaque_data_t merge_z;

    /* execution methods *********************************************************************************************/

    /* forward pass */
    nn_primitives_execute_async_t forward_async;
    
    /* backward pass */
    nn_primitives_execute_backward_async_t backward_async;
    nn_primitives_execute_backward_parameter_async_t backward_parameter_async;

    struct {
        /* convolution

        nn_data_t input/output format: ZXYN

        parameters:
            [0] - weights handle
                    weights nn_data_t dimensions: [kernel_width][kernel_height][number_of_input_feature_maps][number_of_output_feature_maps]
            [1] - bias handle
                    bias nn_data_t dimensions: [number_of_output_feature_maps]
        */
        nn_primitives_convolution_create_t convolution_f32;
        nn_primitives_convolution_create_t convolution_i16;

        /* pooling

        nn_data_t input/output format: ZXYN
        */
        nn_primitives_pooling_create_t pooling_f32;
        nn_primitives_pooling_create_t pooling_i16;

        /* convolution with pooling

        nn_data_t input/output format: ZXYN

        parameters:
            [0] - weights handle
                    weights nn_data_t dimensions: [kernel_width][kernel_height][number_of_input_feature_maps][number_of_output_feature_maps]
            [1] - bias handle
                    bias nn_data_t dimensions: [number_of_output_feature_maps]
        */
        nn_primitives_convolution_pooling_create_t convolution_pooling_f32;
        nn_primitives_convolution_pooling_create_t convolution_pooling_i16;

        /* fully connected

        nn_data_t input format: NXYZ or NX
        nn_data_t output format: NX

        parameters:
            [0] - weights handle
                    weights nn_data_t dimensions:
                        [number_of_input_feature_maps][number_of_output_feature_maps]
                        or
                        [input_width][input_height][number_of_input_feature_maps][number_of_output_feature_maps]
            [1] - bias handle
                    bias nn_data_t dimensions: [number_of_output_feature_maps]
        */
        nn_primitives_fully_connected_create_t fully_connected_f32;
        nn_primitives_fully_connected_create_t fully_connected_i16;
        nn_primitives_fully_connected_create_t fully_connected_i16_i32;

        /* convert layout from zxyn to nz
            use between convolution_f32 and fully_connected_f32

        nn_data_t input format: ZXYN
        nn_data_t output format: NX
        */
        nn_primitives_convert_zxyn_nx_create_t convert_zxyn_nx_f32;

        /* convert layout from z_block_xyz to z2nz
        use between convolution_i16 and fully_connected_i16

        nn_data_t input format: ZXYN
        nn_data_t output format: NX
        */
        nn_primitives_convert_z_block_xyz_x2nx_create_t convert_z_block_xyz_x2nx_i16;

        /* convert layout from z_block_xyz to n8xn
        use between fully_connected_i16_i32 and softmax_i32 when batch_size != 1

        nn_data_t input format: NX
        nn_data_t output format: NX
        */
        nn_primitives_convert_z2nz_n8xn_create_t convert_z2nz_n8xn_i32;

        /* arithmetic operation

        nn_data_t input/output format: ZXYN

        parameters:
            [0] - factor handle
                    factor nn_data_t dimensions: [input_width][input_height][number_of_input_feature_maps]
        */
        nn_primitives_arithmetic_create_t arithmetic_f32;

        /* element wise linear normalization

        out = alpha * in + beta

        nn_data_t input/output format: ZXYN
        */
        nn_primitives_normalization_elementwise_linear_create_t normalization_elementwise_linear_f32;

        /* response normalization across maps

        nn_data_t input/output format: ZXYN
        */
        nn_primitives_normalization_response_across_maps_create_t normalization_response_across_maps_f32;

        /* response normalization across maps

        nn_data_t input/output format: ZXYN
        */
        nn_primitives_normalization_response_across_maps_create_t normalization_response_across_maps_i16;

        /* softmax

        nn_data_t input/output format: NX
        */
        nn_primitives_softmax_create_t softmax_f32;

        /* softmax

        nn_data_t input/output format: NX
        */
        nn_primitives_softmax_create_t softmax_i32;

        /* convert float to int16

        nn_data_t input/output format: ZXYN
        */
        nn_primitives_convert_float_to_i16_create_t convert_float_to_i16;

        /* rectified linear unit activation

        nn_data_t input/output format: ZXYN
        */
        nn_primitives_relu_create_t relu_f32;

        /* loss function
        */
        nn_primitives_loss_function_create_t loss_f32;

        /* dropout
        */
        nn_primitives_dropout_create_t dropout_f32;
    } create_handle;
} nn_primitives_0_t;