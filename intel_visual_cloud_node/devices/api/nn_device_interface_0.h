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
#include "nn_types_0.h"

/*
This file contains API for interface version 0.

Dictionary:

<workflow item>
    Node that performs calculations. It has its arguments (parameters) and inputs. 

<workflow item:argument>s
    Arguments are fixed state, that depend on type of <workflow item>.
    Examples:
        for pooling: padding mode, 2D stride, 2D pooling size & pooling mode
        for full connectivity: biases, weights & activation function with parameters
        for softmax: none

<workflow item:input>s
    Links to other <workflow item>s that create data that current <workflow item> uses.

<workflow>
    Container of <workflow item>s that define data flow graph. Inputs & outputs 
    of this graph become inputs & outputs of compiled <workload>.
    This is why <workflow> itself is a "calculation described as data flow".
    Its data structure allows forward (in->out) & backward (out->in) graph traversal.

<workload>
    Device specific representation, created by compiling <workflow>. 
    Contains count and list of formats for inputs & outputs and reference to
    device that compiled it.
    Other parts of structure are device-specific, and not user-accessable.



From creation to execution:

[create workflow]
    User creates empty <workflow>.
    Workflow has number of inputs & outputs created at compile time.
    Will be filled with <workflow items>.

[fill workflow with items]
    User creates <workflow item>s. Items are connected to each other creating
    a data flow graph. 
    Special "input" and "output" <workflow items>s represent input from and
    output to <workflow>.
    Example of <workflow> filled with <workflow items>:
              _____________________________________________
             |                                             |
      input =] [input]--[convolution]--[pooling]--[output] [= output
     buffer  |_____________________________________________|  buffer

[query workflow metrics]
    User queries particular device for metrics associated with <workflow>.
    Result of the query is an array of entries. Each entry contains:
    - list containing one format per each <workflow> input
    - list containing one format per each <workflow> output
    - batch size (number of input data sets processed in parallel)
    - estimate of time needed to process <workload>
    - estimate of energy required to process <workload>
    This query alows user to choose compilation variant that has best
    metrics (performance, perf/watt) or a certain input/output format
    (what allows chaining workloads between different devices).
    Number of metrics will be extended in the future.

[compilation]
    User submits <workflow> together with batch size, input & output formats
    to device (see previous point). In result <workload>, a device-specific,
    optimized representation of calcuations and data flow is created.
    It's an opaque data structure.

[execute workload]
    User pushes <workload> execution request through the device interface, passing sets of input
    and output buffers along with the <workload> pointer. Execution is done asynchronously.
    User specifies condition variable to be updated with value of execution status.

*/

#if !defined NN_NODE_RUNTIME
#   include<stdint.h>
#endif

/* enumerations & constants **********************************************************************/

/* types of work items
   Adding new work item requires updating this enumeration.
   note: I16QN and I32QN are fixed point data formats stored as either int16 or int32 and parametrized (in layer or
   activation parameters) with number of fractional bits. */
typedef enum {
    /* simple work items */
    NN_WORK_ITEM_TYPE_INPUT,              /* contains input buffer of workflow container */
    NN_WORK_ITEM_TYPE_OUTPUT,             /* contains output buffer of workflow container */
    NN_WORK_ITEM_TYPE_VIEW,               /* returns view of input data */
    NN_WORK_ITEM_TYPE_LOCAL_CONNECTIVITY, /* convolution without weight sharing */
    NN_WORK_ITEM_TYPE_CONVOLUTION,        /* convolution with weight sharing */
    NN_WORK_ITEM_TYPE_FULLY_CONNECTED,    /* fully connected layer */
    NN_WORK_ITEM_TYPE_POOLING,            /* pooling */
    NN_WORK_ITEM_TYPE_NORMALIZATION,      /* signal normalization */
    NN_WORK_ITEM_TYPE_SOFTMAX,            /* normalizes input */
    NN_WORK_ITEM_TYPE_MERGE,              /* merging layer */
    NN_WORK_ITEM_TYPE_ARITHMETIC,         /* arithmetic operations with external data*/

    /* simple work items in fixed point */
    NN_WORK_ITEM_TYPE_CONVOLUTION_INT16_FIXEDPOINT,
    NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I16QN, /* fixed point layer with int16 input and int16 output, this
                                                              layer supports ReLU and Logistic activations */
    NN_WORK_ITEM_TYPE_FULLY_CONNECTED_FORWARD_I16QN_I32QN, /* fixed point layer with int16 input and int32 output, this
                                                              layer supports None activation */
    NN_WORK_ITEM_TYPE_SOFTMAX_FIXEDPOINT,
    NN_WORK_ITEM_TYPE_CONVERT_FLOAT_TO_INT16_FIXEDPOINT,

    /* only max pooling */
    NN_WORK_ITEM_TYPE_MAX_POOLING_INT16_FIXEDPOINT,

    /* lrn normalization */
    NN_WORK_ITEM_TYPE_NORMALIZATION_RESPONSE_ACROSS_MAPS_FORWARD_I16QN,

    /* complex/merged work items */

    /* generic convolution, with non-overlapping max pooling on 2x2 area with 2x2 stride
       result of merging of [convolution with stride 2x2 and relu activation] with [non-overlapping 2x2 pooling] */
    NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2,
    /* ...the same as above, but for fixed point */
    NN_WORK_ITEM_TYPE_CONVOLUTION_POOLING_MAX_2x2_STRIDE_2x2_INT16_FIXEDPOINT,

    /* only for internal use of the device */
    NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT,
    NN_WORK_ITEM_TYPE_LAST = NN_WORK_ITEM_TYPE_CONVERT_DATA_LAYOUT
} NN_WORK_ITEM_TYPE;


/* description of data format
   Types & (implicit)dimensionality fo data produced by <workflow item>. */
typedef enum {
    NN_DATA_FORMAT_INVALID  = 0,    /* invalid value (for zero-initialized structures) */
    NN_DATA_FORMAT_1D       = 1,    /* 1D data, with unspecified data format */
    NN_DATA_FORMAT_2D       = 2,    /* 2D data, with unspecified data format */
    NN_DATA_FORMAT_3D       = 3,    /* 3D data, with unspecified data format */
    NN_DATA_FORMAT_LAST     = NN_DATA_FORMAT_3D
} NN_DATA_FORMAT;

/* padding modes
   Determines what result are returned when sampling data outside of view. */
typedef enum {
    NN_PADDING_MODE_NONE = 0,           /* sampling out of view produces an error (validated at compile time) */
    NN_PADDING_MODE_ZERO,               /* data out of view is zero */
    NN_PADDING_MODE_EXTRAPOLATE,        /* data out of view is a extrapolation based on data within view */
    NN_PADDING_MODE_DATA_OR_ZERO,       /* data out of view is returned from parent containter if possible, otherwise zero */
    NN_PADDING_MODE_DATA_OR_EXTRAPOLATE,/* data out of view is returned from parent contrainer if possible, otherwise extrapolate */
    NN_PADDING_MODE_LAST = NN_PADDING_MODE_DATA_OR_EXTRAPOLATE
} NN_PADDING_MODE;


/* normalization functions
   What type of normalization is used in normalization <workflow item>. */
typedef enum {
    NN_NORMALIZATION_MODE_RESPONSE_SAME_MAP,    /* response normalized per map */
    NN_NORMALIZATION_MODE_RESPONSE_ACROSS_MAPS, /* response normalized across maps */
    NN_NORMALIZATION_MODE_CONTRAST,             /* contrast normalization */
    NN_NORMALIZATION_MODE_LINEAR_SINGLE,        /* linear single input pixel normalization */
    NN_NORMALIZATION_MODE_LAST = NN_NORMALIZATION_MODE_LINEAR_SINGLE
} NN_NORMALIZATION_MODE;


/* parameters for parameter_get_function
   Current unused. */
typedef enum {
    NN_PARAMETER_ = 0,
    NN_PARAMETER_LAST = NN_PARAMETER_
} NN_PARAMETER;


/* types of data provided as input/output to/from workflow.
   Enumeration defines data format but not resolution.
   Will be changed/extended. */
typedef enum {
    NN_WORKLOAD_DATA_TYPE_F32_1D,        /* nn_data_t, 1D float32: single 1D signal */
    NN_WORKLOAD_DATA_TYPE_F32_1D_BATCH,  /* nn_data_t, 2D float32: sequence of 1D signals */
    NN_WORKLOAD_DATA_TYPE_F32_2D,        /* nn_data_t, 2D float32: 2D signal (X, Y) */
    NN_WORKLOAD_DATA_TYPE_F32_2D_BATCH,  /* nn_data_t, 3D float32: sequence of 2D signals */
    NN_WORKLOAD_DATA_TYPE_F32_3D,        /* nn_data_t, 3D float32: 3D signal (X, Y, Z) */
    NN_WORKLOAD_DATA_TYPE_F32_3D_BATCH,  /* nn_data_t, 4D float32: sequence of 3D signals */
    NN_WORKLOAD_DATA_TYPE_F32_ZXY,       /* nn_data_t, 3D float32: 3D signal (Z, X, Y) */
    NN_WORKLOAD_DATA_TYPE_F32_ZXY_BATCH, /* nn_data_t, 3D float32: 3D signal (Z, X, Y) */

    NN_WORKLOAD_DATA_TYPE_I16_1D,        /* nn_data_t, 1D int16: single 1D signal */
    NN_WORKLOAD_DATA_TYPE_I16_1D_BATCH,  /* nn_data_t, 2D int16: sequence of 1D signals */
    NN_WORKLOAD_DATA_TYPE_I16_3D,        /* nn_data_t, 3D int16: 3D signal (X, Y, Z) */
    NN_WORKLOAD_DATA_TYPE_I16_3D_BATCH,  /* nn_data_t, 4D int16: sequence of 3D signals */
    NN_WORKLOAD_DATA_TYPE_I16_ZXY,       /* nn_data_t, 3D int16: 3D signal (Z, X, Y) */
    NN_WORKLOAD_DATA_TYPE_I16_ZXY_BATCH, /* nn_data_t, 3D int16: 3D signal (Z, X, Y) */

    NN_WORKLOAD_DATA_TYPE_I32_1D,        /* nn_data_t, 1D int32: single 1D signal */
    NN_WORKLOAD_DATA_TYPE_I32_1D_BATCH,  /* nn_data_t, 2D int32: sequence of 1D signals */

    NN_WORKLOAD_DATA_TYPE_LAST = NN_WORKLOAD_DATA_TYPE_I32_1D_BATCH
} NN_WORKLOAD_DATA_TYPE;


/* data structures *******************************************************************************/

/* data format of output of work item.
   Contains number of dimensions & size.
   Created as <id ; union-of-parameters> pair to allow addition of new formats. */
typedef struct nn_output_format_1d { uint32_t size[1]; } nn_output_format_1d_t;
typedef struct nn_output_format_2d { uint32_t size[2]; } nn_output_format_2d_t;
typedef struct nn_output_format_3d { uint32_t size[3]; } nn_output_format_3d_t;
typedef struct nn_output_format {
    NN_DATA_FORMAT format;
    union {
        nn_output_format_1d_t     format_1d;
        nn_output_format_2d_t     format_2d;
        nn_output_format_3d_t     format_3d;
    };
} nn_output_format_t;


/* container for activation function in fixed point
   Contains type and parameters per type (if they exist). */
typedef struct nn_argument_activation_fixedpoint {
    nn_argument_activation_t  basic_arguments;
    struct 
    {
        uint16_t accumulator;                   /* number of fractional bits in the accumulator */
        uint16_t output;                        /* number of fractional bits in the output value */
    }  fractions;
} nn_argument_activation_fixedpoint_t;


/* container for normalization function arguments */
typedef struct nn_argument_normalization {
    NN_NORMALIZATION_MODE   mode;               /* type of normalization */
    float                   alpha;              /* alpha in onrmalization equations */
    float                   beta;               /* beta in normalization equations */
    uint32_t                k;                  /* k in normalization equations */
    uint32_t                n;                  /* n in normalization equations */
    uint32_t                size;               /* normalization size */
} nn_argument_normalization_t;


/* symbolic input
   Contains buffer passed to a workload as an input.
   Index identifies input when workload has more than 1 input. */
typedef struct nn_arguments_input {
    uint32_t                    index;          /* input index */
} nn_arguments_input_t;


/* symbolic output 
   Contains buffer passed to a workload as an output.
   Index identifies input when workload has more than 1 output. */
typedef struct nn_arguments_output {
    uint32_t                    index;          /* output index */
} nn_arguments_output_t;


/* returns view of an input as an output
   Size of input view is determined from output buffer. Only origin is necessary. */
typedef struct nn_arguments_view {
    uint32_t                    origin[3];      /* left-up-front corner of view */
} nn_arguments_view_t;

/* arguments for merging layers*/
typedef struct nn_arguments_merge {
    uint16_t                    axis;           /* this param decides on which axis output will be merged. x = 0, y = 1, z = 2 */
} nn_arguments_merge_t;

/* arguments for arithmetic layers*/
typedef struct nn_arguments_arithmetic {

    nn_data_t                 *factor;         /* second argument of arithmetic operation:
                                                  subtrahend in subtraction, multiplier in the multiplication etc.*/
    NN_ARITHMETIC_FUNCTION     arithmetic_function;
} nn_arguments_arithmetic_t;


/* arguments for convolution layers (with shared weights) */
typedef struct nn_arguments_forward_convolution {
    NN_PADDING_MODE             padding;          /* padding mode */
    uint32_t                    center_offset[2]; /* offset of center point in convolution filter */
    uint32_t                    stride[2];        /* stride during filtering operation */
    nn_data_t                  *biases;           /* biases */
    nn_data_t                  *weights;          /* weights */
    nn_argument_activation_t    activation;       /* activation data */
} nn_arguments_forward_convolution_t;


/* arguments for fully connected layers */
typedef struct nn_arguments_forward_fully_connected {
    nn_data_t                  *biases;         /* biases for each neuron */
    nn_data_t                  *weights;        /* weights for each neuron */
    nn_argument_activation_t    activation;     /* activation data */
} nn_arguments_forward_fully_connected_t;


/* arguments for pooling layers */
typedef struct nn_arguments_forward_pooling {
    uint32_t                    stride[2];      /* stride during filtering operation */
    uint32_t                    size[2];        /* pooling area size */
    NN_POOLING_MODE             mode;           /* pooling mode */
} nn_arguments_forward_pooling_t;


/* there's no arguments for softmax layers */


/* arguments for normalization layers */
typedef struct nn_arguments_forward_normalization {
    nn_argument_normalization_t normalization;  /* normalization data */
} nn_arguments_forward_normalization_t;

/* arguments for normalization layers */
typedef struct nn_arguments_forward_normalization_response_across_maps_i16qn {
    float alpha;                                  /* alpha in normalization equations */
    float beta;                                   /* beta in normalization equations */
    uint32_t k;                                   /* k in normalization equations */
    uint32_t n;                                   /* normalization size */
    struct
    {
        uint16_t input;
        uint16_t output;
    }  fractions;
} nn_arguments_normalization_response_across_maps_forward_i16qn_t;

/* arguments for merged convolution and pooling layers (with shared weights) */
typedef struct nn_arguments_forward_convolution_pooling_max_2x2_stride_2x2 {
    nn_data_t                  *biases;           /* biases */
    nn_data_t                  *weights;          /* weights */
    nn_argument_activation_t    activation;       /* activation data */
    NN_PADDING_MODE             padding;          /* padding mode */
    uint32_t                    center_offset[2]; /* offset of center point in convolution filter */
    uint32_t                    stride[2];        /* stride during filtering operation */
} nn_arguments_forward_convolution_pooling_max_2x2_stride_2x2_t;


/* arguments for convolution fixed point layers */
typedef struct nn_arguments_forward_convolution_fixedpoint
{
    nn_data_t                  *biases;         /* biases */
    nn_data_t                  *weights;        /* weights */
    NN_PADDING_MODE             padding;         /* padding mode */
    uint32_t                    stride[2];       /* stride during filtering operation */
    nn_argument_activation_fixedpoint_t    activation;     /* activation data */
    uint32_t                    center_offset[2]; /* offset of center point in convolution filter */
} nn_arguments_forward_convolution_fixedpoint_t;


/* arguments for fully connected fixed point layers */
typedef struct nn_arguments_fully_connected_forward_i16qn_i16qn {
    nn_data_t                  *biases;         /* biases for each neuron */
    nn_data_t                  *weights;        /* weights for each neuron */
    nn_argument_activation_fixedpoint_t    activation;     /* activation data */
} nn_arguments_fully_connected_forward_i16qn_i16qn_t;


/* arguments for fully connected fixed point layers */
typedef struct nn_arguments_fully_connected_forward_i16qn_i32qn {
    nn_data_t                  *biases;         /* biases for each neuron */
    nn_data_t                  *weights;        /* weights for each neuron */
    nn_argument_activation_fixedpoint_t    activation;     /* activation data */
} nn_arguments_fully_connected_forward_i16qn_i32qn_t;


/* arguments for softmax layers fixed point */
typedef struct nn_arguments_forward_softmax_fixedpoint {
    int8_t                      input_fraction; /* number of fractional bits of the input values */
} nn_arguments_forward_softmax_fixedpoint_t;


/* arguments for convert float to fixed point layers */
typedef struct nn_arguments_forward_convert_float_to_int16_fixedpoint {
    int8_t                      output_fraction; /* number of fractional bits of the output values */
} nn_arguments_forward_convert_float_to_int16_fixedpoint_t;


/* arguments for merged convolution and pooling layers fixed point */
typedef struct nn_arguments_forward_merged_convolution_pooling_max_2x2_stride_2x2_fixedpoint
{
    nn_data_t                  *biases;         /* biases */
    nn_data_t                  *weights;        /* weights */
    uint32_t                    center_offset[2]; /* offset of center point in convolution filter */
    NN_PADDING_MODE             padding;          /* padding mode */
    nn_argument_activation_fixedpoint_t    activation;     /* activation data */
    uint32_t                    stride[2];  /* stride during filtering operation */
} nn_arguments_forward_merged_convolution_pooling_max_2x2_stride_2x2_fixedpoint_t;

/* arguments for merged convolution and pooling layers fixed point */
typedef struct nn_arguments_forward_pooling_fixedpoint
{
    uint32_t                    pool_size[2];    /* max-pooling size */
    uint32_t                    pool_stride[2];  /* max-pooling stride */
} nn_arguments_forward_pooling_fixedpoint_t;


/* workflow item is a representation of a single calculation (for example a single convolution).
   It contains asynchronous calculation state (created, in progress, finished, error) and all parameters 
   required for processing. */
typedef struct nn_workflow_item {
    /* following fields are set by the user */
    NN_WORK_ITEM_TYPE           type;           /* work item type */
    nn_output_format_t          output_format;  /* format & size of output */

    /* following fields are populated by creation function */
    uint32_t                    input_count;    /* count of inputs */
    struct nn_workflow_item   **input;          /* input array */

    /* union of packaged arguments for each layer type */
    union {
        /* arguments for simple NN layers */
        nn_arguments_input_t                                            input;
        nn_arguments_output_t                                           output;
        nn_arguments_view_t                                             view;
        nn_arguments_forward_convolution_t                              forward_convolution;
        nn_arguments_forward_fully_connected_t                          forward_fully_connected;
        nn_arguments_forward_pooling_t                                  forward_pooling;
        nn_arguments_forward_normalization_t                            forward_normalization;
        nn_arguments_merge_t                                            forward_merge;
        nn_arguments_arithmetic_t                                       forward_arithmetic;

        /* fixedpoint layers */
        nn_arguments_forward_convolution_fixedpoint_t                   forward_convolution_int16_fixedpoint;
        nn_arguments_fully_connected_forward_i16qn_i16qn_t              fully_connected_forward_i16qn_i16qn;
        nn_arguments_fully_connected_forward_i16qn_i32qn_t              fully_connected_forward_i16qn_i32qn;
        nn_arguments_forward_softmax_fixedpoint_t                       forward_softmax_fixedpoint;
        nn_arguments_forward_pooling_fixedpoint_t                       forward_pooling_fixedpoint;
        nn_arguments_normalization_response_across_maps_forward_i16qn_t normalization_response_across_maps_forward_i16qn;

        /* conversion layers */
        nn_arguments_forward_convert_float_to_int16_fixedpoint_t        forward_convert_float_to_int16_fixedpoint;

        /* arguments for for complex NN layers */
        nn_arguments_forward_convolution_pooling_max_2x2_stride_2x2_t   forward_convolution_pooling_max_2x2_stride_2x2;
        nn_arguments_forward_merged_convolution_pooling_max_2x2_stride_2x2_fixedpoint_t   forward_convolution_pooling_fixedpoint;
    } arguments;

    const char *name;                               /* optional work item name for profiling and debug */

    /* following fields are updated automatically */
    const uint32_t                      use_count;  /* number of work items that use result of current one */
    struct nn_workflow_item **const     use;        /* work items that use result of current one */
} nn_workflow_item_t;


/* workflow contains entire network topolgy
   Workflow is a device-independent representation of data flow and calculations defined by workflow items. */
typedef struct nn_workflow {
    const uint32_t          input_count;    /* number of inputs */
    const uint32_t          output_count;   /* number of outputs */
    nn_workflow_item_t    **input;          /* array of input workflow items */
    nn_workflow_item_t    **output;         /* array of output workflow items */
} nn_workflow_t;


/* workload is a compiled, device-specific representation of workflow
   This is "mostly" opaque data structure. It contains small-user accessible part that contains device
   pointer & data related to input & output for easier parameter validation.
   After that, there opaque (not accessible to user), device specific data structure containing
   compiled state & resources that are used at execution time.
   Warning: this structure cannot be copied by user. Reference & access it by pointer only.*/
typedef struct nn_workload {
    nn_device_t           *const device;        /* device this workload was compiled for  */
    const uint32_t               input_count;   /* count of inputs in this workload */
    const uint32_t               output_count;  /* count of outputs in this workload */
    NN_WORKLOAD_DATA_TYPE *const input_format;  /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE *const output_format; /* array containing formats of outputs */
    const uint32_t               batch;         /* batch size for this workload */
} nn_workload_t;


/* workflow metrics contains information about single compilation variant
   Those metrics currently include:
       - entry 0: time  needed to calulcate workload in nanoseconds
       - entry 1: power needed to calculate workload in nanojoules */
typedef struct nn_workflow_metrics {
    NN_WORKLOAD_DATA_TYPE *const input_format;  /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE *const output_format; /* array containing formats of outputs */
    const uint32_t      batch;                  /* batch size */
    uint64_t            metric[1];              /* table of metrics - will be overindexed */
} nn_workflow_metrics_t;


/* workflow metrics array */
typedef struct nn_workflow_metrics_array {
    const uint32_t      input_count;    /* count of inputs in this workload */
    const uint32_t      output_count;   /* count of outputs in this workload */
    nn_workflow_metrics_t **array;      /* array of variant entries */
} nn_workflow_metrics_array_t;


/* function pointers to API calls ****************************************************************/

/* create empty workflow */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workflow_create_function_t)(
    nn_workflow_t *        *workflow,       /* workflow to be created */
    uint32_t                input_count,    /* number of inputs in created workflow */
    uint32_t                output_count    /* number of outputs in created workflow */
    );


/* delete workflow */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workflow_delete_function_t)(
    nn_workflow_t          *workflow        /* workflow to delete */
    );


/* query workflow for metrics of compilation variants */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workflow_metrics_query_function_t)(
    nn_workflow_metrics_array_t **array,    /* resulting array of variants */
    nn_device_t            *device,         /* device context */
    nn_workflow_t          *workflow        /* workflow to be querried */
    );


/* delete array of workload metrics */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workflow_metrics_delete_function_t)(
    nn_workflow_metrics_array_t *array      /* array to delete */
    );


/* compile workflow into workload */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workflow_compile_function_t)(
    nn_workload_t         **workload,       /* resulting workload */
    nn_device_t            *device,         /* device context */
    nn_workflow_t          *workflow,       /* workflow to be compiled */
    NN_WORKLOAD_DATA_TYPE  *input_format,   /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE  *output_format,  /* array containing formats of outputs */
    uint32_t                batch           /* batch size for compilation */
    );


/* executes workload with given inputs & outputs */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workload_execute_function_t)(
    nn_workload_t          *workload,       /* workload to be started */
    void *                 *input,          /* pointer to array of pointers with input data;  format is in workload->input_format */
    void *                 *output,         /* pointer to array of pointers with output data; format is in workload->output_format */
    NN_API_STATUS          *status          /* asynchronous status to be set/updated */
    );


/* delete workload */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workload_delete_function_t)(
    nn_workload_t          *workload        /* workload to be deleted */
    );


/* create empty work item */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workflow_item_create_function_t)(
    nn_workflow_item_t    **workflow_item,  /* created workflow item */
    uint32_t                input_count,    /* count of inputs */
    nn_workflow_item_t    **input           /* pointer to array of inputs */
    );

/* query workflow for metrics of compilation variants */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workflow_metrics_query_function_t)(
    nn_workflow_metrics_array_t **array,    /* resulting array of variants */
    nn_device_t            *device,         /* device context */
    nn_workflow_t          *workflow        /* workflow to be querried */
    );

/* validate parameters of work_item for this particular device */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workflow_item_validate_function_t)(
    nn_device_t            *device,         /* target device */
    nn_workflow_item_t     *work_item       /* work item to be validated */
    );

/* compile workflow into workload */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workflow_compile_function_t)(
    nn_workload_t         **workload,       /* resulting workload */
    nn_device_t            *device,         /* device context */
    nn_workflow_t          *workflow,       /* workflow to be compiled */
    NN_WORKLOAD_DATA_TYPE  *input_format,   /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE  *output_format,  /* array containing formats of outputs */
    uint32_t                batch           /* batch size for compilation */
    );

/* executes workload with given inputs & outputs */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workload_execute_function_t)(
    nn_workload_t          *workload,       /* workload to be started */
    void *                 *input,          /* array of pointers with input data;  format is in workload->input_format */
    void *                 *output,         /* array of pointers with output data; format is in workload->output_format */
    NN_API_STATUS          *status          /* asynchronous status */
    );

/* delete work item */
typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_workflow_item_delete_function_t)(
    nn_workflow_item_t     *work_item       /* work item to be deleted */
    );


typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_device_parameter_get_function_t)(
    nn_device_t            *device,         /* target context */
    NN_PARAMETER            parameter,      /* parameter to get */
    void                   *buffer,         /* buffer to store result to */
    uint32_t                size            /* size of buffer */
    );


typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_device_parameter_set_function_t)(
    nn_device_t            *device,         /* target context */
    NN_PARAMETER            parameter,      /* parameter to set */
    void                   *buffer,         /* buffer with argument */
    uint32_t                size            /* size of buffer */
    );


typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_translate_api_status_function_t)(
    NN_API_STATUS           status,          /* status code to translate */
    char*                  *brief,           /* one-line explanation */
    char*                  *detailed         /* multi-line explanation */
    );

typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_device_parameter_get_function_t)(
    nn_device_t            *device,         /* target context */
    NN_PARAMETER            parameter,      /* parameter to get */
    void                   *buffer,         /* buffer to store result to */
    uint32_t                size            /* size of buffer */
    );

typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_device_parameter_set_function_t)(
    nn_device_t            *device,         /* target context */
    NN_PARAMETER            parameter,      /* parameter to set */
    void                   *buffer,         /* buffer with argument */
    uint32_t                size            /* size of buffer */
    );

typedef NN_API_STATUS (NN_API_CALL_CONVENTION *nn_translate_api_status_function_t)(
    NN_API_STATUS           status,          /* status code to translate */
    char*                  *brief,           /* one-line explanation */
    char*                  *detailed         /* multi-line explanation */
    );



/* interface structure *******************************************************/


/* interface structure ***************************************************************************/

typedef struct nn_device_interface_0_t {
    uint32_t                                        version;
    nn_device_t                                    *device;
    nn_workflow_create_function_t                   workflow_create_function;
    nn_workflow_delete_function_t                   workflow_delete_function;
    nn_workflow_metrics_query_function_t            workflow_metrics_query_function;
    nn_workflow_metrics_delete_function_t           workflow_metrics_delete_function;
    nn_workflow_compile_function_t                  workflow_compile_function;
    nn_workload_execute_function_t                  workload_execute_function;
    nn_workload_delete_function_t                   workload_delete_function;
    nn_workflow_item_create_function_t              workflow_item_create_function;
    nn_workflow_item_validate_function_t            workflow_item_validate_function;
    nn_workflow_item_delete_function_t              workflow_item_delete_function;
    nn_device_parameter_get_function_t              parameter_get_function;
    nn_device_parameter_set_function_t              parameter_set_function;
    nn_translate_api_status_function_t              translate_api_status_function;
} nn_device_interface_0_t;
