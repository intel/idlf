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
#include "device/api/nn_device_api.h"
#include "device/api/nn_device_interface_0.h"
#include "device/common/nn_workload_data.h"
#include "cpu_device_internal.h"
#include <vector>
#include <deque>
#include <map>

#define ENABLE_WORKLOAD_PROFILING 0

const size_t max_threads = 18;

// NN_UNREACHABLE_CODE signal to supporting compiler that specific location in code cannot be reached
#if defined _MSC_VER 
#   define NN_UNREACHABLE_CODE __assume(0)
#endif

#if defined __GNUC__
#   if (__GNUC__ * 100 + __GNUC_MINOR__) >= 405
#       define NN_UNREACHABLE_CODE __builtin_unreachable()
#   else
#       define NN_UNREACHABLE_CODE
#   endif
#endif

#if defined __clang__
#   if __has_builtin(__builtin_unreachable)
#       define NN_UNREACHABLE_CODE __builtin_unreachable()
#   else
#       define NN_UNREACHABLE_CODE
#   endif
#endif

typedef struct nn_arguments_convert_data_layout {
    uint8_t                     type;
} nn_arguments_convert_data_layout_t;

namespace nn {

/* arguments for merged convolution and pooling layers (with shared weights) */
struct arguments_forward_convolution_pooling_max_2x2_stride_2x2 {
    ::nn_workload_data_t       *biases;           /* biases */
    ::nn_workload_data_t       *weights;          /* weights */
};

/* arguments for convolution fixed point layers */
struct arguments_forward_convolution_fixedpoint
{
    ::nn_workload_data_t       *biases;         /* biases */
    ::nn_workload_data_t       *weights;        /* weights */
    NN_PADDING_MODE             padding;         /* padding mode */
    uint32_t                    stride[2];       /* stride during filtering operation */
    nn_argument_activation_fixedpoint_t    activation;     /* activation data */
    uint32_t                    center_offset[2]; /* offset of center point in convolution filter */
};

/* arguments for fully connected fixed point layers */
struct arguments_fully_connected_forward_i16qn_i16qn {
    ::nn_workload_data_t    *biases;         /* biases for each neuron */
    ::nn_workload_data_t    *weights;        /* weights for each neuron */
    nn_argument_activation_fixedpoint_t    activation;     /* activation data */
};

/* arguments for fully connected fixed point layers */
struct arguments_fully_connected_forward_i16qn_i32qn {
    ::nn_workload_data_t    *biases;         /* biases for each neuron */
    ::nn_workload_data_t    *weights;        /* weights for each neuron */
    nn_argument_activation_fixedpoint_t    activation;     /* activation data */
};

/* arguments for merged convolution and pooling layers fixed point */
struct arguments_forward_merged_convolution_pooling_max_2x2_stride_2x2_fixedpoint
{
    ::nn_workload_data_t       *biases;         /* biases */
    ::nn_workload_data_t       *weights;        /* weights */
    uint32_t                    center_offset[2]; /* offset of center point in convolution filter */
    NN_PADDING_MODE             padding;          /* padding mode */
    nn_argument_activation_fixedpoint_t    activation;     /* activation data */
    uint32_t                    stride[2];  /* stride during filtering operation */
};

} // namespace nn

/* workload item is a "work item" with realized buffers */
typedef struct nn_workload_item {
    /* following fields are copied from work item */
    NN_WORK_ITEM_TYPE                               type;       /* work item type */
    std::vector<nn_workload_data_t *>               output;     /* output of current work item */
    std::vector<struct nn_workload_use_descriptor>  input;      /* input array */

    /* union of packaged arguments for each layer type */
    union {
        /* arguments for simple NN layers */
        nn_arguments_input_t                                            input;
        nn_arguments_output_t                                           output;
        nn_arguments_view_t                                             view;
        nn_arguments_update_t                                           update;
        nn_arguments_loss_function_t                                    loss_function;
        nn_arguments_dropout_t                                          dropout;
        nn_arguments_merge_t                                            forward_merge;

        nn_arguments_forward_normalization_t forward_normalization;

        /* fixedpoint layers */
        nn::arguments_forward_convolution_fixedpoint                    forward_convolution_fixedpoint;
        nn::arguments_fully_connected_forward_i16qn_i16qn               fully_connected_forward_i16qn_i16qn;
        nn::arguments_fully_connected_forward_i16qn_i32qn               fully_connected_forward_i16qn_i32qn;
        nn_arguments_forward_softmax_fixedpoint_t                       forward_softmax_fixedpoint;
        nn_arguments_forward_pooling_fixedpoint_t                       forward_pooling_fixedpoint;
        nn_arguments_normalization_response_across_maps_forward_i16qn_t normalization_response_across_maps_forward_i16qn;

        /* conversion layers */
        nn_arguments_forward_convert_float_to_int16_fixedpoint_t        forward_convert_float_to_fixedpoint;
        nn_arguments_convert_data_layout_t                              convert_data_layout;

        /* arguments for for complex NN layers */
        nn::arguments_forward_convolution_pooling_max_2x2_stride_2x2    forward_convolution_pooling_max_2x2_stride_2x2;
        nn::arguments_forward_merged_convolution_pooling_max_2x2_stride_2x2_fixedpoint   forward_convolution_pooling_max_2x2_stride_2x2_fixedpoint;
    } arguments;

    std::vector<nn_workload_data_t *> parameters;

    std::string                                 name;           /* optional name for profiling and debug */
    std::vector<nn_workload_use_descriptor>     use;            /* work items that use results of current one, along with id of output they use */
    nn_workload_item                           *forward_item;   /* pointer to forward item - used by backward primitives */
    nn_primitive_handle_t                       primitive;
} nn_workload_item_t;


typedef struct nn_workload_use_descriptor
{
    nn_workload_item *  item;
    uint32_t            index;

    nn_workload_data_t* get_data_view()
    {
        return item->output[index];
    }
} nn_workload_use_descriptor_t;

typedef struct profiling_data{
    std::map<nn_workload_item*, std::vector<uint64_t>> work_item_cycles;
} profiling_data_t;

/* opaque (invisible to user) part of workload */
typedef struct nn_workload_opaque {
    std::vector<nn_workload_item_t *> input;
    std::vector<nn_workload_item_t *> output;
    std::deque <nn_workload_item_t *> order_of_execution;
#if ENABLE_WORKLOAD_PROFILING
    profiling_data_t                  profiling_data;
#endif
} nn_workload_opaque_t;

/* create empty workflow */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_create_0_function(
    nn_workflow_t *    *workflow,       /* workflow to be created */
    uint32_t            input_count,    /* number of inputs in created workflow */
    uint32_t            output_count    /* number of outputs in created workflow */
    );

/* delete workflow */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_delete_0_function(
    nn_workflow_t      *workflow        /* workflow to delete */
    );

/* compile workflow into workload */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_compile_0_function(
    nn_workload_t         **workload,       /* resulting workload */
    nn_device_t            *device,         /* device context */
    nn_workflow_t          *workflow,       /* workflow to be compiled */
    NN_WORKLOAD_DATA_TYPE  *input_format,   /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE  *output_format,  /* array containing formats of outputs */
    uint32_t                batch           /* batch size for compilation */
    );

/* executes workload with given inputs & outputs */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workload_execute_0_function(
    nn_workload_t      *workload,       /* workload to be started */
    void *             *input,          /* array of pointers with input data;  format is in workload->input_format */
    void *             *output,         /* array of pointers with output data; format is in workload->output_format */
    NN_API_STATUS      *status          /* asynchronous status */
    );

/* delete workload */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workload_delete_0_function(
    nn_workload_t       *workload       /* workload to be deleted */
    );

/* create empty work item */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_item_create_0_function(
    nn_workflow_item_t             **workflow_item,  /* resulting workflow item */
    uint32_t                         input_count,    /* count of inputs */
    nn_workflow_use_descriptor_t    *input,          /* pointer to array of inputs */
    uint32_t                         output_count    /* count of outputs */
    );

/* query workflow for metrics of compilation variants */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_metrics_query_0_function(
    nn_workflow_metrics_array_t **array,/* resulting array of variants */
    nn_device_t        *device,         /* device context */
    nn_workflow_t      *workflow        /* workflow to be querried */
    );

/* delete array of workload metrics */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_metrics_delete_0_function(
    nn_workflow_metrics_array_t *array  /* array to delete */
    );

/* validate parameters of work_item for this particular device */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_item_validate_0_function(
    nn_device_t        *device,         /* target device */
    nn_workflow_item_t *work_item       /* work item to be validated */
    );

/* delete work item */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_item_delete_0_function(
    nn_workflow_item_t     *work_item   /* work item to be deleted */
    );

NN_API_STATUS NN_API_CALL_CONVENTION nn_device_parameter_get_0_function(
    nn_device_t        *device,         /* target context */
    NN_PARAMETER        parameter,      /* parameter to get */
    void               *buffer,         /* buffer to store result to */
    uint32_t            size            /* size of buffer */
    );

NN_API_STATUS NN_API_CALL_CONVENTION nn_device_parameter_set_0_function(
    nn_device_t        *device,         /* target context */
    NN_PARAMETER        parameter,      /* parameter to set */
    void               *buffer,         /* buffer with argument */
    uint32_t            size            /* size of buffer */
    );

NN_API_STATUS NN_API_CALL_CONVENTION nn_translate_api_status_0_function(
    NN_API_STATUS       status,          /* status code to translate */
    char*              *brief,           /* one-line explanation */
    char*              *detailed         /* multi-line explanation */
    );
