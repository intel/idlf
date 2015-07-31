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

/* query workflow for metrics of compilation variants */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_metrics_query_0x0_function(
    nn_workflow_metrics_array_t **array,/* resulting array of variants */
    nn_device_t        *device,         /* device context */
    nn_workflow_t      *workflow        /* workflow to be querried */
    );

/* delete array of workload metrics */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_metrics_delete_0x0_function(
    nn_workflow_metrics_array_t *array  /* array to delete */
    );

/* compile workflow into workload */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_compile_0x0_function(
    nn_workload_t         **workload,       /* resulting workload */
    nn_device_t            *device,         /* device context */
    nn_workflow_t          *workflow,       /* workflow to be compiled */
    NN_WORKLOAD_DATA_TYPE  *input_format,   /* array containing formats of inputs */
    NN_WORKLOAD_DATA_TYPE  *output_format,  /* array containing formats of outputs */
    uint32_t                batch           /* batch size for compilation */
    );

/* executes workload with given inputs & outputs */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workload_execute_0x0_function(
    nn_workload_t      *workload,       /* workload to be started */
    void *             *input,          /* array of pointers with input data;  format is in workload->input_format */
    void *             *output,         /* array of pointers with output data; format is in workload->output_format */
    NN_API_STATUS      *status          /* asynchronous status */
    );

/* delete workload */
NN_API_STATUS NN_API_CALL_CONVENTION nn_workload_delete_0x0_function(
    nn_workload_t       *workload       /* workload to be deleted */
    );

NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_item_create_0_function(
    nn_workflow_item_t            **workflow_item,  /* resulting workflow item */
    uint32_t                        input_count,    /* count of inputs */
    nn_workflow_use_descriptor_t   *input,          /* pointer to array of inputs */
    uint32_t                        output_count    /* count of outputs */
    );
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_item_validate_0x0_function(
    nn_device_t        *device,         /* target device */
    nn_workflow_item_t *work_item       /* work item to be validated */
    );
NN_API_STATUS NN_API_CALL_CONVENTION nn_workflow_item_delete_0_function(
    nn_workflow_item_t     *work_item       /* work item to be deleted */
    );
NN_API_STATUS NN_API_CALL_CONVENTION nn_device_parameter_get_0x0_function(
    nn_device_t        *device,         /* target context */
    NN_PARAMETER        parameter,      /* parameter to get */
    void               *buffer,         /* buffer to store result to */
    uint32_t            size            /* size of buffer */
    );

NN_API_STATUS NN_API_CALL_CONVENTION nn_device_parameter_set_0x0_function(
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
