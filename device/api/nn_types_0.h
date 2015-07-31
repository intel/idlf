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
#include "nn_call_convention.h"
#include "nn_data_0.h"

typedef struct nn_opaque_data nn_opaque_data_t;
typedef struct nn_device nn_device_t;
typedef struct nn_primitive_t *nn_primitive_handle_t;
typedef struct nn_event{} nn_event_t;

/* status of API call
   All API functions return this enum. */
typedef enum {
    NN_API_STATUS_OK = 0,                                   /* success */
    NN_API_WORK_CREATED     = NN_API_STATUS_OK,             /* async state: work created */
    NN_API_WORK_IN_PROGRESS,                                /* async state: work in progress */
    NN_API_WORK_FINISHED,                                   /* async state: work finished */

    /* errors are negative numbers */
    NN_API_STATUS_ERROR_UNSUPPORTED_VERSION = 0x80000000,   /* unsupported API version */
    NN_API_STATUS_ERROR_OUT_OF_MEMORY,                      /* out of memory during processing */
    NN_API_STATUS_ERROR_INVALID_POINTER,                    /* argument is an invalid pointer */
    NN_API_STATUS_ERROR_STATUS_CODE_NOT_FOUND,              /* invalid error code for translation */
    NN_API_STATUS_ERROR_INVALID_WORK_ITEM_TYPE,             /* invalid work item type */
    NN_API_STATUS_ERROR_INVALID_MEMORY_LAYOUT,              /* buffer has invalid memory layout */
    NN_API_STATUS_ERROR_DATA_NOT_CONSISTENT,                /* inputs & uses graphs are not consistent */
    NN_API_STATUS_ERROR_INVALID_INPUT_COUNT,                /* invalid input count for workflow/item */
    NN_API_STATUS_ERROR_INVALID_OUTPUT_COUNT,               /* invalid output count workflow/item */
    NN_API_STATUS_ERROR_INVALID_WORKFLOW,                   /* workflow structure is invalid */
    NN_API_STATUS_ERROR_OTHER,                              /* other error */
    NN_API_STATUS_LAST                      = NN_API_STATUS_ERROR_OTHER
} NN_API_STATUS;

/* activation functions
   Some of workflow items execute activation function aplied to result of summation. */
typedef enum {
    NN_ACTIVATION_FUNCTION_NONE = 0,    /* f(x) = x */
    NN_ACTIVATION_FUNCTION_ABS,         /* f(x) = abs(x) */
    NN_ACTIVATION_FUNCTION_STEP,        /* f(x) = x<0 ? 0 : 1 */
    NN_ACTIVATION_FUNCTION_RELU,        /* f(x) = max(0, x) */
    NN_ACTIVATION_FUNCTION_SOFTPLUS,    /* f(x) = log(1+exp(x)) */
    NN_ACTIVATION_FUNCTION_LOGISTIC,    /* f(x) = 1/(1+exp(-x)) */
    NN_ACTIVATION_FUNCTION_TANH,        /* f(x) = a*tanh(x*b) | a&b are in nn_argument_activation_t.fp32_tanh*/
    NN_ACTIVATION_FUNCTION_LAST = NN_ACTIVATION_FUNCTION_TANH
} NN_ACTIVATION_FUNCTION;

/* container for activation function data
   Contains type and parameters per type (if they exist). */
typedef struct nn_argument_activation {
    NN_ACTIVATION_FUNCTION  function;           /* activation function */
    union {
        struct {
            float a;                            /* f(x) = [a]*tan(b*x) */
            float b;                            /* f(x) = a*tan([b]*x) */
        } fp32_tanh;
    } data;                                     /* data related for activation function */
} nn_argument_activation_t;

/* pooling types
   What type of pooling/resolution reduction algorithm is used in pooling <workflow item>. */
typedef enum {
    NN_POOLING_MODE_MAX = 0,            /* maximum */
    NN_POOLING_MODE_AVERAGE,            /* average */
    NN_POOLING_MODE_L2,                 /* L2 pooling */
    NN_POOLING_MODE_LAST = NN_POOLING_MODE_L2
} NN_POOLING_MODE;

/* arithmetics functions
Some of workflow items do simple arithmetic operation such as addition , difference, multiplication, quotient
one by one for each element of arguments, which are nn::data structures
*/
typedef enum {
    NN_ARITHMETIC_FUNCTION_NONE = 0,       /* f([x],[y]) = [x]          */
    NN_ARITHMETIC_FUNCTION_ADDITION,       /* f([x],[y]) = [x] + [y]    */
    NN_ARITHMETIC_FUNCTION_SUBTRACTION,    /* f([x],[y]) = [x] - [y]    */
    NN_ARITHMETIC_FUNCTION_MULTIPLICATION, /* f([x],[y]) = [x] .* [y], operator .* - should be understood as
                                           multiplication element by element
                                           and not a matrix multiplication */
    NN_ARITHMETIC_FUNCTION_DIVISION,        /* f([x],[y]) = [x] ./ [y]  operator ./ - as described above, for multiplication*/

    NN_ARITHMETIC_FUNCTION_LAST = NN_ARITHMETIC_FUNCTION_DIVISION
} NN_ARITHMETIC_FUNCTION;


/* Loss functions */
typedef enum
{
    NN_LOSS_FUNCTION_SUM_OF_SQUARES = 0,        /* square error: (y-x)^2 */
    NN_LOSS_FUNCTION_MULTINOMIAL_LOGISTIC,
    NN_LOSS_FUNCTION_LAST = NN_LOSS_FUNCTION_MULTINOMIAL_LOGISTIC
} NN_LOSS_FUNCTION;
