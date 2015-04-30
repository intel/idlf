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

#include "stdio.h"
#include "nn_workload_data.h"
#include "assert.h"

int main_c_api_example()
{
    nn_workload_data_t *data, *view;
    float result_float, value_float;
    int16_t result_int16, value_int16;
    nn_workload_data_coords_t data_size = { 4, 4, 4, 4, 4, 4 };

    nn_workload_data_coords_t nn_view_begin = { 1, 1, 1, 1, 1, 1 };
    nn_workload_data_coords_t nn_view_end = { 3, 3, 3, 3, 3, 3 };

    nn_workload_data_layout_t data_layout_float = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { 0, 1, 2, 3, 4, 5 }, // ordering
        NN_DATATYPE_FLOAT
    };
    nn_workload_data_layout_t data_layout_int16 = {
        { 0, 0, 0, 0, 0, 0 }, // tile in log2(size)
        { 0, 0, 0, 0, 0, 0 }, // alignment
        { 0, 1, 2, 3, 4, 5 }, // ordering
        NN_DATATYPE_INT16
    };

    // FLOAT32
    printf("\nFloat32 C API example\n");
    data = nn_workload_data_create(&data_size, &data_layout_float);
    if (data == NULL) return 1;
    
    value_float = 3.14f;
    nn_workload_data_set_float32(data, value_float, 1, 1, 2, 3, 2, 2);
    
    result_float = nn_workload_data_get_float32(data, 1, 1, 2, 3, 2, 2);
    printf("Value written: %f value read: %f \n", value_float, result_float);

    view = nn_workload_data_create_view(data, &nn_view_begin, &nn_view_end);
    if (view == NULL)
    {
        nn_workload_data_delete(data);
        return 1;
    }

    result_float = nn_workload_data_get_float32(view, 0, 0, 1, 2, 1, 1);
    printf("Value read from the view: %f \n", result_float);

    nn_workload_data_delete(view);
    nn_workload_data_delete(data);

    // INT16
    printf("\nInt16 C API example\n");
    data = nn_workload_data_create(&data_size, &data_layout_int16);
    if (data == NULL) return 1;

    value_int16 = 1234;
    nn_workload_data_set_int16(data, value_int16, 1, 1, 2, 3, 2, 2);

    result_int16 = nn_workload_data_get_int16(data, 1, 1, 2, 3, 2, 2);
    printf("Value written: %d value read: %d \n", value_int16, result_int16);

    view = nn_workload_data_create_view(data, &nn_view_begin, &nn_view_end);
    if (view == NULL)
    {
        nn_workload_data_delete(data);
        return 1;
    }

    result_int16 = nn_workload_data_get_int16(view, 0, 0, 1, 2, 1, 1);
    printf("Value read from the view: %d \n", result_int16);

    nn_workload_data_delete(view);
    nn_workload_data_delete(data);
    return 0;
}
