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

#include "../../api/nn_primitives_api_0.h"
#include "../api_internal/cpu_device_internal.h"
#include "../../common/nn_workload_data.h"

nn_device_t *NN_API_CALL_CONVENTION create_device_with_thread_count(size_t num_threads, NN_API_STATUS *status) {
    SET_STATUS(NN_API_STATUS_OK);
    return new nn_device_internal(num_threads);
}

NN_API_STATUS NN_API_CALL_CONVENTION delete_device(nn_device_t *device){
    delete static_cast<nn_device_internal*>(device);
    return NN_API_STATUS_OK;
}

NN_API_STATUS NN_API_CALL_CONVENTION delete_opaque_data(nn_opaque_data_t *opaque_data){
   delete reinterpret_cast<nn::nn_workload_data_t<float> *>(opaque_data);
   return NN_API_STATUS_OK;
}

NN_API_STATUS NN_API_CALL_CONVENTION delete_event(nn_event_t event){
    return NN_API_STATUS_OK;
}

NN_API_STATUS NN_API_CALL_CONVENTION wait(size_t dependencies_count, nn_event_t *dependencies){
    return NN_API_STATUS_OK;
}

extern nn_primitives_convolution_f32_0_t nn_primitives_convolution_f32_0;
extern nn_primitives_pooling_f32_0_t nn_primitives_pooling_f32_0;
extern nn_primitives_convolution_pooling_f32_2x2stride2_0_t nn_primitives_convolution_pooling_f32_2x2stride2_0;
extern nn_primitives_fully_connected_f32_0_t nn_primitives_fully_connected_f32_0;
extern nn_primitives_convert_zxyn_nx_f32_0_t nn_primitives_convert_zxyn_nx_f32_0;
extern nn_primitives_arithmetic_f32_0_t nn_primitives_arithmetic_f32_0;
extern nn_primitives_normalization_elementwise_linear_f32_0_t nn_primitives_normalization_elementwise_linear_f32_0;
extern nn_primitives_normalization_response_across_maps_f32_0_t nn_primitives_normalization_response_across_maps_f32_0;
extern nn_primitives_softmax_f32_0_t nn_primitives_softmax_f32_0;
extern nn_primitives_relu_f32_0_t nn_primitives_relu_f32_0;

nn_primitives_0_t nn_primitives_0{
    create_device_with_thread_count,
    delete_device,
    delete_opaque_data,
    delete_event,
    wait,

    &nn_primitives_convolution_f32_0,

    nullptr, // &nn_primitives_convolution_i16qn_0,
    nullptr, // &nn_primitives_convolution_i16q78_0,

    &nn_primitives_pooling_f32_0,

    nullptr, // &nn_primitives_convolution_pooling_f32_0,
    nullptr, // &nn_primitives_convolution_pooling_f32_2x2stride2_0,

    nullptr, // &nn_primitives_convolution_pooling_i16qn_0,
    nullptr, // &nn_primitives_convolution_pooling_i16q78_0,
    nullptr, // &nn_primitives_convolution_pooling_i16qn_2x2stride2_0,

    &nn_primitives_fully_connected_f32_0,
    &nn_primitives_convert_zxyn_nx_f32_0,

    nullptr, // &nn_primitives_fully_connected_i16qn_0,
    nullptr, // &nn_primitives_fully_connected_i16q78_0,

    &nn_primitives_arithmetic_f32_0,
    &nn_primitives_normalization_elementwise_linear_f32_0,
    &nn_primitives_normalization_response_across_maps_f32_0,
    &nn_primitives_softmax_f32_0,
    &nn_primitives_relu_f32_0
};

NN_API_CALL int32_t NN_API_CALL_CONVENTION
nn_device_get_primitives(uint32_t version,      /* version of interface to create */
                         void *const primitives /* pointer to interface structure */
                         ) {
    if(!primitives) return -2;
    if (version == 0) {
        *reinterpret_cast<nn_primitives_0_t *>(primitives) = nn_primitives_0;
        return 0;
    } else {
        return -3;
    }
}

NN_API_CALL int32_t NN_API_CALL_CONVENTION
nn_device_get_primitives_description(nn_device_primitives_description_t *const description /* pointer to description structure */
                                     ) {
    if(!description) return -2;
    else {
        *description = nn_device_primitives_description_t{
            NN_DEVICE_PRIMITIVES_TYPE_CPU, // type
            0,                             // version_first
            0,                             // version_last
            "CPU device",                  // name
            "floating point CPU device\n"  // description
            "[TBD]"                        // TODO: fill description
        };

        return 0;
    }
    return -1;
}
