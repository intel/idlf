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

#include "device/api/nn_device_interface_0.h"
#include "nn_workload_data.h"

#include <vector>

struct nn_primitive_t {
    virtual ~nn_primitive_t() {}

    virtual void forward(const std::vector<const nn_workload_data_t*> &inputs,
                         const std::vector<const nn_workload_data_t*> &parameters,
                         const std::vector<nn_workload_data_t *> &outputs) {}

    virtual void backward(const std::vector<nn_workload_data_t *> &inputs,              // TODO: change to "front" or "bottom" ?
                          const std::vector<const nn_workload_data_t *> &parameters,
                          const std::vector<const nn_workload_data_t *> &outputs) {}    // TODO: change to "back" or "top" ?

    virtual void backward_parameter(size_t parameter_index,
                                    const std::vector<const nn_workload_data_t*> &inputs,
                                    const std::vector<nn_workload_data_t *> &parameters,
                                    const std::vector<const nn_workload_data_t *> &outputs) {}

    virtual std::vector<nn_workload_data_t *> create_parameters(bool allocate_delta = false) { return{}; }

    virtual std::vector<nn_workload_data_t *> create_inputs(bool allocate_delta = false) = 0;

    virtual std::vector<nn_workload_data_t *> create_outputs(bool allocate_delta = false) = 0;

    virtual bool validate_input(size_t index, nn_workload_data_t *data) = 0;
};

struct nn_device {};

#define SET_STATUS(value)    \
    if (status != nullptr)   \
        *status = value;

template <size_t T_index>
static inline uint32_t get_format_size(const nn_output_format &output_format) {
    static_assert(T_index <= 2, "output_format only supports 3 dimensions");

    if (T_index >= 1 && !(output_format.format >= NN_DATA_FORMAT_2D))
        return 1;

    if (T_index >= 2 && !(output_format.format >= NN_DATA_FORMAT_3D))
        return 1;

    return output_format.format_3d.size[T_index];
}
