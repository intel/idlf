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

#include <devices/api/nn_primitives_api_0.h>

#include <dlfcn.h>
#include <stdio.h>
#include <math.h>
#include <string>

void cnn_reference(const size_t stride_x,
                   const size_t stride_y,
                   const int32_t center_offset_x,
                   const int32_t center_offset_y,
                   const nn::data<float, 4> &input,
                   const nn::data<float, 4> &weights,
                   const nn::data<float, 1> &bias,
                   nn::data<float, 4> &output){

    assert(output.size[3] == 1); // only batch 1 right now

    for(size_t output_x = 0; output_x < output.size[1]; ++output_x)
        for(size_t output_y = 0; output_y < output.size[2]; ++output_y)
            for(size_t output_z = 0; output_z < output.size[0]; ++output_z){
                auto& output_value = output.at(output_z, output_x, output_y, 0);
                output_value = bias.at(output_z);

                for(size_t kernel_x = 0; kernel_x < weights.size[0]; ++kernel_x)
                    for(size_t kernel_y = 0; kernel_y < weights.size[1]; ++kernel_y)
                        for(size_t kernel_z = 0; kernel_z < weights.size[2]; ++kernel_z){
                            const size_t input_x = output_x * stride_x + kernel_x - center_offset_x;
                            const size_t input_y = output_y * stride_y + kernel_y - center_offset_y;

                            // skip zero-padded inputs
                            if(input_x >= 0 && input_x < input.size[1] && input_y >= 0 && input_y < input.size[2]){
                                const auto& input_value = input.at(kernel_z, input_x, input_y, 0);
                                const auto& weight_value = weights.at(kernel_x, kernel_y, kernel_z, output_z);
                                output_value += input_value * weight_value;
                            }
                        }
            }
}

nn::data<float, 4> create_populated_input(size_t input_w, size_t input_h, size_t num_input_feature_maps, size_t batch_size){
    assert(batch_size == 1); // for now
    
    nn::data<float, 4> result(num_input_feature_maps, input_w, input_h, batch_size);
    for(size_t input_y = 0; input_y < input_h; ++input_y)
        for(size_t input_x = 0; input_x < input_w; ++input_x)
            for(size_t input_z = 0; input_z < num_input_feature_maps; ++input_z)
                result.at(input_z, input_x, input_y, 0) = input_z + input_x * 10.0f + input_y * 100.0f;

    return result;
}

nn::data<float, 4> create_populated_weights(size_t kernel_w, size_t kernel_h, size_t num_input_feature_maps, size_t num_output_feature_maps){
    nn::data<float, 4> result(kernel_w, kernel_h, num_input_feature_maps, num_output_feature_maps);
    for(size_t output = 0; output < num_output_feature_maps; ++output)
        for(size_t input = 0; input < num_input_feature_maps; ++input)
            for(size_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y)
                for(size_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x)
                    result.at(kernel_x, kernel_y, input, output) = kernel_x + kernel_y * 10.0f + input * 100.0f + output * 1000.0f;

    return result;
}

nn::data<float, 1> create_populated_bias(size_t num_output_feature_maps){
    nn::data<float, 1> result(num_output_feature_maps);
    for(size_t output = 0; output < num_output_feature_maps; ++output)
        result.at(output) = num_output_feature_maps;

    return result;
}

int main(){

    std::string lib_path("../../intel_visual_cloud_node/UnixMk/Debug/bin/device_cpu.so");
    
    // load device
    void* library = dlopen(lib_path.c_str(), RTLD_LAZY);
    if(library == nullptr)
        return -1;

    nn_device_get_primitives_description_t get_primitives_description = (nn_device_get_primitives_description_t)dlsym(library, "nn_device_get_primitives_description");
    nn_device_primitives_description_t device_primitives_description;
    get_primitives_description(&device_primitives_description);

    assert(device_primitives_description.version_first <= 0);
    assert(device_primitives_description.version_last >= 0);
    nn_device_get_primitives_t get_primitives = (nn_device_get_primitives_t)dlsym(library, "nn_device_get_primitives");
    nn_primitives_0_t primitives;
    get_primitives(0, &primitives);

    nn_device_t *device = primitives.create_device_with_thread_count(0, nullptr);

    // setup convolution parameters
    const size_t    batch_size              =       1,
                    kernel_w                =       3,
                    kernel_h                =       3,
                    num_input_feature_maps  =      16,
                    num_output_feature_maps =      32,
                    input_w                 =      12,
                    input_h                 =      12,
                    output_w                =      12,
                    output_h                =      12,
                    stride_x                =       1,
                    stride_y                =       1;

    const int32_t   center_offset_x           =     1,
                    center_offset_y           =     1;

    nn_argument_activation_t activation;
    activation.function = NN_ACTIVATION_FUNCTION_NONE;

    // prepare data
    printf("populating data...\n");
    auto input = create_populated_input(input_w, input_h, num_input_feature_maps, batch_size);
    auto weights = create_populated_weights(kernel_w, kernel_h, num_input_feature_maps, num_output_feature_maps);
    auto bias = create_populated_bias(num_output_feature_maps);

    // calculate reference output
    printf("calculating reference output...\n");
    nn::data<float, 4> output_ref(num_output_feature_maps, output_w, output_h, batch_size);
    nn::data<float, 4> output(num_output_feature_maps, output_w, output_h, batch_size);
    cnn_reference(stride_x, stride_y, center_offset_x, center_offset_y, input, weights, bias, output_ref);

    // init primitive handle
    nn_primitive_handle_t primitive = primitives.convolution_f32->create_handle(device,
                                                                                kernel_w,
                                                                                kernel_h,
                                                                                num_input_feature_maps,
                                                                                num_output_feature_maps,
                                                                                output_w,
                                                                                output_h,
                                                                                center_offset_x,
                                                                                center_offset_y,
                                                                                stride_x,
                                                                                stride_y,
                                                                                &activation,
                                                                                batch_size,
                                                                                nullptr);

    // prepare buffers
    nn_opaque_data_t *input_internal, *weights_internal, *bias_internal, *output_internal;
    input_internal = primitives.convolution_f32->create_input(primitive, &input, nullptr);
    weights_internal = primitives.convolution_f32->create_weights(primitive, &weights, nullptr);
    bias_internal = primitives.convolution_f32->create_bias(primitive, &bias, nullptr);
    output_internal = primitives.convolution_f32->create_output_with_padding(primitive, 0, 0, 0, 0, nullptr);

    printf("convolving...\n");
    // execute convolution
    nn_event_t c1 = primitives.convolution_f32->forward_with_weights_and_bias_async(primitive, input_internal, weights_internal, bias_internal, output_internal, 0, nullptr, nullptr);

    printf("extracting output...\n");
    // extract output
    nn_event_t output_ready = primitives.convolution_f32->copy_output_async(primitive, &output, output_internal, 1, &c1, nullptr);

    primitives.wait(1, &output_ready);

    // compare results
    printf("comparing results...\n");
    for(size_t output_x = 0; output_x < output_w; ++output_x)
        for(size_t output_y = 0; output_y < output_h; ++output_y)
            for(size_t output_z =0; output_z < num_output_feature_maps; ++output_z){
                auto &output_val = output.at(output_z, output_x, output_y, 0);
                auto &output_ref_val = output_ref.at(output_z, output_x, output_y, 0);
                auto error = fabs((output_ref_val - output_val)/output_ref_val);
                if(error > 0.10f)
                    printf("error at z= %4zu, x= %4zu, y= %4zu: ref= %10f, test= %10f, err= %10f\n", output_z, output_x, output_y, output_ref_val, output_val, error);
            }

    printf("cleanup...\n");

    primitives.delete_event(output_ready);
    primitives.delete_event(c1);
    primitives.delete_opaque_data(input_internal);
    primitives.delete_opaque_data(weights_internal);
    primitives.delete_opaque_data(bias_internal);
    primitives.delete_opaque_data(output_internal);
    primitives.delete_device(device);

    printf("done\n");
}