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

#include "primitives_workload.h"
#include "common/nn_data_tools.h"

#include <map>
#include <fstream>
#include <algorithm>

struct primitives_workload_caffenet_float_layers
{
    std::map<std::string, nn_opaque_data_t*> inputs;
    std::map<std::string, nn_opaque_data_t*> outputs;
    std::map<std::string, nn_opaque_data_t*> weights;
    std::map<std::string, nn_opaque_data_t*> bias;
    std::map<std::string, nn_event_t> events;
};

class primitives_workload_caffenet_float : public primitives_workload_base {

  private:
    primitives_workload_caffenet_float_layers layers;

    std::map<std::string, nn_primitive_handle_t> handles;
    std::vector<std::vector<std::string>> execution_order;
    std::map<std::string, std::string> layer_to_handle;

  public:
    primitives_workload_caffenet_float() : primitives_workload_base(227, false, fi::resize_image_to_square) {
        read_file_to_vector(labels, "weights_caffenet/names.txt", false);
        read_file_to_vector(wwids, "weights_caffenet/wwids.txt", false);
    }

  private:
    nn::data<float> *load_nn_data_from_file(std::string filename) {
        nn::data<float> *data = nn_data_load_from_file_time_measure(filename);
        if (data == nullptr) {
            std::cerr << "Can't load " << filename << std::endl;
            throw;
        }
        return data;
    }

    nn_primitive_handle_t get_handle(std::string name)
    {
        auto it = layer_to_handle.find(name);
        if (it != layer_to_handle.end())
            name = it->second;
        return handles[name];
    }

    template <class T_primitives>
    void load_weights_and_biases(nn_device_t *device,
                                 T_primitives &primitives,
                                 std::string name,
                                 std::string weights_file,
                                 std::string biases_file,
                                 std::vector<nn_event_t> &ready_events,
                                 std::vector<nn_data_t *> &temporary_data) {
        auto weights = load_nn_data_from_file(weights_file);
        auto bias = load_nn_data_from_file(biases_file);
        temporary_data.push_back(weights);
        temporary_data.push_back(bias);

        assert(get_handle(name) != NULL);

        nn_opaque_data_t *parameters[2];

        primitives.create_parameters(get_handle(name), 2, parameters, 0, nullptr);
        ready_events.push_back(primitives.copy_to_opaque_async(device, parameters[0], weights, 0, nullptr, nullptr));
        ready_events.push_back(primitives.copy_to_opaque_async(device, parameters[1], bias, 0, nullptr, nullptr));

        layers.weights[name] = parameters[0];
        layers.bias[name] = parameters[1];
    }

  public:
    virtual bool is_valid() override { return device != nullptr; }

    virtual void init(nn_primitives_0_t primitives_, nn_device_t *device_, size_t batch_size_) override
    {
        primitives = primitives_;
        device = device_;
        batch_size = batch_size_;
        execution_order = decltype(execution_order)({
                {"A1"},
                {"C1"},
                {"P1"},
                {"N1"},
                {"C2_1", "C2_2"},
                {"P2"},
                {"N2"},
                {"C3"},
                {"C4_1", "C4_2"},
                {"C5_1", "C5_2"},
                {"P5"},
                {"convert_c5fc6"},
                {"FC6"},
                {"FC7"},
                {"FC8"},
                {"SF"}});
        layer_to_handle = decltype(layer_to_handle)({
            {"C2_1", "C2"}, {"C2_2", "C2"},
            {"C2_1_full", "C2_full"}, {"C2_2_full", "C2_full"},
            {"C4_1", "C4"}, {"C4_2", "C4"},
            {"C5_1", "C5"}, {"C5_2", "C5"},
            {"C5_1_full", "C5_full"}, {"C5_2_full", "C5_full"}});

        //Prepare handles for each layer
        {
            nn_primitives_normalization_response_across_maps_hints_t lrn_hints_twos = {};
            lrn_hints_twos.output_padding = {2, 2, 2, 2};
            nn_primitives_normalization_response_across_maps_hints_t lrn_hints_ones = {};
            lrn_hints_ones.output_padding = {1, 1, 1, 1};

            nn_argument_activation activation_relu;
            activation_relu.function = NN_ACTIVATION_FUNCTION_RELU;

            nn_argument_activation activation_none;
            activation_none.function = NN_ACTIVATION_FUNCTION_NONE;

            nn_primitives_convolution_hints_t convolution_hints_ones = {};
            convolution_hints_ones.output_padding = {1, 1, 1, 1};

            handles.insert(
                {"A1", primitives.create_handle.arithmetic_f32(
                    device,
                    227,
                    227,
                    3,
                    NN_ARITHMETIC_FUNCTION_SUBTRACTION,
                    batch_size, // size of input batch_size
                    nullptr)});
            handles.insert(
                {"C1", primitives.create_handle.convolution_f32(
                    device,
                    11, // kernel_w
                    11, // kernel_h
                    3, // num_input
                    96, // num_output
                    55, // output_w
                    55, // output_h
                    0, // center_offset_x
                    0, // center_offset_y
                    4, // stride_x
                    4, // stride_y
                    &activation_relu,
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"P1", primitives.create_handle.pooling_f32(
                    device,
                    NN_POOLING_MODE_MAX, // pooling mode
                    3, // pool size x
                    3, // pool size y
                    2, // pool stride x
                    2, // pool stride y
                    96, // num of feature maps
                    27, // output width
                    27, // output height
                    0, // center offset x
                    0, // center offset y
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"N1", primitives.create_handle.normalization_response_across_maps_f32(
                    device,      // IDLF device handle
                    0.0001f / 5, // sum scale
                    0.75f,       //  sum power
                    1,           // square sum weight
                    5,           // size of moving window on the feature maps
                    27,          // image width
                    27,          // image height
                    96,          // number of feature maps
                    batch_size,  // size of input batch_size,
                    &lrn_hints_twos,
                    nullptr)});
            handles.insert(
                {"C2", primitives.create_handle.convolution_f32(
                    device,
                    5, // kernel_w
                    5, // kernel_h
                    96 / 2, // num_input
                    256 / 2, // num_output
                    27, // output_w
                    27, // output_h
                    2, // center_offset_x
                    2, // center_offset_y
                    1, // stride_x
                    1, // stride_y
                    &activation_relu,
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"C2_full", primitives.create_handle.convolution_f32(
                    device,
                    5, // kernel_w
                    5, // kernel_h
                    96, // num_input
                    256, // num_output
                    27, // output_w
                    27, // output_h
                    2, // center_offset_x
                    2, // center_offset_y
                    1, // stride_x
                    1, // stride_y
                    &activation_relu,
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"P2", primitives.create_handle.pooling_f32(
                    device,
                    NN_POOLING_MODE_MAX, // pooling mode
                    3, // pool size x
                    3, // pool size y
                    2, // pool stride x
                    2, // pool stride y
                    256, // num of feature maps
                    13, // output width
                    13, // output height
                    0, // center offset x
                    0, // center offset y
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"N2", primitives.create_handle.normalization_response_across_maps_f32(
                    device,      // IDLF device handle
                    0.0001f / 5, // sum scale
                    0.75f,       //  sum power
                    1,           // square sum weight
                    5,           // size of moving window on the feature maps
                    13,          // image width
                    13,          // image height
                    256,         // number of feature maps
                    batch_size,  // size of input batch_size
                    &lrn_hints_ones,
                    nullptr)});
            handles.insert(
                {"C3", primitives.create_handle.convolution_f32(
                    device,
                    3, // kernel_w
                    3, // kernel_h
                    256, // num_input
                    384, // num_output
                    13, // output_w
                    13, // output_h
                    1, // center_offset_x
                    1, // center_offset_y
                    1, // stride_x
                    1, // stride_y
                    &activation_relu,
                    batch_size,
                    &convolution_hints_ones,
                    nullptr)});
            handles.insert(
                {"C4", primitives.create_handle.convolution_f32(
                    device,
                    3, // kernel_w
                    3, // kernel_h
                    384 / 2, // num_input
                    384 / 2, // num_output
                    13, // output_w
                    13, // output_h
                    1, // center_offset_x
                    1, // center_offset_y
                    1, // stride_x
                    1, // stride_y
                    &activation_relu,
                    batch_size,
                    &convolution_hints_ones,
                    nullptr)});
            handles.insert(
                {"C5", primitives.create_handle.convolution_f32(
                    device,
                    3, // kernel_w
                    3, // kernel_h
                    384 / 2, // num_input
                    256 / 2, // num_output
                    13, // output_w
                    13, // output_h
                    1, // center_offset_x
                    1, // center_offset_y
                    1, // stride_x
                    1, // stride_y
                    &activation_relu,
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"C5_full", primitives.create_handle.convolution_f32(
                    device,
                    3, // kernel_w
                    3, // kernel_h
                    384, // num_input
                    256, // num_output
                    13, // output_w
                    13, // output_h
                    1, // center_offset_x
                    1, // center_offset_y
                    1, // stride_x
                    1, // stride_y
                    &activation_relu,
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"P5", primitives.create_handle.pooling_f32(
                    device,
                    NN_POOLING_MODE_MAX, // pooling mode
                    3, // pool size x
                    3, // pool size y
                    2, // pool stride x
                    2, // pool stride y
                    256, // num of feature maps
                    6, // output width
                    6, // output height
                    0, // center offset x
                    0, // center offset y
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"convert_c5fc6", primitives.create_handle.convert_zxyn_nx_f32(
                    device,
                    6, // input width
                    6, // input height
                    256, // num of input feature maps
                    batch_size,
                    nullptr)});
            handles.insert(
                {"FC6", primitives.create_handle.fully_connected_f32(
                    device,
                    6 * 6 * 256, // num_input
                    4096,        // num_output
                    &activation_relu,
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"FC7", primitives.create_handle.fully_connected_f32(
                    device,
                    4096, // num_input
                    4096, // num_output
                    &activation_relu,
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"FC8", primitives.create_handle.fully_connected_f32(
                    device,
                    4096, // num_input
                    1000, // num_output
                    &activation_none,
                    batch_size,
                    nullptr,
                    nullptr)});
            handles.insert(
                {"SF", primitives.create_handle.softmax_f32(
                    device,
                    1000, // number of feature maps
                    batch_size,
                    nullptr,
                    nullptr)});
        }

        //Load parameters
        {
            struct LoadElem {
                std::string name;
                std::string weights_file;
                std::string bias_file;
            };
            std::vector<LoadElem> to_load = {
                {"C1", "weights_caffenet/conv1_weights.nnd", "weights_caffenet/conv1_biases.nnd"},
                {"C2_1", "weights_caffenet/conv2_g1_weights.nnd", "weights_caffenet/conv2_g1_biases.nnd"},
                {"C2_2", "weights_caffenet/conv2_g2_weights.nnd", "weights_caffenet/conv2_g2_biases.nnd"},
                {"C3", "weights_caffenet/conv3_weights.nnd", "weights_caffenet/conv3_biases.nnd"},
                {"C4_1", "weights_caffenet/conv4_g1_weights.nnd", "weights_caffenet/conv4_g1_biases.nnd"},
                {"C4_2", "weights_caffenet/conv4_g2_weights.nnd", "weights_caffenet/conv4_g2_biases.nnd"},
                {"C5_1", "weights_caffenet/conv5_g1_weights.nnd", "weights_caffenet/conv5_g1_biases.nnd"},
                {"C5_2", "weights_caffenet/conv5_g2_weights.nnd", "weights_caffenet/conv5_g2_biases.nnd"},
                {"FC6", "weights_caffenet/fc6_weights.nnd", "weights_caffenet/fc6_biases.nnd"},
                {"FC7", "weights_caffenet/fc7_weights.nnd", "weights_caffenet/fc7_biases.nnd"},
                {"FC8", "weights_caffenet/fc8_weights.nnd", "weights_caffenet/fc8_biases.nnd"}};
            std::vector<nn_event_t> parameters_ready_events;
            std::vector<nn_data_t *> temporary_data;
            primitives.create_parameters(handles["A1"], 1, &layers.weights["A1"], 0, nullptr);
            parameters_ready_events.push_back(
                primitives.copy_to_opaque_async(device,
                                                layers.weights["A1"],
                                                load_nn_data_from_file("weights_caffenet/imagenet_mean.nnd"),
                                                0,
                                                nullptr,
                                                nullptr));
            for (auto elem : to_load)
                load_weights_and_biases(
                    device, primitives, elem.name, elem.weights_file, elem.bias_file,
                    parameters_ready_events, temporary_data);
        }

        //outputs
        for (auto layer : execution_order)
            for (auto name : layer)
                primitives.create_outputs(get_handle(name), 1, &layers.outputs[name], 0, nullptr);
        //inputs (and outputs on merges)
        for (auto i = 1u; i < execution_order.size(); ++i)
        {
            if (execution_order[i].size() == execution_order[i - 1].size())
            {
                for (auto j = 0u; j < execution_order[i].size(); ++j)
                    layers.inputs[execution_order[i][j]] = layers.outputs[execution_order[i - 1][j]];
                continue;
            }
            //merge
            if (execution_order[i - 1].size() > execution_order[i].size())
            {
                assert(execution_order[i].size() == 1);
                auto name = execution_order[i].front();
                auto prev_full = get_handle(execution_order[i - 1][0] + "_full");

                primitives.create_outputs(prev_full, 1, &layers.inputs[name], 0, nullptr);

                std::vector<nn_opaque_data_t*> outputs(execution_order[i - 1].size(), nullptr);
                primitives.split_z(outputs.size(), &outputs.front(), layers.inputs[name]);

                for (auto j = 0u; j < outputs.size(); ++j)
                {
                    primitives.delete_opaque_data(layers.outputs[execution_order[i - 1][j]]);
                    layers.outputs[execution_order[i - 1][j]] = outputs[j];
                }
            }
            //split
            if (execution_order[i - 1].size() < execution_order[i].size())
            {
                assert(execution_order[i - 1].size() == 1);
                auto prev_name = execution_order[i - 1].front();

                std::vector<nn_opaque_data_t*> inputs(execution_order[i].size(), nullptr);
                primitives.split_z(inputs.size(), &inputs.front(), layers.outputs[prev_name]);
                for (auto j = 0u; j < inputs.size(); ++j)
                    layers.inputs[execution_order[i][j]] = inputs[j];
            }
        }
        for (auto layer : execution_order)
            for(auto elem : layer)
                if(layers.inputs[elem])
                {
                    try { assert(primitives.validate_input(get_handle(elem),0,layers.inputs[elem])); }
                    catch(std::logic_error&) {} //if unimplemented
                }
    }

    std::vector<nn_opaque_data_t*> get_params(std::string name)
    {
        std::vector<nn_opaque_data_t*> ret;
        if (layers.weights.find(name) != layers.weights.end())
            ret.push_back(layers.weights[name]);
        if (layers.bias.find(name) != layers.bias.end())
            ret.push_back(layers.bias[name]);
        return ret;
    }

    void execute(const nn::data<float, 4> &input, nn::data<float, 2> &output) override
    {
        const size_t batch_size = input.size[3];
        layers.inputs["A1"] = primitives.map_input(handles["A1"], 0, &input, nullptr);

        std::vector<nn_event_t> prev_events;
        for (auto layer : execution_order)
        {
            decltype(prev_events) next_prev_events;
            for (auto name : layer)
            {
                auto params = get_params(name);
                auto event = primitives.forward_async(
                    get_handle(name),
                    1, &layers.inputs[name],
                    params.size(), params.data(),
                    1, &layers.outputs[name],
                    prev_events.size(), prev_events.data(),
                    nullptr);
                next_prev_events.push_back(event);
            }
            prev_events.swap(next_prev_events);
        }
        layers.events["output_ready"] =
            primitives.copy_from_opaque_async(device, &output, layers.outputs["SF"], 1, &layers.events["SF"], nullptr);
        primitives.wait(1, &layers.events["output_ready"]);
    }

    void cleanup() override
    {
        std::cout
            << "Cleanup in memory"
            << std::endl
            << "========================================================"
            << std::endl;

        for (auto& pair : layers.events)
            primitives.delete_event(pair.second);
        for (auto& pair : layers.weights)
            primitives.delete_opaque_data(pair.second);
        for (auto& pair : layers.bias)
            primitives.delete_opaque_data(pair.second);
        for (auto& pair : layers.inputs)
            primitives.delete_opaque_data(pair.second);
        for (auto& pair : layers.outputs)
            primitives.delete_opaque_data(pair.second);
        for (auto& handle : handles)
            primitives.delete_primitive(handle.second);
    }
};

// Code below creates 'attach_' object in anonymous namespace at global scope.
// This ensures, that object itself is not visible to other compilation units
// and it's constructor is ran before main execution starts.
// The sole function of this construction is attaching this workload to
// library of workloads (singleton command pattern).
namespace {
    struct attach {
        primitives_workload_caffenet_float workload;
        attach() {
            primitives_workload::instance().add("caffenet_float", &workload);
        }
    };

    attach attach_;
}
