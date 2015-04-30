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

#include "nn_primitives_api_0.h"
#include "FreeImage_wraps.h"

#include <map>
#include <string>
#include <vector>
#include <fstream>

class primitives_workload_base {
protected:
  primitives_workload_base(uint16_t img_size, bool RGB_order, fi::prepare_image_t image_process)
      : img_size(img_size), RGB_order(RGB_order), image_process(image_process) {}

    const uint16_t img_size;
    nn_device_t *device;
    size_t batch_size;
    nn_primitives_0_t primitives;

public:
    bool RGB_order;
    fi::prepare_image_t image_process;
    std::vector<std::string> labels;
    std::vector<std::string> wwids;

    virtual void init(nn_primitives_0_t primitives, nn_device_t *device, size_t batch_size) = 0;
    virtual void execute(const nn::data<float, 4> &input, nn::data<float, 2> &output) = 0;
    virtual void cleanup() = 0;

    uint16_t  get_img_size() { return img_size; }
    virtual bool is_valid() = 0;
    virtual ~primitives_workload_base(){};
};


class primitives_workload {
    std::map<std::string, primitives_workload_base *> builder_by_name_;
    std::string model_type;

    primitives_workload() {};
    primitives_workload(primitives_workload const&) = delete;
    void operator=(primitives_workload const&)  = delete;
public:
    static primitives_workload &instance() {
        static primitives_workload instance_;
        return instance_;
    }

    void add(std::string name, primitives_workload_base *builder) {
        builder_by_name_[name] = builder;
    }

    primitives_workload_base* get(std::string name) {
        auto result = builder_by_name_.find(name);
        if(result==std::end(builder_by_name_)) throw std::runtime_error(std::string("'")+name+"' topology builder does not exist");
        else return result->second;
    }
};

// utility functions used by workloads
static inline void read_file_to_vector(std::vector<std::string> &container, std::string filename, bool skip_first_line) {
    std::fstream input_file(filename, std::ios::in);
    if(input_file.good() == true) {
        std::string text;
        if(skip_first_line) std::getline(input_file, text);
        while(!input_file.eof()) {
            std::getline(input_file, text);
            if(!text.empty()) container.push_back(text);
        }
        input_file.close();
    } else throw std::runtime_error(std::string("cannot load '")+filename+"'");
}
