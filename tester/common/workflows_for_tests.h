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
#include "tester/common/test_common_tools.h"
#include "common/nn_data_tools.h"
#include "device/api/nn_device_interface_0.h"
#include <map>
#include <string>




class workflows_for_tests_base {
protected:
    workflows_for_tests_base () {};

public:

    virtual nn_workflow_t *init_test_workflow(nn_device_interface_0_t *_di) = 0;
    virtual void cleanup() = 0;
    virtual bool is_valid() = 0;
    virtual ~workflows_for_tests_base(){};
};


class workflows_for_tests {
    std::map<std::string, workflows_for_tests_base *> test_workflow_by_name;
    workflows_for_tests() {};
    workflows_for_tests(workflows_for_tests const&) = delete;
    void operator=(workflows_for_tests const&)  = delete;
public:
    static workflows_for_tests &instance() {
        static workflows_for_tests instance_;
        return instance_;
    }

    void add(std::string name, workflows_for_tests_base *test_workflow) {
        test_workflow_by_name[name] = test_workflow;
    }

    workflows_for_tests_base* get(std::string name) {
        auto result = test_workflow_by_name.find(name);
        if(result==std::end(test_workflow_by_name)) throw std::runtime_error(std::string("'")+name+"' workflow topology, does not exist");
        else return result->second;
    }
};
