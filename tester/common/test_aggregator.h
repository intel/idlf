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

#include <stdint.h>
#include <map>
#include <string>
#include "tester/common/test_base.h"

class test_aggregator {

    std::map<std::string, test_base *> test_by_name_;
    test_aggregator() {};
    test_aggregator(test_aggregator const&) = delete;
    void operator=(test_aggregator const&) = delete;

public:
    static test_aggregator &instance() {
        static test_aggregator instance_;
        return instance_;
    }
    void add(test_base *test) {
        test_by_name_[test->get_description()] = test;
    }
    test_base* get(std::string name) {
        auto result = test_by_name_.find(name);
        if(result==std::end(test_by_name_)) throw std::runtime_error(std::string("'")+name+"' definition does not exist");
        else return result->second;
    }
    size_t get_tests_list(std::vector<std::string> &tests_list) {
        for(auto item:test_by_name_)
            tests_list.push_back(item.first);
        return tests_list.size();
    }
    void init_aggregators(const std::shared_ptr<devices_aggregator>& _devices,
                          const std::shared_ptr<results_aggregator>&_tests_results) {
        for(auto test : test_by_name_)
            test.second->init_aggregators(_devices.get(),_tests_results.get());
    }
};

