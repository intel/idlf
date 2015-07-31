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
#include <iostream>

#include "tester/common/test_aggregator.h"
#include "tester/common/devices_aggregator.h"
#include "tester/common/results_aggregator.h"


int main(int argc,char *argv[])
{
    uint32_t  fails=0;
    std::vector<std::string> test_list;

    std::shared_ptr<results_aggregator>   tests_results(new results_aggregator);
    std::shared_ptr<devices_aggregator>   devices(new devices_aggregator);

    if(!devices.get()->add("device_cpu"+dynamic_library_extension))
        std::cerr  << "Unable to load the CPU device" << std::endl;

    if(!devices.get()->add("device_gpu"+dynamic_library_extension))
       std::cerr  << "Unable to load the GPU device" << std::endl;

    try {
        test_aggregator::instance().init_aggregators(devices,tests_results);
        test_aggregator::instance().get_tests_list(test_list);
        for(auto test_name : test_list)  {
            auto test = test_aggregator::instance().get(test_name);
            if(!test->run()) ++fails;
        }
        tests_results->save_results(get_timestamp()+"_test_results.csv");

        return fails;
    }
    catch(std::runtime_error &error) {
        std::cout << "error: " << error.what() << std::endl;
        return -1;
    }
    catch(...) {
        std::cout << "unknown error" << std::endl;
        return -1;
    }
}