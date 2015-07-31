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

#include <vector>
#include <iostream>
#include "tester/common/devices_aggregator.h"
#include "tester/common/test_common_tools.h"
#include "tester/common/test_aggregator.h"
#include "tester/common/results_aggregator.h"
#include "tester/common/workflows_for_tests.h"


class test_base {

protected:
    devices_aggregator            *devices=nullptr;
    results_aggregator            *tests_results=nullptr;
    std::string                    test_description;

    virtual bool init() = 0;
    virtual bool done() = 0;


public:
    test_base() {};
    virtual ~test_base(){};
    bool init_aggregators(devices_aggregator *_devices,
                          results_aggregator *_tests_results) {

        if(_devices!=nullptr)  devices=_devices;
        else return false;

        if(_tests_results!=nullptr)  tests_results=_tests_results;
        else return false;

        return true;
    };

    virtual bool run() = 0;

    std::string get_description(){
        return test_description;
    };
};
