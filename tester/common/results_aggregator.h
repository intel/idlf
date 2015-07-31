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
#include "tester/common/optional.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>

template <class DataType>
DataType GetMedian(std::vector<DataType> &values)
{
    std::sort(values.begin(),values.end());
    size_t size = values.size();
    if (size % 2 == 0)
        return (values[size / 2 - 1] + values[size / 2]) / 2;
    return values[size / 2];
}

struct test_measurement_result {

private:
    std::vector <std::string>  logs;

public:
    test_measurement_result();
    std::string                description;           // Short description of current test or measurement,
                                                      // which will be put in the output file
    std::string                datetimestamp;
    uint16_t                   loops = 1;
    optional <uint64_t>        time_consumed;         // | one value when one measurement is performed,
    optional <uint64_t>        clocks_consumed;       // } either average value
    optional <double>          power_consumed;        // | in the case of serial measurements

    optional <uint64_t>        time_consumed_min;     // |
    optional <uint64_t>        time_consumed_max;     // } These values will be filled
    optional <uint64_t>        clocks_consumed_min;   // | in case of a series of measurements (loops>1)
    optional <uint64_t>        clocks_consumed_max;   // | Otherwise, the fields remain empty
    optional <double>          power_consumed_min;    // |
    optional <double>          power_consumed_max;    // |

    bool                       passed=true;           // Set the value to false if something goes wrong

    void operator<<(const std::string &memo) {
        logs.push_back(memo);
        std::cout << memo << std::endl;
    }

    friend std::ostream& operator<<(std::ostream& os,const test_measurement_result& obj)
    {
        for(auto piece : obj.logs)
           os << piece << "|";
        return os;
    }

};

class results_aggregator {
private:
    std::vector<test_measurement_result>  results;
    std::string  collect_machine_info();

public:
    results_aggregator();
    ~results_aggregator();
    friend void* operator<<(results_aggregator* _results,const test_measurement_result& result) {
        _results->results.push_back(result);
        return _results;
    }
    void save_results(std::string filename);
};
