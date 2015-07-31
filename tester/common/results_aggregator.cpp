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

#include "tester/common/results_aggregator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

#ifdef __linux__
#  include "common/os_linux.h"
#  include <cpuid.h>
#  define call_cpu_id( CPUInfo, x ); __get_cpuid (x, CPUInfo, CPUInfo + 1, CPUInfo + 2, CPUInfo + 3);
#elif defined(_WIN32) || defined(WIN32)
#  include "common/os_windows.h"
#  include <intrin.h>
#  define call_cpu_id( CPUInfo, x ); __cpuid( CPUInfo, x );
#endif

test_measurement_result::test_measurement_result() {
    datetimestamp = get_timestamp();
}

results_aggregator::results_aggregator() {}

results_aggregator::~results_aggregator() {}

void results_aggregator::save_results( std::string filename ) {
    std::fstream output_file;

    output_file.open( filename, std::ios::out | std::ios::trunc );
    if( output_file.is_open() ) {
        // TODO: Develop a function to retrieve information about the machine
        std::string  machine_description = collect_machine_info();

        output_file
            << "timestamp;"
            << "return;"
            << "description;"
            << "machine;"
            << "loops;"
            << "clocks;"
            << "clocks_min;"
            << "clocks_max;"
            << "time;"
            << "time_min;"
            << "time_max;"
            << "power;"
            << "power_min;"
            << "power_max;"
            << "log"
            << std::endl;
        for( auto result : results ) {
            output_file
                << result.datetimestamp << ";"
                << (result.passed ? "PASS;" : "FAIL;")
                << result.description << ";"
                << machine_description << ";"
                << result.loops << ";"
                << std::to_string( result.clocks_consumed ) << ";"
                << std::to_string( result.clocks_consumed_min ) << ";"
                << std::to_string( result.clocks_consumed_max ) << ";"
                << std::to_string( result.time_consumed ) << ";"
                << std::to_string( result.time_consumed_min ) << ";"
                << std::to_string( result.time_consumed_max ) << ";"
                << std::to_string( result.power_consumed ) << ";"
                << std::to_string( result.power_consumed_min ) << ";"
                << std::to_string( result.power_consumed_max ) << ";"
                << result      //puts result.logs to output stream
                << std::endl;
        }
        output_file.close();
    } else {
        std::cerr << "file access denied" << std::endl;
    }
}
std::string  results_aggregator::collect_machine_info() {
    char cpu_brand[0x40];
#ifdef __linux__
    unsigned //int CPUInfo[4] = { 0 };
#endif
        int CPUInfo[4] = { 0 };
    unsigned n_id;

    call_cpu_id( CPUInfo, 0x80000000 );

    n_id = CPUInfo[0];
    memset( cpu_brand, 0, sizeof(cpu_brand) );

    // Get the information associated with each extended ID.
    for( unsigned i = 0x80000000; i <= n_id; ++i ) {
        call_cpu_id( CPUInfo, i );

        // Interpret CPU brand string and cache information.
        if( i == 0x80000002 )
            memcpy( cpu_brand, CPUInfo, sizeof(CPUInfo) );
        else if( i == 0x80000003 )
            memcpy( cpu_brand + 16, CPUInfo, sizeof(CPUInfo) );
        else if( i == 0x80000004 )
            memcpy( cpu_brand + 32, CPUInfo, sizeof(CPUInfo) );
    }

    return std::string( cpu_brand );
}