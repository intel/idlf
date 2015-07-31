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

#include "gtest/gtest.h"

#include "device/api/nn_device_api.h"
#include "device/api/nn_device_interface_0.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST( cpu_int16_device_load_and_unload, callbacks_test )
{
    // nn_device_load
    EXPECT_LT(nn_device_load(nullptr), 0);          // no error on invalid pointer

    nn_device_description_t dd;
    const auto invalid_first = static_cast<decltype(dd.version_first)>(-1);
    const auto invalid_last = static_cast<decltype(dd.version_first)>(-2);
    dd.version_first = invalid_first;
    dd.version_last = invalid_last;

    EXPECT_EQ(0, nn_device_load(&dd));              // non-zero return code on valid call
    EXPECT_NE(invalid_first, dd.version_first);     // nn_device_description_t::version_first is incorrect
    EXPECT_NE(invalid_last, dd.version_last);       // nn_device_description_t::version_last is incorrect
    EXPECT_LE(dd.version_first, dd.version_last);   // nn_device_description_t::version_first is greater than ::version_last

    // nn_device_interface_open & close
    {
        uint8_t buffer[4096];

        // nn_device_interface_open parameter validation
        EXPECT_GT(0, nn_device_interface_open(invalid_last, buffer));         // no error on invalid version
        EXPECT_GT(0, nn_device_interface_open(dd.version_first, nullptr));    // no error on invalid buffer

        // nn_device_interface_close parameter validation
        EXPECT_GT(0, nn_device_interface_close(nullptr));                                   // no error on invalid interface pointer
    }

    { // interface version 0
        const uint16_t interface_version = 0;
        nn_device_interface_0_t di;
        if(interface_version>=dd.version_first && interface_version<=dd.version_last) {
            EXPECT_EQ(0, nn_device_interface_open(interface_version, &di));
            EXPECT_EQ(interface_version, di.version);                   // returned version matches requested
            EXPECT_NE(nullptr, di.device);                              // non-null device returned
            EXPECT_NE(nullptr, di.workflow_create_function);            // non-null function pointer returned
            EXPECT_NE(nullptr, di.workflow_delete_function);            // non-null function pointer returned
            EXPECT_NE(nullptr, di.workflow_metrics_query_function);     // non-null function pointer returned
            EXPECT_NE(nullptr, di.workflow_metrics_delete_function);    // non-null function pointer returned
            EXPECT_NE(nullptr, di.workflow_compile_function);           // non-null function pointer returned
            EXPECT_NE(nullptr, di.workload_execute_function);           // non-null function pointer returned
            EXPECT_NE(nullptr, di.workload_delete_function);            // non-null function pointer returned
            EXPECT_NE(nullptr, di.workflow_item_create_function);       // non-null function pointer returned
            EXPECT_NE(nullptr, di.workflow_item_validate_function);     // non-null function pointer returned
            EXPECT_NE(nullptr, di.workflow_item_delete_function);       // non-null function pointer returned
            EXPECT_NE(nullptr, di.parameter_get_function);              // non-null function pointer returned
            EXPECT_NE(nullptr, di.parameter_set_function);              // non-null function pointer returned
            EXPECT_NE(nullptr, di.translate_api_status_function);       // non-null function pointer returned
            EXPECT_EQ(0, nn_device_interface_close(&di));               // successful close of interface
        }
    }

    // nn_device_unload
    EXPECT_EQ(0, nn_device_unload()); // successful unload
}
