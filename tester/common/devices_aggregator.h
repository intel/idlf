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

#if defined _WIN32
#   include "common/os_windows.h"
#else
#   include "common/os_linux.h"
#endif

#include <string>
#include <sstream>
#include <iomanip>
#include <map>
#include <memory>
#include "device/api/nn_device_interface_0.h"
#include "device/api/nn_device_api.h"



// -----------------------------------------------------------------------------------------
// ----------------------- tested_device --------------------------------------------------------

class tested_device {
private:
    std::string                             filename;
    void*                                   device_handle;
    nn_device_interface_0_t                 device_interface;
    nn_device_description_t                 description;
    decltype(nn_device_load)               *device_load;
    decltype(nn_device_unload)             *device_unload;
    decltype(nn_device_interface_open)     *interface_open;
    decltype(nn_device_interface_close)    *interface_close;


    tested_device()                            = delete;
    tested_device(tested_device const&)        = delete;
    void operator=(tested_device const&)       = delete;

    template<typename T_type> T_type *dlsym(std::string symbol) {
        if(void *sym=::dlsym(device_handle, symbol.c_str()))
            return reinterpret_cast<T_type*>(sym);
        else throw std::runtime_error(std::string("unable to get symbol '")+symbol+"' from device '"+filename+"'");
    }

public:
    tested_device(std::string);
   ~tested_device();
    nn_device_interface_0_t    *get_device_interface();
    std::string                 get_device_description();
};

// -----------------------------------------------------------------------------------------
// ----------------------- devices_aggregator --------------------------------------------
class  devices_aggregator {
private:
    std::map<std::string, std::shared_ptr<tested_device>> map_of_devices;
public:
    devices_aggregator() {};
    ~devices_aggregator() {};
    bool                     add(std::string);
    tested_device*           get(std::string);
};