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
// OS & compiler-specific constants,  functions & workarounds
// needed Linux APIs missing from other OSes should be emulated
#if defined _WIN32
#   include "common/os_windows.h"
#else
#   include "common/os_linux.h"
#endif

#include "workflow_builder.h"
#include "device/api/nn_device_interface_0.h"
#include "device/api/nn_device_api.h"
#include "common/nn_data_tools.h"

// RAII for library; throws runtime_error when fails
struct scoped_library {
    std::string name;
    void *handle;
public:
    scoped_library(std::string arg_name) : name(arg_name), handle(dlopen(name.c_str(), RTLD_LAZY)) {
        if(!handle) throw std::runtime_error(std::string("failed to open '")+name+"' device");
    }
    ~scoped_library() {
        dlclose(handle);
    }
    template<typename T_type> T_type *dlsym(std::string symbol) {
        if(void *sym=::dlsym(handle, symbol.c_str()))
            return reinterpret_cast<T_type*>(sym);
        else throw std::runtime_error(std::string("unable to get symbol '")+symbol+"' from device '"+name+"'");
    }
};

// RAII for device 
class scoped_device {
    scoped_library              &library_;
    decltype(nn_device_load)    *load_;
    decltype(nn_device_unload)  *unload_;
    friend class scoped_interface_0;
public:
    nn_device_description_t description;
    scoped_device(scoped_library &library) 
    : library_(library)
    , load_(  library_.dlsym<decltype(nn_device_load)>("nn_device_load"))
    , unload_(library_.dlsym<decltype(nn_device_unload)>("nn_device_unload")) {
        if(0!=load_(&description)) throw std::runtime_error(std::string("failed to load device '")+library_.name+"'");
    }
    ~scoped_device() {
        unload_();
    }
};

// RAII for interface
class scoped_interface_0 : public nn_device_interface_0_t {
    scoped_library                         &library_;
    decltype(nn_device_interface_open)     *open_;
    decltype(nn_device_interface_close)    *close_;
public:
    scoped_interface_0(scoped_device &device)
    : library_(device.library_)
    , open_( library_.dlsym<decltype(nn_device_interface_open)>("nn_device_interface_open"))
    , close_(library_.dlsym<decltype(nn_device_interface_close)>("nn_device_interface_close"))
    {
        if(0!=open_(0, this)) throw std::runtime_error(std::string("failed to open interface ")+std::to_string(version)+" from device '"+library_.name+"'");
    }
    ~scoped_interface_0() {
        close_(this);
    }
};

void run_mnist_training( scoped_library      &library,
                         scoped_device       &device,
                         scoped_interface_0  &interface_0,
                         nn_workload_t *workload,
                         workflow_builder_base* builder,
                         char *argv[], 
                         std::map<std::string, std::string> &config, 
                         const int config_batch );

void run_images_classification( scoped_library      &library,
                         scoped_device       &device,
                         scoped_interface_0  &interface_0,
                         nn_workload_t *workload,
                         workflow_builder_base* builder,
                         char *argv[], 
                         std::map<std::string, std::string> &config, 
                         const int config_batch );

void run_mnist_classification( scoped_library      &library,
                         scoped_device       &device,
                         scoped_interface_0  &interface_0,
                         nn_workload_t *workload,
                         workflow_builder_base* builder,
                         char *argv[], 
                         std::map<std::string, std::string> &config, 
                         const int config_batch );
