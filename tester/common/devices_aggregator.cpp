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

#include "tester/common/devices_aggregator.h"

// -----------------------------------------------------------------------------------------
// ----------------------- tested_device --------------------------------------------------------

tested_device::tested_device(std::string _filename): filename(_filename)
{

    device_handle = dlopen( filename.c_str(), RTLD_LAZY);
    if(!device_handle) throw std::runtime_error(std::string("failed to open '"+filename+"' device, reason-> " + dlerror()));

    device_load     = dlsym<decltype(nn_device_load)>("nn_device_load");
    device_unload   = dlsym<decltype(nn_device_unload)>("nn_device_unload");
    interface_open  = dlsym<decltype(nn_device_interface_open)>("nn_device_interface_open");
    interface_close = dlsym<decltype(nn_device_interface_close)>("nn_device_interface_close");

    if(0!=device_load(&description))
        throw std::runtime_error(std::string("failed to load device '")+filename+"'");

    if(0!=interface_open(0,&device_interface))
        throw std::runtime_error(std::string("failed to open interface from device '"+filename+"'"));
}

tested_device::~tested_device()
{
    if(device_handle) {
        interface_close(&device_interface);
        device_unload();
        dlclose(device_handle);
    }
}
nn_device_interface_0_t* tested_device::get_device_interface()
{
    return &device_interface;
}

std::string tested_device::get_device_description() {
    std::ostringstream dscr;
    dscr << "Device ("
         << "name: " << description.name << "|"
         << "type: " << description.type << "|"
         << "version: " << description.version_first << "." << description.version_last << "|"
         << "description: " << description.description
         << ")";
    return dscr.str();
}
// -----------------------------------------------------------------------------------------
// ----------------------- devices_aggregator --------------------------------------------

bool devices_aggregator::add(std::string filename) {
    try{
        std::shared_ptr<tested_device>  device_ptr(new tested_device(filename));
        if(device_ptr!=nullptr)
            map_of_devices.insert(std::make_pair(filename,device_ptr));
        return true;
    }
    catch(...) {
        return false;
    }

}

tested_device* devices_aggregator::get(std::string name) {
    auto result = map_of_devices.find(name);
    if(result==std::end(map_of_devices))
        throw std::runtime_error(std::string("'")+name+"' does not exist");
    else
        return result->second.get();
}
