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

#include "../../api/nn_device_api.h" //DODALEM
#include "../../api/nn_device_interface_0.h"
#include "nn_device_interface_0_functions.h"
#include "../core/layers_opencl.h"
            
#include <cassert>
#include <cstring>
#include <iostream>
#include <ostream>
#include <stdexcept>


// interface tables
nn_device_interface_0_t device_interface_0 = {
    0,              // version
    0,
 nn_workflow_create_0_function,
 nn_workflow_delete_0_function,
 nn_workflow_metrics_query_0x0_function,
 nn_workflow_metrics_delete_0x0_function,
 nn_workflow_compile_0x0_function,
 nn_workload_execute_0x0_function,
 nn_workload_delete_0x0_function,
 nn_workflow_item_create_0_function,
 nn_workflow_item_validate_0x0_function,
 nn_workflow_item_delete_0_function,
 nn_device_parameter_get_0x0_function,
 nn_device_parameter_set_0x0_function,
 nn_translate_api_status_0_function
};


/*
Loads & initializes device.

If succeeds fills device description structure and returns non-negative value.
If fails returns negative error-code.[TODO]
*/
int32_t nn_device_load(
    nn_device_description_t *const description  /* pointer to description structure */
) {

    if( description == nullptr )
    {
        return -1;
    }

    try {
       device_interface_0.device =  reinterpret_cast<nn_device_t*>(new device_gpu::ocl_toolkit());
    }
    catch( device_gpu::runtime_error err ) 
    {
        std::cerr << err.what() << std::endl;
        return -1;
    }
    catch(...) {
       assert(0);   // This should not happen. If we ever throw wrong type of object 
       return -1;   // It should be caught here
    }

    description->version_first = 0;
    description->version_last  = 0;
    return 0;
};


/*
Opens interface: fills structure with function pointers that client/scheduler can use.

If succeeds returns non-negative value.
If fails returns negative error-code.[TODO]
*/
int32_t nn_device_interface_open(
    uint32_t version,               /* version of interface to create */
    void *const device_interface    /* pointer to interface structure */
) {
    if(!device_interface) return -2;

    switch(version) {
    default:
        return -1;
    case 0: 
        std::memcpy(device_interface, &device_interface_0, sizeof(device_interface_0));
        return 0;
    }
}


/*
Closes interface: clears structure with function pointers that client/scheduler can use and releases any outstanding client memory

If succeeds returns non-negative value.
If fails returns negative error-code.[TODO]
*/
int32_t nn_device_interface_close(
    void *const device_interface           /* pointer to interface structure */
) {
    if(!device_interface) return -1;
    switch(*reinterpret_cast<unsigned char *const>(device_interface)) {
    case 0: {
            break;
        }
    default:
        return -2;
    }
    return 0;
}


/*
Unloads & de-initializes device.

If succeeds returns non-negative value.
If fails returns negative error-code.[TODO]
*/
int32_t nn_device_unload() {

    delete reinterpret_cast<device_gpu::ocl_toolkit*>(device_interface_0.device);
    device_interface_0.device = nullptr;
    
    return 0;
}
