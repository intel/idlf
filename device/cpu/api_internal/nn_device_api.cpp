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

#include "nn_device_interface_0_internal.h"

#include <cstring>


// interface tables
nn_device_interface_0_t device_interface_0 = {
    0,              // version
    0,              // device [will be overwritten]
    nn_workflow_create_0_function,
    nn_workflow_delete_0_function,
    nn_workflow_metrics_query_0_function,
    nn_workflow_metrics_delete_0_function,
    nn_workflow_compile_0_function,
    nn_workload_execute_0_function,
    nn_workload_delete_0_function,
    nn_workflow_item_create_0_function,
    nn_workflow_item_validate_0_function,
    nn_workflow_item_delete_0_function,
    nn_device_parameter_get_0_function,
    nn_device_parameter_set_0_function,
    nn_translate_api_status_0_function
};

/* loads & initializes device
   If succeeds fills device description structure and returns non-negative value.
   Returns:
      0: success
     -1: load failed
     -2: invalid pointer
*/

NN_API_CALL int32_t NN_API_CALL_CONVENTION  nn_device_load(
    nn_device_description_t *const description  /* pointer to description structure */
) {
    if(!description) return -2;
    else {

#define TO_STR_INNER(str) #str
#define TO_STR(str) TO_STR_INNER(str)

#ifdef DEBUG
#define COMPILER_CONF "DEBUG configuration";
#else
#define COMPILER_CONF "RELEASE configuration";
#endif
#if defined _WIN32
#   if defined (__ICC) || defined (__INTEL_COMPILER)
#      define COMPILER_DSCR   "ICC: " __VERSION__ "(Windows)"
#   elif defined _MSC_VER
#      define COMPILER_DSCR   "MSC: (Windows)" _MSC_FULL_VER
#   else
#      define COMPILER_DSCR   "Unknown compiler (Windows)"
#   endif
#else
#   if defined (__ICC) || defined (__INTEL_COMPILER)
#      define COMPILER_DSCR   "ICC: " __VERSION__ "(Unix)"
#   elif defined __GNUC__
#      define COMPILER_DSCR   "GNUC: " __VERSION__ "(Unix)"
#   else
#      define COMPILER_DSCR   "Unknown compiler (Unix)"
#   endif
#endif

        const char *device_description = "floating point CPU device|compiled: " __DATE__ __TIME__"|using " TO_STR(COMPILER_DSCR) "|with " COMPILER_CONF;

        *description = nn_device_description_t{
            NN_DEVICE_TYPE_CPU, // type
            0,                  // version_first
            0,                  // version_last
            "CPU device",       // name
            device_description  // description
        };

        return 0;
    }
    return -1;
};


/* opens interface
   Fills interface structure with function pointers that client/scheduler can use.
   Returns:
      0: success
     -1: interface open failed
     -2: invalid pointer
     -3: unsupported version (not in range returned from nn_device_load)
*/
NN_API_CALL int32_t NN_API_CALL_CONVENTION nn_device_interface_open(
    uint32_t version,               /* version of interface to create */
    void *const interface           /* pointer to interface structure */
) {
    if(!interface) return -2;

    switch(version) {
    default:
        return -3;
    case 0: 
        memcpy(interface, &device_interface_0, sizeof(device_interface_0));
        reinterpret_cast<nn_device_interface_0_t*>(interface)->device = reinterpret_cast<nn_device_t*>(new nn_device_internal);
        if (reinterpret_cast<nn_device_interface_0_t*>(interface)->device == nullptr)
        {
            return -1;
        }
        return 0;
    }
}


/* closes interface
   Clears interface structure and releases any outstanding client memory.
   Returns:
      0: success
     -1: interface close failed
     -2: invalid pointer
     -3: trying to release interface by driver that did not create it
*/
NN_API_CALL int32_t NN_API_CALL_CONVENTION nn_device_interface_close(
    void *const interface           /* pointer to interface structure */
) {
    // TODO: nn_device_interface_close: validate that interface is created by driver that created it
    if(!interface) return -2;
    switch(*reinterpret_cast<unsigned char *const>(interface)) {
    case 0: {
        delete reinterpret_cast<nn_device_internal*>(reinterpret_cast<nn_device_interface_0_t*>(interface)->device);
        reinterpret_cast<nn_device_interface_0_t*>(interface)->device = nullptr;
            break;
        }
    default:
        return -3;
    }
    return 0;
}

/* unloads & de-initializes device.
   Performs pre-unload steps, releases all resources & If succeeds returns non-negative value.
   Returns:
      0: success
     -1: unload failed
*/
NN_API_CALL int32_t NN_API_CALL_CONVENTION nn_device_unload() {
    return 0;
}
