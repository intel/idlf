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

/*
This file contains API common to all Deep Learning Framework devices.

Lifetime of device:

[load device]
    Device is loaded into the application.
    Creation of per-device structures is kept to a minimum.
    User gets a range of supported interface versions together with device type (an identifier),
    device name and multi-line description (both in 8-bit ASCII, english).
    Comment: this is conceptual "load" operation. Loading dynamic library containing
             device itself must be done prior to this step.

[open interface]
    User application opens a specific interface version.
    Version must be in range of supported versions returned at [interface load] step.
    At this step all internal data structures needed by selected interface version are
    initialized.

[using the interface]
    Interface is used to create workflows, compile them into workloads and execute
    those on data.
    See nn_device_interface_#.h (# is an interface version in decimal).

[close interface]
    Interface is closed, all internal resources are released.
    Work that is being performed at closing time is cancelled.

[unload device]
    Device is unloaded.
    All per-device structures created at [device load] step are released.
    Comment: this is conceptual "unload" operation. Unloading dynamic library containing
             device itself must be done after this step.
*/

#include <stdint.h>
#include "nn_call_convention.h"


/* enumeration of device types; will be extended in future */
typedef enum {
    NN_DEVICE_TYPE_CPU = 0,
    NN_DEVICE_TYPE_GPU,
    NN_DEVICE_TYPE_LAST = NN_DEVICE_TYPE_GPU
} NN_DEVICE_TYPE;

/* description of device */
typedef struct {
    NN_DEVICE_TYPE  type;           /* device type */
    uint16_t        version_first;  /* first supported API version */
    uint16_t        version_last;   /*  last supported API version */
    const char     *name;           /* pointer to read-only memory with device name (single line) */
    const char     *description;    /* pointer to read-only memory with long device description (multi line)*/
} nn_device_description_t;



/*************************************************************************************************/

/* loads & initializes device
   If succeeds fills device description structure and returns non-negative value.
   Returns:
      0: success
     -1: load failed
     -2: invalid pointer */
NN_API_CALL int32_t NN_API_CALL_CONVENTION nn_device_load(
    nn_device_description_t *const description  /* pointer to description structure */
);



/* opens interface
   Fills interface structure with function pointers that client/scheduler can use.
   Returns:
      0: success
     -1: interface open failed
     -2: invalid pointer
     -3: unsupported version (not in range returned from nn_device_load) */
NN_API_CALL int32_t NN_API_CALL_CONVENTION nn_device_interface_open(
    uint32_t version,               /* version of interface to create */
    void *const device_interface    /* pointer to interface structure */
);



/* closes interface
   Clears interface structure and releases any outstanding client memory.
   Returns:
      0: success
     -1: interface close failed
     -2: invalid pointer
     -3: trying to release interface by driver that did not create it */
NN_API_CALL int32_t NN_API_CALL_CONVENTION nn_device_interface_close(
    void *const device_interface    /* pointer to interface structure */
);



/* unloads & de-initializes device.
   Performs pre-unload steps, releases all resources & If succeeds returns non-negative value.
   Returns:
      0: success
     -1: unload failed */
NN_API_CALL int32_t NN_API_CALL_CONVENTION nn_device_unload();
