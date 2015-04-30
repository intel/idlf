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
#if 0
#include "common.h"
#include "../devices/api/device_api.h"

#if defined _WIN32

#include <windows.h>

void load_devices(
    callback_function_table* &cpu_funcs,
    callback_function_table* &gpu_funcs,
    callback_function_table* &fpga_funcs )
{
    HMODULE cpu_device = LoadLibraryA( "device_cpu.dll" );
    HMODULE gpu_device = LoadLibraryA( "device_gpu.dll" );
    HMODULE fpga_device = LoadLibraryA( "device_fpga.dll" );

    cpu_funcs = NULL;
    gpu_funcs = NULL;
    fpga_funcs = NULL;

    if( cpu_device != NULL )
    {
        bool success = true;
        cpu_funcs = new callback_function_table;

        if( cpu_funcs != NULL )
        {
            api_get_callbacks_t device_get_callbacks = (api_get_callbacks_t)GetProcAddress( cpu_device, "get_functions_callbacks" );

            if( device_get_callbacks != NULL )
            {
                if( ( *device_get_callbacks )( cpu_funcs ) )
                {
                    success = false;
                }
            }
            else
            {
                success = false;
            }

            if( !success )
            {
                delete cpu_funcs;
                cpu_funcs = NULL;

                FreeLibrary( cpu_device );
            }
        }
    }

    if( gpu_device != NULL )
    {
        bool success = true;
        gpu_funcs = new callback_function_table;

        if( gpu_funcs != NULL )
        {
            api_get_callbacks_t device_get_callbacks = (api_get_callbacks_t)GetProcAddress( gpu_device, "get_functions_callbacks" );

            if( device_get_callbacks != NULL )
            {
                if( ( *device_get_callbacks )( gpu_funcs ) )
                {
                    success = false;
                }
            }
            else
            {
                success = false;
            }

            if( !success )
            {
                delete gpu_funcs;
                gpu_funcs = NULL;

                FreeLibrary( gpu_device );
            }
        }
    }

    if( fpga_device != NULL )
    {
        bool success = true;
        fpga_funcs = new callback_function_table;

        if( fpga_funcs != NULL )
        {
            api_get_callbacks_t device_get_callbacks = (api_get_callbacks_t)GetProcAddress( fpga_device, "get_functions_callbacks" );

            if( device_get_callbacks != NULL )
            {
                if( ( *device_get_callbacks )( fpga_funcs ) )
                {
                    success = false;
                }
            }
            else
            {
                success = false;
            }

            if( !success )
            {
                delete fpga_funcs;
                fpga_funcs = NULL;

                FreeLibrary( fpga_device );
            }
        }
    }
}

#elif defined(__linux__)

#include <dlfcn.h>

void load_devices(
    callback_function_table* &cpu_funcs,
    callback_function_table* &gpu_funcs,
    callback_function_table* &fpga_funcs )
{
    void *cpu_device = dlopen("./device_cpu.so", RTLD_LAZY);
    void *gpu_device = dlopen("./device_gpu.so", RTLD_LAZY);
    void *fpga_device = dlopen("./device_fpga.so", RTLD_LAZY);

    cpu_funcs = 0;
    gpu_funcs = 0;
    fpga_funcs = 0;

    if( cpu_device )
    {
        bool success = true;
        cpu_funcs = new callback_function_table;

        if( cpu_funcs )
        {
            api_get_callbacks_t device_get_callbacks = (api_get_callbacks_t)dlsym( cpu_device, "get_functions_callbacks" );

            if( device_get_callbacks )
            {
                if( ( *device_get_callbacks )( cpu_funcs ) )
                {
                    success = false;
                }
            }
            else
            {
                success = false;
            }

            if( !success )
            {
                delete cpu_funcs;
                cpu_funcs = 0;

                dlclose( cpu_device );
            }
        }
    }

    if( gpu_device )
    {
        bool success = true;
        gpu_funcs = new callback_function_table;

        if( gpu_funcs )
        {
            api_get_callbacks_t device_get_callbacks = (api_get_callbacks_t)dlsym( gpu_device, "get_functions_callbacks" );

            if( device_get_callbacks )
            {
                if( ( *device_get_callbacks )( gpu_funcs ) )
                {
                    success = false;
                }
            }
            else
            {
                success = false;
            }

            if( !success )
            {
                delete gpu_funcs;
                gpu_funcs = 0;

                dlclose( gpu_device );
            }
        }
    }

    if( fpga_device )
    {
        bool success = true;
        fpga_funcs = new callback_function_table;

        if( fpga_funcs )
        {
            api_get_callbacks_t device_get_callbacks = (api_get_callbacks_t)dlsym( fpga_device, "get_functions_callbacks" );

            if( device_get_callbacks )
            {
                if( ( *device_get_callbacks )( fpga_funcs ) )
                {
                    success = false;
                }
            }
            else
            {
                success = false;
            }

            if( !success )
            {
                delete fpga_funcs;
                fpga_funcs = 0;

                dlclose( fpga_device );
            }
        }
    }
}

#else
#error "OS not supported!"
#endif
#endif