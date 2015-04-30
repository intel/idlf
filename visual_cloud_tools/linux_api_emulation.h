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

#include <io.h>
#include <string>
#include <windows.h>

// dlopen, dlclose, dlsym
const int RTLD_LAZY = 0;
void *dlopen(const char *filename, int) {
    return LoadLibraryA(filename);
}
int dlclose(void *handle) {
    return FreeLibrary(static_cast<HINSTANCE>(handle))==0 ? 1 : 0;
}
void *dlsym(void *handle, const char *symbol) {
    return GetProcAddress(static_cast<HINSTANCE>(handle), symbol);
}

// opendir, closedir, readdir
const int DT_UNKNOWN = 0;
const int DT_REG = 8;

struct dirent {
    char           *d_name;
    unsigned char   d_type;
    std::string     str_d_name;
};

struct DIR {
    intptr_t        handle; 
    _finddata_t     fileinfo;
    dirent          result;
    std::string     name;
};

DIR *opendir(const char *name)
{
    if(!name || !*name) return nullptr;
    else {
        DIR *dir = new DIR;
        dir->name = name;
        dir->name += dir->name.back()=='/' || dir->name .back()=='\\' ? "*" : "/*";
        if(-1==(dir->handle = _findfirst(dir->name.c_str(), &dir->fileinfo))) {
            delete dir;
            dir = nullptr;
        } else dir->result.d_name = nullptr;
        return dir;
    }
}

int closedir(DIR *dir) {
    if(!dir) return -1;
    else {
        intptr_t handle = dir->handle;
        delete dir;
        return -1==handle ? -1 : _findclose(handle);
    }
}

dirent *readdir(DIR *dir) {
    if(!dir || -1==dir->handle) return nullptr;
    else {
        if(dir->result.d_name && -1==_findnext(dir->handle, &dir->fileinfo)) return nullptr;
        else {
            dirent *result = &dir->result;
            result->d_name = dir->fileinfo.name;
            result->d_type = dir->fileinfo.attrib & (_A_ARCH|_A_NORMAL|_A_RDONLY) ? DT_REG : DT_UNKNOWN;
            return result;
        }
    }
}
