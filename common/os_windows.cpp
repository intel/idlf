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
#ifdef WIN32

#ifndef NOMINMAX
#   define NOMINMAX
#endif

#include <windows.h>
#include <iomanip>
#include <chrono>
#include <sstream>
#include "os_windows.h"

const std::string show_HTML_command("START ");
const std::string dynamic_library_extension(".dll");

void *dlopen(const char *filename,int) {
    return LoadLibraryA(filename);
}

int dlclose(void *handle) {
    return FreeLibrary(static_cast<HINSTANCE>(handle))==0 ? 1 : 0;
}

void *dlsym(void *handle, const char *symbol) {
    return GetProcAddress(static_cast<HINSTANCE>(handle),symbol);
}

char *dlerror(void) {
    DWORD errCode = GetLastError();
    char *err;
    if (!FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
                        NULL,
                        errCode,
                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // default language
                        (LPTSTR) &err,
                        0,
                        NULL))
        return nullptr;
    static char buffer[1024];
    _snprintf_s(buffer, sizeof(buffer), "%s", err);
    LocalFree(err);
    return buffer;
}

DIR *opendir(const char *name)
{
    if(!name || !*name) return nullptr;
    else {
        DIR *dir = new DIR;
        dir->name = name;
        dir->name += dir->name.back()=='/' || dir->name .back()=='\\' ? "*" : "/*";
        if(-1==(dir->handle = _findfirst(dir->name.c_str(),&dir->fileinfo))) {
            delete dir;
            dir = nullptr;
        }
        else dir->result.d_name = nullptr;
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
        if(dir->result.d_name && -1==_findnext(dir->handle,&dir->fileinfo)) return nullptr;
        else {
            dirent *result = &dir->result;
            result->d_name = dir->fileinfo.name;
            result->d_type = dir->fileinfo.attrib & (_A_ARCH|_A_NORMAL|_A_RDONLY) ? DT_REG : DT_UNKNOWN;
            return result;
        }
    }
}

bool is_regular_file(std::string& dirname,struct dirent* folder_entry) {
    return folder_entry->d_type == DT_REG;
}

std::string get_timestamp() {
    std::stringstream timestamp;
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    struct tm _tm;
    localtime_s(&_tm,&in_time_t);
    timestamp<<std::put_time(&_tm,"%Y%m%d%H%M%S");
    return timestamp.str();
}

#endif // windows