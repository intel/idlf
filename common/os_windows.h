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

#ifdef WIN32

#include <string>
#include <io.h>
#include <regex>

// dlopen, dlclose, dlsym, dlerror
const int RTLD_LAZY = 0;

// opendir, closedir, readdir
const int DT_UNKNOWN = 0;
const int DT_REG = 8;

extern const std::string show_HTML_command;
extern const std::string dynamic_library_extension;

void *dlopen(const char *filename,int);
int dlclose(void *handle);
void *dlsym(void *handle, const char *symbol);
char *dlerror(void);

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

DIR *opendir(const char *name);
int closedir(DIR *dir);
dirent *readdir(DIR *dir);
bool is_regular_file(std::string& dirname, struct dirent* folder_entry);
std::string get_timestamp();

#endif //windows