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

// OS & compiler-specific constants,  functions & workarounds
// needed Linux APIs missing from other OSes should be emulated
#if defined _WIN32
#   include "common/os_windows.h"
#else
#   include "common/os_linux.h"
#endif
#include <iostream>
#include "common/common_tools.h"

// returns list of files (path+filename) from specified directory
std::vector<std::string> get_directory_images(std::string images_path) {
    std::vector<std::string> result;
    if(DIR *folder = opendir(images_path.c_str())) {
        dirent *folder_entry;
        const auto image_file = std::regex(".*\\.(jpe?g|png|bmp|gif|j2k|jp2|tiff)$");
        while(folder_entry = readdir(folder))
            if(std::regex_match(folder_entry->d_name, image_file) && is_regular_file(images_path, folder_entry) )
                result.push_back(images_path+ "/" +folder_entry->d_name);
        closedir(folder);
    }
    return result;
}
