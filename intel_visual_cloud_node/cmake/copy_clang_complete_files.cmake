# Copyright (c) 2014, Intel Corporation
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function(copy_clang_complete_files cur_dir)
    message("### Copying .clang_complete files: ###")
    file(GLOB_RECURSE items  RELATIVE ${cur_dir} ${cur_dir}/.clang_complete)
    # Get relative paths , attach them to SOURCE_DIR name (directory where project CMakeLists.txt is)
    # and copy .clang_complete files if possible
    foreach(item ${items})   
        string(REPLACE .clang_complete  "" rel_path ${item})
        if(IS_DIRECTORY ${SOURCE_DIR}/${rel_path})
            file(COPY ${cur_dir}/${item} DESTINATION ${SOURCE_DIR}/${rel_path})
            message("${cur_dir}/${item} ---> ${SOURCE_DIR}/${rel_path}")
        endif()
    endforeach(item) 
endfunction(copy_clang_complete_files)


copy_clang_complete_files(${CMAKE_CURRENT_BINARY_DIR})
