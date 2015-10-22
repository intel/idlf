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

#include <memory>

size_t nn_aligned_size(size_t);
void* nn_allocate_aligned(size_t);
void nn_free_aligned(void*&);
void nn_delete_aligned(void*);

template <typename T>
inline void deleter(void* ptr)
{
    static_cast<T*>(ptr)->~T();
    nn_delete_aligned(ptr);
}

template <typename T>
inline std::unique_ptr<T, decltype(&nn_delete_aligned)> nn_make_unique_aligned(size_t size)
{
    auto buffer = (T*)nn_allocate_aligned(size * sizeof(T));
    if (buffer == nullptr) throw std::bad_alloc();
    return std::unique_ptr<T, decltype(&nn_delete_aligned)>(new (buffer)T, deleter<T>);
}

template <typename T>
inline std::shared_ptr<T> nn_make_shared_aligned(size_t size)
{
    auto buffer = (T*)nn_allocate_aligned(size * sizeof(T));
    if (buffer == nullptr) throw std::bad_alloc();
    return std::shared_ptr<T>(buffer, deleter<T>);
}

