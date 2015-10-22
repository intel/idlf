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
*/#pragma once

#include <cstdint>

void ult_nn_convolve_fst_simplified(
    float* input_ref,
    float* output_ref,
    float* kernel_ref,
    float* bias_ref,
    uint64_t num_output_feature_maps,
    uint64_t num_input_feature_maps,
    uint64_t width,
    uint64_t height,
    uint64_t kernel_width,
    uint64_t kernel_height,
    uint64_t stride_col,
    uint64_t stride_row,
    uint64_t offset_x,
    uint64_t offset_y);

void ult_nn_convolve_fst_simplified_no_offset(
    float* input_ref,
    float* output_ref,
    float* kernel_ref,
    uint64_t num_output_feature_maps,
    uint64_t num_input_feature_maps,
    uint64_t width,
    uint64_t height,
    uint64_t kernel_width,
    uint64_t kernel_height,
    uint64_t stride_col = 1,
    uint64_t stride_row = 1);

