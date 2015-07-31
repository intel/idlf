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
#include "common/FreeImage64/FreeImage.h"
#include <stdint.h>
#include <string>
namespace fi {

#ifdef __linux__
    const FREE_IMAGE_FORMAT common_formats[] = { FIF_JPEG, FIF_J2K, FIF_JP2, FIF_PNG, FIF_BMP, FIF_GIF, FIF_TIFF };
#else
    const FREE_IMAGE_FORMAT common_formats[] = { FIF_JPEG, FIF_J2K, FIF_JP2, FIF_PNG, FIF_BMP, FIF_WEBP, FIF_GIF, FIF_TIFF };
#endif

    FIBITMAP * load_image_from_file( std::string );
    FIBITMAP * crop_image_to_square_and_resize( FIBITMAP *,uint16_t );
    FIBITMAP * resize_image_to_square(FIBITMAP *, uint16_t);

    typedef  FIBITMAP* (*prepare_image_t)(FIBITMAP *, uint16_t);

}
