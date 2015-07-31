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

#include "FreeImage_wraps.h"

FIBITMAP* fi::load_image_from_file( std::string filename )
{
    FIBITMAP          *bitmap=nullptr;
    FREE_IMAGE_FORMAT  image_format= FIF_UNKNOWN;


    // Attempting to obtain information about the file format based on extension
    image_format = FreeImage_GetFIFFromFilename( filename.c_str() );

    if ( image_format != FIF_UNKNOWN ) {
        bitmap = FreeImage_Load( image_format,filename.c_str() );
        if ( bitmap != nullptr ) return bitmap;
    }
    // Attempting to open the file, in spite of the lack of information about the file format
    for ( FREE_IMAGE_FORMAT iff : fi::common_formats ) {
        bitmap = FreeImage_Load( iff,filename.c_str() );
        if ( bitmap != nullptr ) return bitmap;
    }

    return bitmap;
}

FIBITMAP * fi::crop_image_to_square_and_resize( FIBITMAP * bmp_in,uint16_t new_size )
{
    uint16_t  width,height;
    FIBITMAP *bmp_temp1 = nullptr;
    FIBITMAP *bmp_temp2 = nullptr;
    uint16_t  margin;

    width = FreeImage_GetWidth( bmp_in );
    height = FreeImage_GetHeight( bmp_in );

    if ( width!=height ) {
        if ( width>height ) {
            margin = (width - height)/2;
            bmp_temp1 = FreeImage_Copy( bmp_in,margin,0,width-margin-1,height );
        }
        else {
            margin = (height - width)/2;
            bmp_temp1 = FreeImage_Copy( bmp_in,0,margin,width,height-margin-1 );
        }
        FreeImage_Unload( bmp_in );
    }
    else
        bmp_temp1 = bmp_in;

    bmp_temp2 = FreeImage_Rescale( bmp_temp1,new_size,new_size, FILTER_CATMULLROM );

    FreeImage_Unload( bmp_temp1 );

    return bmp_temp2;
}

FIBITMAP * fi::resize_image_to_square(FIBITMAP * bmp_in, uint16_t new_size)
{
    FIBITMAP *bmp_temp1 = nullptr;
    bmp_temp1 = FreeImage_Rescale(bmp_in, new_size, new_size, FILTER_CATMULLROM);
    FreeImage_Unload(bmp_in);
    return bmp_temp1;
}
