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

# Find FreeImage includes and library
#
# This module defines
#  FreeImage_INCLUDE_DIRS
#  FreeImage_LIBRARIES, the libraries to link against to use FreeImage.
#  FreeImage_LIBRARY_DIRS, the location of the libraries
#  FreeImage_FOUND, If false, do not try to use FreeImage
#
# Copyright © 2007, Matt Williams
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

IF (FreeImage_LIBRARIES AND FreeImage_INCLUDE_DIRS)
	SET(FreeImage_FIND_QUIETLY TRUE) # Already in cache, be silent
ELSE(FreeImage_LIBRARIES AND FreeImage_INCLUDE_DIRS)
	MESSAGE(STATUS "Looking for FreeImage")
ENDIF (FreeImage_LIBRARIES AND FreeImage_INCLUDE_DIRS)

SET(FreeImage_INCLUDE_SEARCH_DIRS
	${FreeImage_LIBRARY_SEARCH_DIRS}
	${CMAKE_LIBRARY_PATH}
	/usr/include
	/usr/local/include
	/opt/include
	/opt/freeimage/include
)

SET(FreeImage_LIBRARY_SEARCH_DIRS
	${FreeImage_LIBRARY_SEARCH_DIRS}
	${CMAKE_LIBRARY_PATH}
	/usr/lib
	/usr/local/lib
	/opt/lib
	/opt/freeimage/lib
)

FIND_PATH(FreeImage_INCLUDE_DIRS FreeImage.h ${FreeImage_INCLUDE_SEARCH_DIRS})
FIND_LIBRARY(FreeImage_LIBRARIES freeimage PATHS ${FreeImage_LIBRARY_SEARCH_DIRS})

#Do some preparation
SEPARATE_ARGUMENTS(FreeImage_INCLUDE_DIRS)
SEPARATE_ARGUMENTS(FreeImage_LIBRARIES)

MARK_AS_ADVANCED(FreeImage_INCLUDE_DIRS FreeImage_LIBRARIES FreeImage_LIBRARY_DIRS)

IF (FreeImage_INCLUDE_DIRS AND FreeImage_LIBRARIES)
	SET(FreeImage_FOUND TRUE)
ENDIF (FreeImage_INCLUDE_DIRS AND FreeImage_LIBRARIES)

IF (FreeImage_FOUND)
	IF (NOT FreeImage_FIND_QUIETLY)
		MESSAGE(STATUS "  libraries : ${FreeImage_LIBRARIES} from ${FreeImage_LIBRARY_DIRS}")
		MESSAGE(STATUS "  includes  : ${FreeImage_INCLUDE_DIRS}")
	ENDIF (NOT FreeImage_FIND_QUIETLY)
ELSE (FreeImage_FOUND)
	IF (FreeImage_FIND_REQUIRED)
		MESSAGE(FATAL_ERROR "Could not find FreeImage")
	ENDIF (FreeImage_FIND_REQUIRED)
ENDIF (FreeImage_FOUND)
