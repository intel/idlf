# Do not modify this file

set (TARGET_NAME gtest)

# Main source for gtest
set  (MAIN_SRC
      "src/gtest-all.cc")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("" FILES ${MAIN_SRC})

# Properties->C/C++->General->Additional Include Directories
include_directories ("${CMAKE_SOURCE_DIR}/tester/g_ult/gtest/include" "${CMAKE_SOURCE_DIR}/tester/g_ult/gtest")

# Find Threadslib and create .lib

find_package(Threads REQUIRED)
add_library(${TARGET_NAME} STATIC ${MAIN_SRC})
target_link_libraries(${TARGET_NAME} ${CMAKE_THREAD_LIBS_INIT})
set_target_properties( ${TARGET_NAME}
                       PROPERTIES 
                       FOLDER tester/g_ult
                     )