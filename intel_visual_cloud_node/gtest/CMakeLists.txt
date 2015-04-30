# Do not modify this file

# Main source for gtest
set  (MAIN_SRC
      "src/gtest-all.cc")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("" FILES ${MAIN_SRC})

# Properties->C/C++->General->Additional Include Directories
include_directories ("${PROJECT_SOURCE_DIR}/gtest/include" "${PROJECT_SOURCE_DIR}/gtest")

# Find Threadslib and create .lib

find_package(Threads REQUIRED)
add_library(gtest STATIC ${MAIN_SRC})
target_link_libraries(gtest ${CMAKE_THREAD_LIBS_INIT})