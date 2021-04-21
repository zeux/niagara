cmake_minimum_required(VERSION 3.14)

if(DEFINED ENV{VCPKG_ROOT})
  set(USE_VCPKG YES)
  if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE
      "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  endif()
endif()

project(niagara)
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib") # .lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib") # .dll
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin") # .exe

if(USE_VCPKG)
  find_package(glfw3 CONFIG REQUIRED)
  find_package(glm CONFIG REQUIRED)
  #find_package(volk CONFIG REQUIRED)
  add_subdirectory(extern/volk)
  find_package(meshoptimizer CONFIG REQUIRED)
else()
  add_subdirectory(extern)
  add_library(meshoptimizer::meshoptimizer ALIAS meshoptimizer)
endif()
    
add_subdirectory(src)
