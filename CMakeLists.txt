cmake_minimum_required(VERSION 3.14)

project(niagara)

option(SHADER_DEBUG "Embed debug information into SPIV shaders" OFF)

file(GLOB_RECURSE GLSL_SOURCE_FILES "src/shaders/*.glsl")
file(GLOB_RECURSE GLSL_HEADER_FILES "src/shaders/*.h" "src/config.h")
file(GLOB_RECURSE CPP_SOURCE_FILES "src/*.h" "src/*.cpp")

add_executable(niagara
  ${CPP_SOURCE_FILES}
  ${GLSL_SOURCE_FILES}
  ${GLSL_HEADER_FILES})

set_target_properties(niagara PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

target_compile_definitions(niagara PRIVATE GLFW_INCLUDE_NONE GLM_FORCE_XYZW_ONLY GLM_FORCE_QUAT_DATA_XYZW GLM_FORCE_QUAT_CTOR_XYZW)
target_include_directories(niagara PRIVATE extern/fast_obj extern/cgltf extern/glm)

if(WIN32)
  target_compile_definitions(niagara PRIVATE GLFW_EXPOSE_NATIVE_WIN32)
  target_compile_definitions(niagara PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
endif()

if(UNIX)
  set_source_files_properties(src/extern.cpp PROPERTIES COMPILE_FLAGS $<IF:$<CONFIG:Debug>,,-Os>)
endif()

find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
  message("glfw3 not found, building from source")
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(extern/glfw)
endif()

add_subdirectory(extern/volk)
add_subdirectory(extern/meshoptimizer)

target_link_libraries(niagara
  PRIVATE
    glfw
    volk
    meshoptimizer)

if(UNIX)
  if(DEFINED ENV{VULKAN_SDK})
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/bin/glslangValidator")
  else()
    set(GLSL_VALIDATOR "glslangValidator")
  endif()
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
  set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator.exe")
else()
  set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator.exe")
endif()

set(GLSL_FLAGS --target-env vulkan1.3)

if (SHADER_DEBUG)
  list(APPEND GLSL_FLAGS -gVS)
endif()

# Thanks to: https://gist.github.com/evilactually/a0d191701cb48f157b05be7f74d79396
set(SPIRV_OUTPUT_DIR "${PROJECT_BINARY_DIR}/spirv/")
foreach(GLSL ${GLSL_SOURCE_FILES})
  get_filename_component(STEM ${GLSL} NAME_WLE)
  set(SPIRV "${SPIRV_OUTPUT_DIR}${STEM}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_OUTPUT_DIR}"
    COMMAND ${GLSL_VALIDATOR} -V ${GLSL_FLAGS} --quiet ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL} ${GLSL_HEADER_FILES})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach()

add_custom_target(compile_shaders DEPENDS ${SPIRV_BINARY_FILES})
add_dependencies(niagara compile_shaders)
