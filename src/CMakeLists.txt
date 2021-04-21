file(GLOB_RECURSE GLSL_SOURCE_FILES "shaders/*.glsl")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  PREFIX "Shaders//"
  FILES
    ${GLSL_SOURCE_FILES}
    "shaders/mesh.h")

add_executable(niagara
  "common.h"
  "math.h"
  "device.h"
  "device.cpp"
  "swapchain.h"
  "swapchain.cpp"
  "shaders.h"
  "shaders.cpp"
  "resources.h"
  "resources.cpp"
  "niagara.cpp"
  "fast_obj.cpp"
  
  ${GLSL_SOURCE_FILES}
  "shaders/mesh.h")
target_compile_definitions(niagara
  PRIVATE
    -D_CRT_SECURE_NO_WARNINGS
    -DWIN32_LEAN_AND_MEAN
    -DNOMINMAX
    -DVK_USE_PLATFORM_WIN32_KHR
    -D_GLFW_WIN32
    -DGLFW_EXPOSE_NATIVE_WIN32
    -DGLM_FORCE_XYZW_ONLY )
target_link_libraries(niagara
  PRIVATE
    glfw
    glm::glm
    volk
    meshoptimizer::meshoptimizer)
    
if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
  set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator.exe")
else()
  set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator.exe")
endif()

# Thanks to: https://gist.github.com/evilactually/a0d191701cb48f157b05be7f74d79396
set(SPIRV_OUTPUT_DIR "${PROJECT_BINARY_DIR}/src/shaders/")
foreach(GLSL ${GLSL_SOURCE_FILES})
  get_filename_component(STEM ${GLSL} NAME_WLE)
  set(SPIRV "${SPIRV_OUTPUT_DIR}${STEM}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/src/shaders/"
    COMMAND ${GLSL_VALIDATOR} -V --target-env vulkan1.2 ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL} "shaders/mesh.h")
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach()

add_custom_target(compile_shaders DEPENDS ${SPIRV_BINARY_FILES})
add_dependencies(niagara compile_shaders)
