# Niagara

This is a Vulkan renderer that is written on stream from scratch - without using any third party code that is Vulkan specific. We are using non-Vulkan-specific third party libraries however.

The goal is to experiment with a few modern Vulkan rendering techniques, such as GPU culling & scene submission, cone culling, automatic occlusion culling, task/mesh shading, and whatever else it is that we will want to experiment with.
The code will be written on stream.

# Requirements

The renderer is written using Visual Studio and targets Windows desktops with modern Vulkan drivers. You will need Visual Studio 2017 and Vulkan SDK to follow along

# Building

To build and run the project, clone this repository using --recursive flag:

	git clone https://github.com/zeux/niagara.git --recursive

Make sure you have Vulkan SDK installed; open the Visual Studio project in niagara/src and build it.

To run the program, command line should contain arguments with paths to .obj files; you can use kitten.obj from data/ folder for testing.
On lower-end GPUs you might want to change `drawCount` in `niagara.cpp` to be a value smaller than 1M.

# Stream

The development of this project has streamed on YouTube on weekends in October and November 2018; the project is currently on hold.

Playlist: https://www.youtube.com/playlist?list=PL0JVLUVCkk-l7CWCn3-cdftR0oajugYvd

1. Setting up instance/device and filling the screen with a solid color: https://youtu.be/BR2my8OE1Sc
2. Rendering a triangle on screen: https://youtu.be/5eS3gsL_P-c
3. Cleaning up validation errors and implementing swapchain resize: https://youtu.be/_VU-G5rglnA
4. Rendering a mesh using shader storage buffers and int8: https://youtu.be/nKCzD5iK71M
5. Rendering a mesh using NVidia RTX mesh shading pipeline: https://youtu.be/gbeOKMjmQ-g
6. Optimizing GPU time by using device-local memory and parallelizing mesh shader: https://youtu.be/ayKoqK3kQ9c
7. Using descriptor update templates and parsing SPIRV to extract reflection data: https://youtu.be/3Py4GlWAicY
8. Cluster cone culling using task shaders and subgroup ops: https://youtu.be/KckRq7Rm3Mw
9. Tuning mesh shading pipeline for performance: https://youtu.be/snZkA4D_qjU
10. Depth buffer, perspective projection, 3D transforms and multi draw indirect: https://youtu.be/y4WOsAaXLh0
11. Multiple meshes and GPU frustum culling: https://youtu.be/NGGzk4Fi2iU
12. Draw call compaction using KHR_draw_indirect_count and LOD support: https://youtu.be/IYRgDcnJJ2I
13. Depth pyramid construction and extending SPIRV reflection parser: https://youtu.be/YCteLdYdZWQ
14. Automatic occlusion culling: https://youtu.be/Fj1E1A4CPCM

# Issues

During the streams we find various bugs in parts of the Vulkan stack and report them; bugs marked with ✔️ have been fixed.

* ✔️ vkAcquireNextImageKHR crashes in validation layers if vkGetSwapchainImagesKHR hasn't been called \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/358

* vkGetPhysicalDeviceSurfaceFormatsKHR doesn't fill format count correctly \
https://software.intel.com/en-us/forums/graphics-driver-bug-reporting/topic/797666

* ✔️ Fix NonWritable check when vertexPipelineStoresAndAtomics not enabled \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/73

* Implicit int8->float cast adds Int8 capability to the shader without asking for GL_KHX_shader_explicit_arithmetic_types \
https://github.com/KhronosGroup/glslang/issues/1525

* vkCreateSwapchainKHR crashes in Intel drivers when display is plugged into a dedicated GPU \
https://software.intel.com/en-us/forums/graphics-driver-bug-reporting/topic/797756

* ✔️ Reading uint8_t from storage buffers adds (unnecessarily) UniformAndStorageBuffer8BitAccess capability \
https://github.com/KhronosGroup/glslang/issues/1539

* Binding a buffer with VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as a storage buffer using push descriptors doesn't produce validation errors \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/413

* ✔️ Fragment shader with perprimitiveNV doesn't have OpExtension SPV_NV_mesh_shader \
https://github.com/KhronosGroup/glslang/issues/1541

* ✔️ GL_NV_mesh_shader spec typo for per-primitive fragment shader inputs \
https://github.com/KhronosGroup/GLSL/issues/31

* ✔️ Push descriptors generate false positive DescriptorSetNotBound errors \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/341

* vkCmdDrawIndexedIndirect doesn't issue an error when the buffer wasn't created with VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/450

* ✔️ vkCmdDrawMeshTasksIndirectNV doesn't trigger an error when multiDrawIndirect feature is disabled \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/451

* vkCmdDrawIndexedIndirect is rendering fewer than drawCount draw calls on Intel \
Reproduce using https://github.com/zeux/niagara/commit/fda3d8743c933fb3a533fed560a6671402d6693b

* vkCmdDrawIndexedIndirectCountKHR is causing a GPU crash on Intel \
Reproduce using https://github.com/zeux/niagara/commit/c22c2c56d06249835a474e370ea3218463721f42

* ✔️ Crash during Vulkan replay in push descriptor replay \
https://github.com/baldurk/renderdoc/issues/1182

* NVidia GTX 10xx series GPUs cause VK_ERROR_DEVICE_LOST when drawCount is 1'000'000 \
Reproduce using https://github.com/zeux/niagara/commit/8d69552aede9c429765c8c8afd6687d3f3e53475

* AMD drivers 18.11.2 on Windows don't handle specialization constants correctly, requiring a workaround in drawcull.comp.glsl \
Reproduce using https://github.com/zeux/niagara/commit/6150fbc7e36c64249051227dd9821d5eb6bce9e1; disabling workaround in drawcull.comp.glsl leads to no objects being rendered on screen after the first frame
