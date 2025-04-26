# Niagara

This is a Vulkan renderer that is written on stream from scratch - without using any third party code that is Vulkan specific. We are using non-Vulkan-specific third party libraries however.

The goal is to experiment with a few modern Vulkan rendering techniques, such as GPU culling & scene submission, cone culling, automatic occlusion culling, task/mesh shading, and whatever else it is that we will want to experiment with.
The code will be written on stream.

![image](https://github.com/user-attachments/assets/b102622e-fbe7-4e9c-b575-e4d4533eadfe)

# Requirements

The renderer was originally written using Visual Studio and targeted Windows desktops with modern Vulkan drivers. Since then the development platform has switched to Linux, but you can still build and run it on Windows - via CMake.

# Building

To build and run the project, clone this repository using --recursive flag:

	git clone https://github.com/zeux/niagara --recursive

Make sure you have Vulkan 1.4 SDK and drivers installed; open the folder niagara in Visual Studio (as a CMake project) and build it. On Linux, you can use CMake with your build generator of choice.

To run the program, command line should contain arguments with paths to .obj files or a .gltf scene; you can use kitten.obj from data/ folder for testing.

To use Amazon Lumberyard Bistro scene, clone https://github.com/zeux/niagara_bistro and specify path to bistro.gltf instead.

# Stream

The development of this project is streamed on YouTube, on Saturdays at 11 AM PST with a somewhat irregular schedule.

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
15. Vulkan 1.2 and GPU buffer pointers: https://youtu.be/78tVIA6nRQg
16. Upgrading to Vulkan 1.3: https://youtu.be/Ka30T6BMdhI
17. Implementing triangle culling: https://youtu.be/JKTfAgv3Vlo
18. Meshlet occlusion culling: https://youtu.be/5sBpo5wKmEM
19. Optimizing culling: https://youtu.be/1Tj6bZvZMts
20. Task command submission: https://youtu.be/eYvGruGHhUE
21. Cluster compute culling: https://youtu.be/zROUBE5pLuI
22. Loading glTF scenes: https://youtu.be/9OF6k57orXo
23. Bindless textures: https://youtu.be/n9nqSEyXMeA
24. Tracing rays: https://youtu.be/N1OVfBEcyb8
25. Tracing rays faster: https://youtu.be/U7TGQsjT16E
26. Materials and shadows: https://youtu.be/iZTUjRntMbM
27. Transparent shadows: https://youtu.be/233jxF7irmE
28. Moving objects: https://youtu.be/TcuUz1ib35c
29. Performance nsights: https://youtu.be/qlxrRyRdOcY

# Issues

During the streams we find various bugs in parts of the Vulkan stack and report them; bugs marked with ✔️ have been fixed.

* ✔️ vkAcquireNextImageKHR crashes in validation layers if vkGetSwapchainImagesKHR hasn't been called \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/358

* ✔️ vkGetPhysicalDeviceSurfaceFormatsKHR doesn't fill format count correctly \
https://software.intel.com/en-us/forums/graphics-driver-bug-reporting/topic/797666

* ✔️ Fix NonWritable check when vertexPipelineStoresAndAtomics not enabled \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/73

* ✔️ Implicit int8->float cast adds Int8 capability to the shader without asking for GL_KHX_shader_explicit_arithmetic_types \
https://github.com/KhronosGroup/glslang/issues/1525

* ⁉ vkCreateSwapchainKHR crashes in Intel drivers when display is plugged into a dedicated GPU \
https://software.intel.com/en-us/forums/graphics-driver-bug-reporting/topic/797756

* ✔️ Reading uint8_t from storage buffers adds (unnecessarily) UniformAndStorageBuffer8BitAccess capability \
https://github.com/KhronosGroup/glslang/issues/1539

* ✔️ Binding a buffer with VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as a storage buffer using push descriptors doesn't produce validation errors \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/413

* ✔️ Fragment shader with perprimitiveNV doesn't have OpExtension SPV_NV_mesh_shader \
https://github.com/KhronosGroup/glslang/issues/1541

* ✔️ GL_NV_mesh_shader spec typo for per-primitive fragment shader inputs \
https://github.com/KhronosGroup/GLSL/issues/31

* ✔️ Push descriptors generate false positive DescriptorSetNotBound errors \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/341

* ✔️ vkCmdDrawIndexedIndirect doesn't issue an error when the buffer wasn't created with VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/450

* ✔️ vkCmdDrawMeshTasksIndirectNV doesn't trigger an error when multiDrawIndirect feature is disabled \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/451

* ✔️ vkCmdDrawIndexedIndirect is rendering fewer than drawCount draw calls on Intel \
Reproduce using https://github.com/zeux/niagara/commit/fda3d8743c933fb3a533fed560a6671402d6693b

* ✔️ vkCmdDrawIndexedIndirectCountKHR is causing a GPU crash on Intel \
Reproduce using https://github.com/zeux/niagara/commit/c22c2c56d06249835a474e370ea3218463721f42

* ✔️ Crash during Vulkan replay in push descriptor replay \
https://github.com/baldurk/renderdoc/issues/1182

* ✔️ NVidia GTX 10xx series GPUs cause VK_ERROR_DEVICE_LOST when drawCount is 1'000'000 \
Reproduce using https://github.com/zeux/niagara/commit/8d69552aede9c429765c8c8afd6687d3f3e53475

* ✔️ AMD drivers 18.11.2 on Windows don't handle specialization constants correctly \
Using specialization constant LATE in drawcull.comp.glsl leads to no objects being rendered on screen after the first frame

* ✔️ During validation of pipelines with SPIRV 1.4/1.5 and specialization constants, optimizer isn't configured to use Vulkan 1.2 \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/1512

* ✔️ Crash when calling vkCmdDrawIndexedIndirectCount loaded through GIPA \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/1513

* ✔️ SHADER_MODULE_STATE::has_specialization_constants is not initialized \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/1530

* ✔️ Missing validation: push descriptor updates don't trigger image layout mismatch errors \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/1862

* ✔️ A valid interface block in mesh/task shader is considered invalid \
https://github.com/KhronosGroup/SPIRV-Tools/issues/3653

* ✔️ Usage of any fields of gl_MeshPrimitivesEXT is enabling capability FragmentShadingRateKHR even if gl_PrimitiveShadingRateEXT is not used \
https://github.com/KhronosGroup/glslang/issues/3103

* ✔️ Incomplete mip data is encoded for non-power-of-two textures \
https://github.com/wolfpld/etcpak/pull/43

* radv should use pointer flags on RDNA3 during BVH traversal \
https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/32417

* radv: VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR is not supported for TLAS \
https://gitlab.freedesktop.org/mesa/mesa/-/issues/12346

* ✔️ Missing synchronization validation for ray tracing acceleration updates & uses \
https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/9076

* ✔️ anv: Mesh shaders with two OpSetMeshOutputsEXT instructions are not supported \
https://gitlab.freedesktop.org/mesa/mesa/-/issues/12388
