#pragma once

struct Buffer;

struct Mesh;
struct MeshDraw;
struct Meshlet;

void buildBLAS(VkDevice device, const std::vector<Mesh>& meshes, const Buffer& vb, const Buffer& ib, std::vector<VkAccelerationStructureKHR>& blas, std::vector<VkDeviceSize>& compactedSizes, Buffer& blasBuffer, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const VkPhysicalDeviceMemoryProperties& memoryProperties);
void compactBLAS(VkDevice device, std::vector<VkAccelerationStructureKHR>& blas, const std::vector<VkDeviceSize>& compactedSizes, Buffer& blasBuffer, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const VkPhysicalDeviceMemoryProperties& memoryProperties);

void buildCLAS(VkDevice device, const std::vector<Mesh>& meshes, const std::vector<Meshlet>& meshlets, const Buffer& vxb, const Buffer& mdb, std::vector<VkAccelerationStructureKHR>& blas, Buffer& blasBuffer, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const VkPhysicalDeviceMemoryProperties& memoryProperties);

void fillInstanceRT(VkAccelerationStructureInstanceKHR& instance, const MeshDraw& draw, uint32_t instanceIndex, VkDeviceAddress blas);

VkAccelerationStructureKHR createTLAS(VkDevice device, Buffer& tlasBuffer, Buffer& scratchBuffer, const Buffer& instanceBuffer, uint32_t primitiveCount, const VkPhysicalDeviceMemoryProperties& memoryProperties);

void buildTLAS(VkDevice device, VkCommandBuffer commandBuffer, VkAccelerationStructureKHR tlas, const Buffer& tlasBuffer, const Buffer& scratchBuffer, const Buffer& instanceBuffer, uint32_t primitiveCount, VkBuildAccelerationStructureModeKHR mode);
