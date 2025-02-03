#include "common.h"

#include "scenert.h"
#include "scene.h"
#include "resources.h"

#include "config.h"

#include <string.h>

void buildBLAS(VkDevice device, const std::vector<Mesh>& meshes, const Buffer& vb, const Buffer& ib, std::vector<VkAccelerationStructureKHR>& blas, std::vector<VkDeviceSize>& compactedSizes, Buffer& blasBuffer, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const VkPhysicalDeviceMemoryProperties& memoryProperties)
{
	std::vector<uint32_t> primitiveCounts(meshes.size());
	std::vector<VkAccelerationStructureGeometryKHR> geometries(meshes.size());
	std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(meshes.size());

	const size_t kAlignment = 256;                   // required by spec for acceleration structures, could be smaller for scratch but it's a small waste
	const size_t kDefaultScratch = 32 * 1024 * 1024; // 32 MB scratch by default

	size_t totalAccelerationSize = 0;
	size_t totalPrimitiveCount = 0;
	size_t maxScratchSize = 0;

	std::vector<size_t> accelerationOffsets(meshes.size());
	std::vector<size_t> accelerationSizes(meshes.size());
	std::vector<size_t> scratchSizes(meshes.size());

	VkDeviceAddress vbAddress = getBufferAddress(vb, device);
	VkDeviceAddress ibAddress = getBufferAddress(ib, device);

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		const Mesh& mesh = meshes[i];
		VkAccelerationStructureGeometryKHR& geo = geometries[i];
		VkAccelerationStructureBuildGeometryInfoKHR& buildInfo = buildInfos[i];

		unsigned int lodIndex = 0;

		primitiveCounts[i] = mesh.lods[lodIndex].indexCount / 3;

		geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;

		static_assert(offsetof(Vertex, vz) == offsetof(Vertex, vx) + sizeof(uint16_t) * 2, "Vertex layout mismatch");

		geo.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		geo.geometry.triangles.vertexFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
		geo.geometry.triangles.vertexData.deviceAddress = vbAddress + mesh.vertexOffset * sizeof(Vertex);
		geo.geometry.triangles.vertexStride = sizeof(Vertex);
		geo.geometry.triangles.maxVertex = mesh.vertexCount - 1;
		geo.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
		geo.geometry.triangles.indexData.deviceAddress = ibAddress + mesh.lods[lodIndex].indexOffset * sizeof(uint32_t);

		buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
		buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		buildInfo.geometryCount = 1;
		buildInfo.pGeometries = &geo;

		VkAccelerationStructureBuildSizesInfoKHR sizeInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCounts[i], &sizeInfo);

		accelerationOffsets[i] = totalAccelerationSize;
		accelerationSizes[i] = sizeInfo.accelerationStructureSize;
		scratchSizes[i] = sizeInfo.buildScratchSize;

		totalAccelerationSize = (totalAccelerationSize + sizeInfo.accelerationStructureSize + kAlignment - 1) & ~(kAlignment - 1);
		totalPrimitiveCount += primitiveCounts[i];
		maxScratchSize = std::max(maxScratchSize, size_t(sizeInfo.buildScratchSize));
	}

	createBuffer(blasBuffer, device, memoryProperties, totalAccelerationSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	Buffer scratchBuffer;
	createBuffer(scratchBuffer, device, memoryProperties, std::max(kDefaultScratch, maxScratchSize), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	printf("BLAS accelerationStructureSize: %.2f MB, scratchSize: %.2f MB (max %.2f MB), %.3fM triangles\n", double(totalAccelerationSize) / 1e6, double(scratchBuffer.size) / 1e6, double(maxScratchSize) / 1e6, double(totalPrimitiveCount) / 1e6);

	VkDeviceAddress scratchAddress = getBufferAddress(scratchBuffer, device);

	blas.resize(meshes.size());

	std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges(meshes.size());
	std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> buildRangePtrs(meshes.size());

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		VkAccelerationStructureCreateInfoKHR accelerationInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		accelerationInfo.buffer = blasBuffer.buffer;
		accelerationInfo.offset = accelerationOffsets[i];
		accelerationInfo.size = accelerationSizes[i];
		accelerationInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

		VK_CHECK(vkCreateAccelerationStructureKHR(device, &accelerationInfo, nullptr, &blas[i]));
	}

	VkQueryPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
	createInfo.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
	createInfo.queryCount = blas.size();

	VkQueryPool queryPool = 0;
	VK_CHECK(vkCreateQueryPool(device, &createInfo, 0, &queryPool));

	VK_CHECK(vkResetCommandPool(device, commandPool, 0));

	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	VkBufferMemoryBarrier2 scratchBarrier = bufferBarrier(scratchBuffer.buffer,
	    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
	    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

	for (size_t start = 0; start < meshes.size();)
	{
		size_t scratchOffset = 0;

		// aggregate the range that fits into allocated scratch
		size_t i = start;
		while (i < meshes.size() && scratchOffset + scratchSizes[i] <= scratchBuffer.size)
		{
			buildInfos[i].scratchData.deviceAddress = scratchAddress + scratchOffset;
			buildInfos[i].dstAccelerationStructure = blas[i];
			buildRanges[i].primitiveCount = primitiveCounts[i];
			buildRangePtrs[i] = &buildRanges[i];

			scratchOffset = (scratchOffset + scratchSizes[i] + kAlignment - 1) & ~(kAlignment - 1);
			++i;
		}
		assert(i > start); // guaranteed as scratchBuffer.size >= maxScratchSize

		vkCmdBuildAccelerationStructuresKHR(commandBuffer, uint32_t(i - start), &buildInfos[start], &buildRangePtrs[start]);
		start = i;

		pipelineBarrier(commandBuffer, 0, 1, &scratchBarrier, 0, nullptr);
	}

	vkCmdResetQueryPool(commandBuffer, queryPool, 0, blas.size());
	vkCmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, blas.size(), blas.data(), VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, 0);

	VK_CHECK(vkEndCommandBuffer(commandBuffer));

	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
	VK_CHECK(vkDeviceWaitIdle(device));

	compactedSizes.resize(blas.size());
	VK_CHECK(vkGetQueryPoolResults(device, queryPool, 0, blas.size(), blas.size() * sizeof(VkDeviceSize), compactedSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

	vkDestroyQueryPool(device, queryPool, 0);

	destroyBuffer(scratchBuffer, device);
}

void compactBLAS(VkDevice device, std::vector<VkAccelerationStructureKHR>& blas, const std::vector<VkDeviceSize>& compactedSizes, Buffer& blasBuffer, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const VkPhysicalDeviceMemoryProperties& memoryProperties)
{
	const size_t kAlignment = 256; // required by spec for acceleration structures

	VK_CHECK(vkResetCommandPool(device, commandPool, 0));

	size_t totalCompactedSize = 0;
	std::vector<size_t> compactedOffsets(blas.size());

	for (size_t i = 0; i < blas.size(); ++i)
	{
		compactedOffsets[i] = totalCompactedSize;
		totalCompactedSize = (totalCompactedSize + compactedSizes[i] + kAlignment - 1) & ~(kAlignment - 1);
	}

	printf("BLAS compacted accelerationStructureSize: %.2f MB\n", double(totalCompactedSize) / 1e6);

	Buffer compactedBuffer;
	createBuffer(compactedBuffer, device, memoryProperties, totalCompactedSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	std::vector<VkAccelerationStructureKHR> compactedBlas(blas.size());

	for (size_t i = 0; i < blas.size(); ++i)
	{
		VkAccelerationStructureCreateInfoKHR accelerationInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		accelerationInfo.buffer = compactedBuffer.buffer;
		accelerationInfo.offset = compactedOffsets[i];
		accelerationInfo.size = compactedSizes[i];
		accelerationInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

		VK_CHECK(vkCreateAccelerationStructureKHR(device, &accelerationInfo, nullptr, &compactedBlas[i]));
	}

	VK_CHECK(vkResetCommandPool(device, commandPool, 0));

	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	for (size_t i = 0; i < blas.size(); ++i)
	{
		VkCopyAccelerationStructureInfoKHR copyInfo = { VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR };
		copyInfo.src = blas[i];
		copyInfo.dst = compactedBlas[i];
		copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;

		vkCmdCopyAccelerationStructureKHR(commandBuffer, &copyInfo);
	}

	VK_CHECK(vkEndCommandBuffer(commandBuffer));

	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
	VK_CHECK(vkDeviceWaitIdle(device));

	for (size_t i = 0; i < blas.size(); ++i)
	{
		vkDestroyAccelerationStructureKHR(device, blas[i], nullptr);
		blas[i] = compactedBlas[i];
	}

	destroyBuffer(blasBuffer, device);
	blasBuffer = compactedBuffer;
}

void buildCLAS(VkDevice device, const std::vector<Mesh>& meshes, const std::vector<Meshlet>& meshlets, const Buffer& vxb, const Buffer& mdb, std::vector<VkAccelerationStructureKHR>& blas, Buffer& blasBuffer, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const VkPhysicalDeviceMemoryProperties& memoryProperties)
{
	VkClusterAccelerationStructureTriangleClusterInputNV clusterSizes = { VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV };
	clusterSizes.vertexFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
	clusterSizes.maxGeometryIndexValue = 0;
	clusterSizes.maxClusterUniqueGeometryCount = 1;
	clusterSizes.maxClusterTriangleCount = MESH_MAXTRI;
	clusterSizes.maxClusterVertexCount = MESH_MAXVTX;
	clusterSizes.minPositionTruncateBitCount = 0;

	VkClusterAccelerationStructureInputInfoNV clusterInfo = { VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV };
	clusterInfo.maxAccelerationStructureCount = 0;
	clusterInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	clusterInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
	clusterInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
	clusterInfo.opInput.pTriangleClusters = &clusterSizes;

	size_t maxClustersPerMesh = 0;

	for (const Mesh& mesh : meshes)
	{
		clusterSizes.maxTotalTriangleCount += mesh.lods[0].indexCount / 3;
		clusterInfo.maxAccelerationStructureCount += mesh.lods[0].meshletCount;
		maxClustersPerMesh = std::max(maxClustersPerMesh, size_t(mesh.lods[0].meshletCount));

		for (size_t mi = 0; mi < mesh.lods[0].meshletCount; ++mi)
		{
			const Meshlet& ml = meshlets[mesh.lods[0].meshletOffset + mi];
			clusterSizes.maxTotalVertexCount += ml.vertexCount;
		}
	}

	VkClusterAccelerationStructureClustersBottomLevelInputNV accelSizes = { VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV };
	accelSizes.maxTotalClusterCount = clusterInfo.maxAccelerationStructureCount;
	accelSizes.maxClusterCountPerAccelerationStructure = maxClustersPerMesh;

	VkClusterAccelerationStructureInputInfoNV accelInfo = { VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV };
	accelInfo.maxAccelerationStructureCount = meshes.size();
	accelInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	accelInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
	accelInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
	accelInfo.opInput.pClustersBottomLevel = &accelSizes;

	VkAccelerationStructureBuildSizesInfoKHR csizeInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	vkGetClusterAccelerationStructureBuildSizesNV(device, &clusterInfo, &csizeInfo);

	VkAccelerationStructureBuildSizesInfoKHR bsizeInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	vkGetClusterAccelerationStructureBuildSizesNV(device, &accelInfo, &bsizeInfo);

	printf("CLAS accelerationStructureSize: %.2f MB, scratchSize: %.2f MB\n", double(csizeInfo.accelerationStructureSize) / 1e6, double(csizeInfo.buildScratchSize) / 1e6);
	printf("CBLAS accelerationStructureSize: %.2f MB, scratchSize: %.2f MB\n", double(bsizeInfo.accelerationStructureSize) / 1e6, double(bsizeInfo.buildScratchSize) / 1e6);

	Buffer clasBuffer;
	createBuffer(clasBuffer, device, memoryProperties, csizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	createBuffer(blasBuffer, device, memoryProperties, bsizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	Buffer scratchBuffer;
	createBuffer(scratchBuffer, device, memoryProperties, std::max(bsizeInfo.buildScratchSize, csizeInfo.buildScratchSize), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	Buffer infosBuffer;
	createBuffer(infosBuffer, device, memoryProperties, std::max(clusterInfo.maxAccelerationStructureCount * sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV), accelInfo.maxAccelerationStructureCount * sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV)), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	VkDeviceAddress mdbAddress = getBufferAddress(mdb, device);
	VkDeviceAddress vxbAddress = getBufferAddress(vxb, device);

	VkClusterAccelerationStructureBuildTriangleClusterInfoNV* clusterData = static_cast<VkClusterAccelerationStructureBuildTriangleClusterInfoNV*>(infosBuffer.data);
	size_t vxbOffset = 0;

	for (const Mesh& mesh : meshes)
	{
		for (size_t mi = 0; mi < mesh.lods[0].meshletCount; ++mi)
		{
			const Meshlet& ml = meshlets[mesh.lods[0].meshletOffset + mi];

			VkClusterAccelerationStructureBuildTriangleClusterInfoNV cluster = {};
			cluster.clusterID = uint32_t(mi);
			cluster.triangleCount = ml.triangleCount;
			cluster.vertexCount = ml.vertexCount;
			cluster.positionTruncateBitCount = 0;
			cluster.indexType = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
			cluster.vertexBufferStride = sizeof(uint16_t) * 4;
			cluster.indexBuffer = mdbAddress + (ml.dataOffset + (ml.shortRefs ? (ml.vertexCount + 1) / 2 : ml.vertexCount)) * sizeof(uint32_t);
			cluster.vertexBuffer = vxbAddress + vxbOffset;

			memcpy(clusterData, &cluster, sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV));
			clusterData++;
			vxbOffset += ml.vertexCount * (sizeof(uint16_t) * 4);
		}
	}

	Buffer rangeBuffer;
	// todo host vis -> device local?
	createBuffer(rangeBuffer, device, memoryProperties, (clusterInfo.maxAccelerationStructureCount + accelInfo.maxAccelerationStructureCount) * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	VkClusterAccelerationStructureCommandsInfoNV clusterBuild = { VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV };
	clusterBuild.input = clusterInfo;
	clusterBuild.dstImplicitData = getBufferAddress(clasBuffer, device);
	clusterBuild.scratchData = getBufferAddress(scratchBuffer, device);
	clusterBuild.dstAddressesArray.deviceAddress = getBufferAddress(rangeBuffer, device);
	clusterBuild.dstAddressesArray.size = clusterInfo.maxAccelerationStructureCount * 16;
	clusterBuild.dstAddressesArray.stride = 16;
	clusterBuild.dstSizesArray.deviceAddress = getBufferAddress(rangeBuffer, device) + 8;
	clusterBuild.dstSizesArray.size = clusterInfo.maxAccelerationStructureCount * 16 - 8;
	clusterBuild.dstSizesArray.stride = 16;
	clusterBuild.srcInfosArray.deviceAddress = getBufferAddress(infosBuffer, device);
	clusterBuild.srcInfosArray.size = clusterInfo.maxAccelerationStructureCount * sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV);

	VK_CHECK(vkResetCommandPool(device, commandPool, 0));

	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	vkCmdBuildClusterAccelerationStructureIndirectNV(commandBuffer, &clusterBuild);

	VK_CHECK(vkEndCommandBuffer(commandBuffer));

	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
	VK_CHECK(vkDeviceWaitIdle(device));

	size_t totalClusterSize = 0;
	for (size_t i = 0; i < clusterInfo.maxAccelerationStructureCount; ++i)
	{
		uint64_t addr = ((uint64_t*)rangeBuffer.data)[i * 2];
		uint32_t size = ((uint32_t*)rangeBuffer.data)[i * 4 + 2];

		// printf("cluster %d: %16lx %08x\n", int(i), addr, size);

		totalClusterSize += ((uint32_t*)rangeBuffer.data)[i * 4 + 2];
	}

	printf("CLAS actual total accelerationStructureSize: %.2f MB\n", double(totalClusterSize) / 1e6);

	VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV* accelData = static_cast<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV*>(infosBuffer.data);
	size_t accelOffset = 0;

	printf("max cluster count %d, total cluster count %d, total blas count %d\n", int(maxClustersPerMesh), int(clusterInfo.maxAccelerationStructureCount), int(accelInfo.maxAccelerationStructureCount));

	for (const Mesh& mesh : meshes)
	{
		VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV accel = {};
		accel.clusterReferencesCount = uint32_t(mesh.lods[0].meshletCount);
		accel.clusterReferencesStride = 16;
		accel.clusterReferences = getBufferAddress(rangeBuffer, device) + accelOffset * 16;

		// NOTE: HUGE HACK!!!
		// ON NV 3050 we have a single mesh with over 2000 clusters in Bistro scene
		// if we keep the entire cluster set in the build, the GPU will hang during build
		// we will print here the first cluster we skip from the mesh; not sure what is going on really.
		if (accel.clusterReferencesCount > 2000)
		{
			// IMPORTANT:
			// This is definitely not a "broken" cluster: we can shift the range of clusters we build here by a couple clusters forward, and it still works
			// accel.clusterReferences += 32;
			// ... but if we get just one extra cluster (1537 below), GPU hangs
			// It's suspicious that 1536 == 1024 + 512 but who knows what's going on.
			accel.clusterReferencesCount = 1536;
			printf("HACK: in mesh %d we are going to skip clusters starting from %d\n", int(&mesh - meshes.data()), int(accel.clusterReferencesCount));
			printf("SKIP: cluster blas %016lx size %04x tris %d vertices %d\n",
			    ((uint64_t*)rangeBuffer.data)[(accelOffset + accel.clusterReferencesCount) * 2],
			    ((uint32_t*)rangeBuffer.data)[(accelOffset + accel.clusterReferencesCount) * 4 + 2],
			    meshlets[mesh.lods[0].meshletOffset + accel.clusterReferencesCount].triangleCount,
			    meshlets[mesh.lods[0].meshletOffset + accel.clusterReferencesCount].vertexCount);
		}

		// printf("mesh %d: offset %d count %d\n", int(&mesh - meshes.data()), int(accelOffset), mesh.lods[0].meshletCount);

		memcpy(accelData, &accel, sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV));
		accelData++;
		accelOffset += mesh.lods[0].meshletCount;
	}

	VkClusterAccelerationStructureCommandsInfoNV accelBuild = { VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV };
	accelBuild.input = accelInfo;
	accelBuild.dstImplicitData = getBufferAddress(blasBuffer, device);
	accelBuild.scratchData = getBufferAddress(scratchBuffer, device);
	accelBuild.dstAddressesArray.deviceAddress = getBufferAddress(rangeBuffer, device) + clusterInfo.maxAccelerationStructureCount * 16;
	accelBuild.dstAddressesArray.size = accelInfo.maxAccelerationStructureCount * 16;
	accelBuild.dstAddressesArray.stride = 16;
	accelBuild.dstSizesArray.deviceAddress = getBufferAddress(rangeBuffer, device) + clusterInfo.maxAccelerationStructureCount * 16 + 8;
	accelBuild.dstSizesArray.size = accelInfo.maxAccelerationStructureCount * 16 - 8;
	accelBuild.dstSizesArray.stride = 16;
	accelBuild.srcInfosArray.deviceAddress = getBufferAddress(infosBuffer, device);
	accelBuild.srcInfosArray.size = accelInfo.maxAccelerationStructureCount * sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);

	VK_CHECK(vkResetCommandPool(device, commandPool, 0));
	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	vkCmdBuildClusterAccelerationStructureIndirectNV(commandBuffer, &accelBuild);

	VK_CHECK(vkEndCommandBuffer(commandBuffer));
	VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
	VK_CHECK(vkDeviceWaitIdle(device));

	VkDeviceAddress blasAddress = getBufferAddress(blasBuffer, device);
	uint32_t* rangeAccel = (uint32_t*)rangeBuffer.data + clusterInfo.maxAccelerationStructureCount * 4;

	blas.resize(meshes.size());

	for (size_t i = 0; i < accelInfo.maxAccelerationStructureCount; ++i)
	{
		VkAccelerationStructureCreateInfoKHR accelerationInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		accelerationInfo.buffer = blasBuffer.buffer;
		accelerationInfo.offset = ((uint64_t*)rangeAccel)[i * 2] - blasAddress;
		accelerationInfo.size = rangeAccel[i * 4 + 2];
		accelerationInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

		VK_CHECK(vkCreateAccelerationStructureKHR(device, &accelerationInfo, nullptr, &blas[i]));
	}

	destroyBuffer(scratchBuffer, device);
	destroyBuffer(infosBuffer, device);
	destroyBuffer(rangeBuffer, device);

	// TODO: destroyBuffer(clasBuffer, device);
}

void fillInstanceRT(VkAccelerationStructureInstanceKHR& instance, const MeshDraw& draw, uint32_t instanceIndex, VkDeviceAddress blas)
{
	mat3 xform = transpose(glm::mat3_cast(draw.orientation)) * draw.scale;

	memcpy(instance.transform.matrix[0], &xform[0], sizeof(float) * 3);
	memcpy(instance.transform.matrix[1], &xform[1], sizeof(float) * 3);
	memcpy(instance.transform.matrix[2], &xform[2], sizeof(float) * 3);
	instance.transform.matrix[0][3] = draw.position.x;
	instance.transform.matrix[1][3] = draw.position.y;
	instance.transform.matrix[2][3] = draw.position.z;
	instance.instanceCustomIndex = instanceIndex;
	instance.mask = 1 << draw.postPass;
	instance.flags = draw.postPass ? VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR : VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
	instance.accelerationStructureReference = draw.postPass <= 1 ? blas : 0;
}

VkAccelerationStructureKHR createTLAS(VkDevice device, Buffer& tlasBuffer, Buffer& scratchBuffer, const Buffer& instanceBuffer, uint32_t primitiveCount, const VkPhysicalDeviceMemoryProperties& memoryProperties)
{
	VkAccelerationStructureGeometryKHR geometry = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
	geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	geometry.geometry.instances.data.deviceAddress = getBufferAddress(instanceBuffer, device);

	VkAccelerationStructureBuildGeometryInfoKHR buildInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
	buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
	buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries = &geometry;

	VkAccelerationStructureBuildSizesInfoKHR sizeInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

	printf("TLAS accelerationStructureSize: %.2f MB, scratchSize: %.2f MB, updateScratch: %.2f MB\n", double(sizeInfo.accelerationStructureSize) / 1e6, double(sizeInfo.buildScratchSize) / 1e6, double(sizeInfo.updateScratchSize) / 1e6);

	createBuffer(tlasBuffer, device, memoryProperties, sizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	createBuffer(scratchBuffer, device, memoryProperties, std::max(sizeInfo.buildScratchSize, sizeInfo.updateScratchSize), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VkAccelerationStructureCreateInfoKHR accelerationInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
	accelerationInfo.buffer = tlasBuffer.buffer;
	accelerationInfo.size = sizeInfo.accelerationStructureSize;
	accelerationInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

	VkAccelerationStructureKHR tlas = nullptr;
	VK_CHECK(vkCreateAccelerationStructureKHR(device, &accelerationInfo, nullptr, &tlas));

	return tlas;
}

void buildTLAS(VkDevice device, VkCommandBuffer commandBuffer, VkAccelerationStructureKHR tlas, const Buffer& tlasBuffer, const Buffer& scratchBuffer, const Buffer& instanceBuffer, uint32_t primitiveCount, VkBuildAccelerationStructureModeKHR mode)
{
	VkAccelerationStructureGeometryKHR geometry = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
	geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	geometry.geometry.instances.data.deviceAddress = getBufferAddress(instanceBuffer, device);

	VkAccelerationStructureBuildGeometryInfoKHR buildInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
	buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
	buildInfo.mode = mode;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries = &geometry;

	buildInfo.srcAccelerationStructure = tlas;
	buildInfo.dstAccelerationStructure = tlas;
	buildInfo.scratchData.deviceAddress = getBufferAddress(scratchBuffer, device);

	VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
	buildRange.primitiveCount = primitiveCount;
	const VkAccelerationStructureBuildRangeInfoKHR* buildRangePtr = &buildRange;

	vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildInfo, &buildRangePtr);

	VkBufferMemoryBarrier2 barrier = bufferBarrier(tlasBuffer.buffer,
	    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
	    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

	pipelineBarrier(commandBuffer, 0, 1, &barrier, 0, nullptr);
}
