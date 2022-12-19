#include "common.h"

#include "device.h"
#include "resources.h"
#include "shaders.h"
#include "swapchain.h"

#include "math.h"

#include <stdio.h>
#include <string.h>

#include <vector>
#include <algorithm>

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <fast_obj.h>
#include <meshoptimizer.h>

bool meshShadingEnabled = true;
bool cullingEnabled = true;
bool lodEnabled = true;
bool occlusionEnabled = true;

bool debugPyramid = false;
int debugPyramidLevel = 0;

#define SHADER_PATH "src/shaders/"

VkSemaphore createSemaphore(VkDevice device)
{
	VkSemaphoreCreateInfo createInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

	VkSemaphore semaphore = 0;
	VK_CHECK(vkCreateSemaphore(device, &createInfo, 0, &semaphore));

	return semaphore;
}

VkCommandPool createCommandPool(VkDevice device, uint32_t familyIndex)
{
	VkCommandPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	createInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	createInfo.queueFamilyIndex = familyIndex;

	VkCommandPool commandPool = 0;
	VK_CHECK(vkCreateCommandPool(device, &createInfo, 0, &commandPool));

	return commandPool;
}

VkQueryPool createQueryPool(VkDevice device, uint32_t queryCount, VkQueryType queryType)
{
	VkQueryPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
	createInfo.queryType = queryType;
	createInfo.queryCount = queryCount;

	if (queryType == VK_QUERY_TYPE_PIPELINE_STATISTICS)
	{
		createInfo.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
	}

	VkQueryPool queryPool = 0;
	VK_CHECK(vkCreateQueryPool(device, &createInfo, 0, &queryPool));

	return queryPool;
}

struct alignas(16) Meshlet
{
	vec3 center;
	float radius;
	int8_t cone_axis[3];
	int8_t cone_cutoff;

	uint32_t dataOffset; // dataOffset..dataOffset+vertexCount-1 stores vertex indices, we store indices packed in 4b units after that
	uint8_t vertexCount;
	uint8_t triangleCount;
};

struct alignas(16) Globals
{
	mat4 projection;
};

struct alignas(16) MeshDraw
{
	vec3 position;
	float scale;
	quat orientation;

	uint32_t meshIndex;
	uint32_t vertexOffset; // == meshes[meshIndex].vertexOffset, helps data locality in mesh shader
};

struct MeshDrawCommand
{
	uint32_t drawId;
	VkDrawIndexedIndirectCommand indirect; // 5 uint32_t
	uint32_t taskOffset;
	uint32_t taskCount;
	VkDrawMeshTasksIndirectCommandEXT indirectMS; // 3 uint32_t
};

struct Vertex
{
	float vx, vy, vz;
	uint8_t nx, ny, nz, nw;
	uint16_t tu, tv;
};

struct MeshLod
{
	uint32_t indexOffset;
	uint32_t indexCount;
	uint32_t meshletOffset;
	uint32_t meshletCount;
};

struct alignas(16) Mesh
{
	vec3 center;
	float radius;

	uint32_t vertexOffset;
	uint32_t vertexCount;

	uint32_t lodCount;
	MeshLod lods[8];
};

struct Geometry
{
	// TODO: remove these vectors - they are just scratch copies that waste space
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::vector<Meshlet> meshlets;
	std::vector<uint32_t> meshletdata;
	std::vector<Mesh> meshes;
};

struct alignas(16) DrawCullData
{
	float P00, P11, znear, zfar; // symmetric projection parameters
	float frustum[4]; // data for left/right/top/bottom frustum planes
	float lodBase, lodStep; // lod distance i = base * pow(step, i)
	float pyramidWidth, pyramidHeight; // depth pyramid size in texels

	uint32_t drawCount;

	int cullingEnabled;
	int lodEnabled;
	int occlusionEnabled;

	int lateWorkaroundAMD;
};

struct alignas(16) DepthReduceData
{
	vec2 imageSize;
};

size_t appendMeshlets(Geometry& result, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
	const size_t max_vertices = 64;
	const size_t max_triangles = 124;
	const float cone_weight = 0.5f;

	std::vector<meshopt_Meshlet> meshlets(meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles));
	std::vector<unsigned int> meshlet_vertices(meshlets.size() * max_vertices);
	std::vector<unsigned char> meshlet_triangles(meshlets.size() * max_triangles * 3);

	meshlets.resize(meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(), indices.size(), &vertices[0].vx, vertices.size(), sizeof(Vertex), max_vertices, max_triangles, cone_weight));

	// note: we can append meshlet_vertices & meshlet_triangles buffers more or less directly with small changes in Meshlet struct, but for now keep the GPU side layout flexible and separate
	for (auto& meshlet : meshlets)
	{
		size_t dataOffset = result.meshletdata.size();

		for (unsigned int i = 0; i < meshlet.vertex_count; ++i)
			result.meshletdata.push_back(meshlet_vertices[meshlet.vertex_offset + i]);

		const unsigned int* indexGroups = reinterpret_cast<const unsigned int*>(&meshlet_triangles[0] + meshlet.triangle_offset);
		unsigned int indexGroupCount = (meshlet.triangle_count * 3 + 3) / 4;

		for (unsigned int i = 0; i < indexGroupCount; ++i)
			result.meshletdata.push_back(indexGroups[i]);

		meshopt_Bounds bounds = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, &vertices[0].vx, vertices.size(), sizeof(Vertex));

		Meshlet m = {};
		m.dataOffset = uint32_t(dataOffset);
		m.triangleCount = meshlet.triangle_count;
		m.vertexCount = meshlet.vertex_count;

		m.center = vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
		m.radius = bounds.radius;
		m.cone_axis[0] = bounds.cone_axis_s8[0];
		m.cone_axis[1] = bounds.cone_axis_s8[1];
		m.cone_axis[2] = bounds.cone_axis_s8[2];
		m.cone_cutoff = bounds.cone_cutoff_s8;

		result.meshlets.push_back(m);
	}

	return meshlets.size();
}

bool loadObj(std::vector<Vertex>& vertices, const char* path)
{
	fastObjMesh* obj = fast_obj_read(path);
	if (!obj)
		return false;

	size_t index_count = 0;

	for (unsigned int i = 0; i < obj->face_count; ++i)
		index_count += 3 * (obj->face_vertices[i] - 2);

	vertices.resize(index_count);

	size_t vertex_offset = 0;
	size_t index_offset = 0;

	for (unsigned int i = 0; i < obj->face_count; ++i)
	{
		for (unsigned int j = 0; j < obj->face_vertices[i]; ++j)
		{
			fastObjIndex gi = obj->indices[index_offset + j];

			// triangulate polygon on the fly; offset-3 is always the first polygon vertex
			if (j >= 3)
			{
				vertices[vertex_offset + 0] = vertices[vertex_offset - 3];
				vertices[vertex_offset + 1] = vertices[vertex_offset - 1];
				vertex_offset += 2;
			}

			Vertex& v = vertices[vertex_offset++];

			v.vx = obj->positions[gi.p * 3 + 0];
			v.vy = obj->positions[gi.p * 3 + 1];
			v.vz = obj->positions[gi.p * 3 + 2];
			v.nx = uint8_t(obj->normals[gi.n * 3 + 0] * 127.f + 127.5f);
			v.ny = uint8_t(obj->normals[gi.n * 3 + 1] * 127.f + 127.5f);
			v.nz = uint8_t(obj->normals[gi.n * 3 + 2] * 127.f + 127.5f);
			v.tu = meshopt_quantizeHalf(obj->texcoords[gi.t * 2 + 0]);
			v.tv = meshopt_quantizeHalf(obj->texcoords[gi.t * 2 + 1]);
		}

		index_offset += obj->face_vertices[i];
	}

	assert(vertex_offset == index_count);

	fast_obj_destroy(obj);

	return true;
}

bool loadMesh(Geometry& result, const char* path, bool buildMeshlets)
{
	std::vector<Vertex> triangle_vertices;
	if (!loadObj(triangle_vertices, path))
		return false;

	size_t index_count = triangle_vertices.size();

	std::vector<uint32_t> remap(index_count);
	size_t vertex_count = meshopt_generateVertexRemap(remap.data(), 0, index_count, triangle_vertices.data(), index_count, sizeof(Vertex));

	std::vector<Vertex> vertices(vertex_count);
	std::vector<uint32_t> indices(index_count);

	meshopt_remapVertexBuffer(vertices.data(), triangle_vertices.data(), index_count, sizeof(Vertex), remap.data());
	meshopt_remapIndexBuffer(indices.data(), 0, index_count, remap.data());

	meshopt_optimizeVertexCache(indices.data(), indices.data(), index_count, vertex_count);
	meshopt_optimizeVertexFetch(vertices.data(), indices.data(), index_count, vertices.data(), vertex_count, sizeof(Vertex));

	Mesh mesh = {};

	mesh.vertexOffset = uint32_t(result.vertices.size());
	mesh.vertexCount = uint32_t(vertices.size());

	result.vertices.insert(result.vertices.end(), vertices.begin(), vertices.end());

	vec3 center = vec3(0);

	for (auto& v : vertices)
		center += vec3(v.vx, v.vy, v.vz);

	center /= float(vertices.size());

	float radius = 0;

	for (auto& v : vertices)
		radius = std::max(radius, distance(center, vec3(v.vx, v.vy, v.vz)));

	mesh.center = center;
	mesh.radius = radius;

	std::vector<uint32_t> lodIndices = indices;

	while (mesh.lodCount < COUNTOF(mesh.lods))
	{
		MeshLod& lod = mesh.lods[mesh.lodCount++];

		lod.indexOffset = uint32_t(result.indices.size());
		lod.indexCount = uint32_t(lodIndices.size());

		result.indices.insert(result.indices.end(), lodIndices.begin(), lodIndices.end());

		lod.meshletOffset = uint32_t(result.meshlets.size());
		lod.meshletCount = buildMeshlets ? uint32_t(appendMeshlets(result, vertices, lodIndices)) : 0;

		if (mesh.lodCount < COUNTOF(mesh.lods))
		{
			size_t nextIndicesTarget = size_t(double(lodIndices.size()) * 0.75);
			size_t nextIndices = meshopt_simplify(lodIndices.data(), lodIndices.data(), lodIndices.size(), &vertices[0].vx, vertices.size(), sizeof(Vertex), nextIndicesTarget, 1e-2f);
			assert(nextIndices <= lodIndices.size());

			// we've reached the error bound
			if (nextIndices == lodIndices.size())
				break;

			lodIndices.resize(nextIndices);
			meshopt_optimizeVertexCache(lodIndices.data(), lodIndices.data(), lodIndices.size(), vertex_count);
		}
	}

	// pad meshlets to 64 to allow shaders to over-read when running task shaders
	while (result.meshlets.size() % 64)
		result.meshlets.push_back(Meshlet());

	result.meshes.push_back(mesh);

	return true;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_M)
		{
			meshShadingEnabled = !meshShadingEnabled;
		}
		if (key == GLFW_KEY_C)
		{
			cullingEnabled = !cullingEnabled;
		}
		if (key == GLFW_KEY_O)
		{
			occlusionEnabled = !occlusionEnabled;
		}
		if (key == GLFW_KEY_L)
		{
			lodEnabled = !lodEnabled;
		}
		if (key == GLFW_KEY_P)
		{
			debugPyramid = !debugPyramid;
		}
		if (debugPyramid && (key >= GLFW_KEY_0 && key <= GLFW_KEY_9))
		{
			debugPyramidLevel = key - GLFW_KEY_0;
		}
	}
}

mat4 perspectiveProjection(float fovY, float aspectWbyH, float zNear)
{
	float f = 1.0f / tanf(fovY / 2.0f);
	return mat4(
		f / aspectWbyH, 0.0f, 0.0f, 0.0f,
		0.0f, f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, zNear, 0.0f);
}

vec4 normalizePlane(vec4 p)
{
	return p / length(vec3(p));
}

uint32_t previousPow2(uint32_t v)
{
	uint32_t r = 1;

	while (r * 2 < v)
		r *= 2;

	return r;
}

#define VK_CHECKPOINT(name) do { if (checkpointsSupported) vkCmdSetCheckpointNV(commandBuffer, name); } while (0)

int main(int argc, const char** argv)
{
	if (argc < 2)
	{
		printf("Usage: %s [mesh list]\n", argv[0]);
		return 1;
	}

	int rc = glfwInit();
	assert(rc);

	VK_CHECK(volkInitialize());

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	VkInstance instance = createInstance();
	assert(instance);

	volkLoadInstanceOnly(instance);

#ifdef _DEBUG
	VkDebugReportCallbackEXT debugCallback = registerDebugCallback(instance);
#endif

	VkPhysicalDevice physicalDevices[16];
	uint32_t physicalDeviceCount = sizeof(physicalDevices) / sizeof(physicalDevices[0]);
	VK_CHECK(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices));

	VkPhysicalDevice physicalDevice = pickPhysicalDevice(physicalDevices, physicalDeviceCount);
	assert(physicalDevice);

	uint32_t extensionCount = 0;
	VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0, &extensionCount, 0));

	std::vector<VkExtensionProperties> extensions(extensionCount);
	VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0, &extensionCount, extensions.data()));

	bool pushDescriptorsSupported = false;
	bool checkpointsSupported = false;
	bool meshShadingSupported = false;

	for (auto& ext : extensions)
	{
		pushDescriptorsSupported = pushDescriptorsSupported || strcmp(ext.extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0;
		checkpointsSupported = checkpointsSupported || strcmp(ext.extensionName, VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME) == 0;
		meshShadingSupported = meshShadingSupported || strcmp(ext.extensionName, VK_EXT_MESH_SHADER_EXTENSION_NAME) == 0;
	}

	meshShadingEnabled = meshShadingSupported;

	VkPhysicalDeviceProperties props = {};
	vkGetPhysicalDeviceProperties(physicalDevice, &props);
	assert(props.limits.timestampComputeAndGraphics);

	uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevice);
	assert(familyIndex != VK_QUEUE_FAMILY_IGNORED);

	VkDevice device = createDevice(instance, physicalDevice, familyIndex, pushDescriptorsSupported, checkpointsSupported, meshShadingSupported);
	assert(device);

	volkLoadDevice(device);

	GLFWwindow* window = glfwCreateWindow(1024, 768, "niagara", 0, 0);
	assert(window);

	glfwSetKeyCallback(window, keyCallback);

	VkSurfaceKHR surface = createSurface(instance, window);
	assert(surface);

	VkBool32 presentSupported = 0;
	VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex, surface, &presentSupported));
	assert(presentSupported);

	VkFormat swapchainFormat = getSwapchainFormat(physicalDevice, surface);
	VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

	VkSemaphore acquireSemaphore = createSemaphore(device);
	assert(acquireSemaphore);

	VkSemaphore releaseSemaphore = createSemaphore(device);
	assert(releaseSemaphore);

	VkQueue queue = 0;
	vkGetDeviceQueue(device, familyIndex, 0, &queue);

	VkSampler depthSampler = createSampler(device, VK_SAMPLER_REDUCTION_MODE_MIN);
	assert(depthSampler);

	VkPipelineRenderingCreateInfo renderingInfo = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachmentFormats = &swapchainFormat;
	renderingInfo.depthAttachmentFormat = depthFormat;

	bool rcs = false;

	Shader drawcullCS = {};
	rcs = loadShader(drawcullCS, device, SHADER_PATH "drawcull.comp.spv");
	assert(rcs);

	Shader depthreduceCS = {};
	rcs = loadShader(depthreduceCS, device, SHADER_PATH "depthreduce.comp.spv");
	assert(rcs);

	Shader meshVS = {};
	rcs = loadShader(meshVS, device, SHADER_PATH "mesh.vert.spv");
	assert(rcs);

	Shader meshFS = {};
	rcs = loadShader(meshFS, device, SHADER_PATH "mesh.frag.spv");
	assert(rcs);

	Shader meshletMS = {};
	Shader meshletTS = {};
	if (meshShadingSupported)
	{
		rcs = loadShader(meshletMS, device, SHADER_PATH "meshlet.mesh.spv");
		assert(rcs);

		rcs = loadShader(meshletTS, device, SHADER_PATH "meshlet.task.spv");
		assert(rcs);
	}

	// TODO: this is critical for performance!
	VkPipelineCache pipelineCache = 0;

	Program drawcullProgram = createProgram(device, VK_PIPELINE_BIND_POINT_COMPUTE, { &drawcullCS }, sizeof(DrawCullData), pushDescriptorsSupported);
	VkPipeline drawcullPipeline = createComputePipeline(device, pipelineCache, drawcullCS, drawcullProgram.layout, { /* LATE= */ false });
	VkPipeline drawculllatePipeline = createComputePipeline(device, pipelineCache, drawcullCS, drawcullProgram.layout, { /* LATE= */ true });

	Program depthreduceProgram = createProgram(device, VK_PIPELINE_BIND_POINT_COMPUTE, { &depthreduceCS }, sizeof(DepthReduceData), pushDescriptorsSupported);
	VkPipeline depthreducePipeline = createComputePipeline(device, pipelineCache, depthreduceCS, depthreduceProgram.layout);

	Program meshProgram = createProgram(device, VK_PIPELINE_BIND_POINT_GRAPHICS, { &meshVS, &meshFS }, sizeof(Globals), pushDescriptorsSupported);

	Program meshProgramMS = {};
	if (meshShadingSupported)
		meshProgramMS = createProgram(device, VK_PIPELINE_BIND_POINT_GRAPHICS, { &meshletTS, &meshletMS, &meshFS }, sizeof(Globals), pushDescriptorsSupported);

	VkPipeline meshPipeline = createGraphicsPipeline(device, pipelineCache, renderingInfo, { &meshVS, &meshFS }, meshProgram.layout);
	assert(meshPipeline);

	VkPipeline meshPipelineMS = 0;
	if (meshShadingSupported)
	{
		meshPipelineMS = createGraphicsPipeline(device, pipelineCache, renderingInfo, { &meshletTS, &meshletMS, &meshFS }, meshProgramMS.layout);
		assert(meshPipelineMS);
	}

	Swapchain swapchain;
	createSwapchain(swapchain, physicalDevice, device, surface, familyIndex, swapchainFormat);

	VkQueryPool queryPoolTimestamp = createQueryPool(device, 128, VK_QUERY_TYPE_TIMESTAMP);
	assert(queryPoolTimestamp);

	VkQueryPool queryPoolPipeline = createQueryPool(device, 4, VK_QUERY_TYPE_PIPELINE_STATISTICS);
	assert(queryPoolPipeline);

	VkCommandPool commandPool = createCommandPool(device, familyIndex);
	assert(commandPool);

	VkCommandBufferAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	allocateInfo.commandPool = commandPool;
	allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocateInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer = 0;
	VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer));

	VkDescriptorPool descriptorPool = 0;
	if (!pushDescriptorsSupported)
	{
		uint32_t descriptorCount = 128;

		VkDescriptorPoolSize poolSizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_SAMPLER, descriptorCount },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, descriptorCount },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorCount },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount },
		};

		VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };

		poolInfo.maxSets = descriptorCount;
		poolInfo.poolSizeCount = COUNTOF(poolSizes);
		poolInfo.pPoolSizes = poolSizes;

		VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, 0, &descriptorPool));
	}

	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

	Geometry geometry;

	for (int i = 1; i < argc; ++i)
	{
		if (!loadMesh(geometry, argv[i], meshShadingSupported))
			printf("Error: mesh %s failed to load\n", argv[i]);
	}

	if (geometry.meshes.empty())
	{
		printf("Error: no meshes loaded!\n");
		return 1;
	}

	Buffer scratch = {};
	createBuffer(scratch, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	Buffer mb = {};
	createBuffer(mb, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	Buffer vb = {};
	createBuffer(vb, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	Buffer ib = {};
	createBuffer(ib, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	Buffer mlb = {};
	Buffer mdb = {};
	if (meshShadingSupported)
	{
		createBuffer(mlb, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		createBuffer(mdb, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	}

	uploadBuffer(device, commandPool, commandBuffer, queue, mb, scratch, geometry.meshes.data(), geometry.meshes.size() * sizeof(Mesh));

	uploadBuffer(device, commandPool, commandBuffer, queue, vb, scratch, geometry.vertices.data(), geometry.vertices.size() * sizeof(Vertex));
	uploadBuffer(device, commandPool, commandBuffer, queue, ib, scratch, geometry.indices.data(), geometry.indices.size() * sizeof(uint32_t));

	if (meshShadingSupported)
	{
		uploadBuffer(device, commandPool, commandBuffer, queue, mlb, scratch, geometry.meshlets.data(), geometry.meshlets.size() * sizeof(Meshlet));
		uploadBuffer(device, commandPool, commandBuffer, queue, mdb, scratch, geometry.meshletdata.data(), geometry.meshletdata.size() * sizeof(uint32_t));
	}

	uint32_t drawCount = 1000000;
	std::vector<MeshDraw> draws(drawCount);

	srand(42);

	float sceneRadius = 300;
	float drawDistance = 200;

	for (uint32_t i = 0; i < drawCount; ++i)
	{
		MeshDraw& draw = draws[i];

		size_t meshIndex = rand() % geometry.meshes.size();
		const Mesh& mesh = geometry.meshes[meshIndex];

		draw.position[0] = (float(rand()) / RAND_MAX) * sceneRadius * 2 - sceneRadius;
		draw.position[1] = (float(rand()) / RAND_MAX) * sceneRadius * 2 - sceneRadius;
		draw.position[2] = (float(rand()) / RAND_MAX) * sceneRadius * 2 - sceneRadius;
		draw.scale = (float(rand()) / RAND_MAX) + 1;
		draw.scale *= 2;

		vec3 axis = normalize(vec3((float(rand()) / RAND_MAX) * 2 - 1, (float(rand()) / RAND_MAX) * 2 - 1, (float(rand()) / RAND_MAX) * 2 - 1));
		float angle = glm::radians((float(rand()) / RAND_MAX) * 90.f);

		draw.orientation = quat(cosf(angle * 0.5f), axis * sinf(angle * 0.5f));

		draw.meshIndex = uint32_t(meshIndex);
		draw.vertexOffset = mesh.vertexOffset;
	}

	Buffer db = {};
	createBuffer(db, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	Buffer dvb = {};
	createBuffer(dvb, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	bool dvbCleared = false;

	Buffer dcb = {};
	createBuffer(dcb, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	Buffer dccb = {};
	createBuffer(dccb, device, memoryProperties, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	uploadBuffer(device, commandPool, commandBuffer, queue, db, scratch, draws.data(), draws.size() * sizeof(MeshDraw));

	Image colorTarget = {};
	Image depthTarget = {};

	Image depthPyramid = {};
	VkImageView depthPyramidMips[16] = {};
	uint32_t depthPyramidWidth = 0;
	uint32_t depthPyramidHeight = 0;
	uint32_t depthPyramidLevels = 0;

	double frameCpuAvg = 0;
	double frameGpuAvg = 0;

	uint64_t frameIndex = 0;

	while (!glfwWindowShouldClose(window))
	{
		double frameCpuBegin = glfwGetTime() * 1000;

		glfwPollEvents();

		SwapchainStatus swapchainStatus = updateSwapchain(swapchain, physicalDevice, device, surface, familyIndex, swapchainFormat);

		if (swapchainStatus == Swapchain_NotReady)
			continue;

		if (swapchainStatus == Swapchain_Resized || !colorTarget.image)
		{
			if (colorTarget.image)
				destroyImage(colorTarget, device);
			if (depthTarget.image)
				destroyImage(depthTarget, device);

			if (depthPyramid.image)
			{
				for (uint32_t i = 0; i < depthPyramidLevels; ++i)
					vkDestroyImageView(device, depthPyramidMips[i], 0);
				destroyImage(depthPyramid, device);
			}

			createImage(colorTarget, device, memoryProperties, swapchain.width, swapchain.height, 1, swapchainFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
			createImage(depthTarget, device, memoryProperties, swapchain.width, swapchain.height, 1, depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

			// Note: previousPow2 makes sure all reductions are at most by 2x2 which makes sure they are conservative
			depthPyramidWidth = previousPow2(swapchain.width);
			depthPyramidHeight = previousPow2(swapchain.height);
			depthPyramidLevels = getImageMipLevels(depthPyramidWidth, depthPyramidHeight);

			createImage(depthPyramid, device, memoryProperties, depthPyramidWidth, depthPyramidHeight, depthPyramidLevels, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

			for (uint32_t i = 0; i < depthPyramidLevels; ++i)
			{
				depthPyramidMips[i] = createImageView(device, depthPyramid.image, VK_FORMAT_R32_SFLOAT, i, 1);
				assert(depthPyramidMips[i]);
			}
		}

		uint32_t imageIndex = 0;
		VK_CHECK_SUBOPTIMAL(vkAcquireNextImageKHR(device, swapchain.swapchain, ~0ull, acquireSemaphore, VK_NULL_HANDLE, &imageIndex));

		VK_CHECK(vkResetCommandPool(device, commandPool, 0));

		if (descriptorPool)
			VK_CHECK(vkResetDescriptorPool(device, descriptorPool, 0));

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

		vkCmdResetQueryPool(commandBuffer, queryPoolTimestamp, 0, 128);
		vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPoolTimestamp, 0);

		if (!dvbCleared)
		{
			vkCmdFillBuffer(commandBuffer, dvb.buffer, 0, 4 * drawCount, 0);

			VkBufferMemoryBarrier2 fillBarrier = bufferBarrier(dvb.buffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
			pipelineBarrier(commandBuffer, 0, 1, &fillBarrier, 0, nullptr);

			dvbCleared = true;

			VK_CHECKPOINT("dvb cleared");
		}

		float znear = 0.5f;
		mat4 projection = perspectiveProjection(glm::radians(70.f), float(swapchain.width) / float(swapchain.height), znear);

		mat4 projectionT = transpose(projection);

		vec4 frustumX = normalizePlane(projectionT[3] + projectionT[0]); // x + w < 0
		vec4 frustumY = normalizePlane(projectionT[3] + projectionT[1]); // y + w < 0

		DrawCullData cullData = {};
		cullData.P00 = projection[0][0];
		cullData.P11 = projection[1][1];
		cullData.znear = znear;
		cullData.zfar = drawDistance;
		cullData.frustum[0] = frustumX.x;
		cullData.frustum[1] = frustumX.z;
		cullData.frustum[2] = frustumY.y;
		cullData.frustum[3] = frustumY.z;
		cullData.drawCount = drawCount;
		cullData.cullingEnabled = cullingEnabled;
		cullData.lodEnabled = lodEnabled;
		cullData.occlusionEnabled = occlusionEnabled;
		cullData.lodBase = 10.f;
		cullData.lodStep = 1.5f;
		cullData.pyramidWidth = float(depthPyramidWidth);
		cullData.pyramidHeight = float(depthPyramidHeight);

		Globals globals = {};
		globals.projection = projection;

		auto fullbarrier = [&]()
		{
			VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
			barrier.srcStageMask = barrier.dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
			barrier.srcAccessMask = barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
			VkDependencyInfo dependencyInfo = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
			dependencyInfo.memoryBarrierCount = 1;
			dependencyInfo.pMemoryBarriers = &barrier;
			vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
		};

		auto itsdeadjim = [&]()
		{
			printf("FATAL ERROR: DEVICE LOST (frame %lld)\n", (long long)frameIndex);

			if (checkpointsSupported)
			{
				uint32_t checkpointCount = 0;
				vkGetQueueCheckpointDataNV(queue, &checkpointCount, 0);

				std::vector<VkCheckpointDataNV> checkpoints(checkpointCount, { VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV });
				vkGetQueueCheckpointDataNV(queue, &checkpointCount, checkpoints.data());

				for (auto& cp: checkpoints)
				{
					printf("NV CHECKPOINT: stage %08x name %s\n", cp.stage, cp.pCheckpointMarker ? static_cast<const char*>(cp.pCheckpointMarker) : "??");
				}
			}
		};

		auto flush = [&]()
		{
			VK_CHECK(vkEndCommandBuffer(commandBuffer));

			VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

			VkResult wfi = vkDeviceWaitIdle(device);
			if (wfi == VK_ERROR_DEVICE_LOST)
				itsdeadjim();
			VK_CHECK(wfi);

			VK_CHECK(vkResetCommandPool(device, commandPool, 0));

			VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
		};

		auto pushDescriptors = [&](const Program& program, const DescriptorInfo* descriptors)
		{
			if (pushDescriptorsSupported)
			{
				vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, program.updateTemplate, program.layout, 0, descriptors);
			}
			else
			{
				VkDescriptorSetAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };

				allocateInfo.descriptorPool = descriptorPool;
				allocateInfo.descriptorSetCount = 1;
				allocateInfo.pSetLayouts = &program.setLayout;

				VkDescriptorSet set = 0;
				VK_CHECK(vkAllocateDescriptorSets(device, &allocateInfo, &set));

				vkUpdateDescriptorSetWithTemplate(device, set, program.updateTemplate, descriptors);

				vkCmdBindDescriptorSets(commandBuffer, program.bindPoint, program.layout, 0, 1, &set, 0, 0);
			}
		};

		auto cull = [&](VkPipeline pipeline, uint32_t timestamp, const char* phase, bool late)
		{
			uint32_t rasterizationStage =
				(meshShadingSupported && meshShadingEnabled)
				? VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT
				: VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;

			VK_CHECKPOINT(phase);

			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPoolTimestamp, timestamp + 0);

			VkBufferMemoryBarrier2 prefillBarrier = bufferBarrier(dccb.buffer,
				VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
			pipelineBarrier(commandBuffer, 0, 1, &prefillBarrier, 0, nullptr);

			vkCmdFillBuffer(commandBuffer, dccb.buffer, 0, 4, 0);

			VK_CHECKPOINT("clear buffer");

			// pyramid barrier is tricky: our frame sequence is cull -> render -> pyramid -> cull -> render
			// the first cull (late=0) doesn't read pyramid data BUT the read in the shader is guarded by a push constant value (which could be specialization constant but isn't due to AMD bug)
			// the second cull (late=1) does read pyramid data that was written in the pyramid stage
			// as such, second cull needs to transition GENERAL->GENERAL with a COMPUTE->COMPUTE barrier, but the first cull needs to have a dummy transition because pyramid starts in UNDEFINED state on first frame
			VkImageMemoryBarrier2 pyramidBarrier = imageBarrier(depthPyramid.image,
				late ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0, late ? VK_ACCESS_SHADER_WRITE_BIT : 0, late ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL);

			VkBufferMemoryBarrier2 fillBarriers[] =
			{
				bufferBarrier(dcb.buffer,
					VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | rasterizationStage, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT),
				bufferBarrier(dccb.buffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT),
			};
			pipelineBarrier(commandBuffer, 0, COUNTOF(fillBarriers), fillBarriers, 1, &pyramidBarrier);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

			DescriptorInfo pyramidDesc(depthSampler, depthPyramid.imageView, VK_IMAGE_LAYOUT_GENERAL);
			DescriptorInfo descriptors[] = { db.buffer, mb.buffer, dcb.buffer, dccb.buffer, dvb.buffer, pyramidDesc };
			// vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, drawcullProgram.updateTemplate, drawcullProgram.layout, 0, descriptors);
			pushDescriptors(drawcullProgram, descriptors);

			vkCmdPushConstants(commandBuffer, drawcullProgram.layout, drawcullProgram.pushConstantStages, 0, sizeof(cullData), &cullData);
			vkCmdDispatch(commandBuffer, getGroupCount(uint32_t(draws.size()), drawcullCS.localSizeX), 1, 1);

			VK_CHECKPOINT("culled");

			VkBufferMemoryBarrier2 cullBarriers[] =
			{
				bufferBarrier(dcb.buffer,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
					VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | rasterizationStage, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT),
				bufferBarrier(dccb.buffer,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
					VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
			};

			pipelineBarrier(commandBuffer, 0, COUNTOF(cullBarriers), cullBarriers, 0, nullptr);

			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPoolTimestamp, timestamp + 1);
		};

		auto render = [&](bool late, const VkClearColorValue& colorClear, const VkClearDepthStencilValue& depthClear, uint32_t query, const char* phase)
		{
			VK_CHECKPOINT(phase);

			vkCmdBeginQuery(commandBuffer, queryPoolPipeline, query, 0);

			VkRenderingAttachmentInfo colorAttachment = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
			colorAttachment.imageView = colorTarget.imageView;
			colorAttachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
			colorAttachment.loadOp = late ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachment.clearValue.color = colorClear;

			VkRenderingAttachmentInfo depthAttachment = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
			depthAttachment.imageView = depthTarget.imageView;
			depthAttachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
			depthAttachment.loadOp = late ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
			depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			depthAttachment.clearValue.depthStencil = depthClear;

			VkRenderingInfo passInfo = { VK_STRUCTURE_TYPE_RENDERING_INFO };
			passInfo.renderArea.extent.width = swapchain.width;
			passInfo.renderArea.extent.height = swapchain.height;
			passInfo.layerCount = 1;
			passInfo.colorAttachmentCount = 1;
			passInfo.pColorAttachments = &colorAttachment;
			passInfo.pDepthAttachment = &depthAttachment;

			vkCmdBeginRendering(commandBuffer, &passInfo);

			VkViewport viewport = { 0, float(swapchain.height), float(swapchain.width), -float(swapchain.height), 0, 1 };
			VkRect2D scissor = { {0, 0}, {uint32_t(swapchain.width), uint32_t(swapchain.height)} };

			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			VK_CHECKPOINT("before draw");

			if (meshShadingSupported && meshShadingEnabled)
			{
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineMS);

				DescriptorInfo descriptors[] = { dcb.buffer, db.buffer, mlb.buffer, mdb.buffer, vb.buffer };
				// vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, meshProgramMS.updateTemplate, meshProgramMS.layout, 0, descriptors);
				pushDescriptors(meshProgramMS, descriptors);

				vkCmdPushConstants(commandBuffer, meshProgramMS.layout, meshProgramMS.pushConstantStages, 0, sizeof(globals), &globals);
				vkCmdDrawMeshTasksIndirectCountEXT(commandBuffer, dcb.buffer, offsetof(MeshDrawCommand, indirectMS), dccb.buffer, 0, uint32_t(draws.size()), sizeof(MeshDrawCommand));
			}
			else
			{
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);

				DescriptorInfo descriptors[] = { dcb.buffer, db.buffer, vb.buffer };
				// vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, meshProgram.updateTemplate, meshProgram.layout, 0, descriptors);
				pushDescriptors(meshProgram, descriptors);

				vkCmdBindIndexBuffer(commandBuffer, ib.buffer, 0, VK_INDEX_TYPE_UINT32);

				vkCmdPushConstants(commandBuffer, meshProgram.layout, meshProgram.pushConstantStages, 0, sizeof(globals), &globals);
				vkCmdDrawIndexedIndirectCount(commandBuffer, dcb.buffer, offsetof(MeshDrawCommand, indirect), dccb.buffer, 0, uint32_t(draws.size()), sizeof(MeshDrawCommand));
			}

			VK_CHECKPOINT("after draw");

			vkCmdEndRendering(commandBuffer);

			vkCmdEndQuery(commandBuffer, queryPoolPipeline, query);
		};

		auto pyramid = [&]()
		{
			VK_CHECKPOINT("pyramid");

			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPoolTimestamp, 4);

			VkImageMemoryBarrier2 depthBarriers[] =
			{
				imageBarrier(depthTarget.image,
					VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					VK_IMAGE_ASPECT_DEPTH_BIT),
				imageBarrier(depthPyramid.image,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL)
			};

			pipelineBarrier(commandBuffer, 0, 0, nullptr, COUNTOF(depthBarriers), depthBarriers);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, depthreducePipeline);

			for (uint32_t i = 0; i < depthPyramidLevels; ++i)
			{
				VK_CHECKPOINT("pyramid level");

				DescriptorInfo sourceDepth = (i == 0)
					? DescriptorInfo(depthSampler, depthTarget.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
					: DescriptorInfo(depthSampler, depthPyramidMips[i - 1], VK_IMAGE_LAYOUT_GENERAL);

				DescriptorInfo descriptors[] = { { depthPyramidMips[i], VK_IMAGE_LAYOUT_GENERAL }, sourceDepth };
				// vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, depthreduceProgram.updateTemplate, depthreduceProgram.layout, 0, descriptors);
				pushDescriptors(depthreduceProgram, descriptors);

				uint32_t levelWidth = std::max(1u, depthPyramidWidth >> i);
				uint32_t levelHeight = std::max(1u, depthPyramidHeight >> i);

				DepthReduceData reduceData = { vec2(levelWidth, levelHeight) };

				vkCmdPushConstants(commandBuffer, depthreduceProgram.layout, depthreduceProgram.pushConstantStages, 0, sizeof(reduceData), &reduceData);
				vkCmdDispatch(commandBuffer, getGroupCount(levelWidth, depthreduceCS.localSizeX), getGroupCount(levelHeight, depthreduceCS.localSizeY), 1);

				VkImageMemoryBarrier2 reduceBarrier = imageBarrier(depthPyramid.image,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
					VK_IMAGE_ASPECT_COLOR_BIT, i, 1);

				pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &reduceBarrier);
			}

			VkImageMemoryBarrier2 depthWriteBarrier = imageBarrier(depthTarget.image,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				VK_IMAGE_ASPECT_DEPTH_BIT);

			pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &depthWriteBarrier);

			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPoolTimestamp, 5);
		};

		VkImageMemoryBarrier2 renderBeginBarriers[] =
		{
			imageBarrier(colorTarget.image,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, VK_IMAGE_LAYOUT_UNDEFINED,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL),
			imageBarrier(depthTarget.image,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, VK_IMAGE_LAYOUT_UNDEFINED,
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				VK_IMAGE_ASPECT_DEPTH_BIT),
		};

		pipelineBarrier(commandBuffer, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, COUNTOF(renderBeginBarriers), renderBeginBarriers);

		vkCmdResetQueryPool(commandBuffer, queryPoolPipeline, 0, 4);

		VkClearColorValue colorClear = { 48.f / 255.f, 10.f / 255.f, 36.f / 255.f, 1 };
		VkClearDepthStencilValue depthClear = { 0.f, 0 };

		VK_CHECKPOINT("frame");

		// early cull: frustum cull & fill objects that *were* visible last frame
		cullData.lateWorkaroundAMD = 0;
		cull(drawcullPipeline, 2, "early cull", /* late= */ false);

		// early render: render objects that were visible last frame
		render(/* late= */ false, colorClear, depthClear, 0, "early render");

		// depth pyramid generation
		pyramid();

		// late cull: frustum + occlusion cull & fill objects that were *not* visible last frame
		cullData.lateWorkaroundAMD = 1;
		cull(drawculllatePipeline, 6, "late cull", /* late= */ true);

		// late render: render objects that are visible this frame but weren't drawn in the early pass
		render(/* late= */ true, colorClear, depthClear, 1, "late render");

		VkImageMemoryBarrier2 copyBarriers[] =
		{
			imageBarrier(colorTarget.image,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
			imageBarrier(swapchain.images[imageIndex],
				0, 0, VK_IMAGE_LAYOUT_UNDEFINED,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
			imageBarrier(depthPyramid.image,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL),
		};

		pipelineBarrier(commandBuffer, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, COUNTOF(copyBarriers), copyBarriers);

		VK_CHECKPOINT("swapchain copy");

		if (debugPyramid)
		{
			uint32_t levelWidth = std::max(1u, depthPyramidWidth >> debugPyramidLevel);
			uint32_t levelHeight = std::max(1u, depthPyramidHeight >> debugPyramidLevel);

			VkImageBlit blitRegion = {};
			blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blitRegion.srcSubresource.mipLevel = debugPyramidLevel;
			blitRegion.srcSubresource.layerCount = 1;
			blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blitRegion.dstSubresource.layerCount = 1;
			blitRegion.srcOffsets[0] = { 0, 0, 0 };
			blitRegion.srcOffsets[1] = { int32_t(levelWidth), int32_t(levelHeight), 1 };
			blitRegion.dstOffsets[0] = { 0, 0, 0 };
			blitRegion.dstOffsets[1] = { int32_t(swapchain.width), int32_t(swapchain.height), 1 };

			vkCmdBlitImage(commandBuffer, depthPyramid.image, VK_IMAGE_LAYOUT_GENERAL, swapchain.images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_NEAREST);
		}
		else
		{
			VkImageCopy copyRegion = {};
			copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copyRegion.srcSubresource.layerCount = 1;
			copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copyRegion.dstSubresource.layerCount = 1;
			copyRegion.extent = { swapchain.width, swapchain.height, 1 };

			vkCmdCopyImage(commandBuffer, colorTarget.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain.images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
		}

		VK_CHECKPOINT("present");

		VkImageMemoryBarrier2 presentBarrier = imageBarrier(swapchain.images[imageIndex],
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			0, 0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

		pipelineBarrier(commandBuffer, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &presentBarrier);

		vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPoolTimestamp, 1);

		VK_CHECK(vkEndCommandBuffer(commandBuffer));

		VkPipelineStageFlags submitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &acquireSemaphore;
		submitInfo.pWaitDstStageMask = &submitStageMask;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &releaseSemaphore;

		VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &releaseSemaphore;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &swapchain.swapchain;
		presentInfo.pImageIndices = &imageIndex;

		VK_CHECK_SUBOPTIMAL(vkQueuePresentKHR(queue, &presentInfo));

		VkResult wfi = vkDeviceWaitIdle(device);
		if (wfi == VK_ERROR_DEVICE_LOST)
			itsdeadjim();
		VK_CHECK(wfi);

		uint64_t timestampResults[8] = {};
		VK_CHECK(vkGetQueryPoolResults(device, queryPoolTimestamp, 0, COUNTOF(timestampResults), sizeof(timestampResults), timestampResults, sizeof(timestampResults[0]), VK_QUERY_RESULT_64_BIT));

		uint32_t pipelineResults[2] = {};
		VK_CHECK(vkGetQueryPoolResults(device, queryPoolPipeline, 0, COUNTOF(pipelineResults), sizeof(pipelineResults), pipelineResults, sizeof(pipelineResults[0]), 0));

		uint32_t triangleCount = pipelineResults[0] + pipelineResults[1];

		double frameGpuBegin = double(timestampResults[0]) * props.limits.timestampPeriod * 1e-6;
		double frameGpuEnd = double(timestampResults[1]) * props.limits.timestampPeriod * 1e-6;
		double cullGpuTime = double(timestampResults[3] - timestampResults[2]) * props.limits.timestampPeriod * 1e-6;
		double pyramidGpuTime = double(timestampResults[5] - timestampResults[4]) * props.limits.timestampPeriod * 1e-6;
		double culllateGpuTime = double(timestampResults[7] - timestampResults[6]) * props.limits.timestampPeriod * 1e-6;

		double frameCpuEnd = glfwGetTime() * 1000;

		frameCpuAvg = frameCpuAvg * 0.95 + (frameCpuEnd - frameCpuBegin) * 0.05;
		frameGpuAvg = frameGpuAvg * 0.95 + (frameGpuEnd - frameGpuBegin) * 0.05;

		double trianglesPerSec = double(triangleCount) / double(frameGpuAvg * 1e-3);
		double drawsPerSec = double(drawCount) / double(frameGpuAvg * 1e-3);

		char title[256];
		sprintf(title, "cpu: %.2f ms; gpu: %.2f ms (cull: %.2f ms, pyramid: %.2f ms, cull late: %.2f); triangles %.1fM; %.1fB tri/sec, %.1fM draws/sec; mesh shading %s, frustum culling %s, occlusion culling %s, level-of-detail %s",
			frameCpuAvg, frameGpuAvg, cullGpuTime, pyramidGpuTime, culllateGpuTime,
			double(triangleCount) * 1e-6, trianglesPerSec * 1e-9, drawsPerSec * 1e-6,
			meshShadingSupported && meshShadingEnabled ? "ON" : "OFF", cullingEnabled ? "ON" : "OFF", occlusionEnabled ? "ON" : "OFF", lodEnabled ? "ON" : "OFF");

		glfwSetWindowTitle(window, title);

		frameIndex++;
	}

	VK_CHECK(vkDeviceWaitIdle(device));

	if (colorTarget.image)
		destroyImage(colorTarget, device);
	if (depthTarget.image)
		destroyImage(depthTarget, device);

	if (depthPyramid.image)
	{
		for (uint32_t i = 0; i < depthPyramidLevels; ++i)
			vkDestroyImageView(device, depthPyramidMips[i], 0);
		destroyImage(depthPyramid, device);
	}

	destroyBuffer(mb, device);

	destroyBuffer(db, device);
	destroyBuffer(dvb, device);
	destroyBuffer(dcb, device);
	destroyBuffer(dccb, device);

	if (meshShadingSupported)
	{
		destroyBuffer(mlb, device);
		destroyBuffer(mdb, device);
	}

	destroyBuffer(ib, device);
	destroyBuffer(vb, device);

	destroyBuffer(scratch, device);

	vkDestroyCommandPool(device, commandPool, 0);

	if (descriptorPool)
		vkDestroyDescriptorPool(device, descriptorPool, 0);

	vkDestroyQueryPool(device, queryPoolTimestamp, 0);
	vkDestroyQueryPool(device, queryPoolPipeline, 0);

	destroySwapchain(device, swapchain);

	vkDestroyPipeline(device, drawcullPipeline, 0);
	vkDestroyPipeline(device, drawculllatePipeline, 0);
	destroyProgram(device, drawcullProgram);

	vkDestroyPipeline(device, depthreducePipeline, 0);
	destroyProgram(device, depthreduceProgram);

	vkDestroyPipeline(device, meshPipeline, 0);
	destroyProgram(device, meshProgram);

	if (meshShadingSupported)
	{
		vkDestroyPipeline(device, meshPipelineMS, 0);
		destroyProgram(device, meshProgramMS);
	}

	vkDestroyShaderModule(device, drawcullCS.module, 0);
	vkDestroyShaderModule(device, depthreduceCS.module, 0);

	vkDestroyShaderModule(device, meshVS.module, 0);
	vkDestroyShaderModule(device, meshFS.module, 0);

	if (meshShadingSupported)
	{
		vkDestroyShaderModule(device, meshletTS.module, 0);
		vkDestroyShaderModule(device, meshletMS.module, 0);
	}

	vkDestroySampler(device, depthSampler, 0);

	vkDestroySemaphore(device, releaseSemaphore, 0);
	vkDestroySemaphore(device, acquireSemaphore, 0);

	vkDestroySurfaceKHR(instance, surface, 0);

	glfwDestroyWindow(window);

	vkDestroyDevice(device, 0);

#ifdef _DEBUG
	vkDestroyDebugReportCallbackEXT(instance, debugCallback, 0);
#endif

	vkDestroyInstance(instance, 0);
}
