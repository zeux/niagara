#include "common.h"

#include "device.h"
#include "resources.h"
#include "shaders.h"
#include "swapchain.h"

#include "math.h"

#include <stdio.h>

#include <vector>
#include <algorithm>

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <objparser.h>
#include <meshoptimizer.h>

bool meshShadingEnabled = true;
bool cullingEnabled = true;
bool lodEnabled = true;

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

VkRenderPass createRenderPass(VkDevice device, VkFormat colorFormat, VkFormat depthFormat)
{
	VkAttachmentDescription attachments[2] = {};
	attachments[0].format = colorFormat;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachments[1].format = depthFormat;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorAttachment = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
	VkAttachmentReference depthAttachment = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachment;
	subpass.pDepthStencilAttachment = &depthAttachment;

	VkRenderPassCreateInfo createInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
	createInfo.attachmentCount = sizeof(attachments) / sizeof(attachments[0]);
	createInfo.pAttachments = attachments;
	createInfo.subpassCount = 1;
	createInfo.pSubpasses = &subpass;

	VkRenderPass renderPass = 0;
	VK_CHECK(vkCreateRenderPass(device, &createInfo, 0, &renderPass));

	return renderPass;
}

VkFramebuffer createFramebuffer(VkDevice device, VkRenderPass renderPass, VkImageView colorView, VkImageView depthView, uint32_t width, uint32_t height)
{
	VkImageView attachments[] = { colorView, depthView };

	VkFramebufferCreateInfo createInfo = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
	createInfo.renderPass = renderPass;
	createInfo.attachmentCount = ARRAYSIZE(attachments);
	createInfo.pAttachments = attachments;
	createInfo.width = width;
	createInfo.height = height;
	createInfo.layers = 1;

	VkFramebuffer framebuffer = 0;
	VK_CHECK(vkCreateFramebuffer(device, &createInfo, 0, &framebuffer));

	return framebuffer;
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
	uint32_t vertexOffset;
};

struct MeshDrawCommand
{
	uint32_t drawId;
	VkDrawIndexedIndirectCommand indirect; // 5 uint32_t
	VkDrawMeshTasksIndirectCommandNV indirectMS; // 2 uint32_t
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

struct DrawCullData
{
	vec4 frustum[6];

	uint32_t drawCount;

	int cullingEnabled;
	int lodEnabled;
};

size_t appendMeshlets(Geometry& result, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
	const size_t max_vertices = 64;
	const size_t max_triangles = 124;

	std::vector<meshopt_Meshlet> meshlets(meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles));
	meshlets.resize(meshopt_buildMeshlets(meshlets.data(), indices.data(), indices.size(), vertices.size(), max_vertices, max_triangles));

	for (auto& meshlet : meshlets)
	{
		size_t dataOffset = result.meshletdata.size();

		for (unsigned int i = 0; i < meshlet.vertex_count; ++i)
			result.meshletdata.push_back(meshlet.vertices[i]);

		const unsigned int* indexGroups = reinterpret_cast<const unsigned int*>(meshlet.indices);
		unsigned int indexGroupCount = (meshlet.triangle_count * 3 + 3) / 4;

		for (unsigned int i = 0; i < indexGroupCount; ++i)
			result.meshletdata.push_back(indexGroups[i]);

		meshopt_Bounds bounds = meshopt_computeMeshletBounds(&meshlet, &vertices[0].vx, vertices.size(), sizeof(Vertex));

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

	while (result.meshlets.size() % 32)
		result.meshlets.push_back(Meshlet());

	return meshlets.size();
}

bool loadMesh(Geometry& result, const char* path, bool buildMeshlets)
{
	ObjFile file;
	if (!objParseFile(file, path))
		return false;

	size_t index_count = file.f_size / 3;

	std::vector<Vertex> triangle_vertices(index_count);

	for (size_t i = 0; i < index_count; ++i)
	{
		Vertex& v = triangle_vertices[i];

		int vi = file.f[i * 3 + 0];
		int vti = file.f[i * 3 + 1];
		int vni = file.f[i * 3 + 2];

		float nx = vni < 0 ? 0.f : file.vn[vni * 3 + 0];
		float ny = vni < 0 ? 0.f : file.vn[vni * 3 + 1];
		float nz = vni < 0 ? 1.f : file.vn[vni * 3 + 2];

		v.vx = file.v[vi * 3 + 0];
		v.vy = file.v[vi * 3 + 1];
		v.vz = file.v[vi * 3 + 2];
		v.nx = uint8_t(nx * 127.f + 127.f); // TODO: fix rounding
		v.ny = uint8_t(ny * 127.f + 127.f); // TODO: fix rounding
		v.nz = uint8_t(nz * 127.f + 127.f); // TODO: fix rounding
		v.tu = meshopt_quantizeHalf(vti < 0 ? 0.f : file.vt[vti * 3 + 0]);
		v.tv = meshopt_quantizeHalf(vti < 0 ? 0.f : file.vt[vti * 3 + 1]);
	}

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

	while (mesh.lodCount < ARRAYSIZE(mesh.lods))
	{
		MeshLod& lod = mesh.lods[mesh.lodCount++];

		lod.indexOffset = uint32_t(result.indices.size());
		lod.indexCount = uint32_t(lodIndices.size());

		result.indices.insert(result.indices.end(), lodIndices.begin(), lodIndices.end());

		lod.meshletOffset = uint32_t(result.meshlets.size());
		lod.meshletCount = buildMeshlets ? uint32_t(appendMeshlets(result, vertices, lodIndices)) : 0;

		if (mesh.lodCount < ARRAYSIZE(mesh.lods))
		{
			size_t nextIndicesTarget = size_t(double(lodIndices.size()) * 0.75);
			size_t nextIndices = meshopt_simplify(lodIndices.data(), lodIndices.data(), lodIndices.size(), &vertices[0].vx, vertices.size(), sizeof(Vertex), nextIndicesTarget, 1e-4f);
			assert(nextIndices <= lodIndices.size());

			// we've reached the error bound
			if (nextIndices == lodIndices.size())
				break;

			lodIndices.resize(nextIndices);
			meshopt_optimizeVertexCache(lodIndices.data(), lodIndices.data(), lodIndices.size(), vertex_count);
		}
	}

	result.meshes.push_back(mesh);

	return true;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_R)
		{
			meshShadingEnabled = !meshShadingEnabled;
		}
		if (key == GLFW_KEY_C)
		{
			cullingEnabled = !cullingEnabled;
		}
		if (key == GLFW_KEY_L)
		{
			lodEnabled = !lodEnabled;
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

	volkLoadInstance(instance);

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

	bool meshShadingSupported = false;
	for (auto& ext : extensions)
		if (strcmp(ext.extensionName, "VK_NV_mesh_shader") == 0)
		{
			meshShadingSupported = true;
			break;
		}

	meshShadingEnabled = meshShadingSupported;

	VkPhysicalDeviceProperties props = {};
	vkGetPhysicalDeviceProperties(physicalDevice, &props);
	assert(props.limits.timestampComputeAndGraphics);

	uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevice);
	assert(familyIndex != VK_QUEUE_FAMILY_IGNORED);

	VkDevice device = createDevice(instance, physicalDevice, familyIndex, meshShadingSupported);
	assert(device);

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

	VkRenderPass renderPass = createRenderPass(device, swapchainFormat, depthFormat);
	assert(renderPass);

	bool rcs = false;

	Shader drawcullCS = {};
	rcs = loadShader(drawcullCS, device, "shaders/drawcull.comp.spv");
	assert(rcs);

	Shader meshVS = {};
	rcs = loadShader(meshVS, device, "shaders/mesh.vert.spv");
	assert(rcs);

	Shader meshFS = {};
	rcs = loadShader(meshFS, device, "shaders/mesh.frag.spv");
	assert(rcs);

	Shader meshletMS = {};
	Shader meshletTS = {};
	if (meshShadingSupported)
	{
		rcs = loadShader(meshletMS, device, "shaders/meshlet.mesh.spv");
		assert(rcs);

		rcs = loadShader(meshletTS, device, "shaders/meshlet.task.spv");
		assert(rcs);
	}

	// TODO: this is critical for performance!
	VkPipelineCache pipelineCache = 0;

	Program drawcullProgram = createProgram(device, VK_PIPELINE_BIND_POINT_COMPUTE, { &drawcullCS }, sizeof(DrawCullData));

	VkPipeline drawcullPipeline = createComputePipeline(device, pipelineCache, drawcullCS, drawcullProgram.layout);

	Program meshProgram = createProgram(device, VK_PIPELINE_BIND_POINT_GRAPHICS, { &meshVS, &meshFS }, sizeof(Globals));

	Program meshProgramMS = {};
	if (meshShadingSupported)
		meshProgramMS = createProgram(device, VK_PIPELINE_BIND_POINT_GRAPHICS, { &meshletTS, &meshletMS, &meshFS }, sizeof(Globals));

	VkPipeline meshPipeline = createGraphicsPipeline(device, pipelineCache, renderPass, { &meshVS, &meshFS }, meshProgram.layout);
	assert(meshPipeline);

	VkPipeline meshPipelineMS = 0;
	if (meshShadingSupported)
	{
		meshPipelineMS = createGraphicsPipeline(device, pipelineCache, renderPass, { &meshletTS, &meshletMS, &meshFS }, meshProgramMS.layout);
		assert(meshPipelineMS);
	}

	Swapchain swapchain;
	createSwapchain(swapchain, physicalDevice, device, surface, familyIndex, swapchainFormat, renderPass);

	VkQueryPool queryPoolTimestamp = createQueryPool(device, 128, VK_QUERY_TYPE_TIMESTAMP);
	assert(queryPoolTimestamp);

	VkQueryPool queryPoolPipeline = createQueryPool(device, 1, VK_QUERY_TYPE_PIPELINE_STATISTICS);
	assert(queryPoolPipeline);

	VkCommandPool commandPool = createCommandPool(device, familyIndex);
	assert(commandPool);

	VkCommandBufferAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	allocateInfo.commandPool = commandPool;
	allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocateInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer = 0;
	VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer));

	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

	Geometry geometry;

	for (int i = 1; i < argc; ++i)
	{
		if (!loadMesh(geometry, argv[i], meshShadingSupported))
			printf("Error: mesh %s failed to load\n", argv[i]);
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

	uint32_t drawCount = 1'000'000;
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

		vec3 axis((float(rand()) / RAND_MAX) * 2 - 1, (float(rand()) / RAND_MAX) * 2 - 1, (float(rand()) / RAND_MAX) * 2 - 1);
		float angle = glm::radians((float(rand()) / RAND_MAX) * 90.f);

		draw.orientation = rotate(quat(1, 0, 0, 0), angle, axis);

		draw.meshIndex = uint32_t(meshIndex);
		draw.vertexOffset = mesh.vertexOffset;
	}

	Buffer db = {};
	createBuffer(db, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	Buffer dcb = {};
	createBuffer(dcb, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	Buffer dccb = {};
	createBuffer(dccb, device, memoryProperties, 4, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	uploadBuffer(device, commandPool, commandBuffer, queue, db, scratch, draws.data(), draws.size() * sizeof(MeshDraw));

	Image colorTarget = {};
	Image depthTarget = {};
	VkFramebuffer targetFB = 0;

	double frameCpuAvg = 0;
	double frameGpuAvg = 0;

	while (!glfwWindowShouldClose(window))
	{
		double frameCpuBegin = glfwGetTime() * 1000;

		glfwPollEvents();

		if (resizeSwapchainIfNecessary(swapchain, physicalDevice, device, surface, familyIndex, swapchainFormat, renderPass) || !targetFB)
		{
			if (colorTarget.image)
				destroyImage(colorTarget, device);
			if (depthTarget.image)
				destroyImage(depthTarget, device);
			if (targetFB)
				vkDestroyFramebuffer(device, targetFB, 0);

			createImage(colorTarget, device, memoryProperties, swapchain.width, swapchain.height, swapchainFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
			createImage(depthTarget, device, memoryProperties, swapchain.width, swapchain.height, depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
			targetFB = createFramebuffer(device, renderPass, colorTarget.imageView, depthTarget.imageView, swapchain.width, swapchain.height);
		}

		uint32_t imageIndex = 0;
		VK_CHECK(vkAcquireNextImageKHR(device, swapchain.swapchain, ~0ull, acquireSemaphore, VK_NULL_HANDLE, &imageIndex));

		VK_CHECK(vkResetCommandPool(device, commandPool, 0));

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

		vkCmdResetQueryPool(commandBuffer, queryPoolTimestamp, 0, 128);
		vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPoolTimestamp, 0);

		mat4 projection = perspectiveProjection(glm::radians(50.f), float(swapchain.width) / float(swapchain.height), 0.01f);

		{
			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPoolTimestamp, 2);

			mat4 projectionT = transpose(projection);

			DrawCullData cullData = {};
			cullData.frustum[0] = normalizePlane(projectionT[3] + projectionT[0]); // x + w < 0
			cullData.frustum[1] = normalizePlane(projectionT[3] - projectionT[0]); // x - w > 0
			cullData.frustum[2] = normalizePlane(projectionT[3] + projectionT[1]); // y + w < 0
			cullData.frustum[3] = normalizePlane(projectionT[3] - projectionT[1]); // y - w > 0
			cullData.frustum[4] = normalizePlane(projectionT[3] - projectionT[2]); // z - w > 0 -- reverse z
			cullData.frustum[5] = vec4(0, 0, -1, drawDistance); // reverse z, infinite far plane
			cullData.drawCount = drawCount;
			cullData.cullingEnabled = cullingEnabled;
			cullData.lodEnabled = lodEnabled;

			vkCmdFillBuffer(commandBuffer, dccb.buffer, 0, 4, 0);

			VkBufferMemoryBarrier fillBarrier = bufferBarrier(dccb.buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, 0, 1, &fillBarrier, 0, 0);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, drawcullPipeline);

			DescriptorInfo descriptors[] = { db.buffer, mb.buffer, dcb.buffer, dccb.buffer };
			vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, drawcullProgram.updateTemplate, drawcullProgram.layout, 0, descriptors);

			vkCmdPushConstants(commandBuffer, drawcullProgram.layout, drawcullProgram.pushConstantStages, 0, sizeof(cullData), &cullData);
			vkCmdDispatch(commandBuffer, uint32_t((draws.size() + 31) / 32), 1, 1);

			VkBufferMemoryBarrier cullBarriers[] =
			{
				bufferBarrier(dcb.buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
				bufferBarrier(dccb.buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
			};

			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, 0, ARRAYSIZE(cullBarriers), cullBarriers, 0, 0);

			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPoolTimestamp, 3);
		}

		VkImageMemoryBarrier renderBeginBarriers[] =
		{
			imageBarrier(colorTarget.image, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
			imageBarrier(depthTarget.image, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT),
		};

		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 0, 0, ARRAYSIZE(renderBeginBarriers), renderBeginBarriers);

		vkCmdResetQueryPool(commandBuffer, queryPoolPipeline, 0, 1);
		vkCmdBeginQuery(commandBuffer, queryPoolPipeline, 0, 0);

		VkClearValue clearValues[2] = {};
		clearValues[0].color = { 48.f / 255.f, 10.f / 255.f, 36.f / 255.f, 1 };
		clearValues[1].depthStencil = { 0.f, 0 };

		VkRenderPassBeginInfo passBeginInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
		passBeginInfo.renderPass = renderPass;
		passBeginInfo.framebuffer = targetFB;
		passBeginInfo.renderArea.extent.width = swapchain.width;
		passBeginInfo.renderArea.extent.height = swapchain.height;
		passBeginInfo.clearValueCount = ARRAYSIZE(clearValues);
		passBeginInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(commandBuffer, &passBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkViewport viewport = { 0, float(swapchain.height), float(swapchain.width), -float(swapchain.height), 0, 1 };
		VkRect2D scissor = { {0, 0}, {uint32_t(swapchain.width), uint32_t(swapchain.height)} };

		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		Globals globals = {};
		globals.projection = projection;

		if (meshShadingSupported && meshShadingEnabled)
		{
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineMS);

			DescriptorInfo descriptors[] = { dcb.buffer, db.buffer, mlb.buffer, mdb.buffer, vb.buffer };
			vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, meshProgramMS.updateTemplate, meshProgramMS.layout, 0, descriptors);

			vkCmdPushConstants(commandBuffer, meshProgramMS.layout, meshProgramMS.pushConstantStages, 0, sizeof(globals), &globals);
			vkCmdDrawMeshTasksIndirectCountNV(commandBuffer, dcb.buffer, offsetof(MeshDrawCommand, indirectMS), dccb.buffer, 0, uint32_t(draws.size()), sizeof(MeshDrawCommand));
		}
		else
		{
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);

			DescriptorInfo descriptors[] = { dcb.buffer, db.buffer, vb.buffer };
			vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, meshProgram.updateTemplate, meshProgram.layout, 0, descriptors);

			vkCmdBindIndexBuffer(commandBuffer, ib.buffer, 0, VK_INDEX_TYPE_UINT32);

			vkCmdPushConstants(commandBuffer, meshProgram.layout, meshProgram.pushConstantStages, 0, sizeof(globals), &globals);
			vkCmdDrawIndexedIndirectCountKHR(commandBuffer, dcb.buffer, offsetof(MeshDrawCommand, indirect), dccb.buffer, 0, uint32_t(draws.size()), sizeof(MeshDrawCommand));
		}

		vkCmdEndRenderPass(commandBuffer);

		vkCmdEndQuery(commandBuffer, queryPoolPipeline, 0);

		VkImageMemoryBarrier copyBarriers[] =
		{
			imageBarrier(colorTarget.image, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
			imageBarrier(swapchain.images[imageIndex], 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
		};

		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 0, 0, ARRAYSIZE(copyBarriers), copyBarriers);

		VkImageCopy copyRegion = {};
		copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.srcSubresource.layerCount = 1;
		copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.dstSubresource.layerCount = 1;
		copyRegion.extent = { swapchain.width, swapchain.height, 1 };

		vkCmdCopyImage(commandBuffer, colorTarget.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain.images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

		VkImageMemoryBarrier presentBarrier = imageBarrier(swapchain.images[imageIndex], VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 0, 0, 1, &presentBarrier);

		vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPoolTimestamp, 1);

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

		VK_CHECK(vkQueuePresentKHR(queue, &presentInfo));

		VK_CHECK(vkDeviceWaitIdle(device));

		uint64_t timestampResults[4] = {};
		VK_CHECK(vkGetQueryPoolResults(device, queryPoolTimestamp, 0, ARRAYSIZE(timestampResults), sizeof(timestampResults), timestampResults, sizeof(timestampResults[0]), VK_QUERY_RESULT_64_BIT));

		uint32_t pipelineResults[1] = {};
		VK_CHECK(vkGetQueryPoolResults(device, queryPoolPipeline, 0, 1, sizeof(pipelineResults), pipelineResults, sizeof(pipelineResults[0]), 0));

		uint32_t triangleCount = pipelineResults[0];

		double frameGpuBegin = double(timestampResults[0]) * props.limits.timestampPeriod * 1e-6;
		double frameGpuEnd = double(timestampResults[1]) * props.limits.timestampPeriod * 1e-6;
		double cullGpuTime = double(timestampResults[3] - timestampResults[2]) * props.limits.timestampPeriod * 1e-6;

		double frameCpuEnd = glfwGetTime() * 1000;

		frameCpuAvg = frameCpuAvg * 0.95 + (frameCpuEnd - frameCpuBegin) * 0.05;
		frameGpuAvg = frameGpuAvg * 0.95 + (frameGpuEnd - frameGpuBegin) * 0.05;

		double trianglesPerSec = double(triangleCount) / double(frameGpuAvg * 1e-3);
		double drawsPerSec = double(drawCount) / double(frameGpuAvg * 1e-3);

		char title[256];
		sprintf(title, "cpu: %.2f ms; gpu: %.2f ms (cull: %.2f ms); triangles %.1fM; %.1fB tri/sec, %.1fM draws/sec; mesh shading %s, culling %s, level-of-detail %s",
			frameCpuAvg, frameGpuAvg, cullGpuTime,
			double(triangleCount) * 1e-6, trianglesPerSec * 1e-9, drawsPerSec * 1e-6,
			meshShadingSupported && meshShadingEnabled ? "ON" : "OFF", cullingEnabled ? "ON" : "OFF", lodEnabled ? "ON" : "OFF");

		glfwSetWindowTitle(window, title);
	}

	VK_CHECK(vkDeviceWaitIdle(device));

	if (colorTarget.image)
		destroyImage(colorTarget, device);
	if (depthTarget.image)
		destroyImage(depthTarget, device);
	if (targetFB)
		vkDestroyFramebuffer(device, targetFB, 0);

	destroyBuffer(mb, device);

	destroyBuffer(db, device);
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

	vkDestroyQueryPool(device, queryPoolTimestamp, 0);
	vkDestroyQueryPool(device, queryPoolPipeline, 0);

	destroySwapchain(device, swapchain);

	vkDestroyPipeline(device, drawcullPipeline, 0);
	destroyProgram(device, drawcullProgram);

	vkDestroyPipeline(device, meshPipeline, 0);
	destroyProgram(device, meshProgram);

	if (meshShadingSupported)
	{
		vkDestroyPipeline(device, meshPipelineMS, 0);
		destroyProgram(device, meshProgramMS);
	}

	vkDestroyShaderModule(device, drawcullCS.module, 0);

	vkDestroyShaderModule(device, meshVS.module, 0);
	vkDestroyShaderModule(device, meshFS.module, 0);

	if (meshShadingSupported)
	{
		vkDestroyShaderModule(device, meshletTS.module, 0);
		vkDestroyShaderModule(device, meshletMS.module, 0);
	}

	vkDestroyRenderPass(device, renderPass, 0);

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
