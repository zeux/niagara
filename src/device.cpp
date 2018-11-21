#include "common.h"
#include "device.h"

VkInstance createInstance()
{
	assert(volkGetInstanceVersion() >= VK_API_VERSION_1_1);

	VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	appInfo.apiVersion = VK_API_VERSION_1_1;

	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.pApplicationInfo = &appInfo;

#ifdef _DEBUG
	const char* debugLayers[] =
	{
		"VK_LAYER_LUNARG_standard_validation"
	};

	createInfo.ppEnabledLayerNames = debugLayers;
	createInfo.enabledLayerCount = sizeof(debugLayers) / sizeof(debugLayers[0]);
#endif

	const char* extensions[] =
	{
		VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_WIN32_KHR
		VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#endif
#ifdef _DEBUG
		VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
#endif
	};

	createInfo.ppEnabledExtensionNames = extensions;
	createInfo.enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]);

	VkInstance instance = 0;
	VK_CHECK(vkCreateInstance(&createInfo, 0, &instance));

	return instance;
}

static VkBool32 VKAPI_CALL debugReportCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage, void* pUserData)
{
	// Validation layers don't correctly detect NonWriteable declarations for storage buffers: https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/73
	if (strstr(pMessage, "Shader requires vertexPipelineStoresAndAtomics but is not enabled on the device"))
		return VK_FALSE;

	// Validation layers don't correctly track set assignments when using push descriptors with update templates: https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/341
	if (strstr(pMessage, "uses set #0 but that set is not bound."))
		return VK_FALSE;

	// glslang adds UniformAndStorageBuffer8BitAccess capability to shaders that don't need it: https://github.com/KhronosGroup/glslang/issues/1539
	if (strstr(pMessage, "VkPhysicalDevice8BitStorageFeaturesKHR::uniformAndStorageBuffer8BitAccess"))
		return VK_FALSE;

	// This silences warnings like "For optimal performance image layout should be VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL instead of GENERAL."
	// We'll assume other performance warnings are also not useful.
	if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)
		return VK_FALSE;

	const char* type =
		(flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
		? "ERROR"
		: (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
			? "WARNING"
			: "INFO";

	char message[4096];
	snprintf(message, ARRAYSIZE(message), "%s: %s\n", type, pMessage);

	printf("%s", message);

#ifdef _WIN32
	OutputDebugStringA(message);
#endif

	if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
		assert(!"Validation error encountered!");

	return VK_FALSE;
}

VkDebugReportCallbackEXT registerDebugCallback(VkInstance instance)
{
	VkDebugReportCallbackCreateInfoEXT createInfo = { VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT };
	createInfo.flags = VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT | VK_DEBUG_REPORT_ERROR_BIT_EXT;
	createInfo.pfnCallback = debugReportCallback;

	VkDebugReportCallbackEXT callback = 0;
	VK_CHECK(vkCreateDebugReportCallbackEXT(instance, &createInfo, 0, &callback));

	return callback;
}

uint32_t getGraphicsFamilyIndex(VkPhysicalDevice physicalDevice)
{
	uint32_t queueCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, 0);

	std::vector<VkQueueFamilyProperties> queues(queueCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, queues.data());

	for (uint32_t i = 0; i < queueCount; ++i)
		if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			return i;

	return VK_QUEUE_FAMILY_IGNORED;
}

static bool supportsPresentation(VkPhysicalDevice physicalDevice, uint32_t familyIndex)
{
#if defined(VK_USE_PLATFORM_WIN32_KHR)
	return !!vkGetPhysicalDeviceWin32PresentationSupportKHR(physicalDevice, familyIndex);
#else
	return true;
#endif
}

VkPhysicalDevice pickPhysicalDevice(VkPhysicalDevice* physicalDevices, uint32_t physicalDeviceCount)
{
	VkPhysicalDevice preferred = 0;
	VkPhysicalDevice fallback = 0;

	for (uint32_t i = 0; i < physicalDeviceCount; ++i)
	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(physicalDevices[i], &props);

		printf("GPU%d: %s\n", i, props.deviceName);

		uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevices[i]);
		if (familyIndex == VK_QUEUE_FAMILY_IGNORED)
			continue;

		if (!supportsPresentation(physicalDevices[i], familyIndex))
			continue;

		if (props.apiVersion < VK_API_VERSION_1_1)
			continue;

		if (!preferred && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		{
			preferred = physicalDevices[i];
		}

		if (!fallback)
		{
			fallback = physicalDevices[i];
		}
	}

	VkPhysicalDevice result = preferred ? preferred : fallback;

	if (result)
	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(result, &props);

		printf("Selected GPU %s\n", props.deviceName);
	}
	else
	{
		printf("ERROR: No GPUs found\n");
	}

	return result;
}

VkDevice createDevice(VkInstance instance, VkPhysicalDevice physicalDevice, uint32_t familyIndex, bool checkpointsSupported, bool meshShadingSupported)
{
	float queuePriorities[] = { 1.0f };

	VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	queueInfo.queueFamilyIndex = familyIndex;
	queueInfo.queueCount = 1;
	queueInfo.pQueuePriorities = queuePriorities;

	std::vector<const char*> extensions =
	{
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
		VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
		VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME,
		VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME,
	};

    if (checkpointsSupported)
        extensions.push_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);

	if (meshShadingSupported)
		extensions.push_back(VK_NV_MESH_SHADER_EXTENSION_NAME);

	VkPhysicalDeviceFeatures2 features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
	features.features.multiDrawIndirect = true;
	features.features.pipelineStatisticsQuery = true;

	VkPhysicalDevice16BitStorageFeatures features16 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
	features16.storageBuffer16BitAccess = true;

	VkPhysicalDevice8BitStorageFeaturesKHR features8 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR };
	features8.storageBuffer8BitAccess = true;

	// This will only be used if meshShadingSupported=true (see below)
	VkPhysicalDeviceMeshShaderFeaturesNV featuresMesh = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV };
	featuresMesh.taskShader = true;
	featuresMesh.meshShader = true;

	VkDeviceCreateInfo createInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	createInfo.queueCreateInfoCount = 1;
	createInfo.pQueueCreateInfos = &queueInfo;

	createInfo.ppEnabledExtensionNames = extensions.data();
	createInfo.enabledExtensionCount = uint32_t(extensions.size());

	createInfo.pNext = &features;
	features.pNext = &features16;
	features16.pNext = &features8;

	if (meshShadingSupported)
		features8.pNext = &featuresMesh;

	VkDevice device = 0;
	VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, 0, &device));

	return device;
}
