#include "common.h"
#include "device.h"

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Validation is enabled by default in Debug
#ifndef NDEBUG
#define KHR_VALIDATION 1
#endif

// Synchronization validation is disabled by default in Debug since it's rather slow
#define SYNC_VALIDATION CONFIG_SYNCVAL

#ifdef _WIN32
#include <Windows.h>
#endif

static bool isLayerSupported(const char* name)
{
	uint32_t propertyCount = 0;
	VK_CHECK(vkEnumerateInstanceLayerProperties(&propertyCount, 0));

	std::vector<VkLayerProperties> properties(propertyCount);
	VK_CHECK(vkEnumerateInstanceLayerProperties(&propertyCount, properties.data()));

	for (uint32_t i = 0; i < propertyCount; ++i)
		if (strcmp(name, properties[i].layerName) == 0)
			return true;

	return false;
}

VkInstance createInstance()
{
	assert(volkGetInstanceVersion() >= VK_API_VERSION_1_3);

	VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	appInfo.apiVersion = VK_API_VERSION_1_3;

	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.pApplicationInfo = &appInfo;

#if KHR_VALIDATION || SYNC_VALIDATION
	const char* debugLayers[] =
	{
		"VK_LAYER_KHRONOS_validation"
	};

	if (isLayerSupported("VK_LAYER_KHRONOS_validation"))
	{
		createInfo.ppEnabledLayerNames = debugLayers;
		createInfo.enabledLayerCount = sizeof(debugLayers) / sizeof(debugLayers[0]);
		printf("Enabled Vulkan validation layers (sync validation %s)\n", SYNC_VALIDATION ? "enabled" : "disabled");
	}
	else
	{
		printf("Warning: Vulkan debug layers are not available\n");
	}

#if SYNC_VALIDATION
	VkValidationFeatureEnableEXT enabledValidationFeatures[] =
	{
		VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
	};

	VkValidationFeaturesEXT validationFeatures = { VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT };
	validationFeatures.enabledValidationFeatureCount = sizeof(enabledValidationFeatures) / sizeof(enabledValidationFeatures[0]);
	validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures;

	createInfo.pNext = &validationFeatures;
#endif
#endif

	const char* extensions[] =
	{
		VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_WIN32_KHR
		VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
		VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
		VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
#endif
#ifndef NDEBUG
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
	snprintf(message, COUNTOF(message), "%s: %s\n", type, pMessage);

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
	if (!vkCreateDebugReportCallbackEXT)
		return nullptr;

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

	const char* ngpu = getenv("NGPU");

	for (uint32_t i = 0; i < physicalDeviceCount; ++i)
	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(physicalDevices[i], &props);

		if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
			continue;

		printf("GPU%d: %s\n", i, props.deviceName);

		uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevices[i]);
		if (familyIndex == VK_QUEUE_FAMILY_IGNORED)
			continue;

		if (!supportsPresentation(physicalDevices[i], familyIndex))
			continue;

		if (props.apiVersion < VK_API_VERSION_1_3)
			continue;

		if (ngpu && atoi(ngpu) == i)
		{
			preferred = physicalDevices[i];
		}

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

VkDevice createDevice(VkInstance instance, VkPhysicalDevice physicalDevice, uint32_t familyIndex, bool meshShadingSupported)
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
	};

	if (meshShadingSupported)
		extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);

	VkPhysicalDeviceFeatures2 features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
	features.features.multiDrawIndirect = true;
	features.features.pipelineStatisticsQuery = true;
	features.features.shaderInt16 = true;
	features.features.shaderInt64 = true;

	VkPhysicalDeviceVulkan11Features features11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
	features11.storageBuffer16BitAccess = true;
	features11.shaderDrawParameters = true;

	VkPhysicalDeviceVulkan12Features features12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	features12.drawIndirectCount = true;
	features12.storageBuffer8BitAccess = true;
	features12.uniformAndStorageBuffer8BitAccess = true;
	features12.shaderFloat16 = true;
	features12.shaderInt8 = true;
	features12.samplerFilterMinmax = true;
	features12.scalarBlockLayout = true;

	features12.descriptorIndexing = true;
	features12.shaderSampledImageArrayNonUniformIndexing = true;
	features12.descriptorBindingSampledImageUpdateAfterBind = true;
	features12.descriptorBindingUpdateUnusedWhilePending = true;
	features12.descriptorBindingPartiallyBound = true;
	features12.descriptorBindingVariableDescriptorCount = true;
	features12.runtimeDescriptorArray = true;

	VkPhysicalDeviceVulkan13Features features13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
	features13.dynamicRendering = true;
	features13.synchronization2 = true;
	features13.maintenance4 = true;

	// This will only be used if meshShadingSupported=true (see below)
	VkPhysicalDeviceMeshShaderFeaturesEXT featuresMesh = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT };
	featuresMesh.taskShader = true;
	featuresMesh.meshShader = true;

	VkDeviceCreateInfo createInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	createInfo.queueCreateInfoCount = 1;
	createInfo.pQueueCreateInfos = &queueInfo;

	createInfo.ppEnabledExtensionNames = extensions.data();
	createInfo.enabledExtensionCount = uint32_t(extensions.size());

	createInfo.pNext = &features;
	features.pNext = &features11;
	features11.pNext = &features12;
	features12.pNext = &features13;

	if (meshShadingSupported)
		features13.pNext = &featuresMesh;

	VkDevice device = 0;
	VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, 0, &device));

	return device;
}
