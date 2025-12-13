#include "common.h"
#include "device.h"

#include "config.h"
#include "swapchain.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Validation is enabled by default in Debug
#ifndef NDEBUG
#define KHR_VALIDATION 1
#else
#define KHR_VALIDATION CONFIG_RELVAL
#endif

// Synchronization validation is disabled by default in Debug since it's rather slow
#define SYNC_VALIDATION CONFIG_SYNCVAL

// We have a strict requirement for latest Vulkan version to be available
#define API_VERSION VK_API_VERSION_1_4

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

bool isInstanceExtensionSupported(const char* name)
{
	uint32_t propertyCount = 0;
	VK_CHECK(vkEnumerateInstanceExtensionProperties(NULL, &propertyCount, 0));

	std::vector<VkExtensionProperties> properties(propertyCount);
	VK_CHECK(vkEnumerateInstanceExtensionProperties(NULL, &propertyCount, properties.data()));

	for (uint32_t i = 0; i < propertyCount; ++i)
		if (strcmp(name, properties[i].extensionName) == 0)
			return true;

	return false;
}

VkInstance createInstance()
{
	if (volkGetInstanceVersion() < API_VERSION)
	{
		fprintf(stderr, "ERROR: Vulkan 1.%d instance not found\n", VK_VERSION_MINOR(API_VERSION));
		return 0;
	}

	VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	appInfo.apiVersion = API_VERSION;

	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.pApplicationInfo = &appInfo;

#if KHR_VALIDATION || SYNC_VALIDATION
	const char* debugLayers[] = {
		"VK_LAYER_KHRONOS_validation",
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
	VkValidationFeatureEnableEXT enabledValidationFeatures[] = {
		VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
	};

	VkValidationFeaturesEXT validationFeatures = { VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT };
	validationFeatures.enabledValidationFeatureCount = sizeof(enabledValidationFeatures) / sizeof(enabledValidationFeatures[0]);
	validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures;

	createInfo.pNext = &validationFeatures;
#endif
#endif

	std::vector<const char*> extensions;

	// Query Vulkan instance extensions required by GLFW for creating Vulkan surfaces for GLFW windows.
	uint32_t swapchainExtensionCount;
	if (const char** swapchainExtensions = getSwapchainExtensions(&swapchainExtensionCount))
		extensions.insert(extensions.end(), swapchainExtensions, swapchainExtensions + swapchainExtensionCount);

	if (isInstanceExtensionSupported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

	createInfo.ppEnabledExtensionNames = extensions.data();
	createInfo.enabledExtensionCount = extensions.size();

#ifdef VK_USE_PLATFORM_METAL_EXT
	createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

	VkInstance instance = 0;
	VK_CHECK(vkCreateInstance(&createInfo, 0, &instance));

	return instance;
}

static VkBool32 VKAPI_CALL debugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT types, const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData)
{
	if (severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
		return VK_FALSE;

	// Works around https://github.com/KhronosGroup/Vulkan-Docs/issues/2606
	if (strstr(callbackData->pMessage, "vkCmdBuildClusterAccelerationStructureIndirectNV(): pCommandInfos->srcInfosCount is zero"))
		return VK_FALSE;

	const char* type = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" : "WARNING";

	char message[4096];
	snprintf(message, COUNTOF(message), "%s: %s\n", type, callbackData->pMessage);

	printf("%s", message);

#ifdef _WIN32
	OutputDebugStringA(message);
#endif

	if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
		assert(!"Validation error encountered!");

	return VK_FALSE;
}

VkDebugUtilsMessengerEXT registerDebugCallback(VkInstance instance)
{
	if (!vkCreateDebugUtilsMessengerEXT)
		return nullptr;

	VkDebugUtilsMessengerCreateInfoEXT createInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugUtilsCallback;

	VkDebugUtilsMessengerEXT messenger = 0;
	VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance, &createInfo, 0, &messenger));

	return messenger;
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

		printf("GPU%d: %s (Vulkan 1.%d)\n", i, props.deviceName, VK_VERSION_MINOR(props.apiVersion));

		uint32_t familyIndex = getGraphicsFamilyIndex(physicalDevices[i]);
		if (familyIndex == VK_QUEUE_FAMILY_IGNORED)
			continue;

		if (!supportsPresentation(physicalDevices[i], familyIndex))
			continue;

		if (props.apiVersion < API_VERSION)
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
		fprintf(stderr, "ERROR: No compatible GPU found\n");
	}

	return result;
}

VkDevice createDevice(VkInstance instance, VkPhysicalDevice physicalDevice, uint32_t familyIndex, bool meshShadingSupported, bool raytracingSupported, bool clusterrtSupported)
{
	float queuePriorities[] = { 1.0f };

	VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	queueInfo.queueFamilyIndex = familyIndex;
	queueInfo.queueCount = 1;
	queueInfo.pQueuePriorities = queuePriorities;

	std::vector<const char*> extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};

	if (meshShadingSupported)
		extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);

	if (raytracingSupported)
	{
		extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
		extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
		extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	}

#ifdef VK_NV_cluster_acceleration_structure
	if (clusterrtSupported)
		extensions.push_back(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME);
#endif

	VkPhysicalDeviceFeatures2 features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
	features.features.multiDrawIndirect = true;
	features.features.pipelineStatisticsQuery = true;
	features.features.shaderInt16 = true;
	features.features.shaderInt64 = true;
	features.features.samplerAnisotropy = true;

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

	if (raytracingSupported)
		features12.bufferDeviceAddress = true;

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
	features13.shaderDemoteToHelperInvocation = true; // required for discard; under new glslang rules

	VkPhysicalDeviceVulkan14Features features14 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES };
	features14.maintenance5 = true;
	features14.maintenance6 = true;
	features14.pushDescriptor = true;

	// This will only be used if meshShadingSupported=true (see below)
	VkPhysicalDeviceMeshShaderFeaturesEXT featuresMesh = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT };
	featuresMesh.taskShader = true;
	featuresMesh.meshShader = true;

	// This will only be used if raytracingSupported=true (see below)
	VkPhysicalDeviceRayQueryFeaturesKHR featuresRayQueries = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
	featuresRayQueries.rayQuery = true;

	// This will only be used if raytracingSupported=true (see below)
	VkPhysicalDeviceAccelerationStructureFeaturesKHR featuresAccelerationStructure = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
	featuresAccelerationStructure.accelerationStructure = true;

	// This will only be used if clusterrtSupported=true (see below)
	VkPhysicalDeviceClusterAccelerationStructureFeaturesNV featuresClusterAcceleration = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV };
	featuresClusterAcceleration.clusterAccelerationStructure = true;

	VkDeviceCreateInfo createInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	createInfo.queueCreateInfoCount = 1;
	createInfo.pQueueCreateInfos = &queueInfo;

	createInfo.ppEnabledExtensionNames = extensions.data();
	createInfo.enabledExtensionCount = uint32_t(extensions.size());

	createInfo.pNext = &features;
	features.pNext = &features11;
	features11.pNext = &features12;
	features12.pNext = &features13;
	features13.pNext = &features14;

	void** ppNext = &features14.pNext;

	if (meshShadingSupported)
	{
		*ppNext = &featuresMesh;
		ppNext = &featuresMesh.pNext;
	}

	if (raytracingSupported)
	{
		*ppNext = &featuresRayQueries;
		ppNext = &featuresRayQueries.pNext;

		*ppNext = &featuresAccelerationStructure;
		ppNext = &featuresAccelerationStructure.pNext;
	}

	if (clusterrtSupported)
	{
		*ppNext = &featuresClusterAcceleration;
		ppNext = &featuresClusterAcceleration.pNext;
	}

	VkDevice device = 0;
	VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, 0, &device));

	return device;
}
