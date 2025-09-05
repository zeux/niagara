#include "common.h"
#include "swapchain.h"

#include "config.h"

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <algorithm>

#define VSYNC CONFIG_VSYNC

VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow* window)
{
	// Note: GLFW has a helper glfwCreateWindowSurface but we're going to do this the hard way to reduce our reliance on GLFW Vulkan specifics
	VkSurfaceKHR surface = 0;

#ifdef VK_USE_PLATFORM_WIN32_KHR
	if (glfwGetPlatform() == GLFW_PLATFORM_WIN32)
	{
		VkWin32SurfaceCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
		createInfo.hinstance = GetModuleHandle(0);
		createInfo.hwnd = glfwGetWin32Window(window);
		VK_CHECK(vkCreateWin32SurfaceKHR(instance, &createInfo, 0, &surface));
	}
#endif

#ifdef VK_USE_PLATFORM_WAYLAND_KHR
	if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND)
	{
		VkWaylandSurfaceCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR };
		createInfo.display = glfwGetWaylandDisplay();
		createInfo.surface = glfwGetWaylandWindow(window);
		VK_CHECK(vkCreateWaylandSurfaceKHR(instance, &createInfo, 0, &surface));
	}
#endif

#ifdef VK_USE_PLATFORM_XLIB_KHR
	if (glfwGetPlatform() == GLFW_PLATFORM_X11)
	{
		VkXlibSurfaceCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR };
		createInfo.dpy = glfwGetX11Display();
		createInfo.window = glfwGetX11Window(window);
		VK_CHECK(vkCreateXlibSurfaceKHR(instance, &createInfo, 0, &surface));
	}
#endif

#ifdef VK_USE_PLATFORM_METAL_EXT
	// fallback to GLFW
	if (!surface)
		VK_CHECK(glfwCreateWindowSurface(instance, window, 0, &surface));
#endif

	return surface;
}

VkFormat getSwapchainFormat(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
	uint32_t formatCount = 0;
	VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, 0));
	assert(formatCount > 0);

	std::vector<VkSurfaceFormatKHR> formats(formatCount);
	VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data()));

	if (formatCount == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
		return VK_FORMAT_R8G8B8A8_UNORM;

	for (uint32_t i = 0; i < formatCount; ++i)
		if (formats[i].format == VK_FORMAT_R8G8B8A8_UNORM || formats[i].format == VK_FORMAT_B8G8R8A8_UNORM)
			return formats[i].format;

	return formats[0].format;
}

static VkPresentModeKHR getPresentMode(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
	if (VSYNC)
		return VK_PRESENT_MODE_FIFO_KHR; // guaranteed to be available

	uint32_t presentModeCount = 0;
	VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, 0));
	assert(presentModeCount > 0);

	std::vector<VkPresentModeKHR> presentModes(presentModeCount);
	VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data()));

	for (VkPresentModeKHR mode : presentModes)
	{
		if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
			return mode;
		if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
			return mode;
	}

	// fall back to fifo
	return VK_PRESENT_MODE_FIFO_KHR;
}

void createSwapchain(Swapchain& result, VkPhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface, uint32_t familyIndex, GLFWwindow* window, VkFormat format, VkSwapchainKHR oldSwapchain)
{
	VkSurfaceCapabilitiesKHR surfaceCaps;
	VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps));

	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);

	VkPresentModeKHR presentMode = getPresentMode(physicalDevice, surface);

	VkCompositeAlphaFlagBitsKHR surfaceComposite =
	    (surfaceCaps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
	        ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
	    : (surfaceCaps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
	        ? VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR
	    : (surfaceCaps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
	        ? VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR
	        : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;

	VkSwapchainCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
	createInfo.surface = surface;
	createInfo.minImageCount = std::max(unsigned(MIN_IMAGES), surfaceCaps.minImageCount);
	createInfo.imageFormat = format;
	createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	createInfo.imageExtent.width = width;
	createInfo.imageExtent.height = height;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT;
	createInfo.queueFamilyIndexCount = 1;
	createInfo.pQueueFamilyIndices = &familyIndex;
	createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	createInfo.compositeAlpha = surfaceComposite;
	createInfo.presentMode = presentMode;
	createInfo.oldSwapchain = oldSwapchain;

	VkSwapchainKHR swapchain = 0;
	VK_CHECK(vkCreateSwapchainKHR(device, &createInfo, 0, &swapchain));

	uint32_t imageCount = 0;
	VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, 0));

	std::vector<VkImage> images(imageCount);
	VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.data()));

	result.swapchain = swapchain;
	result.images = images;
	result.width = width;
	result.height = height;
	result.imageCount = imageCount;
}

void destroySwapchain(VkDevice device, const Swapchain& swapchain)
{
	vkDestroySwapchainKHR(device, swapchain.swapchain, 0);
}

SwapchainStatus updateSwapchain(Swapchain& result, VkPhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface, uint32_t familyIndex, GLFWwindow* window, VkFormat format)
{
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);

	if (width == 0 || height == 0)
		return Swapchain_NotReady;

	if (result.width == width && result.height == height)
		return Swapchain_Ready;

	Swapchain old = result;

	createSwapchain(result, physicalDevice, device, surface, familyIndex, window, format, old.swapchain);

	VK_CHECK(vkDeviceWaitIdle(device));

	destroySwapchain(device, old);

	return Swapchain_Resized;
}
