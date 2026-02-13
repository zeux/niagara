#pragma once

bool isInstanceExtensionSupported(const char* name);

VkInstance createInstance();
VkDebugUtilsMessengerEXT registerDebugCallback(VkInstance instance);

uint32_t getGraphicsFamilyIndex(VkPhysicalDevice physicalDevice);
VkPhysicalDevice pickPhysicalDevice(VkPhysicalDevice* physicalDevices, uint32_t physicalDeviceCount);

VkDevice createDevice(VkInstance instance, VkPhysicalDevice physicalDevice, uint32_t familyIndex, bool meshShadingSupported, bool raytracingSupported, bool clusterrtSupported);
