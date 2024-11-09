#pragma once

struct Image;
struct Buffer;

bool loadImage(Image& image, VkDevice device, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const VkPhysicalDeviceMemoryProperties& memoryProperties, const Buffer& scratch, const char* path);
