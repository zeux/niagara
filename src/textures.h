#pragma once

struct Image;
struct Buffer;

bool loadImage(Image& image, VkDevice device, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const VkPhysicalDeviceMemoryProperties& memoryProperties, const Buffer& scratch, const char* path);

unsigned char* decodeImageRGBA(const char* path, int mip, unsigned int& width, unsigned int& height, unsigned int& levels);
