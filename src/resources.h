#pragma once

struct Buffer
{
	VkBuffer buffer;
	VkDeviceMemory memory;
	void* data;
	size_t size;
};

struct Image
{
	VkImage image;
	VkImageView imageView;
	VkDeviceMemory memory;
};

VkImageMemoryBarrier imageBarrier(VkImage image, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);
VkBufferMemoryBarrier bufferBarrier(VkBuffer buffer, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask);

void createBuffer(Buffer& result, VkDevice device, const VkPhysicalDeviceMemoryProperties& memoryProperties, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryFlags);
void uploadBuffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const Buffer& buffer, const Buffer& scratch, const void* data, size_t size);
void destroyBuffer(const Buffer& buffer, VkDevice device);

void createImage(Image& result, VkDevice device, const VkPhysicalDeviceMemoryProperties& memoryProperties, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage);
void destroyImage(const Image& image, VkDevice device);
