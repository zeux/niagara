#include "common.h"
#include "resources.h"

VkImageMemoryBarrier imageBarrier(VkImage image, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageAspectFlags aspectMask)
{
	VkImageMemoryBarrier result = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };

	result.srcAccessMask = srcAccessMask;
	result.dstAccessMask = dstAccessMask;
	result.oldLayout = oldLayout;
	result.newLayout = newLayout;
	result.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	result.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	result.image = image;
	result.subresourceRange.aspectMask = aspectMask;
	result.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
	result.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

	return result;
}

VkBufferMemoryBarrier bufferBarrier(VkBuffer buffer, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask)
{
	VkBufferMemoryBarrier result = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };

	result.srcAccessMask = srcAccessMask;
	result.dstAccessMask = dstAccessMask;
	result.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	result.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	result.buffer = buffer;
	result.offset = 0;
	result.size = VK_WHOLE_SIZE;

	return result;
}

static uint32_t selectMemoryType(const VkPhysicalDeviceMemoryProperties& memoryProperties, uint32_t memoryTypeBits, VkMemoryPropertyFlags flags)
{
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
		if ((memoryTypeBits & (1 << i)) != 0 && (memoryProperties.memoryTypes[i].propertyFlags & flags) == flags)
			return i;

	assert(!"No compatible memory type found");
	return ~0u;
}

void createBuffer(Buffer& result, VkDevice device, const VkPhysicalDeviceMemoryProperties& memoryProperties, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryFlags)
{
	VkBufferCreateInfo createInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	createInfo.size = size;
	createInfo.usage = usage;

	VkBuffer buffer = 0;
	VK_CHECK(vkCreateBuffer(device, &createInfo, 0, &buffer));

	VkMemoryRequirements memoryRequirements;
	vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

	uint32_t memoryTypeIndex = selectMemoryType(memoryProperties, memoryRequirements.memoryTypeBits, memoryFlags);
	assert(memoryTypeIndex != ~0u);

	VkMemoryAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	allocateInfo.allocationSize = memoryRequirements.size;
	allocateInfo.memoryTypeIndex = memoryTypeIndex;

	VkMemoryAllocateFlagsInfo flagInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };

	if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
	{
		allocateInfo.pNext = &flagInfo;
		flagInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
		flagInfo.deviceMask = 1;
	}

	VkDeviceMemory memory = 0;
	VK_CHECK(vkAllocateMemory(device, &allocateInfo, 0, &memory));

	VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));

	void* data = 0;
	if (memoryFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		VK_CHECK(vkMapMemory(device, memory, 0, size, 0, &data));

	result.buffer = buffer;
	result.memory = memory;
	result.data = data;
	result.size = size;
}

void uploadBuffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const Buffer& buffer, const Buffer& scratch, const void* data, size_t size)
{
	// TODO: this function is submitting a command buffer and waiting for device idle for each buffer upload; this is obviously suboptimal and we'd need to batch this later
	assert(scratch.data);
	assert(scratch.size >= size);
	memcpy(scratch.data, data, size);

	VK_CHECK(vkResetCommandPool(device, commandPool, 0));

	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	VkBufferCopy region = { 0, 0, VkDeviceSize(size) };
	vkCmdCopyBuffer(commandBuffer, scratch.buffer, buffer.buffer, 1, &region);

	VkBufferMemoryBarrier copyBarrier = bufferBarrier(buffer.buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 1, &copyBarrier, 0, 0);

	VK_CHECK(vkEndCommandBuffer(commandBuffer));

	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

	VK_CHECK(vkDeviceWaitIdle(device));
}

void destroyBuffer(const Buffer& buffer, VkDevice device)
{
	vkDestroyBuffer(device, buffer.buffer, 0);
	vkFreeMemory(device, buffer.memory, 0);
}

uint64_t getBufferAddress(const Buffer& buffer, VkDevice device)
{
	VkBufferDeviceAddressInfo info = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
	info.buffer = buffer.buffer;

	VkDeviceAddress address = vkGetBufferDeviceAddress(device, &info);
	assert(address != 0);

	return address;
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format, uint32_t mipLevel, uint32_t levelCount)
{
	VkImageAspectFlags aspectMask = (format == VK_FORMAT_D32_SFLOAT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

	VkImageViewCreateInfo createInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
	createInfo.image = image;
	createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	createInfo.format = format;
	createInfo.subresourceRange.aspectMask = aspectMask;
	createInfo.subresourceRange.baseMipLevel = mipLevel;
	createInfo.subresourceRange.levelCount = levelCount;
	createInfo.subresourceRange.layerCount = 1;

	VkImageView view = 0;
	VK_CHECK(vkCreateImageView(device, &createInfo, 0, &view));

	return view;
}

void createImage(Image& result, VkDevice device, const VkPhysicalDeviceMemoryProperties& memoryProperties, uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format, VkImageUsageFlags usage)
{
	VkImageCreateInfo createInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };

	createInfo.imageType = VK_IMAGE_TYPE_2D;
	createInfo.format = format;
	createInfo.extent = { width, height, 1 };
	createInfo.mipLevels = mipLevels;
	createInfo.arrayLayers = 1;
	createInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	createInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	createInfo.usage = usage;
	createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VkImage image = 0;
	VK_CHECK(vkCreateImage(device, &createInfo, 0, &image));

	VkMemoryRequirements memoryRequirements;
	vkGetImageMemoryRequirements(device, image, &memoryRequirements);

	uint32_t memoryTypeIndex = selectMemoryType(memoryProperties, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	assert(memoryTypeIndex != ~0u);

	VkMemoryAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	allocateInfo.allocationSize = memoryRequirements.size;
	allocateInfo.memoryTypeIndex = memoryTypeIndex;

	VkDeviceMemory memory = 0;
	VK_CHECK(vkAllocateMemory(device, &allocateInfo, 0, &memory));

	VK_CHECK(vkBindImageMemory(device, image, memory, 0));

	result.image = image;
	result.imageView = createImageView(device, image, format, 0, mipLevels);
	result.memory = memory;
}

void destroyImage(const Image& image, VkDevice device)
{
	vkDestroyImageView(device, image.imageView, 0);
	vkDestroyImage(device, image.image, 0);
	vkFreeMemory(device, image.memory, 0);
}

uint32_t getImageMipLevels(uint32_t width, uint32_t height)
{
	uint32_t result = 1;

	while (width > 1 || height > 1)
	{
		result++;
		width /= 2;
		height /= 2;
	}

	return result;
}

VkSampler createSampler(VkDevice device, VkSamplerReductionModeEXT reductionMode)
{
	VkSamplerCreateInfo createInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };

	createInfo.magFilter = VK_FILTER_LINEAR;
	createInfo.minFilter = VK_FILTER_LINEAR;
	createInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	createInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.minLod = 0;
	createInfo.maxLod = 16.f;

	VkSamplerReductionModeCreateInfoEXT createInfoReduction = { VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT };

	if (reductionMode != VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE_EXT)
	{
		createInfoReduction.reductionMode = reductionMode;

		createInfo.pNext = &createInfoReduction;
	}

	VkSampler sampler = 0;
	VK_CHECK(vkCreateSampler(device, &createInfo, 0, &sampler));
	return sampler;
}
