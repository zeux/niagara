#pragma once

struct Shader
{
	VkShaderModule module;
	VkShaderStageFlagBits stage;

	VkDescriptorType resourceTypes[32];
	uint32_t resourceMask;

	uint32_t localSizeX;
	uint32_t localSizeY;
	uint32_t localSizeZ;

	bool usesPushConstants;
	bool usesDescriptorArray;
};

struct Program
{
	VkPipelineBindPoint bindPoint;
	VkPipelineLayout layout;
	VkDescriptorSetLayout setLayout;
	VkDescriptorUpdateTemplate updateTemplate;
	VkShaderStageFlags pushConstantStages;
};

bool loadShader(Shader& shader, VkDevice device, const char* path);

using Shaders = std::initializer_list<const Shader*>;
using Constants = std::initializer_list<int>;

VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, const VkPipelineRenderingCreateInfo& renderingInfo, Shaders shaders, VkPipelineLayout layout, Constants constants = {});
VkPipeline createComputePipeline(VkDevice device, VkPipelineCache pipelineCache, const Shader& shader, VkPipelineLayout layout, Constants constants = {});

Program createProgram(VkDevice device, VkPipelineBindPoint bindPoint, Shaders shaders, size_t pushConstantSize);
void destroyProgram(VkDevice device, const Program& program);

inline uint32_t getGroupCount(uint32_t threadCount, uint32_t localSize)
{
	return (threadCount + localSize - 1) / localSize;
}

struct DescriptorInfo
{
	union
	{
		VkDescriptorImageInfo image;
		VkDescriptorBufferInfo buffer;
	};

	DescriptorInfo()
	{
	}

	DescriptorInfo(VkImageView imageView, VkImageLayout imageLayout)
	{
		image.sampler = VK_NULL_HANDLE;
		image.imageView = imageView;
		image.imageLayout = imageLayout;
	}

	DescriptorInfo(VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout)
	{
		image.sampler = sampler;
		image.imageView = imageView;
		image.imageLayout = imageLayout;
	}

	DescriptorInfo(VkBuffer buffer_, VkDeviceSize offset, VkDeviceSize range)
	{
		buffer.buffer = buffer_;
		buffer.offset = offset;
		buffer.range = range;
	}

	DescriptorInfo(VkBuffer buffer_)
	{
		buffer.buffer = buffer_;
		buffer.offset = 0;
		buffer.range = VK_WHOLE_SIZE;
	}
};
