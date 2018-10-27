#pragma once

struct Shader
{
	VkShaderModule module;
	VkShaderStageFlagBits stage;

	uint32_t storageBufferMask;
};

bool loadShader(Shader& shader, VkDevice device, const char* path);
void destroyShader(Shader& shader, VkDevice device);

VkPipelineLayout createPipelineLayout(VkDevice device, const Shader& vs, const Shader& fs);
VkDescriptorUpdateTemplate createUpdateTemplate(VkDevice device, VkPipelineBindPoint bindPoint, VkPipelineLayout layout, const Shader& vs, const Shader& fs);
VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, VkRenderPass renderPass, const Shader& vs, const Shader& fs, VkPipelineLayout layout);

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
