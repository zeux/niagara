#pragma once

#include <string>

struct Shader
{
	std::string name;

	std::vector<char> spirv;
	VkShaderStageFlagBits stage;

	VkDescriptorType resourceTypes[32];
	uint32_t resourceMask;

	uint32_t localSizeX;
	uint32_t localSizeY;
	uint32_t localSizeZ;

	bool usesPushConstants;
	bool usesDescriptorArray;
};

struct ShaderSet
{
	std::vector<Shader> shaders;

	const Shader& operator[](const char* name) const;
};

struct Program
{
	VkPipelineBindPoint bindPoint;
	VkPipelineLayout layout;
	VkDescriptorSetLayout setLayout;
	VkDescriptorUpdateTemplate updateTemplate;

	VkShaderStageFlags pushConstantStages;
	uint32_t pushConstantSize;
	uint32_t pushDescriptorCount;

	size_t descriptorSize; // only used for descriptor heaps
	VkDescriptorType resourceTypes[32] = {};
	uint32_t resourceMask = 0;

	uint32_t localSizeX;
	uint32_t localSizeY;
	uint32_t localSizeZ;

	const Shader* shaders[8];
	size_t shaderCount;
};

bool loadShader(Shader& shader, const char* path);
bool loadShader(Shader& shader, const char* base, const char* path);
bool loadShaders(ShaderSet& shaders, const char* base, const char* path);

using Shaders = std::initializer_list<const Shader*>;
using Constants = std::initializer_list<int>;

VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, const VkPipelineRenderingCreateInfo& renderingInfo, const Program& program, Constants constants = {});
VkPipeline createComputePipeline(VkDevice device, VkPipelineCache pipelineCache, const Program& program, Constants constants = {});

Program createProgram(VkDevice device, VkPipelineBindPoint bindPoint, Shaders shaders, size_t pushConstantSize, size_t descriptorSize = 0, VkDescriptorSetLayout arrayLayout = nullptr);
void destroyProgram(VkDevice device, const Program& program);

VkDescriptorSetLayout createDescriptorArrayLayout(VkDevice device);
std::pair<VkDescriptorPool, VkDescriptorSet> createDescriptorArray(VkDevice device, VkDescriptorSetLayout layout, uint32_t descriptorCount);

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
		VkAccelerationStructureKHR accelerationStructure;
	};

	const void* resource = NULL;
	int resourceMip = -1;

	DescriptorInfo()
	{
	}

	DescriptorInfo(VkAccelerationStructureKHR structure)
	{
		accelerationStructure = structure;
	}

	DescriptorInfo(VkImageView imageView)
	{
		image.sampler = VK_NULL_HANDLE;
		image.imageView = imageView;
		image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}

	DescriptorInfo(VkSampler sampler)
	{
		image.sampler = sampler;
		image.imageView = VK_NULL_HANDLE;
		image.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	DescriptorInfo(VkBuffer buffer_)
	{
		buffer.buffer = buffer_;
		buffer.offset = 0;
		buffer.range = VK_WHOLE_SIZE;
	}

	DescriptorInfo(const struct Buffer& buffer);
	DescriptorInfo(const struct Image& image);
	DescriptorInfo(const struct Image& image, VkImageView mipView, int mipIndex);
};
