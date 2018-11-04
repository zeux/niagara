#include "common.h"
#include "shaders.h"

#include <stdio.h>

#include <vector>

#include <vulkan/spirv.h>

// https://www.khronos.org/registry/spir-v/specs/1.0/SPIRV.pdf
struct Id
{
	enum Kind { Unknown = 0, Variable };

	Kind kind;
	uint32_t type;
	uint32_t storageClass;
	uint32_t binding;
	uint32_t set;
};

static VkShaderStageFlagBits getShaderStage(SpvExecutionModel executionModel)
{
	switch (executionModel)
	{
	case SpvExecutionModelVertex:
		return VK_SHADER_STAGE_VERTEX_BIT;
	case SpvExecutionModelFragment:
		return VK_SHADER_STAGE_FRAGMENT_BIT;
	case SpvExecutionModelTaskNV:
		return VK_SHADER_STAGE_TASK_BIT_NV;
	case SpvExecutionModelMeshNV:
		return VK_SHADER_STAGE_MESH_BIT_NV;

	default:
		assert(!"Unsupported execution model");
		return VkShaderStageFlagBits(0);
	}
}

static void parseShader(Shader& shader, const uint32_t* code, uint32_t codeSize)
{
	assert(code[0] == SpvMagicNumber);

	uint32_t idBound = code[3];

	std::vector<Id> ids(idBound);

	const uint32_t* insn = code + 5;

	while (insn != code + codeSize)
	{
		uint16_t opcode = uint16_t(insn[0]);
		uint16_t wordCount = uint16_t(insn[0] >> 16);

		switch (opcode)
		{
		case SpvOpEntryPoint:
		{
			assert(wordCount >= 2);
			shader.stage = getShaderStage(SpvExecutionModel(insn[1]));
		} break;
		case SpvOpDecorate:
		{
			assert(wordCount >= 3);

			uint32_t id = insn[1];
			assert(id < idBound);

			switch (insn[2])
			{
			case SpvDecorationDescriptorSet:
				assert(wordCount == 4);
				ids[id].set = insn[3];
				break;
			case SpvDecorationBinding:
				assert(wordCount == 4);
				ids[id].binding = insn[3];
				break;
			}
		} break;
		case SpvOpVariable:
		{
			assert(wordCount >= 4);

			uint32_t id = insn[2];
			assert(id < idBound);

			assert(ids[id].kind == Id::Unknown);
			ids[id].kind = Id::Variable;
			ids[id].type = insn[1];
			ids[id].storageClass = insn[3];
		} break;
		}

		assert(insn + wordCount <= code + codeSize);
		insn += wordCount;
	}

	for (auto& id : ids)
	{
		if (id.kind == Id::Variable && id.storageClass == SpvStorageClassUniform)
		{
			// TODO: we currently assume that id.type refers to a pointer to a storage buffer
			assert(id.set == 0);
			assert(id.binding < 32);

			shader.storageBufferMask |= 1 << id.binding;
		}

		if (id.kind == Id::Variable && id.storageClass == SpvStorageClassPushConstant)
		{
			shader.usesPushConstants = true;
		}
	}
}

static VkDescriptorSetLayout createSetLayout(VkDevice device, Shaders shaders)
{
	std::vector<VkDescriptorSetLayoutBinding> setBindings;

	uint32_t storageBufferMask = 0;
	for (const Shader* shader : shaders)
		storageBufferMask |= shader->storageBufferMask;

	for (uint32_t i = 0; i < 32; ++i)
		if (storageBufferMask & (1 << i))
		{
			VkDescriptorSetLayoutBinding binding = {};
			binding.binding = i;
			binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			binding.descriptorCount = 1;

			binding.stageFlags = 0;
			for (const Shader* shader : shaders)
				if (shader->storageBufferMask & (1 << i))
					binding.stageFlags |= shader->stage;

			setBindings.push_back(binding);
		}

	VkDescriptorSetLayoutCreateInfo setCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	setCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
	setCreateInfo.bindingCount = uint32_t(setBindings.size());
	setCreateInfo.pBindings = setBindings.data();

	VkDescriptorSetLayout setLayout = 0;
	VK_CHECK(vkCreateDescriptorSetLayout(device, &setCreateInfo, 0, &setLayout));

	return setLayout;
}

static VkPipelineLayout createPipelineLayout(VkDevice device, Shaders shaders, VkShaderStageFlags pushConstantStages, size_t pushConstantSize)
{
	VkDescriptorSetLayout setLayout = createSetLayout(device, shaders);

	VkPipelineLayoutCreateInfo createInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	createInfo.setLayoutCount = 1;
	createInfo.pSetLayouts = &setLayout;

	VkPushConstantRange pushConstantRange = {};

	if (pushConstantSize)
	{
		pushConstantRange.stageFlags = pushConstantStages;
		pushConstantRange.size = uint32_t(pushConstantSize);

		createInfo.pushConstantRangeCount = 1;
		createInfo.pPushConstantRanges = &pushConstantRange;
	}

	VkPipelineLayout layout = 0;
	VK_CHECK(vkCreatePipelineLayout(device, &createInfo, 0, &layout));

	// TODO: is this safe?
	vkDestroyDescriptorSetLayout(device, setLayout, 0);

	return layout;
}

static VkDescriptorUpdateTemplate createUpdateTemplate(VkDevice device, VkPipelineBindPoint bindPoint, VkPipelineLayout layout, Shaders shaders)
{
	std::vector<VkDescriptorUpdateTemplateEntry> entries;

	uint32_t storageBufferMask = 0;
	for (const Shader* shader : shaders)
		storageBufferMask |= shader->storageBufferMask;

	for (uint32_t i = 0; i < 32; ++i)
		if (storageBufferMask & (1 << i))
		{
			VkDescriptorUpdateTemplateEntry entry = {};
			entry.dstBinding = i;
			entry.dstArrayElement = 0;
			entry.descriptorCount = 1;
			entry.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			entry.offset = sizeof(DescriptorInfo) * i;
			entry.stride = sizeof(DescriptorInfo);

			entries.push_back(entry);
		}

	VkDescriptorUpdateTemplateCreateInfo createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO };

	createInfo.descriptorUpdateEntryCount = uint32_t(entries.size());
	createInfo.pDescriptorUpdateEntries = entries.data();

	createInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
	createInfo.pipelineBindPoint = bindPoint;
	createInfo.pipelineLayout = layout;

	VkDescriptorUpdateTemplate updateTemplate = 0;
	VK_CHECK(vkCreateDescriptorUpdateTemplate(device, &createInfo, 0, &updateTemplate));

	return updateTemplate;
}

bool loadShader(Shader& shader, VkDevice device, const char* path)
{
	FILE* file = fopen(path, "rb");
	if (!file)
		return false;

	fseek(file, 0, SEEK_END);
	long length = ftell(file);
	assert(length >= 0);
	fseek(file, 0, SEEK_SET);

	char* buffer = new char[length];
	assert(buffer);

	size_t rc = fread(buffer, 1, length, file);
	assert(rc == size_t(length));
	fclose(file);

	VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	createInfo.codeSize = length; // note: this needs to be a number of bytes!
	createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer);

	VkShaderModule shaderModule = 0;
	VK_CHECK(vkCreateShaderModule(device, &createInfo, 0, &shaderModule));

	assert(length % 4 == 0);
	parseShader(shader, reinterpret_cast<const uint32_t*>(buffer), length / 4);

	delete[] buffer;

	shader.module = shaderModule;

	return true;
}

VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, VkRenderPass renderPass, Shaders shaders, VkPipelineLayout layout)
{
	VkGraphicsPipelineCreateInfo createInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };

	std::vector<VkPipelineShaderStageCreateInfo> stages;
	for (const Shader* shader : shaders)
	{
		VkPipelineShaderStageCreateInfo stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
		stage.stage = shader->stage;
		stage.module = shader->module;
		stage.pName = "main";

		stages.push_back(stage);
	}

	createInfo.stageCount = uint32_t(stages.size());
	createInfo.pStages = stages.data();

	VkPipelineVertexInputStateCreateInfo vertexInput = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
	createInfo.pVertexInputState = &vertexInput;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	createInfo.pInputAssemblyState = &inputAssembly;

	VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;
	createInfo.pViewportState = &viewportState;

	VkPipelineRasterizationStateCreateInfo rasterizationState = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
	rasterizationState.lineWidth = 1.f;
	rasterizationState.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	createInfo.pRasterizationState = &rasterizationState;

	VkPipelineMultisampleStateCreateInfo multisampleState = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
	multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	createInfo.pMultisampleState = &multisampleState;

	VkPipelineDepthStencilStateCreateInfo depthStencilState = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
	depthStencilState.depthTestEnable = true;
	depthStencilState.depthWriteEnable = true;
	depthStencilState.depthCompareOp = VK_COMPARE_OP_GREATER;
	createInfo.pDepthStencilState = &depthStencilState;

	VkPipelineColorBlendAttachmentState colorAttachmentState = {};
	colorAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo colorBlendState = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
	colorBlendState.attachmentCount = 1;
	colorBlendState.pAttachments = &colorAttachmentState;
	createInfo.pColorBlendState = &colorBlendState;

	VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

	VkPipelineDynamicStateCreateInfo dynamicState = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
	dynamicState.dynamicStateCount = sizeof(dynamicStates) / sizeof(dynamicStates[0]);
	dynamicState.pDynamicStates = dynamicStates;
	createInfo.pDynamicState = &dynamicState;

	createInfo.layout = layout;
	createInfo.renderPass = renderPass;

	VkPipeline pipeline = 0;
	VK_CHECK(vkCreateGraphicsPipelines(device, pipelineCache, 1, &createInfo, 0, &pipeline));

	return pipeline;
}

Program createProgram(VkDevice device, VkPipelineBindPoint bindPoint, Shaders shaders, size_t pushConstantSize)
{
	VkShaderStageFlags pushConstantStages = 0;
	for (const Shader* shader : shaders)
		if (shader->usesPushConstants)
			pushConstantStages |= shader->stage;

	Program program = {};

	program.layout = createPipelineLayout(device, shaders, pushConstantStages, pushConstantSize);
	assert(program.layout);

	program.updateTemplate = createUpdateTemplate(device, bindPoint, program.layout,shaders);
	assert(program.updateTemplate);

	program.pushConstantStages = pushConstantStages;

	return program;
}

void destroyProgram(VkDevice device, const Program& program)
{
	vkDestroyDescriptorUpdateTemplate(device, program.updateTemplate, 0);
	vkDestroyPipelineLayout(device, program.layout, 0);
}