#include "common.h"
#include "shaders.h"
#include "config.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#else
#include <dirent.h>
#endif

#include <string>
#include <vector>

#if defined __APPLE__
#include <spirv_cross/spirv.h>
#elif defined __linux__
#include <spirv/unified1/spirv.h>
#elif VK_HEADER_VERSION >= 135
#include <spirv-headers/spirv.h>
#else
#include <vulkan/spirv.h>
#endif

// https://www.khronos.org/registry/spir-v/specs/1.0/SPIRV.pdf
struct Id
{
	uint32_t opcode;
	uint32_t typeId;
	uint32_t storageClass;
	uint32_t binding;
	uint32_t set;
	uint32_t constant;
};

static VkShaderStageFlagBits getShaderStage(SpvExecutionModel executionModel)
{
	switch (executionModel)
	{
	case SpvExecutionModelVertex:
		return VK_SHADER_STAGE_VERTEX_BIT;
	case SpvExecutionModelGeometry:
		return VK_SHADER_STAGE_GEOMETRY_BIT;
	case SpvExecutionModelFragment:
		return VK_SHADER_STAGE_FRAGMENT_BIT;
	case SpvExecutionModelGLCompute:
		return VK_SHADER_STAGE_COMPUTE_BIT;
	case SpvExecutionModelTaskEXT:
		return VK_SHADER_STAGE_TASK_BIT_EXT;
	case SpvExecutionModelMeshEXT:
		return VK_SHADER_STAGE_MESH_BIT_EXT;

	default:
		assert(!"Unsupported execution model");
		return VkShaderStageFlagBits(0);
	}
}

static VkDescriptorType getDescriptorType(SpvOp op)
{
	switch (op)
	{
	case SpvOpTypeStruct:
		return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	case SpvOpTypeImage:
		return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	case SpvOpTypeSampler:
		return VK_DESCRIPTOR_TYPE_SAMPLER;
	case SpvOpTypeSampledImage:
		return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	case SpvOpTypeAccelerationStructureKHR:
		return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	default:
		assert(!"Unknown resource type");
		return VkDescriptorType(0);
	}
}

static void parseShader(Shader& shader, const uint32_t* code, uint32_t codeSize)
{
	assert(code[0] == SpvMagicNumber);

	uint32_t idBound = code[3];

	std::vector<Id> ids(idBound);

	int localSizeIdX = -1;
	int localSizeIdY = -1;
	int localSizeIdZ = -1;

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
		}
		break;
		case SpvOpExecutionMode:
		{
			assert(wordCount >= 3);
			uint32_t mode = insn[2];

			switch (mode)
			{
			case SpvExecutionModeLocalSize:
				assert(wordCount == 6);
				shader.localSizeX = insn[3];
				shader.localSizeY = insn[4];
				shader.localSizeZ = insn[5];
				break;
			}
		}
		break;
		case SpvOpExecutionModeId:
		{
			assert(wordCount >= 3);
			uint32_t mode = insn[2];

			switch (mode)
			{
			case SpvExecutionModeLocalSizeId:
				assert(wordCount == 6);
				localSizeIdX = int(insn[3]);
				localSizeIdY = int(insn[4]);
				localSizeIdZ = int(insn[5]);
				break;
			}
		}
		break;
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
		}
		break;
		case SpvOpTypeStruct:
		case SpvOpTypeImage:
		case SpvOpTypeSampler:
		case SpvOpTypeSampledImage:
		case SpvOpTypeAccelerationStructureKHR:
		{
			assert(wordCount >= 2);

			uint32_t id = insn[1];
			assert(id < idBound);

			assert(ids[id].opcode == 0);
			ids[id].opcode = opcode;
		}
		break;
		case SpvOpTypePointer:
		{
			assert(wordCount == 4);

			uint32_t id = insn[1];
			assert(id < idBound);

			assert(ids[id].opcode == 0);
			ids[id].opcode = opcode;
			ids[id].typeId = insn[3];
			ids[id].storageClass = insn[2];
		}
		break;
		case SpvOpConstant:
		{
			assert(wordCount >= 4); // we currently only correctly handle 32-bit integer constants

			uint32_t id = insn[2];
			assert(id < idBound);

			assert(ids[id].opcode == 0);
			ids[id].opcode = opcode;
			ids[id].typeId = insn[1];
			ids[id].constant = insn[3]; // note: this is the value, not the id of the constant
		}
		break;
		case SpvOpVariable:
		{
			assert(wordCount >= 4);

			uint32_t id = insn[2];
			assert(id < idBound);

			assert(ids[id].opcode == 0);
			ids[id].opcode = opcode;
			ids[id].typeId = insn[1];
			ids[id].storageClass = insn[3];
		}
		break;
		}

		assert(insn + wordCount <= code + codeSize);
		insn += wordCount;
	}

	for (auto& id : ids)
	{
		// set 0 is reserved for push descriptors
		if (id.opcode == SpvOpVariable && (id.storageClass == SpvStorageClassUniform || id.storageClass == SpvStorageClassUniformConstant || id.storageClass == SpvStorageClassStorageBuffer) && id.set == 0)
		{
			assert(id.binding < 32);
			assert(ids[id.typeId].opcode == SpvOpTypePointer);

			uint32_t typeKind = ids[ids[id.typeId].typeId].opcode;
			VkDescriptorType resourceType = getDescriptorType(SpvOp(typeKind));

			assert((shader.resourceMask & (1 << id.binding)) == 0 || shader.resourceTypes[id.binding] == resourceType);

			shader.resourceTypes[id.binding] = resourceType;
			shader.resourceMask |= 1 << id.binding;
		}

		if (id.opcode == SpvOpVariable && id.storageClass == SpvStorageClassUniformConstant && id.set == 1)
		{
			shader.usesDescriptorArray = true;
		}

		if (id.opcode == SpvOpVariable && id.storageClass == SpvStorageClassPushConstant)
		{
			shader.usesPushConstants = true;
		}
	}

	if (shader.stage == VK_SHADER_STAGE_COMPUTE_BIT)
	{
		if (localSizeIdX >= 0)
		{
			assert(ids[localSizeIdX].opcode == SpvOpConstant);
			shader.localSizeX = ids[localSizeIdX].constant;
		}

		if (localSizeIdY >= 0)
		{
			assert(ids[localSizeIdY].opcode == SpvOpConstant);
			shader.localSizeY = ids[localSizeIdY].constant;
		}

		if (localSizeIdZ >= 0)
		{
			assert(ids[localSizeIdZ].opcode == SpvOpConstant);
			shader.localSizeZ = ids[localSizeIdZ].constant;
		}

		assert(shader.localSizeX && shader.localSizeY && shader.localSizeZ);
	}
}

static uint32_t gatherResources(Shaders shaders, VkDescriptorType (&resourceTypes)[32])
{
	uint32_t resourceMask = 0;

	for (const Shader* shader : shaders)
	{
		for (uint32_t i = 0; i < 32; ++i)
		{
			if (shader->resourceMask & (1 << i))
			{
				if (resourceMask & (1 << i))
				{
					assert(resourceTypes[i] == shader->resourceTypes[i]);
				}
				else
				{
					resourceTypes[i] = shader->resourceTypes[i];
					resourceMask |= 1 << i;
				}
			}
		}
	}

	return resourceMask;
}

static VkDescriptorSetLayout createSetLayout(VkDevice device, Shaders shaders)
{
	std::vector<VkDescriptorSetLayoutBinding> setBindings;

	VkDescriptorType resourceTypes[32] = {};
	uint32_t resourceMask = gatherResources(shaders, resourceTypes);

	for (uint32_t i = 0; i < 32; ++i)
		if (resourceMask & (1 << i))
		{
			VkDescriptorSetLayoutBinding binding = {};
			binding.binding = i;
			binding.descriptorType = resourceTypes[i];
			binding.descriptorCount = 1;

			binding.stageFlags = 0;
			for (const Shader* shader : shaders)
				if (shader->resourceMask & (1 << i))
					binding.stageFlags |= shader->stage;

			setBindings.push_back(binding);
		}

	VkDescriptorSetLayoutCreateInfo setCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	setCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
	setCreateInfo.bindingCount = uint32_t(setBindings.size());
	setCreateInfo.pBindings = setBindings.data();

	VkDescriptorSetLayout setLayout = 0;
	VK_CHECK(vkCreateDescriptorSetLayout(device, &setCreateInfo, 0, &setLayout));

	return setLayout;
}

static VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout setLayout, VkDescriptorSetLayout arrayLayout, VkShaderStageFlags pushConstantStages, size_t pushConstantSize)
{
	VkDescriptorSetLayout layouts[2] = { setLayout, arrayLayout };

	VkPipelineLayoutCreateInfo createInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	createInfo.setLayoutCount = arrayLayout ? 2 : 1;
	createInfo.pSetLayouts = layouts;

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

	return layout;
}

static VkDescriptorUpdateTemplate createUpdateTemplate(VkDevice device, VkPipelineBindPoint bindPoint, VkPipelineLayout layout, Shaders shaders, uint32_t* pushDescriptorCount)
{
	std::vector<VkDescriptorUpdateTemplateEntry> entries;

	VkDescriptorType resourceTypes[32] = {};
	uint32_t resourceMask = gatherResources(shaders, resourceTypes);

	for (uint32_t i = 0; i < 32; ++i)
		if (resourceMask & (1 << i))
		{
			VkDescriptorUpdateTemplateEntry entry = {};
			entry.dstBinding = i;
			entry.dstArrayElement = 0;
			entry.descriptorCount = 1;
			entry.descriptorType = resourceTypes[i];
			entry.offset = sizeof(DescriptorInfo) * i;
			entry.stride = sizeof(DescriptorInfo);

			entries.push_back(entry);
		}

	*pushDescriptorCount = uint32_t(entries.size());

	VkDescriptorUpdateTemplateCreateInfo createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO };

	createInfo.descriptorUpdateEntryCount = uint32_t(entries.size());
	createInfo.pDescriptorUpdateEntries = entries.data();

	createInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS;
	createInfo.pipelineBindPoint = bindPoint;
	createInfo.pipelineLayout = layout;

	VkDescriptorUpdateTemplate updateTemplate = 0;
	VK_CHECK(vkCreateDescriptorUpdateTemplate(device, &createInfo, 0, &updateTemplate));

	return updateTemplate;
}

bool loadShader(Shader& shader, const char* path)
{
	FILE* file = fopen(path, "rb");
	if (!file)
		return false;

	fseek(file, 0, SEEK_END);
	long length = ftell(file);
	assert(length >= 0);
	fseek(file, 0, SEEK_SET);

	std::vector<char> spirv(length);

	size_t rc = fread(spirv.data(), 1, length, file);
	assert(rc == size_t(length));
	fclose(file);

	assert(length % 4 == 0);
	parseShader(shader, reinterpret_cast<const uint32_t*>(spirv.data()), length / 4);

	shader.spirv = spirv;

	return true;
}

bool loadShader(Shader& shader, const char* base, const char* path)
{
	std::string spath = base;
	std::string::size_type pos = spath.find_last_of("/\\");
	if (pos == std::string::npos)
		spath = "";
	else
		spath = spath.substr(0, pos + 1);
	spath += path;

	return loadShader(shader, spath.c_str());
}

bool loadShaders(ShaderSet& shaders, const char* base, const char* path)
{
	std::string spath = base;
	std::string::size_type pos = spath.find_last_of("/\\");
	if (pos == std::string::npos)
		spath = "";
	else
		spath = spath.substr(0, pos + 1);
	spath += path;

#ifdef _WIN32
	_finddata_t finddata;
	intptr_t fh = _findfirst((spath + "/*.spv").c_str(), &finddata);
	if (fh == -1)
		return false;

	do
	{
		const char* ext = strrchr(finddata.name, '.');
		if (!ext)
			continue;

		std::string fpath = spath + '/' + finddata.name;
		Shader shader = {};
		if (!loadShader(shader, fpath.c_str()))
		{
			fprintf(stderr, "Warning: %s is not a valid SPIRV module\n", finddata.name);
			continue;
		}

		shader.name = std::string(finddata.name, ext - finddata.name);
		shaders.shaders.push_back(shader);
	} while (_findnext(fh, &finddata) == 0);

	_findclose(fh);
#else
	DIR* dir = opendir(spath.c_str());
	if (!dir)
		return false;

	while (dirent* de = readdir(dir))
	{
		const char* ext = strstr(de->d_name, ".spv");
		if (!ext || strcmp(ext, ".spv") != 0)
			continue;

		std::string fpath = spath + '/' + de->d_name;
		Shader shader = {};
		if (!loadShader(shader, fpath.c_str()))
		{
			fprintf(stderr, "Warning: %s is not a valid SPIRV module\n", de->d_name);
			continue;
		}

		shader.name = std::string(de->d_name, ext - de->d_name);
		shaders.shaders.push_back(shader);
	}

	closedir(dir);
#endif

	printf("Loaded %d shaders from %s\n", int(shaders.shaders.size()), spath.c_str());
	return true;
}

const Shader& ShaderSet::operator[](const char* name) const
{
	for (const Shader& shader : shaders)
		if (shader.name == name)
			return shader;

	fprintf(stderr, "Error: shader %s could not be loaded\n", name);
	abort();
}

static VkSpecializationInfo fillSpecializationInfo(std::vector<VkSpecializationMapEntry>& entries, const Constants& constants)
{
	for (size_t i = 0; i < constants.size(); ++i)
		entries.push_back({ uint32_t(i), uint32_t(i * 4), 4 });

	VkSpecializationInfo result = {};
	result.mapEntryCount = uint32_t(entries.size());
	result.pMapEntries = entries.data();
	result.dataSize = constants.size() * sizeof(int);
	result.pData = constants.begin();

	return result;
}

VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, const VkPipelineRenderingCreateInfo& renderingInfo, const Program& program, Constants constants)
{
	std::vector<VkSpecializationMapEntry> specializationEntries;
	VkSpecializationInfo specializationInfo = fillSpecializationInfo(specializationEntries, constants);

	VkGraphicsPipelineCreateInfo createInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };

	std::vector<VkPipelineShaderStageCreateInfo> stages(program.shaderCount);
	std::vector<VkShaderModuleCreateInfo> modules(program.shaderCount);
	for (size_t i = 0; i < program.shaderCount; ++i)
	{
		const Shader* shader = program.shaders[i];

		VkShaderModuleCreateInfo& module = modules[i];
		module.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		module.codeSize = shader->spirv.size(); // note: this needs to be a number of bytes!
		module.pCode = reinterpret_cast<const uint32_t*>(shader->spirv.data());

		VkPipelineShaderStageCreateInfo& stage = stages[i];
		stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage = shader->stage;
		stage.pName = "main";
		stage.pSpecializationInfo = &specializationInfo;
		stage.pNext = &module;
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
	rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizationState.depthBiasEnable = true;
	createInfo.pRasterizationState = &rasterizationState;

	VkPipelineMultisampleStateCreateInfo multisampleState = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
	multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	createInfo.pMultisampleState = &multisampleState;

	VkPipelineDepthStencilStateCreateInfo depthStencilState = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
	depthStencilState.depthTestEnable = true;
	depthStencilState.depthWriteEnable = true;
	depthStencilState.depthCompareOp = VK_COMPARE_OP_GREATER;
	createInfo.pDepthStencilState = &depthStencilState;

	VkPipelineColorBlendAttachmentState colorAttachmentStates[8] = {};
	assert(renderingInfo.colorAttachmentCount <= COUNTOF(colorAttachmentStates));
	for (uint32_t i = 0; i < renderingInfo.colorAttachmentCount; ++i)
		colorAttachmentStates[i].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo colorBlendState = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
	colorBlendState.attachmentCount = renderingInfo.colorAttachmentCount;
	colorBlendState.pAttachments = colorAttachmentStates;
	createInfo.pColorBlendState = &colorBlendState;

	VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_DEPTH_BIAS };

	VkPipelineDynamicStateCreateInfo dynamicState = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
	dynamicState.dynamicStateCount = sizeof(dynamicStates) / sizeof(dynamicStates[0]);
	dynamicState.pDynamicStates = dynamicStates;
	createInfo.pDynamicState = &dynamicState;

	createInfo.layout = program.layout;
	createInfo.pNext = &renderingInfo;

	VkPipeline pipeline = 0;
	VK_CHECK(vkCreateGraphicsPipelines(device, pipelineCache, 1, &createInfo, 0, &pipeline));

	return pipeline;
}

VkPipeline createComputePipeline(VkDevice device, VkPipelineCache pipelineCache, const Program& program, Constants constants)
{
	assert(program.shaderCount == 1);
	const Shader& shader = *program.shaders[0];

	assert(shader.stage == VK_SHADER_STAGE_COMPUTE_BIT);

	VkComputePipelineCreateInfo createInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };

	std::vector<VkSpecializationMapEntry> specializationEntries;
	VkSpecializationInfo specializationInfo = fillSpecializationInfo(specializationEntries, constants);

	VkShaderModuleCreateInfo module = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	module.codeSize = shader.spirv.size(); // note: this needs to be a number of bytes!
	module.pCode = reinterpret_cast<const uint32_t*>(shader.spirv.data());

	VkPipelineShaderStageCreateInfo stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	stage.stage = shader.stage;
	stage.pName = "main";
	stage.pSpecializationInfo = &specializationInfo;
	stage.pNext = &module;

	createInfo.stage = stage;
	createInfo.layout = program.layout;

	VkPipeline pipeline = 0;
	VK_CHECK(vkCreateComputePipelines(device, pipelineCache, 1, &createInfo, 0, &pipeline));

	return pipeline;
}

Program createProgram(VkDevice device, VkPipelineBindPoint bindPoint, Shaders shaders, size_t pushConstantSize, VkDescriptorSetLayout arrayLayout)
{
	VkShaderStageFlags pushConstantStages = 0;
	for (const Shader* shader : shaders)
		if (shader->usesPushConstants)
			pushConstantStages |= shader->stage;

	bool usesDescriptorArray = false;
	for (const Shader* shader : shaders)
		usesDescriptorArray |= shader->usesDescriptorArray;

	assert(!usesDescriptorArray || arrayLayout);

	Program program = {};

	program.bindPoint = bindPoint;

	program.setLayout = createSetLayout(device, shaders);
	assert(program.setLayout);

	program.layout = createPipelineLayout(device, program.setLayout, arrayLayout, pushConstantStages, pushConstantSize);
	assert(program.layout);

	program.updateTemplate = createUpdateTemplate(device, bindPoint, program.layout, shaders, &program.pushDescriptorCount);
	assert(program.updateTemplate);

	program.pushConstantStages = pushConstantStages;
	program.pushConstantSize = uint32_t(pushConstantSize);

	const Shader* shader = shaders.size() == 1 ? *shaders.begin() : nullptr;

	if (shader && shader->stage == VK_SHADER_STAGE_COMPUTE_BIT)
	{
		program.localSizeX = shader->localSizeX;
		program.localSizeY = shader->localSizeY;
		program.localSizeZ = shader->localSizeZ;
	}

	memset(program.shaders, 0, sizeof(program.shaders));
	program.shaderCount = 0;

	for (const Shader* shader : shaders)
		program.shaders[program.shaderCount++] = shader;

	return program;
}

void destroyProgram(VkDevice device, const Program& program)
{
	vkDestroyDescriptorUpdateTemplate(device, program.updateTemplate, 0);
	vkDestroyPipelineLayout(device, program.layout, 0);
	vkDestroyDescriptorSetLayout(device, program.setLayout, 0);
}

VkDescriptorSetLayout createDescriptorArrayLayout(VkDevice device)
{
	VkShaderStageFlags stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
	VkDescriptorSetLayoutBinding setBinding = { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, DESCRIPTOR_LIMIT, stageFlags, nullptr };

	VkDescriptorBindingFlags bindingFlags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
	VkDescriptorSetLayoutBindingFlagsCreateInfo setBindingFlags = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
	setBindingFlags.bindingCount = 1;
	setBindingFlags.pBindingFlags = &bindingFlags;

	VkDescriptorSetLayoutCreateInfo setCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	setCreateInfo.pNext = &setBindingFlags;
	setCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
	setCreateInfo.bindingCount = 1;
	setCreateInfo.pBindings = &setBinding;

	VkDescriptorSetLayout setLayout = 0;
	VK_CHECK(vkCreateDescriptorSetLayout(device, &setCreateInfo, 0, &setLayout));

	return setLayout;
}

std::pair<VkDescriptorPool, VkDescriptorSet> createDescriptorArray(VkDevice device, VkDescriptorSetLayout layout, uint32_t descriptorCount)
{
	VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, descriptorCount };
	VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
	poolInfo.maxSets = 1;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;

	VkDescriptorPool pool = nullptr;
	VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, 0, &pool));

	VkDescriptorSetVariableDescriptorCountAllocateInfo setAllocateCountInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO };
	setAllocateCountInfo.descriptorSetCount = 1;
	setAllocateCountInfo.pDescriptorCounts = &descriptorCount;

	VkDescriptorSetAllocateInfo setAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	setAllocateInfo.pNext = &setAllocateCountInfo;
	setAllocateInfo.descriptorPool = pool;
	setAllocateInfo.descriptorSetCount = 1;
	setAllocateInfo.pSetLayouts = &layout;

	VkDescriptorSet set = 0;
	VK_CHECK(vkAllocateDescriptorSets(device, &setAllocateInfo, &set));

	return std::make_pair(pool, set);
}
