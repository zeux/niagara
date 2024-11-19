#include "common.h"
#include "textures.h"

#include "resources.h"

#include <stdio.h>

#include <memory>

struct DDS_PIXELFORMAT
{
	unsigned int dwSize;
	unsigned int dwFlags;
	unsigned int dwFourCC;
	unsigned int dwRGBBitCount;
	unsigned int dwRBitMask;
	unsigned int dwGBitMask;
	unsigned int dwBBitMask;
	unsigned int dwABitMask;
};

struct DDS_HEADER
{
	unsigned int dwSize;
	unsigned int dwFlags;
	unsigned int dwHeight;
	unsigned int dwWidth;
	unsigned int dwPitchOrLinearSize;
	unsigned int dwDepth;
	unsigned int dwMipMapCount;
	unsigned int dwReserved1[11];
	DDS_PIXELFORMAT ddspf;
	unsigned int dwCaps;
	unsigned int dwCaps2;
	unsigned int dwCaps3;
	unsigned int dwCaps4;
	unsigned int dwReserved2;
};

struct DDS_HEADER_DXT10
{
	unsigned int dxgiFormat;
	unsigned int resourceDimension;
	unsigned int miscFlag;
	unsigned int arraySize;
	unsigned int miscFlags2;
};

const unsigned int DDSCAPS2_CUBEMAP = 0x200;
const unsigned int DDSCAPS2_VOLUME = 0x200000;

const unsigned int DDS_DIMENSION_TEXTURE2D = 3;

enum DXGI_FORMAT
{
	DXGI_FORMAT_BC1_UNORM = 71,
	DXGI_FORMAT_BC1_UNORM_SRGB = 72,
	DXGI_FORMAT_BC2_UNORM = 74,
	DXGI_FORMAT_BC2_UNORM_SRGB = 75,
	DXGI_FORMAT_BC3_UNORM = 77,
	DXGI_FORMAT_BC3_UNORM_SRGB = 78,
	DXGI_FORMAT_BC4_UNORM = 80,
	DXGI_FORMAT_BC4_SNORM = 81,
	DXGI_FORMAT_BC5_UNORM = 83,
	DXGI_FORMAT_BC5_SNORM = 84,
	DXGI_FORMAT_BC6H_UF16 = 95,
	DXGI_FORMAT_BC6H_SF16 = 96,
	DXGI_FORMAT_BC7_UNORM = 98,
	DXGI_FORMAT_BC7_UNORM_SRGB = 99,
};

static unsigned int fourCC(const char (&str)[5])
{
	return (unsigned(str[0]) << 0) | (unsigned(str[1]) << 8) | (unsigned(str[2]) << 16) | (unsigned(str[3]) << 24);
}

static VkFormat getFormat(const DDS_HEADER& header, const DDS_HEADER_DXT10& header10)
{
	if (header.ddspf.dwFourCC == fourCC("DXT1"))
		return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
	if (header.ddspf.dwFourCC == fourCC("DXT3"))
		return VK_FORMAT_BC2_UNORM_BLOCK;
	if (header.ddspf.dwFourCC == fourCC("DXT5"))
		return VK_FORMAT_BC3_UNORM_BLOCK;

	if (header.ddspf.dwFourCC == fourCC("DX10"))
	{
		switch (header10.dxgiFormat)
		{
		case DXGI_FORMAT_BC1_UNORM:
		case DXGI_FORMAT_BC1_UNORM_SRGB:
			return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
		case DXGI_FORMAT_BC2_UNORM:
		case DXGI_FORMAT_BC2_UNORM_SRGB:
			return VK_FORMAT_BC2_UNORM_BLOCK;
		case DXGI_FORMAT_BC3_UNORM:
		case DXGI_FORMAT_BC3_UNORM_SRGB:
			return VK_FORMAT_BC3_UNORM_BLOCK;
		case DXGI_FORMAT_BC4_UNORM:
			return VK_FORMAT_BC4_UNORM_BLOCK;
		case DXGI_FORMAT_BC4_SNORM:
			return VK_FORMAT_BC4_SNORM_BLOCK;
		case DXGI_FORMAT_BC5_UNORM:
			return VK_FORMAT_BC5_UNORM_BLOCK;
		case DXGI_FORMAT_BC5_SNORM:
			return VK_FORMAT_BC5_SNORM_BLOCK;
		case DXGI_FORMAT_BC6H_UF16:
			return VK_FORMAT_BC6H_UFLOAT_BLOCK;
		case DXGI_FORMAT_BC6H_SF16:
			return VK_FORMAT_BC6H_SFLOAT_BLOCK;
		case DXGI_FORMAT_BC7_UNORM:
		case DXGI_FORMAT_BC7_UNORM_SRGB:
			return VK_FORMAT_BC7_UNORM_BLOCK;
		}
	}

	return VK_FORMAT_UNDEFINED;
}

static size_t getImageSizeBC(unsigned int width, unsigned int height, unsigned int levels, unsigned int blockSize)
{
	size_t result = 0;

	for (unsigned int i = 0; i < levels; ++i)
	{
		result += ((width + 3) / 4) * ((height + 3) / 4) * blockSize;

		width = width > 1 ? width / 2 : 1;
		height = height > 1 ? height / 2 : 1;
	}

	return result;
}

bool loadImage(Image& image, VkDevice device, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const VkPhysicalDeviceMemoryProperties& memoryProperties, const Buffer& scratch, const char* path)
{
	FILE* file = fopen(path, "rb");
	if (!file)
		return false;

	std::unique_ptr<FILE, int (*)(FILE*)> filePtr(file, fclose);

	unsigned int magic = 0;
	if (fread(&magic, sizeof(magic), 1, file) != 1 || magic != fourCC("DDS "))
		return false;

	DDS_HEADER header = {};
	if (fread(&header, sizeof(header), 1, file) != 1)
		return false;

	DDS_HEADER_DXT10 header10 = {};
	if (header.ddspf.dwFourCC == fourCC("DX10") && fread(&header10, sizeof(header10), 1, file) != 1)
		return false;

	if (header.dwSize != sizeof(header) || header.ddspf.dwSize != sizeof(header.ddspf))
		return false;

	if (header.dwCaps2 & (DDSCAPS2_CUBEMAP | DDSCAPS2_VOLUME))
		return false;

	if (header.ddspf.dwFourCC == fourCC("DX10") && header10.resourceDimension != DDS_DIMENSION_TEXTURE2D)
		return false;

	VkFormat format = getFormat(header, header10);
	if (format == VK_FORMAT_UNDEFINED)
		return false;

	unsigned int blockSize =
	    (format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || format == VK_FORMAT_BC4_SNORM_BLOCK || format == VK_FORMAT_BC4_UNORM_BLOCK) ? 8 : 16;
	size_t imageSize = getImageSizeBC(header.dwWidth, header.dwHeight, header.dwMipMapCount, blockSize);

	if (scratch.size < imageSize)
		return false;

	VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	createImage(image, device, memoryProperties, header.dwWidth, header.dwHeight, header.dwMipMapCount, format, usage);

	size_t readSize = fread(scratch.data, 1, imageSize, file);
	if (readSize != imageSize)
		return false;

	if (fgetc(file) != -1)
		return false;

	filePtr.reset();
	file = nullptr;

	VK_CHECK(vkResetCommandPool(device, commandPool, 0));

	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	VkImageMemoryBarrier2 preBarrier = imageBarrier(image.image,
	    0, 0, VK_IMAGE_LAYOUT_UNDEFINED,
	    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &preBarrier);

	size_t bufferOffset = 0;
	unsigned int mipWidth = header.dwWidth, mipHeight = header.dwHeight;

	for (unsigned int i = 0; i < header.dwMipMapCount; ++i)
	{
		VkBufferImageCopy region = {
			bufferOffset,
			0,
			0,
			{ VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 },
			{ 0, 0, 0 },
			{ mipWidth, mipHeight, 1 },
		};
		vkCmdCopyBufferToImage(commandBuffer, scratch.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		bufferOffset += ((mipWidth + 3) / 4) * ((mipHeight + 3) / 4) * blockSize;

		mipWidth = mipWidth > 1 ? mipWidth / 2 : 1;
		mipHeight = mipHeight > 1 ? mipHeight / 2 : 1;
	}

	assert(bufferOffset == imageSize);

	VkImageMemoryBarrier2 postBarrier = imageBarrier(image.image,
	    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	    VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	pipelineBarrier(commandBuffer, 0, 0, nullptr, 1, &postBarrier);

	VK_CHECK(vkEndCommandBuffer(commandBuffer));

	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

	VK_CHECK(vkDeviceWaitIdle(device));

	return true;
}
