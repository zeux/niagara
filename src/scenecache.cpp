#include "common.h"
#include "scene.h"
#include "config.h"

#include "fileutils.h"

#include "meshoptimizer.h"

#include <stdio.h>
#include <string.h>

const uint32_t kSceneCacheMagic = 0x434E4353; // 'SCNC'
const uint32_t kSceneCacheVersion = 2;

struct SceneHeader
{
	uint32_t magic;
	uint32_t version;

	uint32_t meshletMaxVertices;
	uint32_t meshletMaxTriangles;

	bool clrtMode;
	bool compressed;

	uint32_t compressedVertexBytes;
	uint32_t compressedIndexBytes;
	uint32_t compressedMeshletVtx0Bytes;

	uint32_t vertexCount;
	uint32_t indexCount;
	uint32_t meshletCount;
	uint32_t meshletdataCount;
	uint32_t meshletvtx0Count;
	uint32_t meshCount;

	uint32_t materialCount;
	uint32_t drawCount;
	uint32_t texturePathCount;

	Camera camera;
	vec3 sunDirection;
};

static size_t writeVertexCompressed(const void* vertices, size_t stride, size_t count, FILE* file, int level = 2)
{
	size_t bound = meshopt_encodeVertexBufferBound(count, stride);
	std::vector<unsigned char> buf(bound);
	buf.resize(meshopt_encodeVertexBufferLevel(buf.data(), buf.size(), vertices, count, stride, level));

	fwrite(buf.data(), 1, buf.size(), file);
	return buf.size();
}

static size_t writeIndexCompressed(const uint32_t* indices, size_t count, FILE* file)
{
	size_t bound = meshopt_encodeIndexBufferBound(count, ~0u); // TODO: vertex_count could be optional somehow
	std::vector<unsigned char> buf(bound);
	buf.resize(meshopt_encodeIndexBuffer(buf.data(), buf.size(), indices, count));

	fwrite(buf.data(), 1, buf.size(), file);
	return buf.size();
}

bool saveSceneCache(const char* path, const Geometry& geometry, const std::vector<Material>& materials, const std::vector<MeshDraw>& draws, const std::vector<std::string>& texturePaths, const Camera& camera, const vec3& sunDirection, bool clrtMode, bool compressed, bool verbose)
{
	FILE* file = fopen(path, "wb");
	if (!file)
		return false;

	SceneHeader header;
	memset(&header, 0, sizeof(header));

	header.magic = kSceneCacheMagic;
	header.version = kSceneCacheVersion;

	header.meshletMaxVertices = MESH_MAXVTX;
	header.meshletMaxTriangles = MESH_MAXTRI;
	header.clrtMode = clrtMode;
	header.compressed = compressed;

	header.vertexCount = geometry.vertices.size();
	header.indexCount = geometry.indices.size();
	header.meshletCount = geometry.meshlets.size();
	header.meshletdataCount = geometry.meshletdata.size();
	header.meshletvtx0Count = geometry.meshletvtx0.size();
	header.meshCount = geometry.meshes.size();
	header.materialCount = materials.size();
	header.drawCount = draws.size();
	header.texturePathCount = texturePaths.size();

	header.camera = camera;
	header.sunDirection = sunDirection;

	fwrite(&header, sizeof(header), 1, file);

	if (compressed)
		header.compressedVertexBytes = writeVertexCompressed(geometry.vertices.data(), sizeof(Vertex), geometry.vertices.size(), file);
	else
		fwrite(geometry.vertices.data(), sizeof(Vertex), geometry.vertices.size(), file);

	if (compressed)
		header.compressedIndexBytes = writeIndexCompressed(geometry.indices.data(), geometry.indices.size(), file);
	else
		fwrite(geometry.indices.data(), sizeof(uint32_t), geometry.indices.size(), file);

	fwrite(geometry.meshlets.data(), sizeof(Meshlet), geometry.meshlets.size(), file);
	fwrite(geometry.meshletdata.data(), sizeof(uint32_t), geometry.meshletdata.size(), file);

	if (compressed)
		header.compressedMeshletVtx0Bytes = writeVertexCompressed(geometry.meshletvtx0.data(), sizeof(uint16_t) * 4, geometry.meshletvtx0.size() / 4, file);
	else
		fwrite(geometry.meshletvtx0.data(), sizeof(uint16_t), geometry.meshletvtx0.size(), file);

	fwrite(geometry.meshes.data(), sizeof(Mesh), geometry.meshes.size(), file);
	fwrite(materials.data(), sizeof(Material), materials.size(), file);
	fwrite(draws.data(), sizeof(MeshDraw), draws.size(), file);

	for (const std::string& path : texturePaths)
	{
		char buf[128] = {};
		strncpy(buf, path.c_str(), sizeof(buf) - 1);
		fwrite(buf, sizeof(buf), 1, file);
	}

	// fixup final header
	fseek(file, 0, SEEK_SET);
	fwrite(&header, sizeof(header), 1, file);

	fclose(file);

	if (verbose)
	{
		printf("Scene cache saved to %s\n", path);

		if (compressed)
			printf("Vertex data: %.2f MB (%.2f MB compressed)\n", double(geometry.vertices.size() * sizeof(Vertex)) / 1e6, double(header.compressedVertexBytes) / 1e6);
		else
			printf("Vertex data: %.2f MB\n", double(geometry.vertices.size() * sizeof(Vertex)) / 1e6);

		if (compressed)
			printf("Index data: %.2f MB (%.2f MB compressed)\n", double(geometry.indices.size() * sizeof(uint32_t)) / 1e6, double(header.compressedIndexBytes) / 1e6);
		else
			printf("Index data: %.2f MB\n", double(geometry.indices.size() * sizeof(uint32_t)) / 1e6);

		printf("Meshlet data: %.2f MB\n", double(geometry.meshlets.size() * sizeof(Meshlet) + geometry.meshletdata.size() * sizeof(uint32_t)) / 1e6);

		if (compressed)
			printf("Meshlet RT data: %.2f MB (%.2f MB compressed)\n", double(geometry.meshletvtx0.size() * sizeof(uint16_t)) / 1e6, double(header.compressedMeshletVtx0Bytes) / 1e6);
		else
			printf("Meshlet RT data: %.2f MB\n", double(geometry.meshletvtx0.size() * sizeof(uint16_t)) / 1e6);
	}

	return true;
}

static void read(void* data, size_t size, size_t count, void* fileMemory, size_t& fileOffset)
{
	memcpy(data, (char*)fileMemory + fileOffset, size * count);
	fileOffset += size * count;
}

static void readVertexCompressed(void* data, size_t size, size_t count, size_t compressedBytes, void* fileMemory, size_t& fileOffset)
{
	meshopt_decodeVertexBuffer(data, count, size, (unsigned char*)fileMemory + fileOffset, compressedBytes);
	fileOffset += compressedBytes;
}

static void readIndexCompressed(unsigned int* data, size_t count, size_t compressedBytes, void* fileMemory, size_t& fileOffset)
{
	meshopt_decodeIndexBuffer(data, count, (unsigned char*)fileMemory + fileOffset, compressedBytes);
	fileOffset += compressedBytes;
}

bool loadSceneCache(const char* path, Geometry& geometry, std::vector<Material>& materials, std::vector<MeshDraw>& draws, std::vector<std::string>& texturePaths, Camera& camera, vec3& sunDirection, bool clrtMode)
{
	size_t fileSize;
	void* file = mmapFile(path, &fileSize);
	if (!file || fileSize < sizeof(SceneHeader))
		return false;

	SceneHeader header = {};
	memcpy(&header, file, sizeof(header));

	if (header.magic != kSceneCacheMagic || header.version != kSceneCacheVersion ||
	    header.meshletMaxVertices != MESH_MAXVTX || header.meshletMaxTriangles != MESH_MAXTRI ||
	    header.clrtMode != clrtMode)
	{
		unmapFile(file, fileSize);
		return false;
	}

	size_t fileOffset = sizeof(header);

	geometry.vertices.resize(header.vertexCount);
	geometry.indices.resize(header.indexCount);
	geometry.meshlets.resize(header.meshletCount);
	geometry.meshletdata.resize(header.meshletdataCount);
	geometry.meshletvtx0.resize(header.meshletvtx0Count);
	geometry.meshes.resize(header.meshCount);
	materials.resize(header.materialCount);
	draws.resize(header.drawCount);
	texturePaths.resize(header.texturePathCount);

	if (header.compressed)
		readVertexCompressed(geometry.vertices.data(), sizeof(Vertex), geometry.vertices.size(), header.compressedVertexBytes, file, fileOffset);
	else
		read(geometry.vertices.data(), sizeof(Vertex), geometry.vertices.size(), file, fileOffset);

	if (header.compressed)
		readIndexCompressed(geometry.indices.data(), geometry.indices.size(), header.compressedIndexBytes, file, fileOffset);
	else
		read(geometry.indices.data(), sizeof(uint32_t), geometry.indices.size(), file, fileOffset);

	read(geometry.meshlets.data(), sizeof(Meshlet), geometry.meshlets.size(), file, fileOffset);
	read(geometry.meshletdata.data(), sizeof(uint32_t), geometry.meshletdata.size(), file, fileOffset);

	if (header.compressed)
		readVertexCompressed(geometry.meshletvtx0.data(), sizeof(uint16_t) * 4, geometry.meshletvtx0.size() / 4, header.compressedMeshletVtx0Bytes, file, fileOffset);
	else
		read(geometry.meshletvtx0.data(), sizeof(uint16_t), geometry.meshletvtx0.size(), file, fileOffset);

	read(geometry.meshes.data(), sizeof(Mesh), geometry.meshes.size(), file, fileOffset);
	read(materials.data(), sizeof(Material), materials.size(), file, fileOffset);
	read(draws.data(), sizeof(MeshDraw), draws.size(), file, fileOffset);

	for (std::string& path : texturePaths)
	{
		char buf[128] = {};
		read(buf, sizeof(buf), 1, file, fileOffset);
		buf[sizeof(buf) - 1] = 0;

		path = buf;
	}

	unmapFile(file, fileSize);

	camera = header.camera;
	sunDirection = header.sunDirection;

	return true;
}
