#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#extension GL_GOOGLE_include_directive: require

#include "mesh.h"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform block
{
	DrawCullData cullData;
};

layout(binding = 0) readonly buffer Draws
{
	MeshDraw draws[];
};

layout(binding = 1) readonly buffer Meshes
{
	Mesh meshes[];
};

layout(binding = 2) writeonly buffer DrawCommands
{
	MeshDrawCommand drawCommands[];
};

layout(binding = 3) buffer DrawCommandCount
{
	uint drawCommandCount;
};

layout(binding = 4) buffer DrawVisibility
{
	uint drawVisibility[];
};

void main()
{
	uint di = gl_GlobalInvocationID.x;

	if (di >= cullData.drawCount)
		return;

	uint meshIndex = draws[di].meshIndex;
	Mesh mesh = meshes[meshIndex];

	vec3 center = mesh.center * draws[di].scale + draws[di].position;
	float radius = mesh.radius * draws[di].scale;

	bool visible = true;
	for (int i = 0; i < 6; ++i)
		visible = visible && dot(cullData.frustum[i], vec4(center, 1)) > -radius;

	visible = cullData.cullingEnabled == 1 ? visible : true;

	if (visible && drawVisibility[di] == 0)
	{
		uint dci = atomicAdd(drawCommandCount, 1);

		float lodDistance = log2(max(1, (distance(center, vec3(0)) - radius)));
		uint lodIndex = clamp(int(lodDistance), 0, int(mesh.lodCount) - 1);

		lodIndex = cullData.lodEnabled == 1 ? lodIndex : 0;

		MeshLod lod = meshes[meshIndex].lods[lodIndex];

		drawCommands[dci].drawId = di;
		drawCommands[dci].indexCount = lod.indexCount;
		drawCommands[dci].instanceCount = 1;
		drawCommands[dci].firstIndex = lod.indexOffset;
		drawCommands[dci].vertexOffset = mesh.vertexOffset;
		drawCommands[dci].firstInstance = 0;
		drawCommands[dci].taskCount = (lod.meshletCount + 31) / 32;
		drawCommands[dci].firstTask = lod.meshletOffset / 32;
	}

	drawVisibility[di] = visible ? 1 : 0;
}