#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#extension GL_GOOGLE_include_directive: require

#include "mesh.h"

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform block
{
	vec4 frustum[6];
};

layout(binding = 0) readonly buffer Draws
{
	MeshDraw draws[];
};

layout(binding = 1) writeonly buffer DrawCommands
{
	MeshDrawCommand drawCommands[];
};

void main()
{
	uint ti = gl_LocalInvocationID.x;
	uint gi = gl_WorkGroupID.x;
	uint di = gi * 32 + ti;

	vec3 center = draws[di].center * draws[di].scale + draws[di].position;
	float radius = draws[di].radius * draws[di].scale;

	bool visible = true;
	for (int i = 0; i < 6; ++i)
		visible = visible && dot(frustum[i], vec4(center, 1)) > -radius;

	drawCommands[di].indexCount = draws[di].indexCount;
	drawCommands[di].instanceCount = visible ? 1 : 0;
	drawCommands[di].firstIndex = draws[di].indexOffset;
	drawCommands[di].vertexOffset = draws[di].vertexOffset;
	drawCommands[di].firstInstance = 0;
	drawCommands[di].taskCount = visible ? (draws[di].meshletCount + 31) / 32 : 0;
	drawCommands[di].firstTask = 0;
}