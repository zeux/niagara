#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_NV_mesh_shader: require

#extension GL_GOOGLE_include_directive: require

#extension GL_ARB_shader_draw_parameters: require

#include "mesh.h"

#define DEBUG 0

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

layout(push_constant) uniform block
{
	Globals globals;
};

layout(binding = 0) readonly buffer DrawCommands
{
	MeshDrawCommand drawCommands[];
};

layout(binding = 1) readonly buffer Draws
{
	MeshDraw draws[];
};

layout(binding = 2) readonly buffer Meshlets
{
	Meshlet meshlets[];
};

layout(binding = 3) readonly buffer MeshletData
{
	uint meshletData[];
};

layout(binding = 4) readonly buffer Vertices
{
	Vertex vertices[];
};

in taskNV block
{
	uint meshletIndices[32];
};

layout(location = 0) out vec4 color[];

uint hash(uint a)
{
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}

void main()
{
	uint ti = gl_LocalInvocationID.x;
	uint mi = meshletIndices[gl_WorkGroupID.x];

	MeshDraw meshDraw = draws[drawCommands[gl_DrawIDARB].drawId];

	uint vertexCount = uint(meshlets[mi].vertexCount);
	uint triangleCount = uint(meshlets[mi].triangleCount);
	uint indexCount = triangleCount * 3;

	uint dataOffset = meshlets[mi].dataOffset;
	uint vertexOffset = dataOffset;
	uint indexOffset = dataOffset + vertexCount;

#if DEBUG
	uint mhash = hash(mi);
	vec3 mcolor = vec3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
#endif

	// TODO: if we have meshlets with 62 or 63 vertices then we pay a small penalty for branch divergence here - we can instead redundantly xform the last vertex
	for (uint i = ti; i < vertexCount; i += 32)
	{
		uint vi = meshletData[vertexOffset + i] + meshDraw.vertexOffset;

		vec3 position = vec3(vertices[vi].vx, vertices[vi].vy, vertices[vi].vz);
		vec3 normal = vec3(int(vertices[vi].nx), int(vertices[vi].ny), int(vertices[vi].nz)) / 127.0 - 1.0;
		vec2 texcoord = vec2(vertices[vi].tu, vertices[vi].tv);

		gl_MeshVerticesNV[i].gl_Position = globals.projection * vec4(rotateQuat(position, meshDraw.orientation) * meshDraw.scale + meshDraw.position, 1);
		color[i] = vec4(normal * 0.5 + vec3(0.5), 1.0);

	#if DEBUG
		color[i] = vec4(mcolor, 1.0);
	#endif
	}

	uint indexGroupCount = (indexCount + 3) / 4;

	for (uint i = ti; i < indexGroupCount; i += 32)
	{
		writePackedPrimitiveIndices4x8NV(i * 4, meshletData[indexOffset + i]);
	}

	if (ti == 0)
		gl_PrimitiveCountNV = uint(meshlets[mi].triangleCount);
}