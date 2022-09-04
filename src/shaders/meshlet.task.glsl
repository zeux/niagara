#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_mesh_shader: require

#extension GL_GOOGLE_include_directive: require

#extension GL_KHR_shader_subgroup_ballot: require

#extension GL_ARB_shader_draw_parameters: require

#include "mesh.h"

#define CULL 0 // TODO

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

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

taskPayloadSharedEXT uint meshletIndices[32];

bool coneCull(vec3 center, float radius, vec3 cone_axis, float cone_cutoff, vec3 camera_position)
{
	return dot(center - camera_position, cone_axis) >= cone_cutoff * length(center - camera_position) + radius;
}

void main()
{
	uint ti = gl_LocalInvocationID.x;
	uint mgi = gl_WorkGroupID.x;

	uint drawId = drawCommands[gl_DrawIDARB].drawId;
	MeshDraw meshDraw = draws[drawId];

	uint mi = mgi * 32 + ti + drawCommands[gl_DrawIDARB].taskOffset;

#if CULL
	vec3 center = rotateQuat(meshlets[mi].center, meshDraw.orientation) * meshDraw.scale + meshDraw.position;
	float radius = meshlets[mi].radius * meshDraw.scale;
	vec3 cone_axis = rotateQuat(vec3(int(meshlets[mi].cone_axis[0]) / 127.0, int(meshlets[mi].cone_axis[1]) / 127.0, int(meshlets[mi].cone_axis[2]) / 127.0), meshDraw.orientation);
	float cone_cutoff = int(meshlets[mi].cone_cutoff) / 127.0;

	bool accept = !coneCull(center, radius, cone_axis, cone_cutoff, vec3(0, 0, 0));

	uvec4 ballot = subgroupBallot(accept);

	uint index = subgroupBallotExclusiveBitCount(ballot);

	if (accept)
		meshletIndices[index] = mi;

	uint count = subgroupBallotBitCount(ballot);

	EmitMeshTasksEXT(count, 1, 1);
#else
	meshletIndices[ti] = mi;

	EmitMeshTasksEXT(32, 1, 1);
#endif
}
