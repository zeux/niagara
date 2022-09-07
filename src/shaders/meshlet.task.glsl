#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_mesh_shader: require

#extension GL_GOOGLE_include_directive: require

#extension GL_KHR_shader_subgroup_ballot: require

#extension GL_ARB_shader_draw_parameters: require

#include "mesh.h"

#define CULL 1

layout(local_size_x = TASK_WGSIZE, local_size_y = 1, local_size_z = 1) in;

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

taskPayloadSharedEXT MeshTaskPayload payload;

bool coneCull(vec3 center, float radius, vec3 cone_axis, float cone_cutoff, vec3 camera_position)
{
	return dot(center - camera_position, cone_axis) >= cone_cutoff * length(center - camera_position) + radius;
}

#if CULL
shared int sharedCount;
#endif

void main()
{
	uint drawId = drawCommands[gl_DrawIDARB].drawId;
	MeshDraw meshDraw = draws[drawId];

	uint mgi = gl_GlobalInvocationID.x;
	uint mi = mgi + drawCommands[gl_DrawIDARB].taskOffset;

#if CULL
	sharedCount = 0;
	barrier(); // for sharedCount

	vec3 center = rotateQuat(meshlets[mi].center, meshDraw.orientation) * meshDraw.scale + meshDraw.position;
	float radius = meshlets[mi].radius * meshDraw.scale;
	vec3 cone_axis = rotateQuat(vec3(int(meshlets[mi].cone_axis[0]) / 127.0, int(meshlets[mi].cone_axis[1]) / 127.0, int(meshlets[mi].cone_axis[2]) / 127.0), meshDraw.orientation);
	float cone_cutoff = int(meshlets[mi].cone_cutoff) / 127.0;

	bool accept =
		mgi < drawCommands[gl_DrawIDARB].taskCount &&
		!coneCull(center, radius, cone_axis, cone_cutoff, vec3(0, 0, 0));

	if (accept)
	{
		uint index = atomicAdd(sharedCount, 1);

		payload.meshletIndices[index] = mi;
	}

	payload.drawId = drawId;

	barrier(); // for sharedCount
	EmitMeshTasksEXT(sharedCount, 1, 1);
#else
	payload.drawId = drawId;
	payload.meshletIndices[gl_LocalInvocationID.x] = mi;

	uint count = min(TASK_WGSIZE, drawCommands[gl_DrawIDARB].taskCount - gl_WorkGroupID.x * TASK_WGSIZE);

	EmitMeshTasksEXT(count, 1, 1);
#endif
}
