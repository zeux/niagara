#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#extension GL_GOOGLE_include_directive: require

#include "mesh.h"
#include "math.h"

layout (constant_id = 0) const bool LATE = false;
layout (constant_id = 1) const bool TASK = false;

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

layout(binding = 2) writeonly buffer TaskCommands
{
	MeshTaskCommand taskCommands[];
};

layout(binding = 3) buffer CommandCount
{
	uint commandCount;
};

layout(binding = 4) buffer DrawVisibility
{
	uint drawVisibility[];
};

layout(binding = 5) uniform sampler2D depthPyramid;

void main()
{
	uint di = gl_GlobalInvocationID.x;

	if (di >= cullData.drawCount)
		return;

	// TODO: when occlusion culling is off, can we make sure everything is processed with LATE=false?
	if (!LATE && drawVisibility[di] == 0)
		return;

	uint meshIndex = draws[di].meshIndex;
	Mesh mesh = meshes[meshIndex];

	vec3 center = rotateQuat(mesh.center, draws[di].orientation) * draws[di].scale + draws[di].position;
	float radius = mesh.radius * draws[di].scale;

	bool visible = true;
	// the left/top/right/bottom plane culling utilizes frustum symmetry to cull against two planes at the same time
	visible = visible && center.z * cullData.frustum[1] - abs(center.x) * cullData.frustum[0] > -radius;
	visible = visible && center.z * cullData.frustum[3] - abs(center.y) * cullData.frustum[2] > -radius;
	// the near/far plane culling uses camera space Z directly
	visible = visible && center.z + radius > cullData.znear && center.z - radius < cullData.zfar;

	visible = visible || cullData.cullingEnabled == 0;

	if (LATE && visible && cullData.occlusionEnabled == 1)
	{
		vec4 aabb;
		if (projectSphere(center, radius, cullData.znear, cullData.P00, cullData.P11, aabb))
		{
			float width = (aabb.z - aabb.x) * cullData.pyramidWidth;
			float height = (aabb.w - aabb.y) * cullData.pyramidHeight;

			float level = floor(log2(max(width, height)));

			// Sampler is set up to do min reduction, so this computes the minimum depth of a 2x2 texel quad
			float depth = textureLod(depthPyramid, (aabb.xy + aabb.zw) * 0.5, level).x;
			float depthSphere = cullData.znear / (center.z - radius);

			visible = visible && depthSphere > depth;
		}
	}

	// when meshlet occlusion culling is enabled, we actually *do* need to append the draw command if vis[]==1 in LATE pass,
	// so that we can correctly render now-visible previously-invisible meshlets. we also pass drawvis[] along to task shader
	// so that it can *reject* clusters that we *did* draw in the first pass
	if (visible && (!LATE || cullData.clusterOcclusionEnabled == 1 || drawVisibility[di] == 0))
	{
		// lod distance i = base * pow(step, i)
		// i = log2(distance / base) / log2(step)
		float lodIndexF = log2(length(center) / cullData.lodBase) / log2(cullData.lodStep);
		uint lodIndex = min(uint(max(lodIndexF + 1, 0)), mesh.lodCount - 1);

		lodIndex = cullData.lodEnabled == 1 ? lodIndex : 0;

		MeshLod lod = meshes[meshIndex].lods[lodIndex];

		if (TASK)
		{
			uint taskGroups = (lod.meshletCount + TASK_WGSIZE - 1) / TASK_WGSIZE;
			uint dci = atomicAdd(commandCount, taskGroups);

			uint lateDrawVisibility = drawVisibility[di];
			uint meshletVisibilityOffset = draws[di].meshletVisibilityOffset;

			// TODO: ideally we would abort if commandCount overflows the output buffer
			for (uint i = 0; i < taskGroups; ++i)
			{
				taskCommands[dci + i].drawId = di;
				taskCommands[dci + i].taskOffset = lod.meshletOffset + i * TASK_WGSIZE;
				taskCommands[dci + i].taskCount = min(TASK_WGSIZE, lod.meshletCount - i * TASK_WGSIZE);
				taskCommands[dci + i].lateDrawVisibility = lateDrawVisibility;
				taskCommands[dci + i].meshletVisibilityOffset = meshletVisibilityOffset + i * TASK_WGSIZE;
			}
		}
		else
		{
			uint dci = atomicAdd(commandCount, 1);

			drawCommands[dci].drawId = di;
			drawCommands[dci].indexCount = lod.indexCount;
			drawCommands[dci].instanceCount = 1;
			drawCommands[dci].firstIndex = lod.indexOffset;
			drawCommands[dci].vertexOffset = mesh.vertexOffset;
			drawCommands[dci].firstInstance = 0;
		}
	}

	if (LATE)
		drawVisibility[di] = visible ? 1 : 0;
}
