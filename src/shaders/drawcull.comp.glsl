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
	CullData cullData;
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

layout(binding = 5) uniform texture2D depthPyramid;
layout(binding = 6) uniform sampler depthSampler;

void main()
{
	uint di = gl_GlobalInvocationID.x;

	if (di >= cullData.drawCount)
		return;

	MeshDraw drawData = draws[di];

	if (drawData.postPass != cullData.postPass)
		return;

	// TODO: when occlusion culling is off, can we make sure everything is processed with LATE=false?
	if (!LATE && drawVisibility[di] == 0)
		return;

	uint meshIndex = drawData.meshIndex;
	Mesh mesh = meshes[meshIndex];

	vec3 center = rotateQuat(mesh.center, drawData.orientation) * drawData.scale + drawData.position;
	center = (cullData.view * vec4(center, 1)).xyz;
	float radius = mesh.radius * drawData.scale;

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

			// Because we only consider 2x2 pixels, we need to make sure we are sampling from a mip that reduces the rectangle to 1x1 texel or smaller.
			// Due to the rectangle being arbitrarily offset, a 1x1 rectangle may cover 2x2 texel area. Using floor() here would require sampling 4 corners
			// of AABB (using bilinear fetch), which is a little slower.
			float level = ceil(log2(max(width, height)));

			// Sampler is set up to do min reduction, so this computes the minimum depth of a 2x2 texel quad
			float depth = textureLod(sampler2D(depthPyramid, depthSampler), (aabb.xy + aabb.zw) * 0.5, level).x;
			float depthSphere = cullData.znear / (center.z - radius);

			visible = visible && depthSphere > depth;
		}
	}

	// when meshlet occlusion culling is enabled, we actually *do* need to append the draw command if vis[]==1 in LATE pass,
	// so that we can correctly render now-visible previously-invisible meshlets. we also pass drawvis[] along to task shader
	// so that it can *reject* clusters that we *did* draw in the first pass
	if (visible && (!LATE || (cullData.clusterOcclusionEnabled == 1 && TASK_CULL == 1) || drawVisibility[di] == 0 || cullData.postPass != 0))
	{
		uint lodIndex = 0;

		if (cullData.lodEnabled == 1)
		{
			float distance = max(length(center) - radius, 0);
			float threshold = distance * cullData.lodTarget / drawData.scale;

			for (uint i = 1; i < mesh.lodCount; ++i)
				if (mesh.lods[i].error < threshold)
					lodIndex = i;
		}

		MeshLod lod = meshes[meshIndex].lods[lodIndex];

		if (TASK)
		{
			uint taskGroups = (lod.meshletCount + TASK_WGSIZE - 1) / TASK_WGSIZE;
			uint dci = atomicAdd(commandCount, taskGroups);

			uint lateDrawVisibility = drawVisibility[di];
			uint meshletVisibilityOffset = drawData.meshletVisibilityOffset;

			// drop draw calls on overflow; this limits us to ~4M visible draws or ~32B visible triangles, whichever is larger
			if (dci + taskGroups <= TASK_WGLIMIT)
			{
				for (uint i = 0; i < taskGroups; ++i)
				{
					taskCommands[dci + i].drawId = di;
					taskCommands[dci + i].taskOffset = lod.meshletOffset + i * TASK_WGSIZE;
					taskCommands[dci + i].taskCount = min(TASK_WGSIZE, lod.meshletCount - i * TASK_WGSIZE);
					taskCommands[dci + i].lateDrawVisibility = lateDrawVisibility;
					taskCommands[dci + i].meshletVisibilityOffset = meshletVisibilityOffset + i * TASK_WGSIZE;
				}
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
