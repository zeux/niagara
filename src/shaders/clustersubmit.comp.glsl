#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#extension GL_GOOGLE_include_directive: require

#extension GL_EXT_null_initializer: require

#include "mesh.h"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer ClusterCount
{
	uint clusterCount;
	uint groupCountX;
	uint groupCountY;
	uint groupCountZ;
};

layout(binding = 1) writeonly buffer ClusterIndices
{
	uint clusterIndices[];
};

void main()
{
	uint tid = gl_LocalInvocationID.x;
	uint count = clusterCount;

	// represent cluster count as X*64*1; X has a max of 65535 (per EXT_mesh_shader limits), so this allows us to reach ~4M clusters (???)
	if (tid == 0)
	{
		// TODO: actually use CLUSTER_LIMIT
		groupCountX = min((count + 63) / 64, 65535);
		groupCountY = 64;
		groupCountZ = 1;
	}

	// the above may result in reading command data that was never written; as such, pad the excess entries with dummy commands (up to 63)
	uint boundary = (count + 63) & ~63;

	if (count + tid < boundary)
		clusterIndices[count + tid] = ~0;
}
