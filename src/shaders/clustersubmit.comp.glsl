#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#extension GL_GOOGLE_include_directive: require

#include "mesh.h"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

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
	uint count = min(clusterCount, CLUSTER_LIMIT);

	// represent cluster count as 16*Y*16; Y has a max of 65535 (per EXT_mesh_shader limits), so this allows us to reach ~16M clusters
	// the reason for an odd layout like this is that normally we'd use a 2D 256*Y layout (to maximize locality of access), but that is slower than Y*256 on 7900
	// however, Y*256 is really slow on integrated RDNA2; 16*Y*16 seems to provide a reasonable balance between the two
	if (tid == 0)
	{
		groupCountX = 16;
		groupCountY = min((count + 255) / 256, 65535);
		groupCountZ = 16;
	}

	// the above may result in reading command data that was never written; as such, pad the excess entries with dummy commands (up to 63)
	uint boundary = (count + 255) & ~255;

	if (count + tid < boundary)
		clusterIndices[count + tid] = ~0;
}
