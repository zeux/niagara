#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0, r32f) uniform writeonly image2D outImage;
layout(binding = 1, r32f) uniform readonly image2D inImage;

void main()
{
	uvec2 pos = gl_GlobalInvocationID.xy;

	vec4 depth = imageLoad(inImage, ivec2(pos >> 1));

	imageStore(outImage, ivec2(pos), depth);
}
