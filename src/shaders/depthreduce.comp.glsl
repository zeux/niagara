#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0, r32f) uniform writeonly image2D outImage;
layout(binding = 1) uniform sampler2D inImage;

layout(push_constant) uniform block
{
	vec2 imageSize;
};

void main()
{
	uvec2 pos = gl_GlobalInvocationID.xy;

	// TODO: VK_EXT_sampler_filter_minmax could be used instead to avoid the need to do 3 mins manually
	vec4 depth4 = textureGather(inImage, (vec2(pos) + vec2(0.5)) / imageSize);
	float depth = min(min(depth4.x, depth4.y), min(depth4.z, depth4.w));

	imageStore(outImage, ivec2(pos), vec4(depth));
}
