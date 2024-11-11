#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform block
{
	vec2 imageSize;
};

layout(binding = 0) uniform writeonly image2D outImage;
layout(binding = 1) uniform sampler2D colorImage;

void main()
{
	uvec2 pos = gl_GlobalInvocationID.xy;
	vec2 uv = (vec2(pos) + 0.5) / imageSize;

	vec3 color = texture(colorImage, uv).rgb;

	imageStore(outImage, ivec2(pos), vec4(color, 1.0));
}
