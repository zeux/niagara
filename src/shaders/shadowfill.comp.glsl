#version 460

#extension GL_GOOGLE_include_directive: require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform block
{
	vec2 imageSize;
};

layout(binding = 0, r8) uniform image2D shadowImage;
layout(binding = 1) uniform sampler2D depthImage;

void main()
{
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

	// checkerboard odd
	pos.x *= 2;
	pos.x += pos.y & 1;

	float depth = texelFetch(depthImage, pos, 0).r;

	vec4 depths = vec4(
		texelFetch(depthImage, pos + ivec2(-1, 0), 0).r,
		texelFetch(depthImage, pos + ivec2(+1, 0), 0).r,
		texelFetch(depthImage, pos + ivec2(0, -1), 0).r,
		texelFetch(depthImage, pos + ivec2(0, +1), 0).r
	);

	vec4 shadows = vec4(
		imageLoad(shadowImage, pos + ivec2(-1, 0)).r,
		imageLoad(shadowImage, pos + ivec2(+1, 0)).r,
		imageLoad(shadowImage, pos + ivec2(0, -1)).r,
		imageLoad(shadowImage, pos + ivec2(0, +1)).r
	);

	vec4 weights = exp2(-abs(depths / depth - 1) * 20);

	float shadow = dot(weights, shadows) / (dot(weights, vec4(1)) + 1e-2);

	imageStore(shadowImage, pos, vec4(shadow, 0, 0, 0));
}
