#version 460

#define BLUR 1

#extension GL_GOOGLE_include_directive: require

#include "math.h"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform block
{
	vec2 imageSize;
	float direction;
};

layout(binding = 0) uniform writeonly image2D outImage;

layout(binding = 1) uniform sampler2D shadowImage;
layout(binding = 2) uniform sampler2D depthImage;

void main()
{
	uvec2 pos = gl_GlobalInvocationID.xy;
	vec2 uv = (vec2(pos) + 0.5) / imageSize;

#if BLUR
	float shadow = 0;
	float accumw = 0;

	float znear = 1;
	float depth = znear / texture(depthImage, uv).r;

	vec2 offsetScale = vec2(direction, 1 - direction) / imageSize;

	const int KERNEL = 10;

	for (int i = -KERNEL; i <= KERNEL; ++i)
	{
		vec2 uvoff = uv + vec2(i) * offsetScale;

		// TODO: a lot more tuning required here
		float gw = exp2(-abs(i) / 10);
		float dv = znear / texture(depthImage, uvoff).r;
		float dw = exp2(-abs(depth - dv) * 20);

		shadow += texture(shadowImage, uvoff).r * (dw * gw);
		accumw += dw * gw;
	}

	shadow /= accumw;
#else
	float shadow = texture(shadowImage, uv).r;
#endif

	imageStore(outImage, ivec2(pos), vec4(shadow, 0, 0, 0));
}
