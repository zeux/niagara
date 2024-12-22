#version 460

#define BLUR 1

#extension GL_GOOGLE_include_directive: require

#include "math.h"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform block
{
	vec2 imageSize;
	float direction;
	float znear;
};

layout(binding = 0) uniform writeonly image2D outImage;

layout(binding = 1) uniform sampler2D shadowImage;
layout(binding = 2) uniform sampler2D depthImage;

void main()
{
	uvec2 pos = gl_GlobalInvocationID.xy;

#if BLUR
	float shadow = texelFetch(shadowImage, ivec2(pos), 0).r;
	float accumw = 1;

	float depth = znear / texelFetch(depthImage, ivec2(pos), 0).r;

	ivec2 offsetMask = -ivec2(direction, 1 - direction);

	const int KERNEL = 10;

	for (int sign = -1; sign <= 1; sign += 2)
	{
		ivec2 uvnext = ivec2(pos) + (ivec2(sign) & offsetMask);
		float dnext = znear / texelFetch(depthImage, uvnext, 0).r;
		float dgrad = abs(depth - dnext) < 0.1 ? dnext - depth : 0;

		for (int i = 1; i <= KERNEL; ++i)
		{
	        ivec2 uvoff = ivec2(pos) + (ivec2(i * sign) & offsetMask);

			float gw = exp2(-i * i / 50);
			float dv = znear / texelFetch(depthImage, uvoff, 0).r;
			float dw = exp2(-abs(dv - (depth + dgrad * i)) * 100);
			float fw = gw * dw;

			shadow += texelFetch(shadowImage, uvoff, 0).r * fw;
			accumw += fw;
		}
	}

	shadow /= accumw;
#else
	float shadow = texelFetch(shadowImage, ivec2(pos), 0).r;
#endif

	imageStore(outImage, ivec2(pos), vec4(shadow, 0, 0, 0));
}
