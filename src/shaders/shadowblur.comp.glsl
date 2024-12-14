#version 460

#define BLUR 1
#define BLUROPT 0

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
	float shadow = texture(shadowImage, uv).r;
	float accumw = 1;

	float znear = 1;
	float depth = znear / texture(depthImage, uv).r;

	vec2 offsetScale = vec2(direction, 1 - direction) / imageSize;

	const int KERNEL = 10;

#if BLUROPT
	for (int i = -KERNEL / 2; i <= KERNEL / 2; ++i)
	{
		if (i == 0)
			continue;

		int i0 = i * 2;
		int i1 = i * 2 + 1;

		vec2 uvoffh = uv + vec2((i0 + i1) / 2) * offsetScale;

		vec2 dg = textureGather(depthImage, uvoffh).rg;
		vec2 sg = textureGather(shadowImage, uvoffh).rg;

		vec2 ip = vec2(i0, i1);
		vec2 gw = exp2(-abs(ip) / 10);
		vec2 dv = znear / dg;
		vec2 dw = exp2(-abs(depth - dv) * 20);
		vec2 fw = gw * dw;

		shadow += sg.r * fw.x;
		accumw += fw.x;

		shadow += sg.g * fw.y;
		accumw += fw.y;
	}
#else
	for (int i = -KERNEL; i <= KERNEL; ++i)
	{
		if (i == 0)
			continue;

        vec2 uvoff = uv + vec2(i) * offsetScale;

		float gw = exp2(-abs(i) / 10);
		float dv = znear / texture(depthImage, uvoff).r;
		float dw = exp2(-abs(depth - dv) * 20);
		float fw = gw * dw;

		shadow += texture(shadowImage, uvoff).r * fw;
		accumw += fw;
	}
#endif

	shadow /= accumw;
#else
	float shadow = texture(shadowImage, uv).r;
#endif

	imageStore(outImage, ivec2(pos), vec4(shadow, 0, 0, 0));
}
