#version 460

#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_samplerless_texture_functions: require

#include "math.h"

#define QUALITY 1

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform block
{
	vec2 imageSize;
	int pass; // 0 = downsample from gbuffer, 1 = mip downsample, 2 = mip upsample
	float radius;
};

layout(r11f_g11f_b10f, binding = 0) uniform image2D outImage;
layout(binding = 1) uniform texture2D sourceImage;
layout(binding = 2) uniform sampler filterSampler;

void main()
{
	uvec2 pos = gl_GlobalInvocationID.xy;
	vec2 uv = (vec2(pos) + 0.5) / imageSize;
	vec2 texelSize = 1.0 / imageSize;

	if (pass == 0u)
	{
		// extract emissive from full-res gbuffer and downsample to half-res
		// note: exp2 encoding is not filterable so we sample each source pixel in its center
		vec4 s0 = texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-0.25, -0.25));
		vec4 s1 = texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+0.25, -0.25));
		vec4 s2 = texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-0.25, +0.25));
		vec4 s3 = texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+0.25, +0.25));

		vec3 e0 = fromsrgb(s0.rgb) * (exp2(s0.a * 5) - 1);
		vec3 e1 = fromsrgb(s1.rgb) * (exp2(s1.a * 5) - 1);
		vec3 e2 = fromsrgb(s2.rgb) * (exp2(s2.a * 5) - 1);
		vec3 e3 = fromsrgb(s3.rgb) * (exp2(s3.a * 5) - 1);

		vec3 result = (e0 + e1 + e2 + e3) * 0.25;

		imageStore(outImage, ivec2(pos), vec4(result, 0));
	}
	else if (pass == 1u)
	{
		// downsample blur - read from source mip, write to next smaller mip
		vec3 result = vec3(0);

#if QUALITY
		// Jorge Jimenez. Next Generation Post Processing in Call of Duty: Advanced Warfare.
		result += texture(sampler2D(sourceImage, filterSampler), uv).rgb * 0.125;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+0.5, +0.5)).rgb * (0.5 / 4);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+0.5, -0.5)).rgb * (0.5 / 4);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-0.5, +0.5)).rgb * (0.5 / 4);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-0.5, -0.5)).rgb * (0.5 / 4);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+1, +1)).rgb * (0.125 / 4);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+1, -1)).rgb * (0.125 / 4);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-1, +1)).rgb * (0.125 / 4);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-1, -1)).rgb * (0.125 / 4);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+1, 0)).rgb * (0.125 / 2);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-1, 0)).rgb * (0.125 / 2);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(0, +1)).rgb * (0.125 / 2);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(0, -1)).rgb * (0.125 / 2);
#else
		// Marius Bjørge, Bandwidth-Efficient Rendering. SIGGRAPH 2015
		result += texture(sampler2D(sourceImage, filterSampler), uv).rgb / 2;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+0.5, +0.5)).rgb / 8;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+0.5, -0.5)).rgb / 8;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-0.5, +0.5)).rgb / 8;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-0.5, -0.5)).rgb / 8;
#endif

		imageStore(outImage, ivec2(pos), vec4(result, 0));
	}
	else
	{
		// upsample blur - read from source mip, write to next larger mip, accumulating with existing data
		vec3 result = imageLoad(outImage, ivec2(pos)).rgb;

#if QUALITY
		// Jorge Jimenez. Next Generation Post Processing in Call of Duty: Advanced Warfare.
		result += texture(sampler2D(sourceImage, filterSampler), uv).rgb * (4.0 / 16);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * radius * vec2(+1, 0)).rgb * (2.0 / 16);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * radius * vec2(-1, 0)).rgb * (2.0 / 16);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * radius * vec2(0, +1)).rgb * (2.0 / 16);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * radius * vec2(0, -1)).rgb * (2.0 / 16);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * radius * vec2(+1, +1)).rgb * (1.0 / 16);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * radius * vec2(+1, -1)).rgb * (1.0 / 16);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * radius * vec2(-1, +1)).rgb * (1.0 / 16);
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * radius * vec2(-1, -1)).rgb * (1.0 / 16);
#else
		// Marius Bjørge, Bandwidth-Efficient Rendering. SIGGRAPH 2015
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+1, +1)).rgb / 6;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+1, -1)).rgb / 6;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-1, +1)).rgb / 6;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-1, -1)).rgb / 6;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(-2, 0)).rgb / 12;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(+2, 0)).rgb / 12;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(0, -2)).rgb / 12;
		result += texture(sampler2D(sourceImage, filterSampler), uv + texelSize * vec2(0, +2)).rgb / 12;
#endif

		imageStore(outImage, ivec2(pos), vec4(result, 0));
	}
}
