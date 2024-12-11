#version 460

#extension GL_GOOGLE_include_directive: require

#include "math.h"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

struct ShadeData
{
	vec3 cameraPosition;
	vec3 sunDirection;

	mat4 inverseViewProjection;

	vec2 imageSize;
};

layout(push_constant) uniform block
{
	ShadeData shadeData;
};

layout(binding = 0) uniform writeonly image2D outImage;

layout(binding = 1) uniform sampler2D gbufferImage0;
layout(binding = 2) uniform sampler2D gbufferImage1;
layout(binding = 3) uniform sampler2D depthImage;

layout(binding = 4) uniform sampler2D shadowImage;

void main()
{
	uvec2 pos = gl_GlobalInvocationID.xy;
	vec2 uv = (vec2(pos) + 0.5) / shadeData.imageSize;

	vec4 gbuffer0 = texture(gbufferImage0, uv);
	vec4 gbuffer1 = texture(gbufferImage1, uv);
	float depth = texture(depthImage, uv).r;

	vec3 albedo = fromsrgb(gbuffer0.rgb);
	vec3 emissive = albedo * (exp2(gbuffer0.a * 5) - 1);
	vec3 normal = decodeOct(gbuffer1.rg * 2 - 1);

	float ndotl = max(dot(normal, shadeData.sunDirection), 0.0);

	vec4 clip = vec4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1);
	vec4 wposh = shadeData.inverseViewProjection * clip;
	vec3 wpos = wposh.xyz / wposh.w;

	vec3 view = normalize(shadeData.cameraPosition - wpos);
	vec3 halfv = normalize(view + shadeData.sunDirection);
	float ndoth = max(dot(normal, halfv), 0.0);
	float gloss = gbuffer1.b;

	// TODO: this is not the BRDF we want
	float specular = pow(ndoth, mix(1, 64, gloss)) * gloss;

	float shadow = texture(shadowImage, uv).r;

	float ambient = 0.03;
	float shadowAmbient = 0.03;
	float sunIntensity = 1.5;

	vec3 outputColor = albedo.rgb * (ndotl * min(shadow + shadowAmbient, 1.0) * sunIntensity + ambient) + vec3(specular * shadow) + emissive;

	float deband = gradientNoise(vec2(pos)) * 2 - 1;
	imageStore(outImage, ivec2(pos), vec4(tosrgb(outputColor) + deband * (0.5 / 255), 1.0));
}
