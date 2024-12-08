#version 460

#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_ray_query: require

#include "math.h"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

struct ShadowData
{
	vec3 sunDirection;
	float sunJitter;

	mat4 inverseViewProjection;

	vec2 imageSize;
};

layout(push_constant) uniform block
{
	ShadowData shadowData;
};

layout(binding = 0) uniform writeonly image2D outImage;

layout(binding = 1) uniform sampler2D depthImage;
layout(binding = 2) uniform accelerationStructureEXT tlas;

void main()
{
	uvec2 pos = gl_GlobalInvocationID.xy;
	vec2 uv = (vec2(pos) + 0.5) / shadowData.imageSize;

	float depth = texture(depthImage, uv).r;

	vec4 clip = vec4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1);
	vec4 wposh = shadowData.inverseViewProjection * clip;
	vec3 wpos = wposh.xyz / wposh.w;

	uint rayflags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsCullNoOpaqueEXT;

	vec3 dir = shadowData.sunDirection;
	// TODO: a lot more tuning required here
	// TODO: this should actually be doing cone sampling, not random XZ offsets
	float dir0 = gradientNoise(vec2(pos.xy));
	float dir1 = gradientNoise(vec2(pos.yx));
	dir.x += (dir0 * 2 - 1) * shadowData.sunJitter;
	dir.z += (dir1 * 2 - 1) * shadowData.sunJitter;
	dir = normalize(dir);

	rayQueryEXT rq;
	rayQueryInitializeEXT(rq, tlas, rayflags, 0xff, wpos, 1e-2, dir, 1e3);
	rayQueryProceedEXT(rq);

	float shadow = (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) ? 1.0 : 0.0;

	imageStore(outImage, ivec2(pos), vec4(shadow, 0, 0, 0));
}
