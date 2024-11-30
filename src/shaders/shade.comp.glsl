#version 460

#define RAYTRACE 1

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

struct ShadeData
{
	vec3 sunDirection;
	float padding;

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

#if RAYTRACE
#extension GL_EXT_ray_query: require

layout(binding = 4) uniform accelerationStructureEXT tlas;
#endif

void main()
{
	uvec2 pos = gl_GlobalInvocationID.xy;
	vec2 uv = (vec2(pos) + 0.5) / shadeData.imageSize;

	vec4 gbuffer0 = texture(gbufferImage0, uv);
	vec4 gbuffer1 = texture(gbufferImage1, uv);
	float depth = texture(depthImage, uv).r;

	vec3 albedo = gbuffer0.rgb;
	vec3 emissive = vec3(gbuffer0.a);
	vec3 normal = gbuffer1.rgb * 2 - 1;

	float ndotl = max(dot(normal, shadeData.sunDirection), 0.0);

	vec4 clip = vec4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1);
	vec4 wposh = shadeData.inverseViewProjection * clip;
	vec3 wpos = wposh.xyz / wposh.w;

#if RAYTRACE
	uint rayflags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsCullNoOpaqueEXT;

	rayQueryEXT rq;
	rayQueryInitializeEXT(rq, tlas, rayflags, /* cullMask= */ 1, wpos, 1e-2, shadeData.sunDirection, 1e3);
	rayQueryProceedEXT(rq);

	ndotl *= (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) ? 1.0 : 0.05;
#endif

	vec3 outputColor = albedo.rgb * sqrt(ndotl + 0.05) + emissive;

	imageStore(outImage, ivec2(pos), vec4(outputColor, 1.0));
}
