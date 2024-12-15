#version 460

#extension GL_EXT_ray_query: require
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_nonuniform_qualifier: require

#extension GL_GOOGLE_include_directive: require

#include "math.h"
#include "mesh.h"

layout (constant_id = 0) const int QUALITY = 0;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

struct ShadowData
{
	vec3 sunDirection;
	float sunJitter;

	mat4 inverseViewProjection;

	vec2 imageSize;
	uint checkerboard;
};

layout(push_constant) uniform block
{
	ShadowData shadowData;
};

layout(binding = 0) uniform writeonly image2D outImage;

layout(binding = 1) uniform sampler2D depthImage;
layout(binding = 2) uniform accelerationStructureEXT tlas;

layout(binding = 3) readonly buffer Draws
{
	MeshDraw draws[];
};

layout(binding = 4) readonly buffer Meshes
{
	Mesh meshes[];
};

layout(binding = 5) readonly buffer Materials
{
	Material materials[];
};

layout(binding = 6) readonly buffer Vertices
{
	Vertex vertices[];
};

layout(binding = 7) readonly buffer Indices
{
	uint indices[];
};

layout(binding = 8) uniform sampler textureSampler;

layout(binding = 0, set = 1) uniform texture2D textures[];

#define SAMP(id) sampler2D(textures[nonuniformEXT(id)], textureSampler)

bool shadowTrace(vec3 wpos, vec3 dir, uint rayflags)
{
	rayQueryEXT rq;
	rayQueryInitializeEXT(rq, tlas, rayflags, 0xff, wpos, 1e-2, dir, 1e3);
	rayQueryProceedEXT(rq);
	return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

bool shadowTraceTransparent(vec3 wpos, vec3 dir, uint rayflags)
{
	rayQueryEXT rq;
	rayQueryInitializeEXT(rq, tlas, rayflags, 0xff, wpos, 1e-2, dir, 1e3);
	while (rayQueryProceedEXT(rq))
	{
		int objid = rayQueryGetIntersectionInstanceIdEXT(rq, false);
		int triid = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
		vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, false);

		MeshDraw draw = draws[objid];
		Material material = materials[draw.materialIndex];
		Mesh mesh = meshes[draw.meshIndex];

		uint vertexOffset = mesh.vertexOffset;
		uint indexOffset = mesh.lods[0].indexOffset;

		// TODO: It might be worth repacking some of this data for RT to reduce indirections
		// However, attempting to do this gained us zero performance back, so maybe not?
		uint tria = indices[indexOffset + triid * 3 + 0];
		uint trib = indices[indexOffset + triid * 3 + 1];
		uint tric = indices[indexOffset + triid * 3 + 2];

		vec2 uva = vec2(vertices[vertexOffset + tria].tu, vertices[vertexOffset + tria].tv);
		vec2 uvb = vec2(vertices[vertexOffset + trib].tu, vertices[vertexOffset + trib].tv);
		vec2 uvc = vec2(vertices[vertexOffset + tric].tu, vertices[vertexOffset + tric].tv);

		vec2 uv = uva * (1 - bary.x - bary.y) + uvb * bary.x + uvc * bary.y;

		float alpha = 1.0;
		if (material.albedoTexture > 0)
			alpha = textureLod(SAMP(material.albedoTexture), uv, 0).a;

		if (alpha >= 0.5)
			rayQueryConfirmIntersectionEXT(rq);
	}
	return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

void main()
{
	uvec2 pos = gl_GlobalInvocationID.xy;

	if (shadowData.checkerboard == 1)
	{
		// checkerboard even
		pos.x *= 2;
		pos.x += pos.y & 1;
	}

	vec2 uv = (vec2(pos) + 0.5) / shadowData.imageSize;
	float depth = texture(depthImage, uv).r;

	vec4 clip = vec4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1);
	vec4 wposh = shadowData.inverseViewProjection * clip;
	vec3 wpos = wposh.xyz / wposh.w;

	vec3 dir = shadowData.sunDirection;

	// TODO: a lot more tuning required here
	// TODO: this should actually be doing cone sampling, not random XZ offsets
	float dir0 = gradientNoise(vec2(pos.xy));
	float dir1 = gradientNoise(vec2(pos.yx));
	dir.x += (dir0 * 2 - 1) * shadowData.sunJitter;
	dir.z += (dir1 * 2 - 1) * shadowData.sunJitter;
	dir = normalize(dir);

	// On AMDVLK + RDNA3, two shadow traces are faster in practice than one; this may be different on other vendors/drivers
	// For example, for now on radv + RDNA3 one trace is faster, but radv is missing pointer flags optimizations
#if 1
	bool shadowhit = shadowTrace(wpos, dir, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsCullNoOpaqueEXT);

	if (!shadowhit && QUALITY != 0)
		shadowhit = shadowTraceTransparent(wpos, dir, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsCullOpaqueEXT);
#else
	bool shadowhit = QUALITY == 0
		? shadowTrace(wpos, dir, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsCullNoOpaqueEXT)
		: shadowTraceTransparent(wpos, dir, gl_RayFlagsTerminateOnFirstHitEXT);
#endif

	float shadow = shadowhit ? 0.0 : 1.0;

	imageStore(outImage, ivec2(pos), vec4(shadow, 0, 0, 0));
}
