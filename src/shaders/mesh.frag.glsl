#version 460

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require

#include "mesh.h"

layout (constant_id = 2) const int POST = 0;

#define DEBUG 0

layout(push_constant) uniform block
{
	Globals globals;
};

layout(binding = 1) readonly buffer Draws
{
	MeshDraw draws[];
};

layout(location = 0) out vec4 gbuffer[2];

layout(location = 0) in flat uint drawId;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec4 tangent;
layout(location = 4) in vec3 wpos;

layout(binding = 7) uniform sampler textureSampler;

layout(binding = 0, set = 1) uniform texture2D textures[];

#define SAMP(id) sampler2D(textures[nonuniformEXT(id)], textureSampler)

uint hash(uint a)
{
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}

void main()
{
	MeshDraw meshDraw = draws[drawId];

	vec4 albedo = vec4(0.5f, 0.5f, 0.5f, 1);
	if (meshDraw.albedoTexture > 0)
		albedo = texture(SAMP(meshDraw.albedoTexture), uv);

	vec3 nmap = vec3(0, 0, 1);
	if (meshDraw.normalTexture > 0)
		nmap = texture(SAMP(meshDraw.normalTexture), uv).rgb * 2 - 1;

	vec4 specgloss = vec4(0.0f);
	if (meshDraw.specularTexture > 0)
		specgloss = texture(SAMP(meshDraw.specularTexture), uv);

	vec3 emissive = vec3(0.0f);
	if (meshDraw.emissiveTexture > 0)
		emissive = texture(SAMP(meshDraw.emissiveTexture), uv).rgb;

	vec3 bitangent = cross(normal, tangent.xyz) * tangent.w;

	vec3 nrm = normalize(nmap.r * tangent.xyz + nmap.g * bitangent + nmap.b * normal);

	// TODO: emissive encoding should support colored & HDR emissive
	gbuffer[0] = vec4(albedo.rgb, emissive.r);
	// TODO: specular glossiness encoding is missing; we should encode roughness+metalness somehow in gbuffer1.z and use oct encoding for normal
	gbuffer[1] = vec4(nrm * 0.5 + 0.5, 0.0);

	if (POST > 0 && albedo.a < 0.5)
		discard;

#if DEBUG
	uint mhash = hash(drawId);
	gbuffer[0] = vec4(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255), 255) / 255.0;
#endif
}
