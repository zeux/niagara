#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require

#include "mesh.h"

#define DEBUG 0

layout(binding = 1) readonly buffer Draws
{
	MeshDraw draws[];
};

layout(location = 0) out vec4 outputColor;

layout(location = 0) in flat uint drawId;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec4 tangent;

layout(binding = 0, set = 1) uniform sampler2D textures[];

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
		albedo = texture(textures[nonuniformEXT(meshDraw.albedoTexture)], uv);

	vec4 nmap = vec4(0, 0, 1, 0);
	if (meshDraw.normalTexture > 0)
		nmap = texture(textures[nonuniformEXT(meshDraw.normalTexture)], uv) * 2 - 1;

	vec3 bitangent = cross(normal, tangent.xyz) * tangent.w;

	vec3 nrm = normalize(nmap.x * tangent.xyz + nmap.y * bitangent + nmap.z * normal);

	float ndotl = max(dot(nrm, normalize(vec3(-1, 1, -1))), 0.0);

	outputColor = albedo * sqrt(ndotl + 0.05);

	// outputColor = vec4(nrm * 0.5 + 0.5, 1.0);
	// outputColor = vec4(normal * 0.5 + 0.5, 1.0);
	// outputColor = albedo;

#if DEBUG
	uint mhash = hash(drawId);
	outputColor = vec4(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255), 255) / 255.0;
#endif
}
