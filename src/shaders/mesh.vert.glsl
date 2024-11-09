#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#extension GL_GOOGLE_include_directive: require

#extension GL_ARB_shader_draw_parameters: require

#include "mesh.h"
#include "math.h"

layout(push_constant) uniform block
{
	Globals globals;
};

layout(binding = 0) readonly buffer DrawCommands
{
	MeshDrawCommand drawCommands[];
};

layout(binding = 1) readonly buffer Draws
{
	MeshDraw draws[];
};

layout(binding = 2) readonly buffer Vertices
{
	Vertex vertices[];
};

layout(location = 0) out flat uint out_drawId;
layout(location = 1) out vec2 out_uv;
layout(location = 2) out vec3 out_normal;
layout(location = 3) out vec4 out_tangent;

void main()
{
	uint drawId = drawCommands[gl_DrawIDARB].drawId;
	MeshDraw meshDraw = draws[drawId];

	vec3 position = vec3(vertices[gl_VertexIndex].vx, vertices[gl_VertexIndex].vy, vertices[gl_VertexIndex].vz);
	vec3 normal = vec3(int(vertices[gl_VertexIndex].nx), int(vertices[gl_VertexIndex].ny), int(vertices[gl_VertexIndex].nz)) / 127.0 - 1.0;
	vec4 tangent = vec4(int(vertices[gl_VertexIndex].tx), int(vertices[gl_VertexIndex].ty), int(vertices[gl_VertexIndex].tz), int(vertices[gl_VertexIndex].tw)) / 127.0 - 1.0;
	vec2 texcoord = vec2(vertices[gl_VertexIndex].tu, vertices[gl_VertexIndex].tv);

	normal = rotateQuat(normal, meshDraw.orientation);
	tangent.xyz = rotateQuat(tangent.xyz, meshDraw.orientation);

	gl_Position = globals.projection * (globals.cullData.view * vec4(rotateQuat(position, meshDraw.orientation) * meshDraw.scale + meshDraw.position, 1));
	out_drawId = drawId;
	out_uv = texcoord;
	out_normal = normal;
	out_tangent = tangent;
}
