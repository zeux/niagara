#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_mesh_shader: require

#extension GL_GOOGLE_include_directive: require

#include "mesh.h"
#include "math.h"

#define DEBUG 0
#define CULL MESH_CULL

layout(local_size_x = MESH_WGSIZE, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = MESH_MAXVTX, max_primitives = MESH_MAXTRI) out;

layout(push_constant) uniform block
{
	Globals globals;
};

layout(binding = 1) readonly buffer Draws
{
	MeshDraw draws[];
};

layout(binding = 2) readonly buffer Meshlets
{
	Meshlet meshlets[];
};

layout(binding = 3) readonly buffer MeshletData
{
	uint meshletData[];
};

layout(binding = 3) readonly buffer MeshletData8
{
	uint8_t meshletData8[];
};

layout(binding = 4) readonly buffer Vertices
{
	Vertex vertices[];
};

layout(location = 0) out vec4 color[];

taskPayloadSharedEXT MeshTaskPayload payload;

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

#if CULL
shared vec3 vertexClip[MESH_MAXVTX];
#endif

void main()
{
	uint ti = gl_LocalInvocationIndex;
	uint mi = payload.meshletIndices[gl_WorkGroupID.x];

	MeshDraw meshDraw = draws[payload.drawId];

	uint vertexCount = uint(meshlets[mi].vertexCount);
	uint triangleCount = uint(meshlets[mi].triangleCount);

	SetMeshOutputsEXT(vertexCount, triangleCount);

	uint dataOffset = meshlets[mi].dataOffset;
	uint vertexOffset = dataOffset;
	uint indexOffset = dataOffset + vertexCount;

#if DEBUG
	uint mhash = hash(mi);
	vec3 mcolor = vec3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
#endif

	vec2 screen = vec2(globals.screenWidth, globals.screenHeight);

	for (uint i = ti; i < vertexCount; )
	{
		uint vi = meshletData[vertexOffset + i] + meshDraw.vertexOffset;

		vec3 position = vec3(vertices[vi].vx, vertices[vi].vy, vertices[vi].vz);
		vec3 normal = vec3(int(vertices[vi].nx), int(vertices[vi].ny), int(vertices[vi].nz)) / 127.0 - 1.0;
		vec2 texcoord = vec2(vertices[vi].tu, vertices[vi].tv);

		vec4 clip = globals.projection * vec4(rotateQuat(position, meshDraw.orientation) * meshDraw.scale + meshDraw.position, 1);

		gl_MeshVerticesEXT[i].gl_Position = clip;
		color[i] = vec4(normal * 0.5 + vec3(0.5), 1.0);

	#if CULL
		vertexClip[i] = vec3((clip.xy / clip.w * 0.5 + vec2(0.5)) * screen, clip.w);
	#endif

	#if DEBUG
		color[i] = vec4(mcolor, 1.0);
	#endif

	#if MESH_MAXVTX <= MESH_WGSIZE
		break;
	#else
		i += MESH_WGSIZE;
	#endif
	}

#if CULL
	barrier();
#endif

	for (uint i = ti; i < triangleCount; )
	{
		uint offset = indexOffset * 4 + i * 3;
		uint a = uint(meshletData8[offset]), b = uint(meshletData8[offset + 1]), c = uint(meshletData8[offset + 2]);

		gl_PrimitiveTriangleIndicesEXT[i] = uvec3(a, b, c);

	#if CULL
		bool culled = false;

		vec2 pa = vertexClip[a].xy, pb = vertexClip[b].xy, pc = vertexClip[c].xy;

		// backface culling + zero-area culling
		vec2 eb = pb - pa;
		vec2 ec = pc - pa;

		culled = culled || (eb.x * ec.y >= eb.y * ec.x);

		// small primitive culling
		vec2 bmin = min(pa, min(pb, pc));
		vec2 bmax = max(pa, max(pb, pc));
		float sbprec = 1.0 / 256.0; // note: this can be set to 1/2^subpixelPrecisionBits

		// note: this is slightly imprecise (doesn't fully match hw behavior and is both too loose and too strict)
		culled = culled || (round(bmin.x - sbprec) == round(bmax.x) || round(bmin.y) == round(bmax.y + sbprec));

		// the computations above are only valid if all vertices are in front of perspective plane
		culled = culled && (vertexClip[a].z > 0 && vertexClip[b].z > 0 && vertexClip[c].z > 0);

		gl_MeshPrimitivesEXT[i].gl_CullPrimitiveEXT = culled;
	#endif

	#if MESH_MAXTRI <= MESH_WGSIZE
		break;
	#else
		i += MESH_WGSIZE;
	#endif
	}
}
