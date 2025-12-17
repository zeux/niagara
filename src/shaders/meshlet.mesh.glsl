#version 450

#extension GL_GOOGLE_include_directive: require
#include "../config.h"

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#if NV_MESH
#extension GL_NV_mesh_shader: require
#extension GL_KHR_shader_subgroup_ballot: require
#else
#extension GL_EXT_mesh_shader: require
#endif

#include "mesh.h"
#include "math.h"

layout (constant_id = 1) const bool TASK = false;

#define DEBUG 0
#define CULL MESH_CULL
#define FASTCULL 1

layout(local_size_x = MESH_WGSIZE, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = MESH_MAXVTX, max_primitives = MESH_MAXTRI) out;

layout(push_constant) uniform block
{
	Globals globals;
};

layout(binding = 0) readonly buffer TaskCommands
{
	MeshTaskCommand taskCommands[];
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

layout(binding = 3) readonly buffer MeshletData16
{
	uint16_t meshletData16[];
};

layout(binding = 3) readonly buffer MeshletData8
{
	uint8_t meshletData8[];
};

layout(binding = 4) readonly buffer Vertices
{
	Vertex vertices[];
};

layout(binding = 5) readonly buffer ClusterIndices
{
	uint clusterIndices[];
};

layout(location = 0) out flat uint out_drawId[];
layout(location = 1) out vec2 out_uv[];
layout(location = 2) out vec3 out_normal[];
layout(location = 3) out vec4 out_tangent[];
layout(location = 4) out vec3 out_wpos[];

// only usable with task shader (TASK=true)
#if NV_MESH
in taskNV block { MeshTaskPayload payload; };
#else
taskPayloadSharedEXT MeshTaskPayload payload;
#endif

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

#if CULL && !NV_MESH
shared vec3 vertexClip[MESH_MAXVTX];
#endif

void main()
{
	uint ti = gl_LocalInvocationIndex;

#if NV_MESH
	uint ci = TASK ? payload.clusterIndices[gl_WorkGroupID.x] : clusterIndices[gl_WorkGroupID.x];
#else
	// we convert 3D index to 1D index using a fixed *256 factor, see clustersubmit.comp.glsl
	uint ci = TASK ? payload.clusterIndices[gl_WorkGroupID.x] : clusterIndices[gl_WorkGroupID.x + gl_WorkGroupID.y * 256 + gl_WorkGroupID.z * CLUSTER_TILE];
#endif

	if (ci == ~0)
	{
#if NV_MESH
		gl_PrimitiveCountNV = 0;
#else
		SetMeshOutputsEXT(0, 0);
#endif
		return;
	}

	MeshTaskCommand	command = taskCommands[ci & 0xffffff];
	uint mi = command.taskOffset + (ci >> 24);

	MeshDraw meshDraw = draws[command.drawId];

	uint vertexCount = uint(meshlets[mi].vertexCount);
	uint triangleCount = uint(meshlets[mi].triangleCount);

#if !NV_MESH
	SetMeshOutputsEXT(vertexCount, triangleCount);
#endif

	uint dataOffset = meshlets[mi].dataOffset;
	uint baseVertex = meshlets[mi].baseVertex;
	bool shortRefs = uint(meshlets[mi].shortRefs) == 1;
	uint vertexOffset = dataOffset;
	uint indexOffset = dataOffset + (shortRefs ? (vertexCount + 1) / 2 : vertexCount);

#if DEBUG
	uint mhash = hash(mi);
	vec3 mcolor = vec3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0;
#endif

	vec2 screen = vec2(globals.screenWidth, globals.screenHeight);

	for (uint i = ti; i < vertexCount; )
	{
		uint vi = shortRefs ? uint(meshletData16[vertexOffset * 2 + i]) + baseVertex : meshletData[vertexOffset + i] + baseVertex;

		vec3 position = vec3(vertices[vi].vx, vertices[vi].vy, vertices[vi].vz);
		vec2 texcoord = vec2(vertices[vi].tu, vertices[vi].tv);

		vec3 normal;
		vec4 tangent;
		unpackTBN(vertices[vi].np, uint(vertices[vi].tp), normal, tangent);

		normal = rotateQuat(normal, meshDraw.orientation);
		tangent.xyz = rotateQuat(tangent.xyz, meshDraw.orientation);

		vec3 wpos = rotateQuat(position, meshDraw.orientation) * meshDraw.scale + meshDraw.position;
		vec4 clip = globals.projection * (globals.cullData.view * vec4(wpos, 1));

#if NV_MESH
		gl_MeshVerticesNV[i].gl_Position = clip;
#else
		gl_MeshVerticesEXT[i].gl_Position = clip;
#endif
		out_drawId[i] = command.drawId;
		out_uv[i] = texcoord;
		out_normal[i] = normal;
		out_tangent[i] = tangent;
		out_wpos[i] = wpos;

	#if CULL && !NV_MESH
		vertexClip[i] = vec3((clip.xy / clip.w * 0.5 + vec2(0.5)) * screen, clip.w);
	#endif

	#if DEBUG
		out_normal[i] = mcolor;
	#endif

	#if MESH_MAXVTX <= MESH_WGSIZE
		break;
	#else
		i += MESH_WGSIZE;
	#endif
	}

#if CULL
#if NV_MESH
	subgroupMemoryBarrier();
#else
	barrier();
#endif
#endif

#if CULL && NV_MESH
	uint toff = 0;
#endif

	for (uint i = ti; i < triangleCount; )
	{
		uint offset = indexOffset * 4 + i * 3;
		uint a = uint(meshletData8[offset]), b = uint(meshletData8[offset + 1]), c = uint(meshletData8[offset + 2]);

#if !NV_MESH
		gl_PrimitiveTriangleIndicesEXT[i] = uvec3(a, b, c);
#elif !CULL
		gl_PrimitiveIndicesNV[i * 3 + 0] = a;
		gl_PrimitiveIndicesNV[i * 3 + 1] = b;
		gl_PrimitiveIndicesNV[i * 3 + 2] = c;
#endif

	#if CULL
		bool culled = false;

#if NV_MESH && FASTCULL
		culled = determinant(mat3(gl_MeshVerticesNV[a].gl_Position.xyw, gl_MeshVerticesNV[b].gl_Position.xyw, gl_MeshVerticesNV[c].gl_Position.xyw)) <= 0;
#else
#if NV_MESH
		vec3 ca = gl_MeshVerticesNV[a].gl_Position.xyw;
		vec3 cb = gl_MeshVerticesNV[b].gl_Position.xyw;
		vec3 cc = gl_MeshVerticesNV[c].gl_Position.xyw;

		ca.xy /= ca.z;
		cb.xy /= cb.z;
		cc.xy /= cc.z;
#else
		vec3 ca = vertexClip[a], cb = vertexClip[b], cc = vertexClip[c];
#endif

		vec2 pa = ca.xy, pb = cb.xy, pc = cc.xy;

		// backface culling + zero-area culling
		vec2 eb = pb - pa;
		vec2 ec = pc - pa;

		culled = culled || (eb.x * ec.y <= eb.y * ec.x);

		// small primitive culling
		vec2 bmin = min(pa, min(pb, pc));
		vec2 bmax = max(pa, max(pb, pc));
		float sbprec = 1.0 / 256.0; // note: this can be set to 1/2^subpixelPrecisionBits

	#if NV_MESH
		bmin = (bmin * 0.5 + vec2(0.5)) * screen;
		bmax = (bmax * 0.5 + vec2(0.5)) * screen;
	#endif

		// note: this is slightly imprecise (doesn't fully match hw behavior and is both too loose and too strict)
		culled = culled || (round(bmin.x - sbprec) == round(bmax.x) || round(bmin.y) == round(bmax.y + sbprec));

		// the computations above are only valid if all vertices are in front of perspective plane
		culled = culled && (ca.z > 0 && cb.z > 0 && cc.z > 0);
#endif

#if NV_MESH
		uvec4 cballot = subgroupBallot(!culled);
		uint coff = subgroupBallotExclusiveBitCount(cballot);

		if (!culled)
		{
			gl_PrimitiveIndicesNV[(toff + coff) * 3 + 0] = a;
			gl_PrimitiveIndicesNV[(toff + coff) * 3 + 1] = b;
			gl_PrimitiveIndicesNV[(toff + coff) * 3 + 2] = c;
		}

		toff += subgroupBallotBitCount(cballot);
#else
		gl_MeshPrimitivesEXT[i].gl_CullPrimitiveEXT = culled;
#endif
	#endif

	#if MESH_MAXTRI <= MESH_WGSIZE
		break;
	#else
		i += MESH_WGSIZE;
	#endif
	}

#if NV_MESH
#if CULL
	gl_PrimitiveCountNV = toff;
#else
	gl_PrimitiveCountNV = triangleCount;
#endif
#endif
}
