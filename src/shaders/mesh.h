struct Vertex
{
	float vx, vy, vz;
	uint8_t nx, ny, nz, nw;
	float16_t tu, tv;
};

struct Meshlet
{
	// vec3 keeps Meshlet aligned to 16 bytes which is important because C++ has an alignas() directive
	vec3 center;
	float radius;
	int8_t cone_axis[3];
	int8_t cone_cutoff;

	uint dataOffset;
	uint8_t vertexCount;
	uint8_t triangleCount;
};

struct Globals
{
	mat4 projection;
};

struct MeshDraw
{
	vec3 position;
	float scale;
	vec4 orientation;

	vec3 center;
	float radius;

	uint vertexOffset;
	uint indexOffset;
	uint indexCount;
	uint meshletOffset;
	uint meshletCount;
};

struct MeshDrawCommand
{
	uint drawId;
	uint indexCount;
	uint instanceCount;
	uint firstIndex;
	uint vertexOffset;
	uint firstInstance;
    uint taskCount;
    uint firstTask;
};

vec3 rotateQuat(vec3 v, vec4 q)
{
	return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
