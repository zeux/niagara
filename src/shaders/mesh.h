struct Vertex
{
	float vx, vy, vz;
	uint8_t nx, ny, nz, nw;
	float16_t tu, tv;
};

struct Meshlet
{
	// vec4 keeps Meshlet aligned to 16 bytes which is important because C++ has an alignas() directive
	vec4 cone;
	uint dataOffset;
	uint8_t vertexCount;
	uint8_t triangleCount;
};

struct MeshDraw
{
	mat4 projection;
	vec3 position;
	float scale;
	vec4 orientation;
};

vec3 rotateQuat(vec3 v, vec4 q)
{
	return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
