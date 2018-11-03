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
	vec2 offset;
	vec2 scale;
};