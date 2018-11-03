struct Vertex
{
	float vx, vy, vz;
	uint8_t nx, ny, nz, nw;
	float16_t tu, tv;
};

struct Meshlet
{
	vec4 cone;
	uint vertices[64];
	uint8_t indices[126*3]; // up to 126 triangles
	uint8_t triangleCount;
	uint8_t vertexCount;
};
