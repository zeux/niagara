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
	uint indicesPacked[124*3/4]; // up to 124 triangles
	uint8_t triangleCount;
	uint8_t vertexCount;
};
