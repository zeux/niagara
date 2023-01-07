#define TASK_WGSIZE 64
#define MESH_WGSIZE 64

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

	float screenWidth, screenHeight, znear, zfar; // symmetric projection parameters
	float frustum[4]; // data for left/right/top/bottom frustum planes

	float pyramidWidth, pyramidHeight; // depth pyramid size in texels
	int clusterOcclusionEnabled;
	bool lateWorkaroundAMD; // TODO: rename
};

struct DrawCullData
{
	float P00, P11, znear, zfar; // symmetric projection parameters
	float frustum[4]; // data for left/right/top/bottom frustum planes
	float lodBase, lodStep; // lod distance i = base * pow(step, i)
	float pyramidWidth, pyramidHeight; // depth pyramid size in texels

	uint drawCount;

	int cullingEnabled;
	int lodEnabled;
	int occlusionEnabled;
	int meshShadingEnabled;

	int clusterOcclusionEnabled;
	bool lateWorkaroundAMD;
};

struct MeshLod
{
	uint indexOffset;
	uint indexCount;
	uint meshletOffset;
	uint meshletCount;
};

struct Mesh
{
	vec3 center;
	float radius;

	uint vertexOffset;
	uint vertexCount;

	uint lodCount;
	MeshLod lods[8];
};

struct MeshDraw
{
	vec3 position;
	float scale;
	vec4 orientation;

	uint meshIndex;
	uint vertexOffset; // == meshes[meshIndex].vertexOffset, helps data locality in mesh shader
	uint meshletVisibilityOffset;
};

struct MeshDrawCommand
{
	uint drawId;

	// used by traditional raster
	uint indexCount;
	uint instanceCount;
	uint firstIndex;
	uint vertexOffset;
	uint firstInstance;

	// used by mesh shading path
	uint lateDrawVisibility;
	uint meshletVisibilityOffset;
	uint taskOffset;
	uint taskCount;
	uint taskX;
	uint taskY;
	uint taskZ;
};

struct MeshTaskPayload
{
	uint drawId;
	uint meshletIndices[TASK_WGSIZE];
};

vec3 rotateQuat(vec3 v, vec4 q)
{
	return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
