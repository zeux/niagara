// Workgroup size for task shader; each task shader thread produces up to one meshlet
#define TASK_WGSIZE 32

// Workgroup size for mesh shader; mesh shader workgroup processes the entire meshlet in parallel
#define MESH_WGSIZE 32

// Should we use NV_mesh_shader instead of EXT_mesh_shader? NV extension has different indirect dispatch layout.
#define NV_MESH 1

// Should we do meshlet frustum, occlusion and backface culling in task shader?
#define TASK_CULL 1

// Should we do triangle frustum and backface culling in mesh shader?
#define MESH_CULL 0

// Maximum number of vertices and triangles in a meshlet
#define MESH_MAXVTX 64
#define MESH_MAXTRI 96

// Meshlet build configuration for raster/RT
#define MESHLET_CONE_WEIGHT 0.25f
#define MESHLET_FILL_WEIGHT 0.5f

// Number of clusters along X dimension in a 3D tiled dispatch (must be a divisor of 256)
#define CLUSTER_TILE 16

// Maximum number of total task shader workgroups; 4M workgroups ~= 256M meshlets ~= 16B triangles if TASK_WGSIZE=64 and MESH_MAXTRI=64
#define TASK_WGLIMIT (1 << 22)

// Maximum number of total visible clusters; 16M meshlets ~= 64MB buffer with cluster indices
#define CLUSTER_LIMIT (1 << 24)

// Maximum number of frames in flight
#define MAX_FRAMES 2

// Minimum number of images in flight
#define MIN_IMAGES 3

// Should we enable vertical sync during presentation? Worth setting to 0 when doing perf profiling to avoid GPU downclock during idle
#define CONFIG_VSYNC 1

// Should we enable validation layers in release? (they are always enabled in debug)
#define CONFIG_RELVAL 0

// Should we enable synchronization validation? Worth running with 1 occasionally to check correctness.
#define CONFIG_SYNCVAL 0

// Maximum number of texture descriptors in the pool
#define DESCRIPTOR_LIMIT 65536
