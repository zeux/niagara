#include "common.h"
#include "scene.h"

#include "config.h"

#include <fast_obj.h>
#include <cgltf.h>
#include <meshoptimizer.h>

#include <time.h>

#include <algorithm>
#include <memory>

static size_t appendMeshlets(Geometry& result, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, bool fast = false)
{
	const size_t max_vertices = MESH_MAXVTX;
	const size_t max_triangles = MESH_MAXTRI;
	const float cone_weight = 0.25f;

	std::vector<meshopt_Meshlet> meshlets(meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles));
	std::vector<unsigned int> meshlet_vertices(meshlets.size() * max_vertices);
	std::vector<unsigned char> meshlet_triangles(meshlets.size() * max_triangles * 3);

	if (fast)
		meshlets.resize(meshopt_buildMeshletsScan(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(), indices.size(), vertices.size(), max_vertices, max_triangles));
	else
		meshlets.resize(meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(), indices.size(), &vertices[0].vx, vertices.size(), sizeof(Vertex), max_vertices, max_triangles, cone_weight));

	// note: we can append meshlet_vertices & meshlet_triangles buffers more or less directly with small changes in Meshlet struct, but for now keep the GPU side layout flexible and separate
	for (auto& meshlet : meshlets)
	{
		meshopt_optimizeMeshlet(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, meshlet.vertex_count);

		size_t dataOffset = result.meshletdata.size();

		for (unsigned int i = 0; i < meshlet.vertex_count; ++i)
			result.meshletdata.push_back(meshlet_vertices[meshlet.vertex_offset + i]);

		const unsigned int* indexGroups = reinterpret_cast<const unsigned int*>(&meshlet_triangles[0] + meshlet.triangle_offset);
		unsigned int indexGroupCount = (meshlet.triangle_count * 3 + 3) / 4;

		for (unsigned int i = 0; i < indexGroupCount; ++i)
			result.meshletdata.push_back(indexGroups[i]);

		meshopt_Bounds bounds = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, &vertices[0].vx, vertices.size(), sizeof(Vertex));

		Meshlet m = {};
		m.dataOffset = uint32_t(dataOffset);
		m.triangleCount = meshlet.triangle_count;
		m.vertexCount = meshlet.vertex_count;

		m.center = vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
		m.radius = bounds.radius;
		m.cone_axis[0] = bounds.cone_axis_s8[0];
		m.cone_axis[1] = bounds.cone_axis_s8[1];
		m.cone_axis[2] = bounds.cone_axis_s8[2];
		m.cone_cutoff = bounds.cone_cutoff_s8;

		result.meshlets.push_back(m);
	}

	return meshlets.size();
}

static bool loadObj(std::vector<Vertex>& vertices, const char* path)
{
	fastObjMesh* obj = fast_obj_read(path);
	if (!obj)
		return false;

	size_t index_count = 0;

	for (unsigned int i = 0; i < obj->face_count; ++i)
		index_count += 3 * (obj->face_vertices[i] - 2);

	vertices.resize(index_count);

	size_t vertex_offset = 0;
	size_t index_offset = 0;

	for (unsigned int i = 0; i < obj->face_count; ++i)
	{
		for (unsigned int j = 0; j < obj->face_vertices[i]; ++j)
		{
			fastObjIndex gi = obj->indices[index_offset + j];

			// triangulate polygon on the fly; offset-3 is always the first polygon vertex
			if (j >= 3)
			{
				vertices[vertex_offset + 0] = vertices[vertex_offset - 3];
				vertices[vertex_offset + 1] = vertices[vertex_offset - 1];
				vertex_offset += 2;
			}

			Vertex& v = vertices[vertex_offset++];

			v.vx = obj->positions[gi.p * 3 + 0];
			v.vy = obj->positions[gi.p * 3 + 1];
			v.vz = obj->positions[gi.p * 3 + 2];
			v.nx = uint8_t(obj->normals[gi.n * 3 + 0] * 127.f + 127.5f);
			v.ny = uint8_t(obj->normals[gi.n * 3 + 1] * 127.f + 127.5f);
			v.nz = uint8_t(obj->normals[gi.n * 3 + 2] * 127.f + 127.5f);
			v.tx = v.ty = v.tz = 127;
			v.tw = 254;
			v.tu = meshopt_quantizeHalf(obj->texcoords[gi.t * 2 + 0]);
			v.tv = meshopt_quantizeHalf(obj->texcoords[gi.t * 2 + 1]);
		}

		index_offset += obj->face_vertices[i];
	}

	assert(vertex_offset == index_count);

	fast_obj_destroy(obj);

	return true;
}

static void appendMesh(Geometry& result, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, bool buildMeshlets, bool fast = false)
{
	std::vector<uint32_t> remap(indices.size());
	size_t uniqueVertices = meshopt_generateVertexRemap(remap.data(), indices.data(), indices.size(), vertices.data(), vertices.size(), sizeof(Vertex));

	meshopt_remapVertexBuffer(vertices.data(), vertices.data(), vertices.size(), sizeof(Vertex), remap.data());
	meshopt_remapIndexBuffer(indices.data(), indices.data(), indices.size(), remap.data());

	vertices.resize(uniqueVertices);

	if (fast)
		meshopt_optimizeVertexCacheFifo(indices.data(), indices.data(), indices.size(), vertices.size(), 16);
	else
		meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), vertices.size());

	meshopt_optimizeVertexFetch(vertices.data(), indices.data(), indices.size(), vertices.data(), vertices.size(), sizeof(Vertex));

	Mesh mesh = {};

	mesh.vertexOffset = uint32_t(result.vertices.size());
	mesh.vertexCount = uint32_t(vertices.size());

	result.vertices.insert(result.vertices.end(), vertices.begin(), vertices.end());

	std::vector<vec3> normals(vertices.size());
	for (size_t i = 0; i < vertices.size(); ++i)
	{
		Vertex& v = vertices[i];
		normals[i] = vec3(v.nx / 127.f - 1.f, v.ny / 127.f - 1.f, v.nz / 127.f - 1.f);
	}

	vec3 center = vec3(0);

	for (auto& v : vertices)
		center += vec3(v.vx, v.vy, v.vz);

	center /= float(vertices.size());

	float radius = 0;

	for (auto& v : vertices)
		radius = std::max(radius, distance(center, vec3(v.vx, v.vy, v.vz)));

	mesh.center = center;
	mesh.radius = radius;

	float lodScale = meshopt_simplifyScale(&vertices[0].vx, vertices.size(), sizeof(Vertex));

	std::vector<uint32_t> lodIndices = indices;
	float lodError = 0.f;

	float normalWeights[3] = { 1.f, 1.f, 1.f };

	while (mesh.lodCount < COUNTOF(mesh.lods))
	{
		MeshLod& lod = mesh.lods[mesh.lodCount++];

		lod.indexOffset = uint32_t(result.indices.size());
		lod.indexCount = uint32_t(lodIndices.size());

		result.indices.insert(result.indices.end(), lodIndices.begin(), lodIndices.end());

		lod.meshletOffset = uint32_t(result.meshlets.size());
		lod.meshletCount = buildMeshlets ? uint32_t(appendMeshlets(result, vertices, lodIndices, fast)) : 0;

		lod.error = lodError * lodScale;

		if (mesh.lodCount < COUNTOF(mesh.lods))
		{
			// note: we're using the same value for all LODs; if this changes, we need to remove/change 95% exit criteria below
			const float maxError = 1e-1f;
			const unsigned int options = 0;

			size_t nextIndicesTarget = (size_t(double(lodIndices.size()) * 0.65) / 3) * 3;
			float nextError = 0.f;
			size_t nextIndices = meshopt_simplifyWithAttributes(lodIndices.data(), lodIndices.data(), lodIndices.size(), &vertices[0].vx, vertices.size(), sizeof(Vertex), &normals[0].x, sizeof(vec3), normalWeights, 3, NULL, nextIndicesTarget, maxError, options, &nextError);
			assert(nextIndices <= lodIndices.size());

			// we've reached the error bound
			if (nextIndices == lodIndices.size() || nextIndices == 0)
				break;

			// while we could keep this LOD, it's too close to the last one (and it can't go below that due to constant error bound above)
			if (nextIndices >= size_t(double(lodIndices.size()) * 0.95))
				break;

			lodIndices.resize(nextIndices);
			lodError = std::max(lodError, nextError); // important! since we start from last LOD, we need to accumulate the error

			if (fast)
				meshopt_optimizeVertexCacheFifo(lodIndices.data(), lodIndices.data(), lodIndices.size(), vertices.size(), 16);
			else
				meshopt_optimizeVertexCache(lodIndices.data(), lodIndices.data(), lodIndices.size(), vertices.size());
		}
	}

	// pad meshlets to 64 to allow shaders to over-read when running task shaders
	while (result.meshlets.size() % 64)
		result.meshlets.push_back(Meshlet());

	result.meshes.push_back(mesh);
}

bool loadMesh(Geometry& geometry, const char* path, bool buildMeshlets, bool fast)
{
	std::vector<Vertex> vertices;
	if (!loadObj(vertices, path))
		return false;

	std::vector<uint32_t> indices(vertices.size());
	for (size_t i = 0; i < indices.size(); ++i)
		indices[i] = uint32_t(i);

	appendMesh(geometry, vertices, indices, buildMeshlets, fast);
	return true;
}

static void decomposeTransform(float translation[3], float rotation[4], float scale[3], const float* transform)
{
	float m[4][4] = {};
	memcpy(m, transform, 16 * sizeof(float));

	// extract translation from last row
	translation[0] = m[3][0];
	translation[1] = m[3][1];
	translation[2] = m[3][2];

	// compute determinant to determine handedness
	float det =
	    m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
	    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
	    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

	float sign = (det < 0.f) ? -1.f : 1.f;

	// recover scale from axis lengths
	scale[0] = sqrtf(m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]) * sign;
	scale[1] = sqrtf(m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2]) * sign;
	scale[2] = sqrtf(m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]) * sign;

	// normalize axes to get a pure rotation matrix
	float rsx = (scale[0] == 0.f) ? 0.f : 1.f / scale[0];
	float rsy = (scale[1] == 0.f) ? 0.f : 1.f / scale[1];
	float rsz = (scale[2] == 0.f) ? 0.f : 1.f / scale[2];

	float r00 = m[0][0] * rsx, r10 = m[1][0] * rsy, r20 = m[2][0] * rsz;
	float r01 = m[0][1] * rsx, r11 = m[1][1] * rsy, r21 = m[2][1] * rsz;
	float r02 = m[0][2] * rsx, r12 = m[1][2] * rsy, r22 = m[2][2] * rsz;

	// "branchless" version of Mike Day's matrix to quaternion conversion
	int qc = r22 < 0 ? (r00 > r11 ? 0 : 1) : (r00 < -r11 ? 2 : 3);
	float qs1 = qc & 2 ? -1.f : 1.f;
	float qs2 = qc & 1 ? -1.f : 1.f;
	float qs3 = (qc - 1) & 2 ? -1.f : 1.f;

	float qt = 1.f - qs3 * r00 - qs2 * r11 - qs1 * r22;
	float qs = 0.5f / sqrtf(qt);

	rotation[qc ^ 0] = qs * qt;
	rotation[qc ^ 1] = qs * (r01 + qs1 * r10);
	rotation[qc ^ 2] = qs * (r20 + qs2 * r02);
	rotation[qc ^ 3] = qs * (r12 + qs3 * r21);
}

bool loadScene(Geometry& geometry, std::vector<MeshDraw>& draws, std::vector<std::string>& texturePaths, Camera& camera, vec3& sunDirection, const char* path, bool buildMeshlets, bool fast)
{
	clock_t timer = clock();

	cgltf_options options = {};
	cgltf_data* data = NULL;
	cgltf_result res = cgltf_parse_file(&options, path, &data);
	if (res != cgltf_result_success)
		return false;

	std::unique_ptr<cgltf_data, void (*)(cgltf_data*)> dataPtr(data, &cgltf_free);

	res = cgltf_load_buffers(&options, data, path);
	if (res != cgltf_result_success)
		return false;

	res = cgltf_validate(data);
	if (res != cgltf_result_success)
		return false;

	std::vector<std::pair<unsigned int, unsigned int>> primitives;
	std::vector<cgltf_material*> primitiveMaterials;

	size_t firstMeshOffset = geometry.meshes.size();

	for (size_t i = 0; i < data->meshes_count; ++i)
	{
		const cgltf_mesh& mesh = data->meshes[i];

		size_t meshOffset = geometry.meshes.size();

		for (size_t pi = 0; pi < mesh.primitives_count; ++pi)
		{
			const cgltf_primitive& prim = mesh.primitives[pi];
			if (prim.type != cgltf_primitive_type_triangles || !prim.indices)
				continue;

			size_t vertexCount = prim.attributes[0].data->count;
			std::vector<Vertex> vertices(vertexCount);

			std::vector<float> scratch(vertexCount * 4);

			if (const cgltf_accessor* pos = cgltf_find_accessor(&prim, cgltf_attribute_type_position, 0))
			{
				assert(cgltf_num_components(pos->type) == 3);
				cgltf_accessor_unpack_floats(pos, scratch.data(), vertexCount * 3);

				for (size_t j = 0; j < vertexCount; ++j)
				{
					vertices[j].vx = scratch[j * 3 + 0];
					vertices[j].vy = scratch[j * 3 + 1];
					vertices[j].vz = scratch[j * 3 + 2];
				}
			}

			if (const cgltf_accessor* nrm = cgltf_find_accessor(&prim, cgltf_attribute_type_normal, 0))
			{
				assert(cgltf_num_components(nrm->type) == 3);
				cgltf_accessor_unpack_floats(nrm, scratch.data(), vertexCount * 3);

				for (size_t j = 0; j < vertexCount; ++j)
				{
					vertices[j].nx = uint8_t(scratch[j * 3 + 0] * 127.f + 127.5f);
					vertices[j].ny = uint8_t(scratch[j * 3 + 1] * 127.f + 127.5f);
					vertices[j].nz = uint8_t(scratch[j * 3 + 2] * 127.f + 127.5f);
				}
			}

			if (const cgltf_accessor* tan = cgltf_find_accessor(&prim, cgltf_attribute_type_tangent, 0))
			{
				assert(cgltf_num_components(tan->type) == 4);
				cgltf_accessor_unpack_floats(tan, scratch.data(), vertexCount * 4);

				for (size_t j = 0; j < vertexCount; ++j)
				{
					vertices[j].tx = uint8_t(scratch[j * 4 + 0] * 127.f + 127.5f);
					vertices[j].ty = uint8_t(scratch[j * 4 + 1] * 127.f + 127.5f);
					vertices[j].tz = uint8_t(scratch[j * 4 + 2] * 127.f + 127.5f);
					vertices[j].tw = uint8_t(scratch[j * 4 + 3] * 127.f + 127.5f);
				}
			}

			if (const cgltf_accessor* tex = cgltf_find_accessor(&prim, cgltf_attribute_type_texcoord, 0))
			{
				assert(cgltf_num_components(tex->type) == 2);
				cgltf_accessor_unpack_floats(tex, scratch.data(), vertexCount * 2);

				for (size_t j = 0; j < vertexCount; ++j)
				{
					vertices[j].tu = meshopt_quantizeHalf(scratch[j * 2 + 0]);
					vertices[j].tv = meshopt_quantizeHalf(scratch[j * 2 + 1]);
				}
			}

			std::vector<uint32_t> indices(prim.indices->count);
			cgltf_accessor_unpack_indices(prim.indices, indices.data(), 4, indices.size());

			appendMesh(geometry, vertices, indices, buildMeshlets, fast);
			primitiveMaterials.push_back(prim.material);
		}

		primitives.push_back(std::make_pair(unsigned(meshOffset), unsigned(geometry.meshes.size() - meshOffset)));
	}

	assert(primitiveMaterials.size() + firstMeshOffset == geometry.meshes.size());

	for (size_t i = 0; i < data->nodes_count; ++i)
	{
		const cgltf_node* node = &data->nodes[i];

		if (node->mesh)
		{
			float matrix[16];
			cgltf_node_transform_world(node, matrix);

			float translation[3];
			float rotation[4];
			float scale[3];
			decomposeTransform(translation, rotation, scale, matrix);

			// TODO: better warnings for non-uniform or negative scale

			std::pair<unsigned int, unsigned int> range = primitives[cgltf_mesh_index(data, node->mesh)];

			for (unsigned int j = 0; j < range.second; ++j)
			{
				MeshDraw draw = {};
				draw.position = vec3(translation[0], translation[1], translation[2]);
				draw.scale = std::max(scale[0], std::max(scale[1], scale[2]));
				draw.orientation = quat(rotation[0], rotation[1], rotation[2], rotation[3]);
				draw.meshIndex = range.first + j;
				draw.vertexOffset = geometry.meshes[range.first + j].vertexOffset;

				cgltf_material* material = primitiveMaterials[range.first + j - firstMeshOffset];

				draw.albedoTexture =
				    material && material->pbr_metallic_roughness.base_color_texture.texture
				        ? 1 + cgltf_texture_index(data, material->pbr_metallic_roughness.base_color_texture.texture)
				    : material && material->pbr_specular_glossiness.diffuse_texture.texture
				        ? 1 + cgltf_texture_index(data, material->pbr_specular_glossiness.diffuse_texture.texture)
				        : 0;
				draw.normalTexture =
				    material && material->normal_texture.texture
				        ? 1 + cgltf_texture_index(data, material->normal_texture.texture)
				        : 0;
				draw.specularTexture =
				    material && material->pbr_specular_glossiness.specular_glossiness_texture.texture
				        ? 1 + cgltf_texture_index(data, material->pbr_specular_glossiness.specular_glossiness_texture.texture)
				        : 0;
				draw.emissiveTexture =
				    material && material->emissive_texture.texture
				        ? 1 + cgltf_texture_index(data, material->emissive_texture.texture)
				        : 0;

				if (material && material->alpha_mode != cgltf_alpha_mode_opaque)
					draw.postPass = 1;

				draws.push_back(draw);
			}
		}

		if (node->camera)
		{
			float matrix[16];
			cgltf_node_transform_world(node, matrix);

			float translation[3];
			float rotation[4];
			float scale[3];
			decomposeTransform(translation, rotation, scale, matrix);

			assert(node->camera->type == cgltf_camera_type_perspective);

			camera.position = vec3(translation[0], translation[1], translation[2]);
			camera.orientation = quat(rotation[0], rotation[1], rotation[2], rotation[3]);
			camera.fovY = node->camera->data.perspective.yfov;
		}

		if (node->light && node->light->type == cgltf_light_type_directional)
		{
			float matrix[16];
			cgltf_node_transform_world(node, matrix);

			sunDirection = vec3(matrix[8], matrix[9], matrix[10]);
		}
	}

	for (size_t i = 0; i < data->textures_count; ++i)
	{
		cgltf_texture* texture = &data->textures[i];
		assert(texture->image);

		cgltf_image* image = texture->image;
		assert(image->uri);

		std::string ipath = path;
		std::string::size_type pos = ipath.find_last_of('/');
		if (pos == std::string::npos)
			ipath = "";
		else
			ipath = ipath.substr(0, pos + 1);

		std::string uri = image->uri;
		uri.resize(cgltf_decode_uri(&uri[0]));

		std::string::size_type dot = uri.find_last_of('.');
		if (dot != std::string::npos)
			uri.replace(dot, uri.size() - dot, ".dds");

		texturePaths.push_back(ipath + uri);
	}

	printf("Loaded %s: %d meshes, %d draws, %d vertices in %.2f sec\n",
	    path, int(geometry.meshes.size()), int(draws.size()), int(geometry.vertices.size()),
	    double(clock() - timer) / CLOCKS_PER_SEC);

	if (buildMeshlets)
	{
		unsigned int meshletVtxs = 0, meshletTris = 0;

		for (Meshlet& meshlet : geometry.meshlets)
		{
			meshletVtxs += meshlet.vertexCount;
			meshletTris += meshlet.triangleCount;
		}

		printf("Meshlets: %d meshlets, %d triangles, %d vertex refs\n", int(geometry.meshlets.size()), int(meshletTris), int(meshletVtxs));
	}

	return true;
}
