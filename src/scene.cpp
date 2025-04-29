#include "common.h"
#include "scene.h"

#include "config.h"

#include <fast_obj.h>
#include <cgltf.h>
#include <meshoptimizer.h>

#include <time.h>

#include <algorithm>
#include <memory>
#include <cstring>

static void appendMeshlet(Geometry& result, const meshopt_Meshlet& meshlet, const std::vector<vec3>& vertices, const std::vector<unsigned int>& meshlet_vertices, const std::vector<unsigned char>& meshlet_triangles, uint32_t baseVertex, bool lod0)
{
	size_t dataOffset = result.meshletdata.size();

	unsigned int minVertex = ~0u, maxVertex = 0;
	for (unsigned int i = 0; i < meshlet.vertex_count; ++i)
	{
		minVertex = std::min(meshlet_vertices[meshlet.vertex_offset + i], minVertex);
		maxVertex = std::max(meshlet_vertices[meshlet.vertex_offset + i], maxVertex);
	}

	bool shortRefs = maxVertex - minVertex < (1 << 16);

	for (unsigned int i = 0; i < meshlet.vertex_count; ++i)
	{
		unsigned int ref = meshlet_vertices[meshlet.vertex_offset + i] - minVertex;
		if (shortRefs && i % 2)
			result.meshletdata.back() |= ref << 16;
		else
			result.meshletdata.push_back(ref);
	}

	const unsigned int* indexGroups = reinterpret_cast<const unsigned int*>(&meshlet_triangles[0] + meshlet.triangle_offset);
	unsigned int indexGroupCount = (meshlet.triangle_count * 3 + 3) / 4;

	for (unsigned int i = 0; i < indexGroupCount; ++i)
		result.meshletdata.push_back(indexGroups[i]);

	if (lod0)
	{
		for (unsigned int i = 0; i < meshlet.vertex_count; ++i)
		{
			unsigned int vtx = meshlet_vertices[meshlet.vertex_offset + i];

			unsigned short hx = meshopt_quantizeHalf(vertices[vtx].x);
			unsigned short hy = meshopt_quantizeHalf(vertices[vtx].y);
			unsigned short hz = meshopt_quantizeHalf(vertices[vtx].z);

			result.meshletvtx0.push_back(hx);
			result.meshletvtx0.push_back(hy);
			result.meshletvtx0.push_back(hz);
			result.meshletvtx0.push_back(0);
		}
	}

	meshopt_Bounds bounds = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, &vertices[0].x, vertices.size(), sizeof(vec3));

	Meshlet m = {};
	m.dataOffset = uint32_t(dataOffset);
	m.baseVertex = baseVertex + minVertex;
	m.triangleCount = meshlet.triangle_count;
	m.vertexCount = meshlet.vertex_count;
	m.shortRefs = shortRefs;

	m.center = vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
	m.radius = bounds.radius;
	m.cone_axis[0] = bounds.cone_axis_s8[0];
	m.cone_axis[1] = bounds.cone_axis_s8[1];
	m.cone_axis[2] = bounds.cone_axis_s8[2];
	m.cone_cutoff = bounds.cone_cutoff_s8;

	result.meshlets.push_back(m);
}

static size_t appendMeshlets(Geometry& result, const std::vector<vec3>& vertices, std::vector<uint32_t>& indices, uint32_t baseVertex, bool lod0, bool fast, bool clrt)
{
	const size_t max_vertices = MESH_MAXVTX;
	const size_t min_triangles = MESH_MAXTRI / 4;
	const size_t max_triangles = MESH_MAXTRI;
	const float cone_weight = 0.25f;
	const float fill_weight = 0.5f;

	std::vector<meshopt_Meshlet> meshlets(indices.size() / 3);
	std::vector<unsigned int> meshlet_vertices(meshlets.size() * max_vertices);
	std::vector<unsigned char> meshlet_triangles(meshlets.size() * max_triangles * 3);

	if (fast)
		meshlets.resize(meshopt_buildMeshletsScan(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(), indices.size(), vertices.size(), max_vertices, max_triangles));
	else if (clrt && lod0) // only use split algo for lod0 as this is the only lod that is used for raytracing
		meshlets.resize(meshopt_buildMeshletsSplit(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(), indices.size(), &vertices[0].x, vertices.size(), sizeof(vec3), max_vertices, min_triangles, max_triangles, fill_weight));
	else
		meshlets.resize(meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(), indices.size(), &vertices[0].x, vertices.size(), sizeof(vec3), max_vertices, max_triangles, cone_weight));

	for (auto& meshlet : meshlets)
	{
		meshopt_optimizeMeshlet(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, meshlet.vertex_count);

		appendMeshlet(result, meshlet, vertices, meshlet_vertices, meshlet_triangles, baseVertex, lod0);
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

			v.vx = meshopt_quantizeHalf(obj->positions[gi.p * 3 + 0]);
			v.vy = meshopt_quantizeHalf(obj->positions[gi.p * 3 + 1]);
			v.vz = meshopt_quantizeHalf(obj->positions[gi.p * 3 + 2]);
			v.tp = 0;
			v.np = (meshopt_quantizeSnorm(obj->normals[gi.n * 3 + 0], 10) + 511) |
			       (meshopt_quantizeSnorm(obj->normals[gi.n * 3 + 1], 10) + 511) << 10 |
			       (meshopt_quantizeSnorm(obj->normals[gi.n * 3 + 1], 10) + 511) << 20;
			v.tu = meshopt_quantizeHalf(obj->texcoords[gi.t * 2 + 0]);
			v.tv = meshopt_quantizeHalf(obj->texcoords[gi.t * 2 + 1]);
		}

		index_offset += obj->face_vertices[i];
	}

	assert(vertex_offset == index_count);

	fast_obj_destroy(obj);

	return true;
}

static void appendMesh(Geometry& result, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, bool buildMeshlets, bool fast, bool clrt)
{
	std::vector<uint32_t> remap(vertices.size());
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

	std::vector<vec3> positions(vertices.size());
	for (size_t i = 0; i < vertices.size(); ++i)
	{
		Vertex& v = vertices[i];
		positions[i] = vec3(meshopt_dequantizeHalf(v.vx), meshopt_dequantizeHalf(v.vy), meshopt_dequantizeHalf(v.vz));
	}

	std::vector<vec3> normals(vertices.size());
	for (size_t i = 0; i < vertices.size(); ++i)
	{
		Vertex& v = vertices[i];
		normals[i] = vec3((v.np & 1023) / 511.f - 1.f, ((v.np >> 10) & 1023) / 511.f - 1.f, ((v.np >> 20) & 1023) / 511.f - 1.f);
	}

	vec3 center = vec3(0);

	for (auto& v : positions)
		center += v;

	center /= float(vertices.size());

	float radius = 0;

	for (auto& v : positions)
		radius = std::max(radius, distance(center, v));

	mesh.center = center;
	mesh.radius = radius;

	float lodScale = meshopt_simplifyScale(&positions[0].x, vertices.size(), sizeof(vec3));

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
		lod.meshletCount = buildMeshlets ? uint32_t(appendMeshlets(result, positions, lodIndices, mesh.vertexOffset, &lod == mesh.lods, fast, clrt)) : 0;

		lod.error = lodError * lodScale;

		if (mesh.lodCount < COUNTOF(mesh.lods))
		{
			// note: we're using the same value for all LODs; if this changes, we need to remove/change 95% exit criteria below
			const float maxError = 1e-1f;
			const unsigned int options = 0;

			size_t nextIndicesTarget = (size_t(double(lodIndices.size()) * 0.65) / 3) * 3;
			float nextError = 0.f;
			size_t nextIndices = meshopt_simplifyWithAttributes(lodIndices.data(), lodIndices.data(), lodIndices.size(), &positions[0].x, vertices.size(), sizeof(vec3), &normals[0].x, sizeof(vec3), normalWeights, 3, NULL, nextIndicesTarget, maxError, options, &nextError);
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

	result.meshes.push_back(mesh);
}

bool loadMesh(Geometry& geometry, const char* path, bool buildMeshlets, bool fast, bool clrt)
{
	std::vector<Vertex> vertices;
	if (!loadObj(vertices, path))
		return false;

	std::vector<uint32_t> indices(vertices.size());
	for (size_t i = 0; i < indices.size(); ++i)
		indices[i] = uint32_t(i);

	appendMesh(geometry, vertices, indices, buildMeshlets, fast, clrt);
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

static void loadVertices(std::vector<Vertex>& vertices, const cgltf_primitive& prim)
{
	size_t vertexCount = vertices.size();
	std::vector<float> scratch(vertexCount * 4);

	if (const cgltf_accessor* pos = cgltf_find_accessor(&prim, cgltf_attribute_type_position, 0))
	{
		assert(cgltf_num_components(pos->type) == 3);
		cgltf_accessor_unpack_floats(pos, scratch.data(), vertexCount * 3);

		for (size_t j = 0; j < vertexCount; ++j)
		{
			vertices[j].vx = meshopt_quantizeHalf(scratch[j * 3 + 0]);
			vertices[j].vy = meshopt_quantizeHalf(scratch[j * 3 + 1]);
			vertices[j].vz = meshopt_quantizeHalf(scratch[j * 3 + 2]);
		}
	}

	if (const cgltf_accessor* nrm = cgltf_find_accessor(&prim, cgltf_attribute_type_normal, 0))
	{
		assert(cgltf_num_components(nrm->type) == 3);
		cgltf_accessor_unpack_floats(nrm, scratch.data(), vertexCount * 3);

		for (size_t j = 0; j < vertexCount; ++j)
		{
			float nx = scratch[j * 3 + 0], ny = scratch[j * 3 + 1], nz = scratch[j * 3 + 2];

			vertices[j].np = (meshopt_quantizeSnorm(nx, 10) + 511) |
			                 (meshopt_quantizeSnorm(ny, 10) + 511) << 10 |
			                 (meshopt_quantizeSnorm(nz, 10) + 511) << 20;
		}
	}

	if (const cgltf_accessor* tan = cgltf_find_accessor(&prim, cgltf_attribute_type_tangent, 0))
	{
		assert(cgltf_num_components(tan->type) == 4);
		cgltf_accessor_unpack_floats(tan, scratch.data(), vertexCount * 4);

		for (size_t j = 0; j < vertexCount; ++j)
		{
			float tx = scratch[j * 4 + 0], ty = scratch[j * 4 + 1], tz = scratch[j * 4 + 2];
			float tsum = fabsf(tx) + fabsf(ty) + fabsf(tz);
			float tu = tz >= 0 ? tx / tsum : (1 - fabsf(ty / tsum)) * (tx >= 0 ? 1 : -1);
			float tv = tz >= 0 ? ty / tsum : (1 - fabsf(tx / tsum)) * (ty >= 0 ? 1 : -1);

			vertices[j].tp = (meshopt_quantizeSnorm(tu, 8) + 127) | (meshopt_quantizeSnorm(tv, 8) + 127) << 8;
			vertices[j].np |= (scratch[j * 4 + 3] >= 0 ? 0 : 1) << 30;
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
}

bool loadScene(Geometry& geometry, std::vector<Material>& materials, std::vector<MeshDraw>& draws, std::vector<std::string>& texturePaths, std::vector<Animation>& animations, Camera& camera, vec3& sunDirection, const char* path, bool buildMeshlets, bool fast, bool clrt)
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

			std::vector<Vertex> vertices(prim.attributes[0].data->count);
			loadVertices(vertices, prim);

			std::vector<uint32_t> indices(prim.indices->count);
			cgltf_accessor_unpack_indices(prim.indices, indices.data(), 4, indices.size());

			appendMesh(geometry, vertices, indices, buildMeshlets, fast, clrt);
			primitiveMaterials.push_back(prim.material);
		}

		primitives.push_back(std::make_pair(unsigned(meshOffset), unsigned(geometry.meshes.size() - meshOffset)));
	}

	assert(primitiveMaterials.size() + firstMeshOffset == geometry.meshes.size());

	std::vector<int> nodeDraws(data->nodes_count, -1); // for animations

	size_t materialOffset = materials.size();
	assert(materialOffset > 0); // index 0 = dummy materials

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

				cgltf_material* material = primitiveMaterials[range.first + j - firstMeshOffset];

				draw.materialIndex = material ? materialOffset + int(cgltf_material_index(data, material)) : 0;

				if (material && material->alpha_mode != cgltf_alpha_mode_opaque)
					draw.postPass = 1;

				if (material && material->has_transmission)
					draw.postPass = 2;

				nodeDraws[i] = int(draws.size());

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

	int textureOffset = 1 + int(texturePaths.size());

	for (size_t i = 0; i < data->materials_count; ++i)
	{
		cgltf_material* material = &data->materials[i];
		Material mat = {};

		mat.diffuseFactor = vec4(1);

		if (material->has_pbr_specular_glossiness)
		{
			if (material->pbr_specular_glossiness.diffuse_texture.texture)
				mat.albedoTexture = textureOffset + int(cgltf_texture_index(data, material->pbr_specular_glossiness.diffuse_texture.texture));

			mat.diffuseFactor = vec4(material->pbr_specular_glossiness.diffuse_factor[0], material->pbr_specular_glossiness.diffuse_factor[1], material->pbr_specular_glossiness.diffuse_factor[2], material->pbr_specular_glossiness.diffuse_factor[3]);

			if (material->pbr_specular_glossiness.specular_glossiness_texture.texture)
				mat.specularTexture = textureOffset + int(cgltf_texture_index(data, material->pbr_specular_glossiness.specular_glossiness_texture.texture));

			mat.specularFactor = vec4(material->pbr_specular_glossiness.specular_factor[0], material->pbr_specular_glossiness.specular_factor[1], material->pbr_specular_glossiness.specular_factor[2], material->pbr_specular_glossiness.glossiness_factor);
		}
		else if (material->has_pbr_metallic_roughness)
		{
			if (material->pbr_metallic_roughness.base_color_texture.texture)
				mat.albedoTexture = textureOffset + int(cgltf_texture_index(data, material->pbr_metallic_roughness.base_color_texture.texture));

			mat.diffuseFactor = vec4(material->pbr_metallic_roughness.base_color_factor[0], material->pbr_metallic_roughness.base_color_factor[1], material->pbr_metallic_roughness.base_color_factor[2], material->pbr_metallic_roughness.base_color_factor[3]);

			if (material->pbr_metallic_roughness.metallic_roughness_texture.texture)
				mat.specularTexture = textureOffset + int(cgltf_texture_index(data, material->pbr_metallic_roughness.metallic_roughness_texture.texture));

			mat.specularFactor = vec4(1, 1, 1, 1 - material->pbr_metallic_roughness.roughness_factor);
		}

		if (material->normal_texture.texture)
			mat.normalTexture = textureOffset + int(cgltf_texture_index(data, material->normal_texture.texture));

		if (material->emissive_texture.texture)
			mat.emissiveTexture = textureOffset + int(cgltf_texture_index(data, material->emissive_texture.texture));

		mat.emissiveFactor = vec3(material->emissive_factor[0], material->emissive_factor[1], material->emissive_factor[2]);

		materials.push_back(mat);
	}

	for (size_t i = 0; i < data->textures_count; ++i)
	{
		cgltf_texture* texture = &data->textures[i];
		assert(texture->image);

		cgltf_image* image = texture->image;
		assert(image->uri);

		std::string ipath = path;
		std::string::size_type pos = ipath.find_last_of("/\\");
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

	std::vector<cgltf_animation_sampler*> samplersT(data->nodes_count);
	std::vector<cgltf_animation_sampler*> samplersR(data->nodes_count);
	std::vector<cgltf_animation_sampler*> samplersS(data->nodes_count);

	for (size_t i = 0; i < data->animations_count; ++i)
	{
		cgltf_animation* anim = &data->animations[i];

		for (size_t j = 0; j < anim->channels_count; ++j)
		{
			cgltf_animation_channel* channel = &anim->channels[j];
			cgltf_animation_sampler* sampler = channel->sampler;

			if (!channel->target_node)
				continue;

			if (channel->target_path == cgltf_animation_path_type_translation)
				samplersT[cgltf_node_index(data, channel->target_node)] = sampler;
			else if (channel->target_path == cgltf_animation_path_type_rotation)
				samplersR[cgltf_node_index(data, channel->target_node)] = sampler;
			else if (channel->target_path == cgltf_animation_path_type_scale)
				samplersS[cgltf_node_index(data, channel->target_node)] = sampler;
		}
	}

	for (size_t i = 0; i < data->nodes_count; ++i)
	{
		if (!samplersR[i] && !samplersT[i] && !samplersS[i])
			continue;

		if (nodeDraws[i] == -1)
		{
			fprintf(stderr, "Warning: skipping animation for node %d without draw\n", int(i));
			continue;
		}

		cgltf_accessor* input = 0;
		if (samplersT[i])
			input = samplersT[i]->input;
		else if (samplersR[i])
			input = samplersR[i]->input;
		else if (samplersS[i])
			input = samplersS[i]->input;

		if ((samplersT[i] && samplersT[i]->input->count != input->count) ||
		    (samplersR[i] && samplersR[i]->input->count != input->count) ||
		    (samplersS[i] && samplersS[i]->input->count != input->count))
		{
			fprintf(stderr, "Warning: skipping animation for node %d due to mismatched sampler counts\n", int(i));
			continue;
		}

		if ((samplersT[i] && samplersT[i]->interpolation != cgltf_interpolation_type_linear) ||
		    (samplersR[i] && samplersR[i]->interpolation != cgltf_interpolation_type_linear) ||
		    (samplersS[i] && samplersS[i]->interpolation != cgltf_interpolation_type_linear))
		{
			fprintf(stderr, "Warning: skipping animation for node %d due to mismatched sampler counts\n", int(i));
			continue;
		}

		if (input->count < 2)
		{
			fprintf(stderr, "Warning: skipping animation for node %d with %d keyframes\n", int(i), int(input->count));
			continue;
		}

		std::vector<float> times(input->count);
		cgltf_accessor_unpack_floats(input, times.data(), times.size());

		Animation animation = {};
		animation.drawIndex = nodeDraws[i];
		animation.startTime = times[0];
		animation.period = times[1] - times[0];

		std::vector<float> valuesR, valuesT, valuesS;

		if (samplersT[i])
		{
			valuesT.resize(samplersT[i]->output->count * 3);
			cgltf_accessor_unpack_floats(samplersT[i]->output, valuesT.data(), valuesT.size());
		}

		if (samplersR[i])
		{
			valuesR.resize(samplersR[i]->output->count * 4);
			cgltf_accessor_unpack_floats(samplersR[i]->output, valuesR.data(), valuesR.size());
		}

		if (samplersS[i])
		{
			valuesS.resize(samplersS[i]->output->count * 3);
			cgltf_accessor_unpack_floats(samplersS[i]->output, valuesS.data(), valuesS.size());
		}

		cgltf_node nodeCopy = data->nodes[i];

		for (size_t j = 0; j < input->count; ++j)
		{
			if (samplersT[i])
				memcpy(nodeCopy.translation, &valuesT[j * 3], 3 * sizeof(float));

			if (samplersR[i])
				memcpy(nodeCopy.rotation, &valuesR[j * 4], 4 * sizeof(float));

			if (samplersS[i])
				memcpy(nodeCopy.scale, &valuesS[j * 3], 3 * sizeof(float));

			float matrix[16];
			cgltf_node_transform_world(&nodeCopy, matrix);

			float translation[3];
			float rotation[4];
			float scale[3];
			decomposeTransform(translation, rotation, scale, matrix);

			Keyframe kf = {};
			kf.translation = vec3(translation[0], translation[1], translation[2]);
			kf.rotation = quat(rotation[0], rotation[1], rotation[2], rotation[3]);
			kf.scale = std::max(scale[0], std::max(scale[1], scale[2]));

			animation.keyframes.push_back(kf);
		}

		animations.push_back(std::move(animation));
	}

	printf("Loaded %s: %d meshes, %d draws, %d animations, %d vertices in %.2f sec\n",
	    path, int(geometry.meshes.size()), int(draws.size()), int(animations.size()), int(geometry.vertices.size()),
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
