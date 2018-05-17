#include "TestData.h"

TestData::TestData(float hitrate, size_t triangleCount)
{
	this->triangleCount = triangleCount;

	Triangle* cpuTriangles = new Triangle[triangleCount];
	
	// generate triangles and ray
	for (size_t i = 0; i < triangleCount; ++i)
	{
		cpuTriangles[i][0] = { -1.0f, -1.0f, 1.0f };
		cpuTriangles[i][1] = { 0.0f, 1.0f, 1.0f };
		cpuTriangles[i][2] = { 1.0f, -1.0f, 1.0f };
	}
	Ray* cpuRay = new Ray();
	cpuRay->origin = vec3(0.0f);
	cpuRay->direction = vec3(0.0f, 0.0f, 1.0f);

	// Allocate on GPU
	cudaMalloc((void**) &triangles, triangleCount * sizeof(Triangle));	cudaMalloc((void**) &ray, sizeof(Ray));		// Copy to GPU	cudaMemcpy(triangles, cpuTriangles, triangleCount * sizeof(Triangle), cudaMemcpyHostToDevice);	cudaMemcpy(ray, cpuRay, sizeof(Ray), cudaMemcpyHostToDevice);

	// Delete CPU allocations
	delete cpuTriangles;
	delete cpuRay;
}

TestData::~TestData()
{
	cudaFree(ray);
	cudaFree(triangles);
}

__host__ __device__
float dot(vec3 a, vec3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
vec3 cross(vec3 a, vec3 b)
{
	return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}