#include "TestData.h"

TestData::TestData(float hitrate, size_t triangleCount)
{
	this->triangleCount = triangleCount;

	Triangle* cpuTriangles = new Triangle[triangleCount];
	// generate triangles and ray
	Ray* cpuRay = new Ray();
	cpuRay->origin = vec3(0.0f);
	cpuRay->direction = vec3(0.0f, 0.0f, 1.0f);

	// Allocate on GPU
	cudaMalloc((void**) &triangles, triangleCount * sizeof(Triangle));	cudaMalloc((void**) &ray, sizeof(Ray));		// Copy to GPU	cudaMemcpy(cpuTriangles, triangles, triangleCount * sizeof(Triangle), cudaMemcpyHostToDevice);	cudaMemcpy(cpuRay, ray, sizeof(Ray), cudaMemcpyHostToDevice);

	// Delete CPU allocations
	delete cpuTriangles;
	delete cpuRay;
}

TestData::~TestData()
{
	cudaFree(ray);
	cudaFree(triangles);
}