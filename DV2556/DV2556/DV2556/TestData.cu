#include "TestData.h"
#include <iostream>

#define RANDMAX 20
#define RANDMIN -RANDMAX
#define FLOATRAND RANDMIN + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX) / (RANDMAX - RANDMIN))

TestData::TestData(float hitrate, size_t triangleCount)
{
	this->triangleCount = triangleCount;

	Triangle* cpuTriangles = new Triangle[triangleCount];
	// generate ray
	Ray* cpuRay = new Ray();
	cpuRay->origin = vec3(0.0f);
	cpuRay->direction = vec3(0.0f, 0.0f, 1.0f);
	// generate triangles (Disregarding hitrate for now)
	srand(1);
	for (size_t i = 0; i < triangleCount; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cpuTriangles[i].vertices[j] = vec3(FLOATRAND, FLOATRAND, FLOATRAND);
		}
	}

	// Keeping the cudaMallocs and cudaMemcpys in separate functions seems to prevent them from failing
	// Triangles
	prepareTriangles(cpuTriangles);

	// Ray
	prepareRay(cpuRay);

	// Delete CPU allocations
	delete cpuTriangles;
	delete cpuRay;
}

TestData::~TestData()
{
	cudaFree(ray);
	cudaFree(triangles);
}

void TestData::prepareTriangles(Triangle* cpuTriangles)
{
	cudaMalloc((void**)&triangles, triangleCount * sizeof(Triangle));	cudaMemcpy(triangles, cpuTriangles, triangleCount * sizeof(Triangle), cudaMemcpyKind::cudaMemcpyHostToDevice);
}

void TestData::prepareRay(Ray* cpuRay)
{
	cudaMalloc((void**)&ray, sizeof(Ray));	cudaMemcpy(ray, cpuRay, sizeof(Ray), cudaMemcpyKind::cudaMemcpyHostToDevice);
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