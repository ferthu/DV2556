#include "TestData.h"
#include "DefineFuncs.h"
#include <iostream>

#define RANDMAX 20
#define RANDMIN -RANDMAX
#define FLOATRAND RANDMIN + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX) / (RANDMAX - RANDMIN))

TestData::TestData(float hitrate, size_t triangleCount)
{
	this->triangleCount = triangleCount;

	Triangle* cpuTriangles = new Triangle[triangleCount];
	BaldwinTransformation* cpuBaldwinTransformations = new BaldwinTransformation[triangleCount];
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

		// Precalculate data for Baldwin test

		// Build transformation from global to barycentric coordinates.
		vec3 v0 = cpuTriangles[i].vertices[0];
		vec3 v1 = cpuTriangles[i].vertices[1];
		vec3 v2 = cpuTriangles[i].vertices[2];

		// Calculate edges and normal
		vec3 edge1;
		SUB(edge1, v1, v0);

		vec3 edge2;
		SUB(edge2, v2, v0);

		vec3 normal;
		CROSS(normal, edge1, edge2);

		float x1, x2;
		float num = DOT(v0, normal);                 // Element (3,4) of each transformation matrix

		if (fabs(normal[0]) > fabs(normal[1]) && fabs(normal[0]) > fabs(normal[2]))
		{
			x1 = v1[1] * v0[2] - v1[2] * v0[1];
			x2 = v2[1] * v0[2] - v2[2] * v0[1];

			// Do matrix set up here for when a = 1, b = c = 0 formula

			cpuBaldwinTransformations[i].transformation[0] = 0.0f;
			cpuBaldwinTransformations[i].transformation[1] = edge2[2] / normal[0];
			cpuBaldwinTransformations[i].transformation[2] = -edge2[1] / normal[0];
			cpuBaldwinTransformations[i].transformation[3] = x2 / normal[0];

			cpuBaldwinTransformations[i].transformation[4] = 0.0f;
			cpuBaldwinTransformations[i].transformation[5] = -edge1[2] / normal[0];
			cpuBaldwinTransformations[i].transformation[6] = edge1[1] / normal[0];
			cpuBaldwinTransformations[i].transformation[7] = -x1 / normal[0];

			cpuBaldwinTransformations[i].transformation[8] = 1.0f;
			cpuBaldwinTransformations[i].transformation[9] = normal[1] / normal[0];
			cpuBaldwinTransformations[i].transformation[10] = normal[2] / normal[0];
			cpuBaldwinTransformations[i].transformation[11] = -num / normal[0];
		}
		else if (fabs(normal[1]) > fabs(normal[2]))
		{
			x1 = v1[2] * v0[0] - v1[0] * v0[2];
			x2 = v2[2] * v0[0] - v2[0] * v0[2];

			// b = 1 case

			cpuBaldwinTransformations[i].transformation[0] = -edge2[2] / normal[1];
			cpuBaldwinTransformations[i].transformation[1] = 0.0f;
			cpuBaldwinTransformations[i].transformation[2] = edge2[0] / normal[1];
			cpuBaldwinTransformations[i].transformation[3] = x2 / normal[1];
			
			cpuBaldwinTransformations[i].transformation[4] = edge1[2] / normal[1];
			cpuBaldwinTransformations[i].transformation[5] = 0.0f;
			cpuBaldwinTransformations[i].transformation[6] = -edge1[0] / normal[1];
			cpuBaldwinTransformations[i].transformation[7] = -x1 / normal[1];
			
			cpuBaldwinTransformations[i].transformation[8] = normal[0] / normal[1];
			cpuBaldwinTransformations[i].transformation[9] = 1.0f;
			cpuBaldwinTransformations[i].transformation[10] = normal[2] / normal[1];
			cpuBaldwinTransformations[i].transformation[11] = -num / normal[1];
		}
		else if (fabs(normal[2]) > 0.0f)
		{
			x1 = v1[0] * v0[1] - v1[1] * v0[0];
			x2 = v2[0] * v0[1] - v2[1] * v0[0];

			// c = 1 case

			cpuBaldwinTransformations[i].transformation[0] = edge2[1] / normal[2];
			cpuBaldwinTransformations[i].transformation[1] = -edge2[0] / normal[2];
			cpuBaldwinTransformations[i].transformation[2] = 0.0f;
			cpuBaldwinTransformations[i].transformation[3] = x2 / normal[2];
			
			cpuBaldwinTransformations[i].transformation[4] = -edge1[1] / normal[2];
			cpuBaldwinTransformations[i].transformation[5] = edge1[0] / normal[2];
			cpuBaldwinTransformations[i].transformation[6] = 0.0f;
			cpuBaldwinTransformations[i].transformation[7] = -x1 / normal[2];
			
			cpuBaldwinTransformations[i].transformation[8] = normal[0] / normal[2];
			cpuBaldwinTransformations[i].transformation[9] = normal[1] / normal[2];
			cpuBaldwinTransformations[i].transformation[10] = 1.0f;
			cpuBaldwinTransformations[i].transformation[11] = -num / normal[2];
		}
		else
		{
			return;
		}
	}

	// Keeping the cudaMallocs and cudaMemcpys in separate functions seems to prevent them from failing
	// Triangles
	prepareTriangles(cpuTriangles);

	// Baldwin triangles
	prepareBaldwinTransformations(cpuBaldwinTransformations);

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

void TestData::prepareBaldwinTransformations(BaldwinTransformation* cpuBaldwinTransformations)
{
	cudaMalloc((void**)&baldwinTransformations, triangleCount * sizeof(BaldwinTransformation));	cudaMemcpy(baldwinTransformations, cpuBaldwinTransformations, triangleCount * sizeof(BaldwinTransformation), cudaMemcpyKind::cudaMemcpyHostToDevice);
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