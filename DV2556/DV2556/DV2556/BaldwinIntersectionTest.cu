#include "BaldwinIntersectionTest.h"
#include "DefineFuncs.h"

void BaldwinIntersectionTest::test(TestData* data)
{
	baldwinTest<<<NUM_BLOCKS, NUM_THREADSPERBLOCK>>>(data->triangles, data->baldwinTransformations, data->ray, data->triangleCount, result);
}

__global__ void baldwinTest(Triangle* triangles, BaldwinTransformation* transformations, Ray* ray, size_t triangleCount, IntersectionResult* resultArray)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < triangleCount)
	{
		float t;
		if (baldwinIntersectTriangle(*ray, triangles[index], transformations[index], &t))
		{
			resultArray[index].hit = true;
			resultArray[index].distance = t;
		}
		else
		{
			resultArray[index].hit = false;
		}
		index += stride;
	}
	return;
}

__device__ int baldwinIntersectTriangle(Ray ray, Triangle tri, BaldwinTransformation trf, float * t)
{
	vec3 dir = ray.direction;
	vec3 org = ray.origin;

	vec3 v0 = tri[0];
	vec3 v1 = tri[1];
	vec3 v2 = tri[2];

	float transformation[12];// = trf.transformation;

	for (int i = 0; i < 12; i++)
	{
		transformation[i] = trf.transformation[i];
	}

	// Get barycentric z components of ray origin and direction for calculation of t value

	const float transS = transformation[8] * org[0] + transformation[9] * org[1] + transformation[10] * org[2] + transformation[11];
	const float transD = transformation[8] * dir[0] + transformation[9] * dir[1] + transformation[10] * dir[2];

	const float ta = -transS / transD;


	// Reject negative t values and rays parallel to triangle
	if (ta < 0.0f)
		return 0;


	// Get global coordinates of ray's intersection with triangle's plane.
	const vec3 wr(org[0] + ta * dir[0], org[1] + ta * dir[1], org[2] + ta * dir[2]);


	// Calculate "x" and "y" barycentric coordinates
	const float xg = transformation[0] * wr[0] + transformation[1] * wr[1] + transformation[2] * wr[2] + transformation[3];
	const float yg = transformation[4] * wr[0] + transformation[5] * wr[1] + transformation[6] * wr[2] + transformation[7];


	// Final intersection test
	if (xg >= 0.0f  &&  yg >= 0.0f  &&  yg + xg < 1.0f)
	{
		*t = ta;
		return 1;
	}

	return 0;
}
