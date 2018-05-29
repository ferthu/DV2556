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

#define tFar   10000.0f
#define tNear  0.0000001f
__device__ int baldwinIntersectTriangle(Ray ray, Triangle tri, BaldwinTransformation trf, float * t)
{
	vec3 dir = ray.direction;
	vec3 org = ray.origin;

	vec3 v0 = tri[0];
	vec3 v1 = tri[1];
	vec3 v2 = tri[2];

	float xg, yg;                       // The barycentric coordinates

	if (trf.fixedColumn == 1)
	{

		const float transS = org[0] + trf.transformation[6] * org[1] + trf.transformation[7] * org[2] + trf.transformation[8];
		const float transD = dir[0] + trf.transformation[6] * dir[1] + trf.transformation[7] * dir[2];
		const float ta = -transS / transD;

		if (ta <= tNear || ta >= tFar)
			return 0;

		const float wr[3] = { org[0] + ta * dir[0], org[1] + ta * dir[1], org[2] + ta * dir[2] };

		xg = trf.transformation[0] * wr[1] + trf.transformation[1] * wr[2] + trf.transformation[2];
		yg = trf.transformation[3] * wr[1] + trf.transformation[4] * wr[2] + trf.transformation[5];

		*t = ta;
	}
	else if (trf.fixedColumn == 2)
	{

		const float transS = trf.transformation[6] * org[0] + org[1] + trf.transformation[7] * org[2] + trf.transformation[8];
		const float transD = trf.transformation[6] * dir[0] + dir[1] + trf.transformation[7] * dir[2];
		const float ta = -transS / transD;

		if (ta <= tNear || ta >= tFar)
			return 0;

		const float wr[3] = { org[0] + ta * dir[0], org[1] + ta * dir[1], org[2] + ta * dir[2] };

		xg = trf.transformation[0] * wr[0] + trf.transformation[1] * wr[2] + trf.transformation[2];
		yg = trf.transformation[3] * wr[0] + trf.transformation[4] * wr[2] + trf.transformation[5];

		*t = ta;
	}
	else if (trf.fixedColumn == 3)
	{

		const float transS = trf.transformation[6] * org[0] + trf.transformation[7] * org[1] + org[2] + trf.transformation[8];
		const float transD = trf.transformation[6] * dir[0] + trf.transformation[7] * dir[1] + dir[2];
		const float ta = -transS / transD;

		if (ta <= tNear || ta >= tFar)
			return 0;

		const float wr[3] = { org[0] + ta * dir[0], org[1] + ta * dir[1], org[2] + ta * dir[2] };

		xg = trf.transformation[0] * wr[0] + trf.transformation[1] * wr[1] + trf.transformation[2];
		yg = trf.transformation[3] * wr[0] + trf.transformation[4] * wr[1] + trf.transformation[5];

		*t = ta;
	}
	else {
		// Invalid fixed-column code, treat ray as missing triangle
		return 0;
	}


	// Final check of barycentric coordinates to see if intersection is inside or outside triangle

	if (xg >= 0.0f  &&  yg >= 0.0f  &&  yg + xg < 1.0f) {
		return 1;
	}

	return 0;
}
