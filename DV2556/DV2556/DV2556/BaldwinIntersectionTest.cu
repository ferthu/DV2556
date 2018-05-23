#include "BaldwinIntersectionTest.h"
#include "DefineFuncs.h"

__global__ void baldwinTest(Triangle* triangles, BaldwinTransformation* transformations, Ray* ray, size_t triangleCount, IntersectionResult* resultArray)
{
	size_t index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < triangleCount)
	{
		// Get triangle and ray data
		IntersectionResult res;
		res.hit = false;
		res.distance = FLT_MAX;
		vec3 dir = ray->direction;
		vec3 org = ray->origin;

		vec3 v0 = triangles[index][0];
		vec3 v1 = triangles[index][1];
		vec3 v2 = triangles[index][2];

		float* transformation = transformations[index].transformation;

		// Get barycentric z components of ray origin and direction for calculation of t value

		const float transS = transformation[8] * org[0] + transformation[9] * org[1] + transformation[10] * org[2] + transformation[11];
		const float transD = transformation[8] * dir[0] + transformation[9] * dir[1] + transformation[10] * dir[2];

		const float ta = -transS / transD;


		// Reject negative t values and rays parallel to triangle
		if (ta < 0.0f)
			return;


		// Get global coordinates of ray's intersection with triangle's plane.
		const vec3 wr(org[0] + ta * dir[0], org[1] + ta * dir[1], org[2] + ta * dir[2]);


		// Calculate "x" and "y" barycentric coordinates
		const float xg = transformation[0] * wr[0] + transformation[1] * wr[1] + transformation[2] * wr[2] + transformation[3];
		const float yg = transformation[4] * wr[0] + transformation[5] * wr[1] + transformation[6] * wr[2] + transformation[7];


		// final intersection test

		if (xg >= 0.0f  &&  yg >= 0.0f  &&  yg + xg < 1.0f)
		{
			res.distance = ta;
			res.hit = true;

			resultArray[index] = res;
		}

		return;
	}
}

void BaldwinIntersectionTest::test(TestData* data)
{
	const int threadsPerBlock = 256;
	size_t blocks = (data->triangleCount + threadsPerBlock - 1) / threadsPerBlock;
	baldwinTest << <blocks, threadsPerBlock >> >(data->triangles, data->baldwinTransformations, data->ray, data->triangleCount, result);
}
