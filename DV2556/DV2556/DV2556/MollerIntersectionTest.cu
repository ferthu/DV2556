#include "MollerIntersectionTest.h"
#include "DefineFuncs.h"

#define NUM_BLOCKS 32
#define NUM_THREADSPERBLOCK 256


void MollerIntersectionTest::test(TestData * data)
{
	GPUMollerIntersectionTests << <NUM_BLOCKS, NUM_THREADSPERBLOCK >> > (data->ray, data->triangles, result, data->triangleCount);
}

__global__ void GPUMollerIntersectionTests(Ray * ray, Triangle * tri, IntersectionResult* res, size_t triCount)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < triCount)
	{
		float t;
		if (MollerIntersectTriangle(*ray, tri[index], &t) == 1)
		{
			res[index].hit = true;
			res[index].distance = t;
		}
		else
		{
			res[index].hit = false;
		}
		index += stride;
	}
	return ;
}

//Intersection test functions on GPU and defines
#define EPSILON 1e-5
#define TEST_CULL
__device__ int MollerIntersectTriangle(Ray ray, Triangle tri, float * t)
{
	vec3 edge1, edge2, tvec, pvec, qvec;
	float det, inv_det , u, v;

	//find vecotrs fo two edges sharing vert0
	SUB(edge1, tri.vertices[1], tri.vertices[0]);
	SUB(edge2, tri.vertices[2], tri.vertices[0]);

	//begin calculationg determinant
	CROSS(pvec, ray.direction, edge2);

	//if determinant is near zero, ray lies in plane of triangle
	det = DOT(edge1, pvec);

#ifdef TEST_CULL
	if (det < EPSILON)
		return 0;
	
	//calculate distance from vert0 to ray origin
	SUB(tvec, ray.origin, tri.vertices[0]);

	//calculate U parameter and test bounds
	u = DOT(tvec, pvec);
	if (u < 0.0f || u > det)
		return 0;

	//prepare to test V parameter
	CROSS(qvec, tvec, edge1);

	//calculate V parameter and test bounds
	v = DOT(ray.direction, qvec);
	if (v <0.0f || u + v > det)
		return 0;

	//calculate t, scale parameters, ray intersect triangle
	*t = DOT(edge2, qvec);
	inv_det = 1.0f / det;
	*t *= inv_det;

	//fixing u and v values if we want them
	//*u *= inv_det;
	//*v *= inv_det;
#else
	if (det > -EPSILON && det < EPSILON)
		return 0;
	inv_det = 1 / det;

	//calculate distance from vert0 to ray origin
	SUB(tvec, ray.origin, tri.vertices[0]);
	u = DOT(tvec, pvec);
	if (u < 0.0f || u > 1.0f)
		return 0;

	//prepare to test V parameter
	CROSS(qvec, tvec, edge1);

	//calculate V parameter and test bounds
	v = DOT(ray.direction, qvec) * inv_det;
	if (v < 0.0f || u + v > 1.0f)
		return 0;

	//calculate t, ray intersection triangle
	*t = DOT(edge2, qvec) * inv_det;
#endif
	return 1;
}
