#include <cfloat>

#include "Watertight.h"
#include "DefineFuncs.h"

__device__
int sign_mask(float f)
{
	float n = 1.0f;
	float nn = -n;

	int* pn = reinterpret_cast<int*>(&n);
	int* pnn = reinterpret_cast<int*>(&nn);

	// mask is the bit difference of n and nn, only the sign bit is 1, the rest are 0
	int mask = *pn ^ *pnn;

	int* pf = reinterpret_cast<int*>(&f);

	// Return the sign bit of f
	return *pf & mask;
}

__device__
float xorf(float f, int mask)
{
	int* pf = reinterpret_cast<int*>(&f);

	int res = *pf ^ mask;

	return *reinterpret_cast<float*>(&res);
}

__device__
vec3 abs(vec3 a)
{
	vec3 res;
	res.x = (a.x > 0.0f) ? a.x : -a.x;
	res.y = (a.y > 0.0f) ? a.y : -a.y;
	res.z = (a.z > 0.0f) ? a.z : -a.z;
	return res;
}

__device__
int max_dim(vec3 a)
{
	int max_dim = 0;

	if (a[max_dim] < a[1])
		max_dim = 1;

	if (a[max_dim] < a[2])
		max_dim = 2;

	return max_dim;
}

void Watertight::test(TestData* data)
{
	watertightTest<<<NUM_BLOCKS, NUM_THREADSPERBLOCK>>>(data->triangles, data->ray, data->triangleCount, result);
}

//#define BACKFACE_CULLING

__global__ void watertightTest(Triangle* triangles, Ray* ray, size_t triangleCount, IntersectionResult* resultArray)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < triangleCount)
	{
		float t;
		if (watertightIntersectTriangle(*ray, triangles[index], &t))
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

__device__ int watertightIntersectTriangle(Ray ray, Triangle tri, float* t)
{
	vec3 dir = ray.direction;
	vec3 org = ray.origin;

	// Calculate dimension where the ray direction is maximal
	int kz = max_dim(abs(dir));
	int kx = kz + 1; if (kx == 3) kx = 0;
	int ky = kx + 1; if (ky == 3) ky = 0;

	// Swap kx and ky dimension to preserve winding direction of triangles
	if (dir[kz] < 0.0f)
	{
		int temp = kx;
		kx = ky;
		ky = temp;
	}

	// Calculate shear constants
	float Sx = dir[kx] / dir[kz];
	float Sy = dir[ky] / dir[kz];
	float Sz = 1.0f / dir[kz];

	// Calculate vertices relative to ray origin
	vec3 A; SUB(A, tri[0], org);
	vec3 B; SUB(B, tri[1], org);
	vec3 C; SUB(C, tri[2], org);

	// Perform shear and scale of vertices
	const float Ax = A[kx] - Sx * A[kz];
	const float Ay = A[ky] - Sy * A[kz];
	const float Bx = B[kx] - Sx * B[kz];
	const float By = B[ky] - Sy * B[kz];
	const float Cx = C[kx] - Sx * C[kz];
	const float Cy = C[ky] - Sy * C[kz];

	// Calculate scaled barycentric coordinates
	float U = Cx * By - Cy * Bx;
	float V = Ax * Cy - Ay * Cx;
	float W = Bx * Ay - By * Ax;

#ifdef BACKFACE_CULLING
	if (U < 0.0f || V < 0.0f || W < 0.0f) return 0;
#else
	if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f)) return 0;
#endif

	// Fallback to test against edges using double precision
	if (U == 0.0f || V == 0.0f || W == 0.0f)
	{
		double CxBy = (double)Cx * (double)By;
		double CyBx = (double)Cy * (double)Bx;
		U = (float)(CxBy - CyBx);

		double AxCy = (double)Ax * (double)Cy;
		double AyCx = (double)Ay * (double)Cx;
		V = (float)(AxCy - AyCx);

		double BxAy = (double)Bx * (double)Ay;
		double ByAx = (double)By * (double)Ax;
		W = (float)(BxAy - ByAx);

#ifdef BACKFACE_CULLING
		if (U < 0.0f || V < 0.0f || W < 0.0f) return 0;
#else
		if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f)) return 0;
#endif
	}

	// Calculate determinant
	float det = U + V + W;
	if (det == 0.0f) return 0;

	// Calculate scaled z-coordinates of vertices and use them to calculate the hit distance
	const float Az = Sz * A[kz];
	const float Bz = Sz * B[kz];
	const float Cz = Sz * C[kz];
	const float T = U * Az + V * Bz + W * Cz;

#ifdef BACKFACE_CULLING
	if (T < 0.0f /*|| T > res.distance * det*/) return 0;
#else
	int det_sign = sign_mask(det);
	if (xorf(T, det_sign) < 0.0f /*|| xorf(T, det_sign) > res.distance * xorf(det, det_sign)*/) return 0;
#endif

	// normalize U, V, W and T
	const float rcpDet = 1.0f / det;
	//hit.u = U * rcpDet;
	//hit.v = V * rcpDet;
	//hit.w = W * rcpDet;
	*t = T * rcpDet;
	return 1;
}

