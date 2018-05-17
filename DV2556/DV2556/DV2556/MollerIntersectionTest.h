#include "IntersectionTest.h"
#include "TestData.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"


class MollerIntersectionTest : public IntersectionTest
{
public:
	void test(TestData* data);
};
__global__ void GPUMollerIntersectionTests(Ray* ray, Triangle* tri, IntersectionResult* res, size_t triCount);
__device__ int MollerIntersectTriangle(Ray ray, Triangle tri, float* t);