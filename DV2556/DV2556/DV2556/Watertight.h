#pragma once
#include "IntersectionTest.h"
#include "TestData.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

class Watertight : public IntersectionTest
{
protected:
	 void test(TestData* data) override;
};

__global__ void watertightTest(Triangle* triangles, Ray* ray, size_t triangleCount, IntersectionResult* resultArray);