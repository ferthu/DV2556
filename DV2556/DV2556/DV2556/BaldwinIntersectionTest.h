#pragma once
#include "IntersectionTest.h"
#include "TestData.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

class BaldwinIntersectionTest : public IntersectionTest
{
protected:
	void test(TestData* data) override;
};

__global__ void baldwinTest(Triangle* triangles, BaldwinTransformation* transformations, Ray* ray, size_t triangleCount, IntersectionResult* resultArray);