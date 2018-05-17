#pragma once
#include <string>
#include "TestData.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

// Parent class of individual tests

struct IntersectionResult
{
	bool hit;
	float distance;
	// ...

	__host__ __device__
	IntersectionResult& operator=(const IntersectionResult& rhs)
	{
		hit = rhs.hit;
		distance = rhs.distance;
		return *this;
	}
};

class IntersectionTest
{
public:
	// Calls collectData, then starts timer and calls test(), writes results to file
	std::vector<IntersectionResult> runTest(TestData* data);

	virtual ~IntersectionTest();

protected:
	// Collects relevant data from the TestData and stores it in the class
	// Probably not needed
	//virtual void collectData(TestData* data) = 0;

	// test() stores result here
	IntersectionResult* result = nullptr;

	// Runs the ray-triangle intersection test
	virtual void test(TestData* data) = 0;
};