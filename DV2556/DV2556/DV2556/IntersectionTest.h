#pragma once
#include <string>
#include "TestData.h"
#include "StopWatch.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

// Parent class of individual tests
#define NUM_BLOCKS 256
#define NUM_THREADSPERBLOCK 256

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

struct TestResult
{
	float duration = 0.0f;
	std::vector<IntersectionResult> intersectionResults;
};

class IntersectionTest
{
public:
	// Calls collectData, then starts timer and calls test(), writes results to file
	TestResult runTest(TestData* data);

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