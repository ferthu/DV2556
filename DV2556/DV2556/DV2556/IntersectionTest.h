#pragma once
#include <string>
#include "TestData.h"

// Parent class of individual tests

struct IntersectionResult
{
	bool hit;
	float distance;
	// ...
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