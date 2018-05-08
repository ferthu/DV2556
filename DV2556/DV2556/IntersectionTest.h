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

protected:
	// Collects relevant data from the TestData and stores it in the class
	virtual void collectData(TestData* data) = 0;

	// Runs the ray-triangle intersection test
	virtual std::vector<IntersectionResult> test() = 0;
};