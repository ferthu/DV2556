#include "IntersectionTest.h"

std::vector<IntersectionResult> IntersectionTest::runTest(TestData* data)
{
	std::vector<IntersectionResult> resultVector;

	if (result != nullptr)
	{
		cudaFree(result);
	}

	cudaMalloc((void**) &result, data->triangleCount * sizeof(IntersectionResult));

	// start timer
	test(data);
	// end timer

	// collect results

	return resultVector;
}


IntersectionTest::~IntersectionTest()
{
	cudaFree(result);
}