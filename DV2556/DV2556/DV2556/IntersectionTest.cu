#include "IntersectionTest.h"

std::vector<IntersectionResult> IntersectionTest::runTest(TestData* data)
{
	std::vector<IntersectionResult> resultVector(data->triangleCount);

	if (result != nullptr)
	{
		cudaFree(result);
	}

	cudaMalloc((void**) &result, data->triangleCount * sizeof(IntersectionResult));

	// start timer
	test(data);
	cudaDeviceSynchronize();
	// end timer

	// collect results
	cudaMemcpy(resultVector.data(), result, data->triangleCount * sizeof(IntersectionResult), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	return resultVector;
}


IntersectionTest::~IntersectionTest()
{
	cudaFree(result);
}