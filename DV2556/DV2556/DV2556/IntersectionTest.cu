#include "IntersectionTest.h"

TestResult IntersectionTest::runTest(TestData* data)
{
	TestResult retResult;
	retResult.intersectionResults.resize(data->triangleCount);
	//std::vector<IntersectionResult> resultVector(data->triangleCount);

	//if (result != nullptr)
	//{
	//	cudaFree(result);
	//}
	
	if (result == nullptr)
		cudaMalloc((void**) &result, data->triangleCount * sizeof(IntersectionResult));

	// start timer
	StopWatch sw;
	sw.start();

	test(data);
	cudaDeviceSynchronize();
	// end timer
	retResult.duration = sw.getTimeInSeconds();
	// collect results
	cudaMemcpy(retResult.intersectionResults.data(), result, data->triangleCount * sizeof(IntersectionResult), cudaMemcpyKind::cudaMemcpyDeviceToHost);


	return retResult;
}


IntersectionTest::~IntersectionTest()
{
	cudaFree(result);
}