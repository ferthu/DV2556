#include "IntersectionTest.h"

std::vector<IntersectionResult> IntersectionTest::runTest(TestData* data)
{
	std::vector<IntersectionResult> result;

	// start timer
	test(data);
	// end timer

	return result;
}
