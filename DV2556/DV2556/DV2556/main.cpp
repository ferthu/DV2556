#include "TestData.h"
#include "IntersectionTest.h"
#include "Watertight.h"
#include "DefineFuncs.h"

#include <iostream>
#include <vector>

int main()
{
	TestData data(0.1f, 10);

	Watertight wt;

	std::vector<IntersectionResult> res = wt.runTest(&data);

	getchar();

	// Create IntersectionTests and call runTest() with &data
}