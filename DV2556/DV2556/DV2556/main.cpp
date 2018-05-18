#include "TestData.h"
#include "IntersectionTest.h"
#include "Watertight.h"
#include "DefineFuncs.h"

#include <iostream>
#include <vector>

int main()
{
	TestData data(0.1f, 1000);

	Watertight wt;

	std::vector<IntersectionResult> res = wt.runTest(&data);

	getchar();
}