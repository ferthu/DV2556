#include "TestData.h"
#include "IntersectionTest.h"
#include "Watertight.h"
#include "DefineFuncs.h"
#include "MollerIntersectionTest.h"

#include <iostream>
#include <vector>


float calcHitrate(std::vector<IntersectionResult> &res)
{
	float sum = 0.0f;
	for (unsigned int i = 0; i < res.size(); i++)
	{
		if (res[i].hit)
			sum += 1;
	}
	return sum / ((float)res.size());
}


int main()
{
	TestData data(0.1f, 1000);

	Watertight wt;

	std::vector<IntersectionResult> res = wt.runTest(&data);

	printf("Watertight hitrate: %f\n", calcHitrate(res));

	MollerIntersectionTest moller;
	std::vector<IntersectionResult> resMoller = moller.runTest(&data);
	
	printf("Moller hitrate: %f\n", calcHitrate(resMoller));

	getchar();
}