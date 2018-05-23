#include "TestData.h"
#include "IntersectionTest.h"
#include "Watertight.h"
#include "DefineFuncs.h"
#include "MollerIntersectionTest.h"
#include "BaldwinIntersectionTest.h"

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
	TestData data(0.1f, 5000000);

	Watertight wt;
	TestResult res = wt.runTest(&data);

	printf("Watertight\n Hitrate: %f\n Time: %f s\n\n", calcHitrate(res.intersectionResults), res.duration);

	MollerIntersectionTest moller;
	TestResult resMoller = moller.runTest(&data);
	
	printf("Moller\n Hitrate: %f\n Time: %f s\n\n", calcHitrate(resMoller.intersectionResults), resMoller.duration);

	BaldwinIntersectionTest baldwin;
	TestResult resBaldwin = baldwin.runTest(&data);

	printf("Baldwin\n Hitrate: %f\n Time: %f s\n\n", calcHitrate(resBaldwin.intersectionResults), resBaldwin.duration);

	getchar();
}