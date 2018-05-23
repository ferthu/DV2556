#include "TestData.h"
#include "IntersectionTest.h"
#include "Watertight.h"
#include "DefineFuncs.h"
#include "MollerIntersectionTest.h"

#include <iostream>
#include <vector>
#include <fstream>


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

void saveTestResult(TestResult &res, std::string fileName)
{
	std::fstream fs;
	fs.open(fileName, std::fstream::out);
	fs << "#\tHitrate\tDuration" << std::endl;
	fs << "1\t" << calcHitrate(res.intersectionResults) << "\t" << res.duration << std::endl;
	fs.close();
}


int main()
{
	TestData data(0.1f, 100000);

	Watertight wt;

	TestResult res = wt.runTest(&data);

	printf("Watertight\n Hitrate: %f\n Time: %f s\n\n", calcHitrate(res.intersectionResults), res.duration);

	MollerIntersectionTest moller;
	TestResult resMoller = moller.runTest(&data);
	
	printf("Moller\n Hitrate: %f\n Time: %f s\n\ns", calcHitrate(resMoller.intersectionResults), resMoller.duration);
	saveTestResult(resMoller, "MöllerResult.txt");
	getchar();
}