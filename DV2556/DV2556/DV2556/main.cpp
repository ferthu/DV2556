#include "TestData.h"
#include "IntersectionTest.h"
#include "Watertight.h"
#include "DefineFuncs.h"
#include "MollerIntersectionTest.h"
#include "BaldwinIntersectionTest.h"

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

void saveTestResult(TestResult &res, std::string fileName, int iter)
{
	std::fstream fs;
	fs.open(fileName, std::fstream::out | std::fstream::app);
	fs << iter << "\t" << calcHitrate(res.intersectionResults) << "\t" << res.duration << std::endl;
	fs.close();
}

void resetFile(std::string fileName)
{
	std::fstream fs;
	fs.open(fileName, std::fstream::out | std::fstream::trunc);
	fs << "Iter\tHitrate\tDuration" << std::endl;
	fs.close();
}

void saveUploadTimes(TestData &data, std::string fileName)
{
	std::fstream fs;
	fs.open(fileName, std::fstream::out);
	fs << "Triangles\tBaldwin Transformations" << std::endl;
	fs << data.triangleUploadTime << "\t" << data.baldwinTransformationUploadTime << std::endl;
	fs.close();
}

#define NUMBER_OF_RUNS 10

int main()
{
	// Warmup
	TestData* data = new TestData(0.1f, 500000);

	Watertight warmup;
	TestResult warmupRes = warmup.runTest(data);

	delete data;

	// Real test
	data = new TestData(0.1f, 10000000);
	saveUploadTimes(*data, "UploadTimes.txt");

	//Clear previous results in result files
	resetFile("WatertightResult.txt");
	resetFile("MollerResult.txt");
	resetFile("BaldwinResult.txt");

	for (int i = 0; i < NUMBER_OF_RUNS; i++)
	{
		Watertight wt;
		TestResult res = wt.runTest(data);

		printf("Watertight\n Hitrate: %f\n Time: %f s\n\n", calcHitrate(res.intersectionResults), res.duration);
		saveTestResult(res, "WatertightResult.txt", i);

		MollerIntersectionTest moller;
		TestResult resMoller = moller.runTest(data);

		printf("Moller\n Hitrate: %f\n Time: %f s\n\n", calcHitrate(resMoller.intersectionResults), resMoller.duration);
		saveTestResult(resMoller, "MollerResult.txt", i);

		BaldwinIntersectionTest baldwin;
		TestResult resBaldwin = baldwin.runTest(data);

		printf("Baldwin\n Hitrate: %f\n Time: %f s\n\n", calcHitrate(resBaldwin.intersectionResults), resBaldwin.duration);
		saveTestResult(resBaldwin, "BaldwinResult.txt", i);
	}
	getchar();

	delete data;
}