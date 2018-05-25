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

void saveTestResult(TestResult &res, std::string fileName)
{
	std::fstream fs;
	fs.open(fileName, std::fstream::out);
	fs << "#\tHitrate\tDuration" << std::endl;
	fs << "1\t" << calcHitrate(res.intersectionResults) << "\t" << res.duration << std::endl;
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

int main()
{
	// Warmup
	TestData* data = new TestData(0.1f, 5000000);

	Watertight warmup;
	TestResult warmupRes = warmup.runTest(data);

	delete data;

	// Real test
	data = new TestData(0.1f, 50000000);
	saveUploadTimes(*data, "UploadTimes.txt");

	Watertight wt;
	TestResult res = wt.runTest(data);

	printf("Watertight\n Hitrate: %f\n Time: %f s\n\n", calcHitrate(res.intersectionResults), res.duration);
	saveTestResult(res, "WatertightResult.txt");

	MollerIntersectionTest moller;
	TestResult resMoller = moller.runTest(data);
	
	printf("Moller\n Hitrate: %f\n Time: %f s\n\n", calcHitrate(resMoller.intersectionResults), resMoller.duration);
	saveTestResult(resMoller, "M�llerResult.txt");

	BaldwinIntersectionTest baldwin;
	TestResult resBaldwin = baldwin.runTest(data);

	printf("Baldwin\n Hitrate: %f\n Time: %f s\n\n", calcHitrate(resBaldwin.intersectionResults), resBaldwin.duration);
	saveTestResult(resBaldwin, "BaldwinResult.txt");
	getchar();

	delete data;
}