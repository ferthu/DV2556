#include "StopWatch.h"

void StopWatch::start()
{
	s = clock();
}

float StopWatch::getTimeInSeconds()
{
	clock_t difference = clock() - s;
	return difference / CLOCKS_PER_SEC;
}

int StopWatch::getTimeInMilliSeconds()
{
	clock_t difference = clock() - s;
	return difference * 1000 / CLOCKS_PER_SEC;
}
