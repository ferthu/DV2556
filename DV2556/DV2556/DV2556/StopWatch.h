#pragma once
#include <time.h>
class StopWatch
{
private:
	clock_t s;
public:
	void start();
	float getTimeInSeconds();
	int getTimeInMilliSeconds();
};