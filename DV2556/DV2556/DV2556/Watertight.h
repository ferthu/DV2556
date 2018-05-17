#pragma once
#include "IntersectionTest.h"
#include "TestData.h"

class Watertight : public IntersectionTest
{
protected:
	 void test(TestData* data) override;
};