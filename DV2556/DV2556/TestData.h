#pragma once
#include <vector>

// Contains all data that any test will need. Individual tests will create their own rays 
// and triangles containing only the data they need

struct Triangle
{

};

struct Ray
{

};

class TestData
{
public:
	TestData() = delete;
	TestData(float hitrate);
	TestData();

	Ray ray;
	std::vector<Triangle> triangles;
};