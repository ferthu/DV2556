#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>

// Contains all data that any test will need. Individual tests will create their own rays 
// and triangles containing only the data they need

struct vec3
{
	union
	{
		struct
		{
			float x, y, z;
		};
		float coords[3];
	};
	__host__ __device__
	vec3()
	{
		x = y = z = 0.0f;
	}
	__host__ __device__
	vec3(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}
	__host__ __device__
	vec3(float val)
	{
		x = y = z = val;
	}
	__host__ __device__
	float& operator[](int i)
	{
		return coords[i];
	}

	__host__ __device__
	const float& operator[](int i) const
	{
		return coords[i];
	}
};


__host__ __device__ float dot(vec3 a, vec3 b);

__host__ __device__ vec3 cross(vec3 a, vec3 b);

struct Triangle
{
	vec3 vertices[3];

	__host__ __device__
	vec3& operator[](int i)
	{
		return vertices[i];
	}

	__host__ __device__
	const vec3& operator[](int i) const
	{
		return vertices[i];
	}
};

struct Ray
{
	vec3 origin;
	vec3 direction;
};

class TestData
{
public:
	TestData() = delete;
	TestData(float hitrate, size_t triangleCount);
	~TestData();

	// Pointers to GPU data
	Ray* ray;
	Triangle* triangles;
	size_t triangleCount;
};