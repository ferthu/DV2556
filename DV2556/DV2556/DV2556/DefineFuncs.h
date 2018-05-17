#pragma once
#define CROSS(dest, v1, v2) \
			dest.x = v1.y*v2.z-v1.z*v2.y; \
			dest.y = v1.z*v2.x-v1.x*v2.z; \
			dest.z = v1.x*v2.y-v1.y*v2.x;
#define DOT(v1,v2) (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z)
#define SUB(dest,v1,v2) \
			dest.x = v1.x - v2.x; \
			dest.y = v1.y - v2.y; \
			dest.z = v1.z - v2.z; 