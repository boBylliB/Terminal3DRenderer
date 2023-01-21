// Container Class
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Point.h"
#include "Vector.cuh"

class Triangle {
public:
	Point verts[3];
	Vector normal;

	Triangle(void);
	Triangle(const Point[3]);

	__host__ __device__ bool checkWithin(Vector, const Point&) const;
	void calculateNormal(void);
};

#endif