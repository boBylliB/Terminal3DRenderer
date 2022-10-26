#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Point.h"
#include "Vector.h"

class Triangle {
public:
	Point verts[3];
	Vector normal;
	float D;

	bool checkWithin(const Point&);
	void calculateNormal(void);
	void calculateD(void);
	Triangle(const Point[3]);
};

#endif