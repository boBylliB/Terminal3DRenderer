#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Point.h"
#include "Vector.h"

class Triangle {
public:
	Point verts[3];
	Vector normal;
	double D; // D is the dot product of the normal and any point on the triangle, used for ray intersection

	bool checkWithin(const Point&);
	void calculateNormal(void);
	void calculateD(void);
	Triangle(const Point[3]);
};

#endif