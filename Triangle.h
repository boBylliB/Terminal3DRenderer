// Container Class
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Point.h"
#include "Vector.h"

class Triangle {
public:
	Point verts[3];
	Vector normal;

	Triangle(void);
	Triangle(const Point[3]);

	bool checkWithin(Vector, const Point&) const;
	void calculateNormal(void);
};

#endif