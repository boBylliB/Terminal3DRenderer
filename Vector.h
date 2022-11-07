#ifndef VECTOR_H	
#define VECTOR_H

#include "Angle.h"
#include "Point.h"

class Vector {
public:
	double I;
	double J;
	double K;

	Angle toAngle(void);
	void fromAngle(const Angle&);
	void rotate(const Angle&);
	void normalize(void);
	void scale(const double);
	double magnitude(void);
	double dot(const Vector&);
	Vector cross(const Vector&);
	void fromPoint(const Point&);
	void difference(const Point&, const Point&);
	Point toPoint(void);
	Vector(double=0, double=0, double=0);
};

#endif