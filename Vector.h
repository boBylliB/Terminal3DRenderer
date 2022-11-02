#ifndef VECTOR_H	
#define VECTOR_H

#include "Angle.h"

class Vector {
public:
	double I;
	double J;
	double K;

	Angle toAngle(void);
	void fromAngle(const Angle&);
	void rotate(const Angle&);
	void normalize(void);
	double magnitude(void);
	double dot(const Vector&);
	Vector cross(const Vector&);
	Vector(double=0, double=0, double=0);
};

#endif