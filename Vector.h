#ifndef VECTOR_H	
#define VECTOR_H

#include "Angle.h"

class Vector {
public:
	float I;
	float J;
	float K;

	Angle toAngle(void);
	void fromAngle(const Angle&);
	void rotate(const Angle&);
	void normalize(void);
	float magnitude(void);
	float dot(const Vector&);
	Vector cross(const Vector&);
	Vector(float=0, float=0, float=0);
};

#endif