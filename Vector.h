#ifndef VECTOR_H	
#define VECTOR_H

#include "Point.h"
#include "Angle.h"

class Vector {
public:
	float I;
	float J;
	float K;

	Angle toAngle(void);
	void fromAngle(Angle);
	void rotate(Angle);
	float magnitude(void);
	float dot(Vector);
	Vector cross(Vector);
	Vector(float=1, float=1, float=1);
};

#endif