#ifndef VECTOR_H	
#define VECTOR_H

#include "Point.h"
#include "Angle.h"

class Vector {
private:
	float I;
	float J;
	float K;
public:
	void setValues(float i, float j, float k);
	float getValues(void);
	Angle toAngle(void);
	void fromAngle(Angle);
	void rotate(Angle);
	float magnitude(void);
	Vector dot(Vector);
	Vector cross(Vector);
};

#endif