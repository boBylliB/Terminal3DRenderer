#ifndef RENDERMATH_CPP
#define RENDERMATH_CPP

#include <cmath>
#include "RenderMath.h"

float degToRad(float deg) {
	return (deg / 180.0) * atan(1.0) * 4.0;
}
double radToDeg(double rad)
{
	return (rad / (atan(1.0) * 4.0)) * 180.0;
}

#endif