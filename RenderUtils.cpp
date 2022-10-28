#ifndef RENDERUTILS_CPP
#define RENDERUTILS_CPP

#include <cmath>
#include "RenderUtils.h"

using namespace std;

float degToRad(float deg) {
	return (deg / 180.0) * atan(1.0) * 4.0;
}
float radToDeg(float rad)
{
	return (rad / (atan(1.0) * 4.0)) * 180.0;
}
float stringToFloat(string str) {
	float output = 0;
	bool negative = false;
	if (str.length() > 0 && str.at(0) == '-')
		negative = true;


}

#endif