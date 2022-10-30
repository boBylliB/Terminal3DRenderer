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
	int decimalPt = -1;
	
	for (int idx = 0; idx < str.length(); idx++) {
		int digitCalc = pow(10, str.length() - idx - 1);

		switch (str.at(idx)) {
		case '-':
			negative = !negative;
			break;
		case '.':
			decimalPt = idx;
			break;
		case '1':
			output += 1 * digitCalc;
			break;
		case '2':
			output += 2 * digitCalc;
			break;
		case '3':
			output += 3 * digitCalc;
			break;
		case '4':
			output += 4 * digitCalc;
			break;
		case '5':
			output += 5 * digitCalc;
			break;
		case '6':
			output += 6 * digitCalc;
			break;
		case '7':
			output += 7 * digitCalc;
			break;
		case '8':
			output += 8 * digitCalc;
			break;
		case '9':
			output += 9 * digitCalc;
			break;
		}
	}

	if (negative)
		output *= -1;

	output *= pow(10, -1 * (int)(str.length() - decimalPt - 1));
	output = (intPart(output) / 10) + decPart(output);

	return output;
}
int intPart(float f) {
	return (int)f;
}
float decPart(float f) {
	return f - intPart(f);
}

#endif