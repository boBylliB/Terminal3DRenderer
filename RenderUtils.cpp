#ifndef RENDERUTILS_CPP
#define RENDERUTILS_CPP

#include <cmath>
#include "RenderUtils.h"

using namespace std;

double degToRad(double deg) {
	return (deg / 180.0) * atan(1.0) * 4.0;
}
double radToDeg(double rad)
{
	return (rad / (atan(1.0) * 4.0)) * 180.0;
}
float stringToFloat(string str) {
	float output = 0;
	int decimalPt = str.find('.');
	string intPortion;
	string decPortion;

	if (decimalPt != string::npos) {
		intPortion = str.substr(0, decimalPt);
		decPortion = str.substr(decimalPt + 1, str.length());
	}
	else {
		intPortion = str;
		decPortion = "0";
	}
	
	int intValue = stringToInt(intPortion);
	int decValue = stringToInt(decPortion);

	output = intValue + decValue * pow(10, -1.0 * (float)decPortion.length());

	return output;
}
double stringToDouble(string str) {
	double output = 0;
	int decimalPt = str.find('.');
	string intPortion;
	string decPortion;

	if (decimalPt != string::npos) {
		intPortion = str.substr(0, decimalPt);
		decPortion = str.substr(decimalPt + 1, str.length());
	}
	else {
		intPortion = str;
		decPortion = "0";
	}

	int intValue = stringToInt(intPortion);
	int decValue = stringToInt(decPortion);

	output = intValue + decValue * pow(10, -1.0 * (double)decPortion.length());

	return output;
}
int stringToInt(string str) {
	int output = 0;
	bool negative = false;

	for (int idx = 0; idx < str.length(); idx++) {
		int digitCalc = pow(10, str.length() - idx - 1);

		switch (str.at(idx)) {
		case '-':
			negative = !negative;
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

	return output;
}
int intPart(double d) {
	return (int)d;
}
double decPart(double d) {
	return d - intPart(d);
}

#endif