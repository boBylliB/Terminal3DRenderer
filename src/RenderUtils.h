#ifndef RENDERUTILS_H
#define RENDERUTILS_H

#include <string>
#include <vector>
#include <iostream>

double degToRad(double);
double radToDeg(double);
void trimString(std::string&);
float stringToFloat(std::string);
double stringToDouble(std::string);
int stringToInt(std::string);
int intPart(double);
double decPart(double);
void printLargeDouble(double, int = 2);
void printLargeInt(int);
std::vector<std::string> getLargeDigit(int);

#endif