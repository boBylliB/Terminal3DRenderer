// Type Class
#ifndef POINT_H	
#define POINT_H

#include "Vector.h"

class Point {
public:
	double x;
	double y;
	double z;

	Point(double = 0, double = 0, double = 0);

	Point operator+(const Point&) const;
	Point operator+(const Vector&) const;
	Point& operator+=(const Point&);
	Point operator-(const Point&) const;
	Point operator-(const Vector&) const;
	Point& operator-=(const Point&);
	Point operator*(const Point&) const;
	Point& operator*=(const Point&);
	Point operator/(const Point&) const;
	Point& operator/=(const Point&);
};

#endif