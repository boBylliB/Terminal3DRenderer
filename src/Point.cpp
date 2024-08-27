#ifndef POINT_CPP
#define POINT_CPP

#include "Point.h"

Point::Point(double x, double y, double z) {
	this->x = x;
	this->y = y;
	this->z = z;
}

Point Point::operator+(const Point& pt) const {
	Point output(pt.x + x, pt.y + y, pt.z + z);

	return output;
}
Point Point::operator+(const Vector& vec) const {
	Point output(vec.I + x, vec.J + y, vec.K + z);

	return output;
}
Point& Point::operator+=(const Point& pt) {
	x += pt.x;
	y += pt.y;
	z += pt.z;

	return *this;
}
Point Point::operator-(const Point& pt) const {
	Point output(x - pt.x, y - pt.y, z - pt.z);

	return output;
}
Point Point::operator-(const Vector& vec) const {
	Point output(x - vec.I, y - vec.J, z - vec.K);

	return output;
}
Point& Point::operator-=(const Point& pt) {
	x -= pt.x;
	y -= pt.y;
	z -= pt.z;

	return *this;
}
Point Point::operator*(const Point& pt) const {
	Point output(x * pt.x, y * pt.y, z * pt.z);

	return output;
}
Point& Point::operator*=(const Point& pt) {
	x *= pt.x;
	y *= pt.y;
	z *= pt.z;

	return *this;
}
Point Point::operator/(const Point& pt) const {
	Point output(x / pt.x, y / pt.y, z / pt.z);

	return output;
}
Point& Point::operator/=(const Point& pt) {
	x /= pt.x;
	y /= pt.y;
	z /= pt.z;

	return *this;
}

#endif