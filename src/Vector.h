// Type Class
#ifndef VECTOR_H	
#define VECTOR_H

#include "Angle.h"
#include "Point.h"

class Vector {
public:
	double I;
	double J;
	double K;

	Vector(const double = 0, const double = 0, const double = 0);
	Vector(const Angle&);
	Vector(const Point&);
	Vector(const Point&, const Point&);

	Vector operator+(const Vector&);
	Vector& operator+=(const Vector&);
	Vector& operator+=(const Point&);
	Vector operator-(const Vector&);
	Vector& operator-=(const Vector&);
	Vector& operator-=(const Point&);
	Vector operator*(const Vector&);
	Vector& operator*=(const Vector&);
	Vector& operator*=(const Point&);
	Vector operator/(const Vector&);
	Vector& operator/=(const Vector&);
	Vector& operator/=(const Point&);

	Angle toAngle(void) const;
	void fromAngle(const Angle&);
	void rotate(const Angle&);
	void normalize(void);
	void scale(const double); // multiplies each direction by the given double
	double magnitude(void) const;
	double dot(const Vector&) const; // dot product of the current vector (dot) the given vector
	Vector cross(const Vector&) const; // cross product of the current vector (cross) the given vector
	void fromPoint(const Point&); // vector from 0,0,0 to the given point
	void difference(const Point&, const Point&); // vector from point 1 to point 2
	Point toPoint(void) const; // Point defined as where this vector is pointing if the origin was 0,0,0
};

#endif