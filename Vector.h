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
	void scale(const double);
	double magnitude(void) const;
	double dot(const Vector&) const;
	Vector cross(const Vector&) const;
	void fromPoint(const Point&);
	void difference(const Point&, const Point&);
	Point toPoint(void) const;
};

#endif