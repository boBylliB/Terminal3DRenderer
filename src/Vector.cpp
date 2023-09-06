#ifndef VECTOR_CPP	
#define VECTOR_CPP

#include <cmath>
#include "Vector.h"
#include "RenderUtils.h"

Vector::Vector(const double i, const double j, const double k) {
    I = i;
    J = j;
    K = k;
}
Vector::Vector(const Angle& a) {
    fromAngle(a);
}
Vector::Vector(const Point& pt) {
    fromPoint(pt);
}
Vector::Vector(const Point& pt1, const Point& pt2) {
    difference(pt1, pt2);
}

Vector Vector::operator+(const Vector& vec) {
    Vector output(I + vec.I, J + vec.J, K + vec.K);

    return output;
}
Vector& Vector::operator+=(const Vector& vec) {
    I += vec.I;
    J += vec.J;
    K += vec.K;

    return *this;
}
Vector& Vector::operator+=(const Point& pt) {
    I += pt.x;
    J += pt.y;
    K += pt.z;

    return *this;
}
Vector Vector::operator-(const Vector& vec) {
    Vector output(I - vec.I, J - vec.J, K - vec.K);

    return output;
}
Vector& Vector::operator-=(const Vector& vec) {
    I -= vec.I;
    J -= vec.J;
    K -= vec.K;

    return *this;
}
Vector& Vector::operator-=(const Point& pt) {
    I -= pt.x;
    J -= pt.y;
    K -= pt.z;

    return *this;
}
Vector Vector::operator*(const Vector& vec) {
    Vector output(I * vec.I, J * vec.J, K * vec.K);

    return output;
}
Vector& Vector::operator*=(const Vector& vec) {
    I *= vec.I;
    J *= vec.J;
    K *= vec.K;

    return *this;
}
Vector& Vector::operator*=(const Point& pt) {
    I *= pt.x;
    J *= pt.y;
    K *= pt.z;

    return *this;
}
Vector Vector::operator/(const Vector& vec) {
    Vector output(I / vec.I, J / vec.J, K / vec.K);

    return output;
}
Vector& Vector::operator/=(const Vector& vec) {
    I /= vec.I;
    J /= vec.J;
    K /= vec.K;

    return *this;
}
Vector& Vector::operator/=(const Point& pt) {
    I /= pt.x;
    J /= pt.y;
    K /= pt.z;

    return *this;
}
bool Vector::operator==(const Vector& vec) {
    return I == vec.I && J == vec.J && K == vec.K;
}

Angle Vector::toAngle(void) const {
    Angle a;

    // The reason why I do this instead of testing directly for zero is because of floating point math (in this case, with doubles)
    // After doing floating point math on a value, there is a chance that it isn't perfectly precise
    // Therefore, I test within reasonable doubt that the value SHOULD be zero in this case
    // Issues this close to zero could lead to undefined behaviour with my math, hence the concern
    if (K < 0.0000000001 && K > -0.0000000001) {
        if (I > 0.0)
            a.theta = 0;
        else
            a.theta = 180.0;
    }
    else if (I > 0.0000000001 || I < -0.0000000001) {
        double ratio = K / I;
        if (ratio < 0)
            ratio *= -1;
        a.theta = radToDeg(atan(ratio));
        if (K > 0.0 && I < 0.0)
            a.theta = 180.0 - a.theta;
        else if (K < 0.0 && I < 0.0)
            a.theta += 180.0;
        else if (K < 0.0 && I > 0.0)
            a.theta = 360.0 - a.theta;
    }
    else {
        if (K > 0.0)
            a.theta = 90.0;
        else
            a.theta = 270.0;
    }
    a.phi = radToDeg(acos(J / magnitude()));

    return a;
}
void Vector::fromAngle(const Angle &a) {
    I = cos(degToRad(a.theta)) * sin(degToRad(a.phi));
    J = cos(degToRad(a.phi));
    K = sin(degToRad(a.theta)) * sin(degToRad(a.phi));

    // If there's no way to tell how large a vector should be, I just normalize it so that it's easier to scale later
	normalize();
}
void Vector::rotate(const Angle &a) {
    Angle vecAngle = toAngle();
    double mag = magnitude();
    vecAngle += a;

    fromAngle(vecAngle);
    scale(mag);
}
double Vector::magnitude(void) const {
	return sqrt(I*I + J*J + K*K);
}
void Vector::normalize(void) {
	double mag = magnitude();
	I /= mag;
	J /= mag;
	K /= mag;
}
void Vector::scale(const double factor) {
    if (factor != 0) {
        I *= factor;
        J *= factor;
        K *= factor;
    }
}
double Vector::dot(const Vector &vec) const {
	return vec.I * I + vec.J * J + vec.K * K;
}
Vector Vector::cross(const Vector &vec) const {
	double newI = J * vec.K - K * vec.J;
	double newJ = K * vec.I - I * vec.K;
	double newK = I * vec.J - J * vec.I;
	Vector out(newI, newJ, newK);
	return out;
}
void Vector::fromPoint(const Point& pt) {
    I = pt.x;
    J = pt.y;
    K = pt.z;
}
void Vector::difference(const Point& a, const Point& b) {
    I = b.x - a.x;
    J = b.y - a.y;
    K = b.z - a.z;
}
Point Vector::toPoint(void) const {
    Point pt = { I, J, K };
    return pt;
}

#endif