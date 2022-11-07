#ifndef VECTOR_CPP	
#define VECTOR_CPP

#include <cmath>
#include "Vector.h"
#include "RenderUtils.h"

Vector::Vector(double i, double j, double k) {
    I = i;
    J = j;
    K = k;
}
Angle Vector::toAngle(void) {
    Angle a;

    if (K < 0.00001 && K > -0.00001) {
        /* printf("K was zero\n"); */
        if (I > 0.0)
            a.theta = 90.0;
        else
            a.theta = 270.0;
    }
    else if (I > 0.00001 || I < -0.00001) {
        if (K > 0.0 && I > 0.0)
            a.theta = radToDeg(atan(K / I));
        else if (K > 0.0 && I < 0.0)
            a.theta = radToDeg(-1.0 * atan(K / I)) + 90.0;
        else if (K < 0.0 && I < 0.0)
            a.theta = radToDeg(atan(K / I)) + 180.0;
        else
            a.theta = radToDeg(-1.0 * atan(K / I)) + 270.0;
    }
    else {
        /* printf("I was zero\n"); */
        if (K > 0.0)
            a.theta = 0.0;
        else
            a.theta = 180.0;
    }
    a.phi = radToDeg(acos(J / magnitude()));

    return a;
}
void Vector::fromAngle(const Angle &a) {
    double magnitude = 0;

    I = cos(degToRad(a.theta)) * sin(degToRad(a.phi));
    J = sin(degToRad(a.phi));
    K = sin(degToRad(a.theta)) * sin(degToRad(a.phi));

	normalize();
}
void Vector::rotate(const Angle &a) {
    Angle vecAngle = toAngle();
    vecAngle.phi += a.phi;
    vecAngle.theta += a.theta;

    // Phi is clamped from 0 to 180, theta should roll over to stay between the range of 0 to 360
    if (vecAngle.phi > 180)
        vecAngle.phi = 180;
    if (vecAngle.phi < 0)
        vecAngle.phi = 0;
    while (vecAngle.theta > 360)
        vecAngle.theta - 360;
    while (vecAngle.theta < 0)
        vecAngle.theta + 360;

    fromAngle(vecAngle);
}
double Vector::magnitude(void) {
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
double Vector::dot(const Vector &vec) {
	return vec.I * I + vec.J * J + vec.K * K;
}
Vector Vector::cross(const Vector &vec) {
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
Point Vector::toPoint(void) {
    Point pt = { I, J, K };
    return pt;
}

#endif