#ifndef VECTOR_CPP	
#define VECTOR_CPP

#include <cmath>
#include "Vector.h"
#include "RenderUtils.h"

Vector::Vector(float i, float j, float k) {
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
float Vector::magnitude(void) {
	return sqrt(I*I + J*J + K*K);
}
void Vector::normalize(void) {
	float mag = magnitude();
	I /= mag;
	J /= mag;
	K /= mag;
}
float Vector::dot(const Vector &vec) {
	return vec.I * I + vec.J * J + vec.K * K;
}
Vector Vector::cross(const Vector &vec) {
	float newI = J * vec.K - K * vec.J;
	float newJ = K * vec.I - I * vec.K;
	float newK = I * vec.J - J * vec.I;
	Vector out(newI, newJ, newK);
	return out;
}

#endif