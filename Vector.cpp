#ifndef VECTOR_CPP	
#define VECTOR_CPP

#include <cmath>
#include "Vector.h"

Angle Vector::toAngle(void) {

}
void Vector::fromAngle(Angle a) {

}
void Vector::rotate(Angle a) {

}
float Vector::magnitude(void) {
	return sqrt(I*I + J*J + K*K);
}
float Vector::dot(Vector vec) {
	return vec.I * I + vec.J * J + vec.K * K;
}
Vector Vector::cross(Vector vec) {
	float newI = J * vec.K - K * vec.J;
	float newJ = K * vec.I - I * vec.K;
	float newK = I * vec.J - J * vec.I;
	Vector out(newI, newJ, newK);
	return out;
}

#endif