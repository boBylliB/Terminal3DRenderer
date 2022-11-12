#ifndef ANGLE_CPP
#define ANGLE_CPP

#include "Angle.h"

Angle::Angle(double theta, double phi) {
	this->theta = theta;
	this->phi = phi;

	clamp();
}

Angle Angle::operator+(const Angle& a) {
	Angle output(theta + a.theta, phi + a.phi);

	return output;
}
Angle& Angle::operator+=(const Angle& a) {
	theta += a.theta;
	phi += a.phi;

	clamp();

	return *this;
}
Angle Angle::operator-(const Angle& a) {
	Angle output(theta - a.theta, phi - a.phi);

	return output;
}
Angle& Angle::operator-=(const Angle& a) {
	theta -= a.theta;
	phi -= a.phi;

	clamp();

	return *this;
}
Angle Angle::operator*(const Angle& a) {
	Angle output(theta * a.theta, phi * a.phi);

	return output;
}
Angle& Angle::operator*=(const Angle& a) {
	theta *= a.theta;
	phi *= a.phi;

	clamp();

	return *this;
}
Angle Angle::operator/(const Angle& a) {
	Angle output(theta / a.theta, phi / a.phi);

	return output;
}
Angle& Angle::operator/=(const Angle& a) {
	theta /= a.theta;
	phi /= a.phi;

	clamp();

	return *this;
}

// Theta is clamped between 0 and 360, for rotation about the vertical axis
// Phi is clamped between -180 and 180, for rotation about the horizontal axis, with 0 being straight up
void Angle::clamp(void) {
	while (theta > 360)
		theta -= 360;
	while (theta < 0)
		theta += 360;
	while (phi > 180)
		phi -= 360;
	while (phi < -180)
		phi += 360;
}

#endif