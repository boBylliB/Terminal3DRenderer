// Type Class
#ifndef ANGLE_H	
#define ANGLE_H

// Theta is clamped between 0 and 360, for rotation about the vertical axis
// Phi is clamped between -180 and 180, for rotation about the horizontal axis, with 0 being straight up
class Angle {
public:
	double theta;
	double phi;

	Angle(double = 0, double = 0);

	Angle operator+(const Angle&);
	Angle operator+(const double);
	Angle& operator+=(const Angle&);
	Angle& operator+=(const double);
	Angle operator-(const Angle&);
	Angle operator-(const double);
	Angle& operator-=(const Angle&);
	Angle& operator-=(const double);
	Angle operator*(const Angle&);
	Angle operator*(const double);
	Angle& operator*=(const Angle&);
	Angle& operator*=(const double);
	Angle operator/(const Angle&);
	Angle operator/(const double);
	Angle& operator/=(const Angle&);
	Angle& operator/=(const double);

	// Theta is clamped between 0 and 360, for rotation about the vertical axis
	// Phi is clamped between -180 and 180, for rotation about the horizontal axis, with 0 being straight up
	void clamp(void);
};

#endif