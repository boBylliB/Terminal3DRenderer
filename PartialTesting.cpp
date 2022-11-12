#include <iostream>
#include <cstdlib>
#include <string>

#include "RenderUtils.h"
#include "Angle.h"

using namespace std;

int main(void) {
	cout << "Enter theta1: ";
	double t1;
	cin >> t1;
	cout << "Enter phi1: ";
	double p1;
	cin >> p1;
	cout << "Enter theta2: ";
	double t2;
	cin >> t2;
	cout << "Enter phi2: ";
	double p2;
	cin >> p2;
	cout << "Enter numRays: ";
	int numRays;
	cin >> numRays;

	Angle a1(t1, p1);
	Angle a2(t2, p2);

	Angle add = a1 + a2;
	Angle sub = a1 - a2;
	Angle mlt = a1 * a2;
	Angle div = a1 / a2;

	cout << add.theta << ", " << add.phi << endl;
	cout << t1 + t2 << ", " << p1 + p2 << endl;
	cout << sub.theta << ", " << sub.phi << endl;
	cout << t1 - t2 << ", " << p1 - p2 << endl;
	cout << mlt.theta << ", " << mlt.phi << endl;
	cout << t1 * t2 << ", " << p1 * p2 << endl;
	cout << div.theta << ", " << div.phi << endl;
	cout << t1 / t2 << ", " << p1 / p2 << endl << endl;

	add.theta = a1.theta;
	add.phi = a1.phi;
	sub.theta = a1.theta;
	sub.phi = a1.phi;
	mlt.theta = a1.theta;
	mlt.phi = a1.phi;
	div.theta = a1.theta;
	div.phi = a1.phi;

	add += a2;
	sub -= a2;
	mlt *= a2;
	div /= a2;

	cout << add.theta << ", " << add.phi << endl;
	cout << sub.theta << ", " << sub.phi << endl;
	cout << mlt.theta << ", " << mlt.phi << endl;
	cout << div.theta << ", " << div.phi << endl << endl;

	Angle startingAngle((a1.theta * (numRays / 2) * -1.0), 90 - (a1.phi * (numRays / 2)));
	cout << startingAngle.theta << ", " << startingAngle.phi << endl;

	return 0;
}