// Control Class
#ifndef THREADEDCAMERA_H
#define THREADEDCAMERA_H

#include <vector>
#include <atomic>

#include "Mesh.h"
#include "Vector.h"
#include "Camera.h"

struct threadedCalculationArgs {
	std::vector<double> *distances;
	const Point& pos;
	const Mesh& m;
	std::vector<Vector> *rays;
	const bool showProgress;
};

class ThreadedCamera : public Camera {
public:
	ThreadedCamera(const Point&, const Vector&, const double, const double, const int, const int);
	ThreadedCamera(Camera&);

	// Utility Functions
	static void threadedCalculateIntersectDistances(void*);
	// Core Functions
	// Functions the same as the standard camera display, but does the math in parallel for speed
	void threadedDisplay(const Mesh&, const bool = false);
	// Just the math of the display function above, outputting to a Frame to be displayed later
	Frame threadedDisplayMath(const Mesh&, const bool = false);
};

#endif