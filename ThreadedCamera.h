// Control Class
#ifndef THREADEDCAMERA_H
#define THREADEDCAMERA_H

#include <vector>

#include "Mesh.h"
#include "Vector.h"
#include "Camera.h"

struct threadedCalculationArgs {
	std::vector<double> distances;
	Point& pos;
	Mesh& m;
	vector<Vector> rays;
	bool showProgress;
};

class ThreadedCamera : public Camera {
public:
	ThreadedCamera(const Point&, const Vector&, const double, const double, const int, const int);
	ThreadedCamera(Camera&);

	// Utility Functions
	void threadedCalculateIntersectDistances(struct threadedCalculationArgs*);
	// Core Functions
	// Functions the same as the standard camera display, but does the math in parallel for speed
	void threadedDisplay(const Mesh&, const bool = false);
};

#endif