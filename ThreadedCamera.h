// Control Class
#ifndef THREADEDCAMERA_H
#define THREADEDCAMERA_H

#include <vector>

#include "Mesh.h"
#include "Vector.h"
#include "Camera.h"

class ThreadedCamera : public Camera {
public:
	ThreadedCamera(const Point&, const Vector&, const double, const double, const int, const int);
	ThreadedCamera(Camera&);

	// Utility Functions
	void threadedCalculateIntersectDistances(std::vector<double>, const Point&, const Mesh&, const std::vector<Vector>, const bool = false);
	// Core Functions
	// Functions the same as the standard camera display, but does the math in parallel for speed
	void threadedDisplay(const Mesh&, const bool = false);
};

#endif