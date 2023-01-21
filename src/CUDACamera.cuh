// Control Class
#ifndef CUDACAMERA_CUH
#define CUDACAMERA_CUH

#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Mesh.h"
#include "Vector.h"
#include "Camera.h"

// CUDA kernel function to run in parallel on the GPU (cannot be a member function)
__global__ void CUDACalculateIntersectDistances(int, double*, const Point&, const Mesh&, Vector*);

class CUDACamera : public Camera {
public:
	CUDACamera(const Point&, const Vector&, const double, const double, const int, const int);
	CUDACamera(Camera&);

	// Core Functions
	// Functions the same as the standard camera display, but does the math in parallel on the GPU for speed
	void CUDADisplay(const Mesh&);
};

#endif