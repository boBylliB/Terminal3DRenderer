// Control Class
#ifndef CUDACAMERA_CUH
#define CUDACAMERA_CUH

#include <vector>
#include <chrono>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nppdefs.h"

#include "Mesh.h"
#include "Vector.h"
#include "Camera.h"
#include "Frame.h"
#include "Profiler.h"

// CUDA kernel functions to run in parallel on the GPU (cannot be a member function)
__global__ void CUDACalculateIntersectDistances(int*, double*, Point*, Triangle*, int*, Vector*);
__global__ void CUDACalculateIntersectDistancesSegmented(int*, double*, Point*, Triangle*, Triangle**, int*, int*, Vector*, int*, bool**);
__global__ void CUDACalculateRays(int*, Angle*, int*, Vector*);

// CUDA device functions
__device__ double CUDACalculateIntersectDistance(Triangle*, int, Point, Vector);
__device__ bool CUDACheckWithin(Triangle, Vector, Point);
__device__ Vector CUDADifferenceVector(Vector, Point, Point);
__device__ Vector CUDAAngleVector(Vector, Angle);
__device__ Vector CUDANormalize(Vector);
__device__ double CUDADot(Vector, Vector);
__device__ Vector CUDACross(Vector, Vector);

class CUDACamera : public Camera {
public:
	CUDACamera(const Point&, const Vector&, const double, const double, const int, const int);
	CUDACamera(Camera&);

	// Core Functions
	// Functions the same as the standard camera display, but does the math in parallel on the GPU for speed
	void CUDADisplay(const Mesh&, const bool = true);
	// Just the math of the display function above, outputting to a Frame to be displayed later
	Frame CUDADisplayMath(const Mesh&, const bool = true);
};

#endif