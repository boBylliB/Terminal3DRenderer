#ifndef CUDACAMERA_CU
#define CUDACAMERA_CU

#include <thread>

#include "CUDACamera.cuh"

using namespace std;

// CUDA kernel function to run in parallel on the GPU (cannot be a member function)
__global__
void CUDACalculateIntersectDistances(int numRays, double* distances, const Point& pos, const Mesh& m, Vector* rays) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < numRays; idx += stride) {
		m.calculateIntersectDistance(pos, rays[idx]);
	}
}

CUDACamera::CUDACamera(const Point& position, const Vector& direction, const double fov = FOV, const double roll = 0, const int outHeight = HEIGHT, const int outWidth = WIDTH) : Camera(position, direction, fov, roll, outHeight, outWidth) {

}
CUDACamera::CUDACamera(Camera& cam) : Camera(cam.getPosition(), cam.getDirection(), cam.getFOV().theta, cam.getRoll(), cam.getOutputSize()[0], cam.getOutputSize()[1]) {

}

// Core Functions
// Functions the same as the standard camera display, but does the math in parallel on the GPU for speed
void CUDACamera::CUDADisplay(const Mesh& m) {
	// Display a line for the output width for verification that the whole display fits on screen
	for (int idx = 0; idx < outputWidth; idx++) {
		cout << "@";
	}
	cout << endl << endl;
	// Calculate the angle between pixels
	Angle angleBetween(fieldOfView.theta / outputWidth, fieldOfView.phi / outputHeight);
	// Create 9 rays, evenly spaced, per on-screen "pixel"
	angleBetween /= 3;

	Angle startingAngle((angleBetween.theta * (3.0 * outputWidth / 2.0) * -1.0), (angleBetween.phi * (3.0 * outputHeight / 2.0) * -1.0));
	startingAngle += direction.toAngle();

	// Create rays array on shared CPU & GPU memory
	int numRays = outputHeight * outputWidth * 9;
	Vector* rays;
	cudaMallocManaged(&rays, numRays * sizeof(Vector));

	for (int row = 0; row < outputHeight; row++) {
		for (int col = 0; col < outputWidth; col++) {
			for (int subrow = row * 3; subrow < row * 3 + 3; subrow++) {
				for (int subcol = col * 3; subcol < col * 3 + 3; subcol++) {
					Angle rayAngle = startingAngle;
					rayAngle.theta += subcol * angleBetween.theta;
					rayAngle.phi += subrow * angleBetween.phi;

					Vector ray(rayAngle);
					rays[subrow + subcol] = ray;
				}
			}
		}
	}
	cout << "Rays created, calculating intersects" << std::endl;
	// Calculates the number of thread blocks needed, making sure to round up if needed
	int numBlocks = (numRays + NUMTHREADSPERBLOCK - 1) / NUMTHREADSPERBLOCK;
	// Create required arrays on shared GPU & CPU memory
	double* intersectDistances;
	cudaMallocManaged(&intersectDistances, numRays * sizeof(double));
	// Calculate the intersection distances for each ray in parallel
	cout << "Beginning GPU Distance Calculations" << endl;
	CUDACalculateIntersectDistances<<<numBlocks, NUMTHREADSPERBLOCK>>>(numRays, intersectDistances, position, m, rays);
	cout << "Calculated Distances" << endl;
	// Now we wait for all threads to finish and collect the results
	cudaDeviceSynchronize();
	// Free memory
	for (int idx = 0; idx < NUMTHREADS; idx++) {
		cudaFree(rays);
		cudaFree(intersectDistances);
	}
	// Calculate the minimum distance for brightness falloff
	double minDist = DBL_MAX;
	double maxDist = 0;
	for (int idx = 0; idx < numRays; idx++) {
		if (intersectDistances[idx] < minDist && intersectDistances[idx] > 0)
			minDist = intersectDistances[idx];
		if (intersectDistances[idx] > maxDist)
			maxDist = intersectDistances[idx];
	}
	if (maxDist <= minDist)
		maxDist = DBL_MAX;
	double falloff = maxDist - minDist;
	// For each pixel, the "brightness" is the number of rays in that pixel that have an intersection, scaled linearly by distance
	cout << "Calculating brightness values" << std::endl;
	vector<double> pixelBrightness(outputHeight * outputWidth, 0.0);
	for (int idx = 0; idx < numRays; idx++) {
		int interIdx = idx;
		int pixIdx = idx / 9;
		if (intersectDistances[interIdx] > 0) {
			double brightnessScale = 1.0 - (intersectDistances[interIdx] - minDist) / falloff;
			// Make sure value is between FALLOFFMIN and 1
			brightnessScale = (brightnessScale >= 1) ? 1 : brightnessScale;
			brightnessScale = (brightnessScale <= FALLOFFMIN) ? FALLOFFMIN : brightnessScale;
			pixelBrightness[pixIdx] += brightnessScale;
		}
	}
	// Display the calculated image to the screen
	for (int row = 0; row < outputHeight; row++) {
		for (int col = 0; col < outputWidth; col++) {
			double brightness = pixelBrightness[row * outputWidth + col];
			string grayscale = GRAYSCALE;
			if (brightness - intPart(brightness) >= 0.5)
				brightness += 1;
			int brightInt = brightness;
			if (brightInt > grayscale.length() - 1)
				brightInt = grayscale.length() - 1;
			if (brightInt < 0)
				brightInt = 0;
			cout << grayscale[brightInt];
		}
		cout << std::endl;
	}
}

#endif