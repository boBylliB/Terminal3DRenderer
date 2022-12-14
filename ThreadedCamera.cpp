#ifndef THREADEDCAMERA_CPP
#define THREADEDCAMERA_CPP

#include <thread>
#include <atomic>

#include "ThreadedCamera.h"

using namespace std;

ThreadedCamera::ThreadedCamera(const Point& position, const Vector& direction, const double fov = FOV, const double roll = 0, const int outHeight = HEIGHT, const int outWidth = WIDTH) : Camera(position, direction, fov, roll, outHeight, outWidth) {

}
ThreadedCamera::ThreadedCamera(Camera& cam) : Camera(cam.getPosition(), cam.getDirection(), cam.getFOV().theta, cam.getRoll(), cam.getOutputSize()[0], cam.getOutputSize()[1]) {

}

// Utility Functions
void ThreadedCamera::threadedCalculateIntersectDistances(void* arguments) {
	struct threadedCalculationArgs* args = (struct threadedCalculationArgs*)arguments;
	*(args->distances) = args->m.calculateIntersectDistances(args->pos, *(args->rays), args->showProgress);
	cout << "Calculated distances" << endl;
}

// Core Functions
// Functions the same as the standard camera display, but does the math in parallel for speed
void ThreadedCamera::threadedDisplay(const Mesh& m, const bool showProgress) {
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

	vector<Vector> rays;

	for (int row = 0; row < outputHeight; row++) {
		for (int col = 0; col < outputWidth; col++) {
			for (int subrow = row * 3; subrow < row * 3 + 3; subrow++) {
				for (int subcol = col * 3; subcol < col * 3 + 3; subcol++) {
					Angle rayAngle = startingAngle;
					rayAngle.theta += subcol * angleBetween.theta;
					rayAngle.phi += subrow * angleBetween.phi;

					Vector ray(rayAngle);
					rays.push_back(ray);
				}
			}
		}
	}
	cout << "Rays created, calculating intersects" << std::endl;
	// Split rays into chunks for threads
	int raysPerThread = rays.size() / NUMTHREADS;

	vector<Vector> *partialRays[NUMTHREADS];

	for (int idx = 0; idx < NUMTHREADS - 1; idx++) {
		partialRays[idx] = new vector<Vector>();
		for (int rayIdx = 0; rayIdx < raysPerThread; rayIdx++) {
			partialRays[idx]->emplace_back(rays[rayIdx + idx * raysPerThread]);
		}
	}
	partialRays[NUMTHREADS - 1] = new vector<Vector>();
	for (int rayIdx = (NUMTHREADS - 1) * raysPerThread; rayIdx < rays.size(); rayIdx++) {
		partialRays[NUMTHREADS - 1]->emplace_back(rays[rayIdx]);
	}
	// Calculate the intersection distances for each ray in parallel
	vector<double> *intersectDistanceSections[NUMTHREADS];
	threadedCalculationArgs *args[NUMTHREADS];
	thread threadpool[NUMTHREADS];
	for (int idx = 0; idx < NUMTHREADS; idx++) {
		intersectDistanceSections[idx] = new vector<double>();
		args[idx] = new threadedCalculationArgs{ intersectDistanceSections[idx], position, m, partialRays[idx], showProgress };
		threadpool[idx] = thread(&ThreadedCamera::threadedCalculateIntersectDistances, args[idx]);
	}
	// Now we wait for all threads to finish and merge each intersect distance section into a complete intersect distances array
	vector<double> intersectDistances;
	for (int idx = 0; idx < NUMTHREADS; idx++) {
		threadpool[idx].join();
		for (int jdx = 0; jdx < intersectDistanceSections[idx]->size(); jdx++) {
			intersectDistances.push_back(intersectDistanceSections[idx]->at(jdx));
		}
	}
	// Free memory
	for (int idx = 0; idx < NUMTHREADS; idx++) {
		delete intersectDistanceSections[idx];
		delete partialRays[idx];
	}
	// Calculate the minimum distance for brightness falloff
	double minDist = DBL_MAX;
	double maxDist = 0;
	for (int idx = 0; idx < intersectDistances.size(); idx++) {
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
	for (int idx = 0; idx < intersectDistances.size(); idx++) {
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