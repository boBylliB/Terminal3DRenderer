#ifndef THREADEDCAMERA_CPP
#define THREADEDCAMERA_CPP

#include <thread>

#include "ThreadedCamera.h"

using namespace std;

ThreadedCamera::ThreadedCamera(const Point& position, const Vector& direction, const double fov = FOV, const double roll = 0, const int outHeight = HEIGHT, const int outWidth = WIDTH) : Camera(position, direction, fov, roll, outHeight, outWidth) {

}
ThreadedCamera::ThreadedCamera(Camera& cam) : Camera(cam.getPosition(), cam.getDirection(), cam.getFOV().theta, cam.getRoll(), cam.getOutputSize()[0], cam.getOutputSize()[1]) {

}

// Utility Functions
void ThreadedCamera::threadedCalculateIntersectDistances(vector<double> distances, const Point& pos, const Mesh& m, const vector<Vector> rays, const bool showProgress) {
	distances = m.calculateIntersectDistances(pos, rays, showProgress);
}

// Core Functions
// Functions the same as the standard camera display, but does the math in parallel for speed
void ThreadedCamera::threadedDisplay(const Mesh& m, const bool showProgress) {
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

	vector<Vector> partialRays[NUMTHREADS];

	for (int idx = 0; idx < NUMTHREADS - 1; idx++) {
		for (int rayIdx = 0; rayIdx < raysPerThread; rayIdx++) {
			partialRays[idx].push_back(rays[rayIdx + idx * raysPerThread]);
		}
	}
	for (int rayIdx = (NUMTHREADS - 1) * raysPerThread; rayIdx < rays.size(); rayIdx++) {
		partialRays[NUMTHREADS - 1].push_back(rays[rayIdx]);
	}
	// Calculate the intersection distances for each ray in parallel
	vector<double> intersectDistanceSections[NUMTHREADS];
	vector<thread> threadpool;
	for (int idx = 0; idx < NUMTHREADS; idx++) {
		threadpool.emplace_back(&ThreadedCamera::threadedCalculateIntersectDistances, intersectDistanceSections[idx], position, m, partialRays[idx], showProgress);
	}
	// Now we wait for all threads to finish and merge each intersect distance section into a complete intersect distances array
	vector<double> intersectDistances;
	for (int idx = 0; idx < NUMTHREADS; idx++) {
		threadpool[idx].join();
		intersectDistances.insert(intersectDistances.end(), intersectDistanceSections[idx].begin(), intersectDistanceSections[idx].end());
	}
	// For each pixel, the "brightness" is the number of rays in that pixel that have an intersection, scaled linearly by distance
	cout << "Calculating brightness values" << std::endl;
	vector<double> pixelBrightness(outputHeight * outputWidth, 0.0);
	for (int idx = 0; idx < outputHeight * outputWidth * 9; idx++) {
		int interIdx = idx;
		int pixIdx = idx / 9;
		if (intersectDistances[interIdx] > 0)
			pixelBrightness[pixIdx] += 1.0 / (intersectDistances[interIdx] * FALLOFF);
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