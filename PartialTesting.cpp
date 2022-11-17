#include <iostream>
#include <cstdlib>
#include <string>
#include <pthread.h>

#include "Angle.h"
#include "Camera.h"
#include "Defines.h"
#include "Mesh.h"
#include "MeshFile.h"
#include "Point.h"
#include "RenderUtils.h"
#include "ThreadedCamera.h"
#include "Triangle.h"
#include "Vector.h"

using namespace std;
/*
struct calculateIntersectArgs {
	vector<double> distances;
	Point& pos;
	Mesh& m;
	vector<Vector> rays;
	bool showProgress;
};

int main(void) {
	Mesh m;
	Angle fieldOfView(30, CHARRATIO * 30);
	int outputWidth = 300;
	int outputHeight = 300;
	Point position(150, 30, 75);
	Vector direction(position, m.center);

	vector<Vector> rays;
	rays.reserve(1000);
	for (int idx = 0; idx < 1000; idx++) {
		rays.emplace_back(Vector());
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
	pthread_t threadpool[NUMTHREADS];
	for (int idx = 0; idx < NUMTHREADS; idx++) {
		struct calculateIntersectArgs args = { intersectDistanceSections[idx], position, m, partialRays[idx], true };
		pthread_create(&threadpool[idx], NULL, &ThreadedCamera::threadedCalculateIntersectDistances, &args);
	}
	// Now we wait for all threads to finish and merge each intersect distance section into a complete intersect distances array
	vector<double> intersectDistances;
	for (int idx = 0; idx < NUMTHREADS; idx++) {
		threadpool[idx].join();
		intersectDistances.insert(intersectDistances.end(), intersectDistanceSections[idx].begin(), intersectDistanceSections[idx].end());
	}

	return 0;
}*/