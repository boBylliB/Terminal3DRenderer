#include <iostream>
#include <string>
#include <climits>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <chrono>

#include "Angle.h"
#include "Camera.h"
#include "CUDACamera.cuh"
#include "Defines.h"
#include "Mesh.h"
#include "MeshFile.h"
#include "Point.h"
#include "RenderUtils.h"
#include "ThreadedCamera.h"
#include "Triangle.h"
#include "Vector.h"

using namespace std;

int main(void) {
	MeshFile mf("C:/Users/jrsan/Documents/GitHub/Terminal3DRenderer/models/Doge.obj");
	Mesh m(mf);
	cout << "Triangle count: " << m.numTris << endl;

	Point camPos(150, 50, 75);
	Vector camDir(camPos, m.center);
	/*
	Camera cam(camPos, camDir, 35, 0, 5, 5);
	cam.display(m, true);*/

	int framesPerSecond = 20;
	double delay = 1.0 / framesPerSecond;

	CUDACamera tcam(camPos, camDir, 30, 0, 500, 500);
	double theta = 45;
	double phi = 0;
	Angle orbitAng(theta, phi);
	Point orbitCenter;
	int numFrames = 360 / abs(theta);
	vector<Frame> frames;
	frames.push_back(tcam.CUDADisplayMath(m));
	// Since we always generate the first frame, we skip index 0 in the optional for loop
	for (int idx = 1; idx < numFrames; idx++) {
		tcam.orbit(orbitAng, orbitCenter);
		frames.push_back(tcam.CUDADisplayMath(m));
		cout << "Progress: " << (((double)idx) / (numFrames - 1)) * 100 << "%" << endl;
	}
	
	ios_base::sync_with_stdio(false);
	int idx = 0;
	auto oldTime = std::chrono::high_resolution_clock::now();
	while (true) {
		auto newTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed = newTime - oldTime;
		if (elapsed.count() > delay) {
			// system("cls") is generally considered terrible practice, but this is simply a test file and therefore it doesn't matter since it won't make it into the other libraries
			system("cls");
			frames[idx].print();
			idx++;
			//if (idx >= frames.size()) idx = 0;
			if (idx >= frames.size()) break;
			auto oldTime = std::chrono::high_resolution_clock::now();
		}
	}

	Profiler profiler;
	profiler.printArchiveComparison();

	return 0;
}