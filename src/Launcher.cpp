#include <iostream>
#include <string>
#include <climits>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cfloat>

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
	Mesh m;
	cout << "Triangle count: " << m.numTris << endl;

	Point camPos(150, 50, 75);
	Vector camDir(camPos, m.center);
	/*
	Camera cam(camPos, camDir, 35, 0, 5, 5);
	cam.display(m, true);*/

	CUDACamera tcam(camPos, camDir, 30, 0, 1000, 1000);
	double theta = 20;
	double phi = 0;
	Angle orbitAng(theta, phi);
	Point orbitCenter;
	int numFrames = 360 / abs(theta);
	vector<Frame> frames;
	frames.push_back(tcam.CUDADisplayMath(m));
	for (int idx = 0; idx < numFrames; idx++) {
		tcam.orbit(orbitAng, orbitCenter);
		frames.push_back(tcam.CUDADisplayMath(m));
		cout << "Progress: " << (((double)idx) / numFrames) * 100 << "%" << endl;
	}

	for (int idx = 0; idx < frames.size(); idx++) {
		frames[idx].print();
	}

	return 0;
}