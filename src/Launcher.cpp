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

	Point camPos(7, 4, 7);
	Vector camDir(camPos, m.center);
	/*
	Camera cam(camPos, camDir, 35, 0, 5, 5);
	cam.display(m, true);*/

	CUDACamera tcam(camPos, camDir, 30, 0, 1000, 1000);
	double theta = 5;
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
	
	int idx = 0;
	while (true) {
		// system("cls") is generally considered terrible practice, but this is simply a test file and therefore it doesn't matter since it won't make it into the other libraries
		system("cls");
		frames[idx].print();
		idx++;
		if (idx >= frames.size()) idx = 0;
	}

	return 0;
}