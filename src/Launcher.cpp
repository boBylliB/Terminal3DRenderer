#include <iostream>
#include <string>
#include <climits>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cfloat>

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

int main(void) {
	Mesh m;
	cout << "Triangle count: " << m.numTris << endl;

	Point camPos(150, 50, 75);
	Vector camDir(camPos, m.center);
	/*
	Camera cam(camPos, camDir, 35, 0, 5, 5);
	cam.display(m, true);*/
	
	ThreadedCamera tcam(camPos, camDir, 30, 0, 500, 500);
	tcam.threadedDisplay(m, true);

	return 0;
}