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
#include "Triangle.h"
#include "Vector.h"

using namespace std;
/*
int main(void) {
	Mesh m;

	Point camPos(10, 10, 10);
	Vector camDir(camPos, m.center);

	Camera cam(camPos, camDir, 40, 0, 100, 100);
	cam.display(m);

	return 0;
}*/