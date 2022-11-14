#include <iostream>
#include <cstdlib>
#include <string>

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

int main(void) {
	MeshFile mf;
	Mesh m;

	m.buildMesh(mf);

	for (int i = 0; i < m.numTris; i++) {
		for (int j = 0; j < 3; j++) {
			cout << m.tris[i].verts[j].x << " " << m.tris[i].verts[j].y << " " << m.tris[i].verts[j].z << endl;
		}
		cout << endl;
	}

	Point camPos(7, 0, 0);
	Vector camDir(camPos, m.center);

	Camera cam(camPos, camDir, 30, 0, 100, 100);
	cam.display(m);

	return 0;
}