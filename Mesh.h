#ifndef MESH_H
#define MESH_H

#include <vector>
#include "Defines.h"
#include "Triangle.h"
#include "MeshFile.h"

class Mesh {
public:
	std::vector<Triangle> tris;
	int numTris;
	Point center;

	void buildMesh(MeshFile);
	std::vector<double> calculateIntersectDistances(const Point, const std::vector<Vector>);
	void calcCenter(void);
};

#endif