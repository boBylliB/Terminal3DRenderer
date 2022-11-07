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

	void buildMesh(MeshFile);
	std::vector<double> calculateIntersectDistances(const Point origin, const std::vector<Vector> rays, const int rayCount);
	Point getCenter(void);
};

#endif