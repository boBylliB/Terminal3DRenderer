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

	void buildMesh(MeshFile&);
	void calcCenter(void);

	// Calculates the intersect distances for each given ray when compared to this mesh
	// The return value will be a vector of doubles that are either the distance to intersection or -1 for "do not display"
	// The returned double will be -1 if it doesn't intersect any triangle
	std::vector<double> calculateIntersectDistances(const Point&, const std::vector<Vector>) const;
};

#endif