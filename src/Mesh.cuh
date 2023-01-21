// Container Class
#ifndef MESH_H
#define MESH_H

#include <vector>
#include "Defines.h"
#include "Triangle.cuh"
#include "Vector.cuh"
#include "MeshFile.h"

class Mesh {
public:
	std::vector<Triangle> tris;
	int numTris;
	Point center;

	Mesh(void);
	Mesh(MeshFile&);

	void buildMesh(MeshFile&);
	void calcCenter(void);

	// Calculates the intersect distance for a given ray when compared to this mesh
	// The returned double will be -1 if it doesn't intersect any triangle
	double calculateIntersectDistance(const Point&, const Vector&) const;
	// Calculates the intersect distances for each given ray when compared to this mesh
	// The return value will be a vector of doubles that are either the distance to intersection or -1 for "do not display"
	std::vector<double> calculateIntersectDistances(const Point&, const std::vector<Vector>, const bool) const;
};

#endif