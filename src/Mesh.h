// Container Class
#ifndef MESH_H
#define MESH_H

#include <vector>
#include "Defines.h"
#include "Triangle.h"
#include "Vector.h"
#include "MeshFile.h"

class Mesh {
public:
	std::vector<Triangle> tris;
	std::vector<std::vector<Triangle>> triSegments;
	std::vector<Point> segmentCenters;
	std::vector<Vector> segmentBounds;
	std::vector<int> numTrisPerSegment;
	int numTris;
	int numSegments;
	Point center;
	Vector bounds;

	Mesh(void);
	Mesh(MeshFile&);

	void buildMesh(MeshFile&);
	void calcBounds(void);
	void calcSegments(void);

	// Calculates the intersect distance for a given ray when compared to this mesh
	// The returned double will be -1 if it doesn't intersect any triangle
	double calculateIntersectDistance(const Point&, const Vector&) const;
	// Calculates the intersect distances for each given ray when compared to this mesh
	// The return value will be a vector of doubles that are either the distance to intersection or -1 for "do not display"
	std::vector<double> calculateIntersectDistances(const Point&, const std::vector<Vector>, const bool) const;
};

#endif