// Container Class
#ifndef SEGMENTEDMESH_H
#define SEGMENTEDMESH_H

#include <vector>
#include "Defines.h"
#include "Triangle.h"
#include "Vector.h"
#include "Mesh.h"
#include "MeshFile.h"

class SegmentedMesh {
public:
	std::vector<std::vector<Triangle>> triSegments;
	std::vector<Point> segmentCenters;
	std::vector<Vector> segmentBounds;
	std::vector<int> numTris;
	int numSegments;
	Point center;
	Vector bounds;

	SegmentedMesh(void);
	SegmentedMesh(MeshFile&);
	SegmentedMesh(Mesh);

	void calcSegments(Mesh);
};

#endif