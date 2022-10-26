#ifndef MESH_H
#define MESH_H

#include "Defines.h"
#include "Triangle.h"
#include "MeshFile.h"

class Mesh {
public:
	Triangle tris[MAXTRIS];
	int numTris;

	void buildMesh(MeshFile);
};

#endif