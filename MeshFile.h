#ifndef MESHFILE_H
#define MESHFILE_H

#include "Defines.h"
#include "Triangle.h"
#include <cstdlib>
#include <fstream>

enum Filetype { OBJ, STL, PLY, TXT};

class MeshFile {
public:
	std::string filename;
	Filetype type;
	
	MeshFile(void);
};

#endif