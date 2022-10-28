#ifndef MESHFILE_H
#define MESHFILE_H

#include "Defines.h"
#include "Triangle.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

enum class Filetype { OBJ, STL, PLY, TXT};

class MeshFile {
public:
	std::string filename;
	Filetype type;
	std::ifstream fin;
	
	void open(void);
	void close(void);
	~MeshFile(void);
};

#endif