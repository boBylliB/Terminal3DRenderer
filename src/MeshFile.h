// Container Class
#ifndef MESHFILE_H
#define MESHFILE_H

#include "Defines.h"
#include "Triangle.cuh"
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
	
	MeshFile(void);
	MeshFile(const std::string&); 
	~MeshFile(void);

	void open(void);
	void open(const std::string&);
	void close(void);
};

#endif