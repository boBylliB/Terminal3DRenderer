#ifndef MESHFILE_CPP
#define MESHFILE_CPP

#include "MeshFile.h"

using namespace std;

void MeshFile::open(void) {
	string givenFilename;
	string extension;
	bool valid = false;

	while (!valid) {
		cout << "Enter the name/path of the file you'd like to view: ";
		getline(cin, givenFilename);
		cin.clear();
		cin.ignore(INT_MAX, '\n');

		fin.open(givenFilename);
		if (fin.good()) {
			extension = givenFilename.substr(givenFilename.length() - 3);
			if (extension == "obj") {
				filename = givenFilename;
				type = Filetype::OBJ;
				cout << "Detected file extension: " << extension << endl;
				valid = true;
			}
			else if (extension == "ply") {
				filename = givenFilename;
				type = Filetype::PLY;
				cout << "Detected file extension: " << extension << endl;
				valid = true;
			}
			else if (extension == "stl") {
				filename = givenFilename;
				type = Filetype::STL;
				cout << "Detected file extension: " << extension << endl;
				valid = true;
			}
			else if (extension == "txt") {
				filename = givenFilename;
				type = Filetype::TXT;
				cout << "Detected file extension: " << extension << endl;
				valid = true;
			}
			else {
				cout << "Unknown file extension: " << extension << endl;
				cout << "Please enter a file of type .obj, .stl, .ply, or custom .txt" << endl;
			}
		}
	}
}
void MeshFile::close(void) {
	if (fin.is_open()) {
		fin.close();
	}
}
MeshFile::~MeshFile(void) {
	close();
}

#endif