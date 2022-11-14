#ifndef MESHFILE_CPP
#define MESHFILE_CPP

#include "MeshFile.h"

using namespace std;

MeshFile::MeshFile(void) {
	open();
}
MeshFile::MeshFile(const string& givenFilename) {
	open(givenFilename);
}
MeshFile::~MeshFile(void) {
	close();
}

void MeshFile::open(void) {
	string givenFilename;
	bool valid = false;

	do {
		cout << "Enter the name/path of the file you'd like to view: ";
		getline(cin, givenFilename);
		cin.clear();
		cin.ignore(INT_MAX, '\n');

		valid = true;

		try {
			open(givenFilename);
		}
		catch (string err) {
			cout << "Error reading file: " << err << endl;
			cout << "Please try again. The file should be of type obj, stl, ply, or txt (custom filetype)" << endl;
			valid = false;
		}
	} while (!valid);
}
void MeshFile::open(const string& givenFilename) {
	string extension;

	system("cd");

	fin.open(givenFilename.c_str(), ifstream::in);
	if (!fin.is_open())
		perror(("error while opening file: " + givenFilename).c_str());
	if (fin.good()) {
		extension = givenFilename.substr(givenFilename.length() - 3);
		if (extension == "obj") {
			filename = givenFilename;
			type = Filetype::OBJ;
			cout << "Detected file extension: " << extension << " as obj" << endl;
		}
		else if (extension == "ply") {
			filename = givenFilename;
			type = Filetype::PLY;
			cout << "Detected file extension: " << extension << " as ply" << endl;
		}
		else if (extension == "stl") {
			filename = givenFilename;
			type = Filetype::STL;
			cout << "Detected file extension: " << extension << " as stl" << endl;
		}
		else if (extension == "txt") {
			filename = givenFilename;
			type = Filetype::TXT;
			cout << "Detected file extension: " << extension << " as txt" << endl;
		}
		else {
			throw "Unknown file extension";
		}
	}
}
void MeshFile::close(void) {
	if (fin.is_open()) {
		fin.close();
	}
}

#endif