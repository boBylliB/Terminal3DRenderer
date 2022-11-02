#ifndef MESH_CPP	
#define MESH_CPP

#include <fstream>
#include "Mesh.h"
#include "RenderUtils.h"

using namespace std;

void Mesh::buildMesh(MeshFile mf) {
    int vert = 0;
    int tri = 0;
    int i = 0;
    int vertIdx = 0;
    int triIdx = 0;
    Point* verts = new Point[MAXTRIS * 3]{};
    
    switch (mf.type) {
    case Filetype::TXT:
        while (mf.fin.good() && tri < MAXTRIS) {
            if (i > 2) {
                i = 0;
                vert++;
            }
            if (vert > 2) {
                vert = 0;
                tri++;
            }

            string inputStr;
            mf.fin >> inputStr;
            double input = stringToDouble(inputStr);

            switch (i) {
            case 0:
                tris[tri].verts[vert].x = input;
                break;
            case 1:
                tris[tri].verts[vert].y = input;
                break;
            case 2:
                tris[tri].verts[vert].z = input;
                break;
            }
            ++i;
        }
        for (i = 0; i < numTris; ++i) {
            tris[i].calculateNormal();
            tris[i].calculateD();
        }
        break;
    case Filetype::OBJ:
        while (mf.fin.good() && triIdx < MAXTRIS && vertIdx < MAXTRIS * 3) {
            string line;
            getline(mf.fin, line);

            if (line.length() > 2 && line.at(0) == 'v' && line.at(1) == ' ') {
                line = line.substr(2);
                int split = 0;
                int idx = 0;
                for (int i = 0; i < line.length(); i++) {
                    if (line.at(i) == ' ') {
                        double value = stringToDouble(line.substr(split, i));
                        switch (idx) {
                        case 0:
                            verts[vertIdx].x = value;
                            break;
                        case 1:
                            verts[vertIdx].y = value;
                            break;
                        case 2:
                            verts[vertIdx].z = value;
                            break;
                        }
                        split = i + 1;
                        idx++;
                    }
                }
                if (idx < 3) {
                    double value = stringToDouble(line.substr(split, i));
                    switch (idx) {
                    case 0:
                        verts[vertIdx].x = value;
                        break;
                    case 1:
                        verts[vertIdx].y = value;
                        break;
                    case 2:
                        verts[vertIdx].z = value;
                        break;
                    }
                }
                vertIdx++;
            }
            if (line.length() > 2 && line.at(0) == 'f' && line.at(1) == ' ') {
                line = line.substr(2);
                bool validSplit = true;
                int split = 0;
                int idx = 0;
                for (int i = 0; i < line.length(); i++) {
                    if (line.at(i) == ' ' && !validSplit) {
                        split = i + 1;
                        validSplit = true;
                    }
                    if ((line.at(i) == '/' || line.at(i) == ' ') && validSplit) {
                        validSplit = false;
                        int value = stringToInt(line.substr(split, i));
                        tris[triIdx].verts[idx] = verts[value];
                        if (line.at(i) == ' ')
                            split = i + 1;
                        idx++;
                    }
                }
                if (idx < 3 && validSplit) {
                    int value = stringToInt(line.substr(split, i));
                    tris[triIdx].verts[idx] = verts[value];
                }
                triIdx++;
            }
        }
        break;
    case Filetype::PLY:

        break;
    case Filetype::STL:

        break;
    }

    delete[] verts;
}

#endif