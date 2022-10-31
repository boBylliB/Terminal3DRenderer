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
            float input = stringToFloat(inputStr);

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
        int vertIdx = 0;
        int triIdx = 0;
        Point verts[MAXTRIS * 3];
        while (mf.fin.good() && triIdx < MAXTRIS && vertIdx < MAXTRIS * 3) {
            string line;
            getline(mf.fin, line);

            if (line.length() > 2 && line.at(0) == 'v' && line.at(1) == ' ') {
                line = line.substr(2);
                int split = 0;
                int idx = 0;
                for (int i = 0; i < line.length(); i++) {
                    if (line.at(i) == ' ') {
                        float value = stringToFloat(line.substr(split, i));
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
                vertIdx++;
            }
        }
        break;
    case Filetype::PLY:

        break;
    case Filetype::STL:

        break;
    }
}

#endif