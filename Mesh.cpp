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

        break;
    case Filetype::PLY:

        break;
    case Filetype::STL:

        break;
    }
}

#endif