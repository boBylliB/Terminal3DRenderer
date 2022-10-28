#ifndef MESH_CPP	
#define MESH_CPP

#include "Mesh.h"

void Mesh::buildMesh(MeshFile mf) {
    int vert = 0;
    int tri = 0;
    int i = 0;
    
    switch (mf.type) {
    case Filetype::TXT:
        while (fin.good() && tri < MAXTRIS) {
            if (i > 2) {
                i = 0;
                vert++;
            }
            if (vert > 2) {
                vert = 0;
                tri++;
            }

            float input[3];

            switch (i) {
            case 0:
                tris[tri].verts[vert].x = input;
                break;
            case 1:
                mesh->tris[tri].verts[vert].y = input;
                break;
            case 2:
                mesh->tris[tri].verts[vert].z = input;
                break;
            }
            ++i;
        }
        mesh->triCount = i + 1;
        for (i = 0; i < mesh->triCount; ++i) {
            vertVec.I = mesh->tris[i].verts[0].x;
            vertVec.J = mesh->tris[i].verts[0].y;
            vertVec.K = mesh->tris[i].verts[0].z;
            mesh->tris[i].normal = calculateNormal(&mesh->tris[i]);
            mesh->tris[i].D = dotProduct(mesh->tris[i].normal, vertVec);
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