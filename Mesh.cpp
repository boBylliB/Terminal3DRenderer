#ifndef MESH_CPP	
#define MESH_CPP

#include <fstream>
#include <vector>
#include "Mesh.h"
#include "RenderUtils.h"

using namespace std;

void Mesh::buildMesh(MeshFile mf) {
    int vert = 0;
    int tri = 0;
    int i = 0;
    int vertIdx = 0;
    int triIdx = 0;
    vector<Point> verts = vector<Point>();
    
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
        while (mf.fin.good()) {
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
}
void Mesh::calcCenter(void) {
    double xMin = 0;
    double xMax = 0;
    double yMin = 0;
    double yMax = 0;
    double zMin = 0;
    double zMax = 0;

    for (int i = 0; i < numTris; ++i) {
        for (int j = 0; j < 3; ++j) {
            Point pt = tris[i].verts[j];
            if (i == 0 && j == 0) {
                xMin = pt.x;
                xMax = pt.x;
                yMin = pt.y;
                yMax = pt.y;
                zMin = pt.z;
                zMax = pt.z;
            }
            if (pt.x < xMin)
                xMin = pt.x;
            if (pt.x > xMax)
                xMax = pt.x;
            if (pt.y < yMin)
                yMin = pt.y;
            if (pt.y > yMax)
                yMax = pt.y;
            if (pt.z < zMin)
                zMin = pt.z;
            if (pt.z > zMax)
                zMax = pt.z;
        }
    }

    center.x = (xMin + xMax) / 2;
    center.y = (yMin + yMax) / 2;
    center.z = (zMin + zMax) / 2;
}
std::vector<double> Mesh::calculateIntersectDistances(const Point origin, const std::vector<Vector> rays) const {
    // Get the normal of the triangle
    // Get D, the dot product of the normal and any point on the triangle
    // Distance = (nd - origin dot normal) / (normalized ray dot normal)
    // If the distance is negative, it is behind the origin of the ray
    // Find the point of intersection by adding distance * normalized ray to the origin coordinates
    // Check for intersection within triangle
    std::vector<double> distances;
    for (int rayIdx = 0; rayIdx < rays.size(); rayIdx++) {
        std::vector<double> rayDistances;
        for (int triIdx = 0; triIdx < tris.size(); triIdx++) {
            Vector originVector;
            Vector vertVector;
            Vector ray = rays[rayIdx];
            originVector.fromPoint(origin);
            vertVector.fromPoint(tris[triIdx].verts[0]);
            ray.normalize();
            
            double dist = (tris[triIdx].D - originVector.dot(vertVector)) / ray.dot(tris[triIdx].normal);
            if (dist > 0) {
                ray.scale(dist);
                Point diff = ray.toPoint();
                Point intersection;
                intersection.x = origin.x + diff.x;
                intersection.y = origin.y + diff.y;
                intersection.z = origin.z + diff.z;

                if (tris[triIdx].checkWithin(intersection))
                    rayDistances.push_back(dist);
            }
        }

        double min = -1;
        for (double dist : rayDistances) {
            if (dist > 0) {
                if (min == -1 || dist < min)
                    min = dist;
            }
        }
        distances[rayIdx] = min;
    }
    return distances;
}

#endif