#ifndef MESH_CPP	
#define MESH_CPP

#include <fstream>
#include <vector>
#include "Mesh.h"
#include "RenderUtils.h"
#include "QuickSort.h"

using namespace std;

void Mesh::buildMesh(MeshFile& mf) {
    int vert = 0;
    int tri = 0;
    int i = 0;
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

            Triangle currentTri;
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
        numTris = tris.size();
        for (i = 0; i < numTris; ++i) {
            tris[i].calculateNormal();
        }
        break;
    case Filetype::OBJ:
        while (mf.fin.good()) {
            string line;
            getline(mf.fin, line);

            if (line.length() > 2 && line.at(0) == 'v' && line.at(1) == ' ') {
                line = line.substr(2);
                Point currentPt;
                int split = 0;
                int idx = 0;
                for (int i = 0; i < line.length(); i++) {
                    if (line.at(i) == ' ') {
                        double value = stringToDouble(line.substr(split, i - split));
                        switch (idx) {
                        case 0:
                            currentPt.x = value;
                            break;
                        case 1:
                            currentPt.y = value;
                            break;
                        case 2:
                            currentPt.z = value;
                            break;
                        }
                        split = i + 1;
                        idx++;
                    }
                }
                if (idx < 3) {
                    double value = stringToDouble(line.substr(split, i - split));
                    switch (idx) {
                    case 0:
                        currentPt.x = value;
                        break;
                    case 1:
                        currentPt.y = value;
                        break;
                    case 2:
                        currentPt.z = value;
                        break;
                    }
                }
                verts.push_back(currentPt);
            }
            if (line.length() > 2 && line.at(0) == 'f' && line.at(1) == ' ') {
                line = line.substr(2);
                bool validSplit = true;
                int split = 0;
                int idx = 0;
                Triangle currentTri;
                for (int i = 0; i < line.length(); i++) {
                    if ((line.at(i) == '/' || line.at(i) == ' ') && validSplit) {
                        validSplit = false;
                        int value = stringToInt(line.substr(split, i - split)) - 1;
                        currentTri.verts[idx] = verts[value];
                        if (line.at(i) == ' ')
                            split = i + 1;
                        idx++;
                    }
                    else if (line.at(i) == ' ' && !validSplit) {
                        split = i + 1;
                        validSplit = true;
                    }
                }
                if (idx < 3 && validSplit) {
                    int value = stringToInt(line.substr(split, i - split)) - 1;
                    currentTri.verts[idx] = verts[value];
                }
                currentTri.calculateNormal();
                tris.push_back(currentTri);
                triIdx++;
            }
        }
        numTris = tris.size();
        break;
    case Filetype::PLY:

        break;
    case Filetype::STL:

        break;
    }

    calcCenter();
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
std::vector<double> Mesh::calculateIntersectDistances(const Point& origin, const std::vector<Vector> rays) const {
    // Get the normal of the triangle
    // Get the vector from any point on the triangle to the origin
    // Distance = (normal dot (triangle to origin)) / (normal dot normalized ray)
    // If the distance is negative, it is behind the origin of the ray
    // Find the point of intersection by adding distance * normalized ray to the origin coordinates
    // Check for intersection within triangle
    std::vector<double> distances;
    for (int rayIdx = 0; rayIdx < rays.size(); rayIdx++) {
        std::vector<double> rayDistances;
        for (int triIdx = 0; triIdx < tris.size(); triIdx++) {
            if (tris[triIdx].checkWithin(rays[rayIdx], origin)) {
                rayDistances.push_back(1.0);
            }
        }
        
        double min = DBL_MAX;
        bool set = false;
        for (double dist : rayDistances) {
            if (dist < min) {
                min = dist;
                set = true;
            }
        }
        if (set)
            distances.push_back(min);
        else
            distances.push_back(-1);
    }
    return distances;
}

#endif