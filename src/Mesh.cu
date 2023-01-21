#ifndef MESH_CPP	
#define MESH_CPP

#include <fstream>
#include <vector>
#include "Mesh.cuh"
#include "RenderUtils.h"

using namespace std;

Mesh::Mesh(void) {
    MeshFile mf;
    buildMesh(mf);
}
Mesh::Mesh(MeshFile& mf) {
    buildMesh(mf);
}

void Mesh::buildMesh(MeshFile& mf) {
    int vert = 0;
    int tri = 0;
    int i = 0;
    int triIdx = 0;
    vector<Point> verts = vector<Point>();
    
    switch (mf.type) {
    case Filetype::TXT:
        /* TXT FILE FORMAT (custom):
        *  1 triangle {
        *  x, y, z
        *  x, y, z
        *  x, y, z
        *
        *  }
        *   triangles are separated by a blank line, doesn't matter to the program but makes it easier to read
        *   points should be listed counterclockwise in order to calculate normals correctly (right hand rule)
        */
        while (mf.fin.good()) {
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
                currentTri.verts[vert].x = input;
                break;
            case 1:
                currentTri.verts[vert].y = input;
                break;
            case 2:
                currentTri.verts[vert].z = input;
                break;
            }
            tris.push_back(currentTri);
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
        // Space for future implementation (MeshFile.cpp should block code execution from reaching here until this is implemented)
        break;
    case Filetype::STL:
        // Space for future implementation
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

    // Find the maximum and minimum displacements from origin for each coordinate axis
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
    // Center is exactly between each extreme
    center.x = (xMin + xMax) / 2;
    center.y = (yMin + yMax) / 2;
    center.z = (zMin + zMax) / 2;
}
double Mesh::calculateIntersectDistance(const Point& origin, const Vector& ray) const {
    // Get the normal of the triangle
    // Get the vector from any point on the triangle to the origin
    // Distance = (normal dot (triangle to origin)) / (normal dot normalized ray)
    // If the distance is negative, it is behind the origin of the ray
    // Find the point of intersection by adding distance * normalized ray to the origin coordinates
    // Check for intersection within triangle
    double rayDistance = DBL_MAX;
    bool set = false;
    for (int triIdx = 0; triIdx < tris.size(); triIdx++) {
        if (tris[triIdx].checkWithin(ray, origin)) {
            Vector originVector(origin, tris[triIdx].verts[0]);
            Vector normal = tris[triIdx].normal;
            Vector normRay = ray;
            normRay.normalize();

            double dist = (normal.dot(originVector)) / normal.dot(normRay);
            if (dist < rayDistance) {
                rayDistance = dist;
                set = true;
            }
        }
    }
    if (set)
        return rayDistance;
    else
        return -1;
}
std::vector<double> Mesh::calculateIntersectDistances(const Point& origin, const std::vector<Vector> rays, const bool showProgress) const {
    int progress = 0;
    int progressTarget = 0;
    std::vector<double> distances;
    for (int rayIdx = 0; rayIdx < rays.size(); rayIdx++) {
        distances.push_back(calculateIntersectDistance(origin, rays[rayIdx]));

        // Displays progress in percentage to the screen
        if (rayIdx > progressTarget && showProgress) {
            cout << "Intersect Distance Progress: " << progress << "%" << endl;
            progress += 1;
            double progressPct = progress / 100.0;
            progressTarget = rays.size() * progressPct;
        }
    }
    return distances;
}

#endif