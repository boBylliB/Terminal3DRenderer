#ifndef MESH_CPP	
#define MESH_CPP

#include <fstream>
#include <vector>
#include "Mesh.h"
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
        // FUTURE IMPLEMENTATION! Need to write program to convert binary IEEE 754 little-endian 32-bit floating point nums in the binary STL to data in a mesh
        bool ascii = false;
        string header = "";
        if (mf.fin.good()) {
            char start[5];
            mf.fin.get(start, 5);
            if (start == "solid") {
                ascii = true;
                string restOfHeader;
                getline(mf.fin, restOfHeader);
                header += restOfHeader;
            }
            else {
                header += start;
                char restOfHeader[75];
                mf.fin.get(restOfHeader, 75);
                header += restOfHeader;
            }
        }
        while (mf.fin.good()) {
            if (ascii) {
                string line;
                getline(mf.fin, line);
                trimString(line);

                string keyword;
                vector<string> data;
                int split = 0;
                for (int idx = 0; idx < line.length(); idx++) {
                    if (line.at(idx) == ' ') {
                        if (split == 0)
                            keyword = line.substr(split, idx - split);
                        else
                            data.push_back(line.substr(split, idx - split));
                        split = idx + 1;
                    }
                }

                if (keyword == "vertex") {
                    Point currentPt;
                    currentPt.x = stringToDouble(data[0]);
                    currentPt.y = stringToDouble(data[1]);
                    currentPt.z = stringToDouble(data[2]);
                    verts[vert] = currentPt;
                    vert++;

                    if (vert > 2) {
                        vert = 0;
                        Triangle currentTri;
                        currentTri.verts[0] = verts[0];
                        currentTri.verts[1] = verts[1];
                        currentTri.verts[2] = verts[2];
                        currentTri.calculateNormal();
                        tris.push_back(currentTri);
                    }
                }
            }
            else {

            }
        }
        numTris = tris.size();
        break;
    }

    calcBounds();
}
void Mesh::calcBounds(void) {
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
    // Extremes now need to be adjusted relative to the center
    bounds.I = abs(xMax - center.x);
    bounds.J = abs(yMax - center.y);
    bounds.K = abs(zMax - center.z);
}
void Mesh::calcSegments(void) {
    // Calculate split offsets
    double initialX = center.x - bounds.I;
    double stepX = (2 * bounds.I) / (NUMSPLITSX + 1);
    double initialY = center.y - bounds.J;
    double stepY = (2 * bounds.J) / (NUMSPLITSY + 1);
    double initialZ = center.z - bounds.K;
    double stepZ = (2 * bounds.K) / (NUMSPLITSZ + 1);
    // Move triangles into temporary vector to be able to drop them once processed
    std::vector<Triangle> tempTris;
    for (int idx = 0; idx < numTris; idx++) {
        tempTris.push_back(tris[idx]);
    }
    // Pull triangles into segments
    for (int idx = 0; idx < NUMSPLITSX; idx++) {
        for (int jdx = 0; jdx < NUMSPLITSY; jdx++) {
            for (int kdx = 0; kdx < NUMSPLITSZ; kdx++) {
                // Create initial bounding box
                Vector bnds;
                bnds.I = stepX / 2;
                bnds.J = stepY / 2;
                bnds.K = stepZ / 2;
                Point ctr;
                ctr.x = initialX + bnds.I + stepX * idx;
                ctr.y = initialY + bnds.J + stepY * jdx;
                ctr.z = initialZ + bnds.K + stepZ * kdx;
                // Collect all triangles within that bounding box (taking into account a small buffer for floating point errors)
                float minX = ctr.x - bnds.I - SEGMENTBUFFER;
                float maxX = ctr.x + bnds.I + SEGMENTBUFFER;
                float minY = ctr.y - bnds.J - SEGMENTBUFFER;
                float maxY = ctr.y + bnds.J + SEGMENTBUFFER;
                float minZ = ctr.z - bnds.K - SEGMENTBUFFER;
                float maxZ = ctr.z + bnds.K + SEGMENTBUFFER;
                std::vector<Triangle> segTris;
                for (int triIdx = 0; triIdx < tempTris.size(); triIdx++) {
                    // If all vertices exist within the bounding box, we can safely assume we never have to check it again
                    // Otherwise, it likely will exist within multiple segments, and we shouldn't remove it
                    bool allWithin = true;
                    bool within = false;
                    for (int vertIdx = 0; vertIdx < 3; vertIdx++) {
                        Point vert = tempTris[triIdx].verts[vertIdx];
                        if (vert.x > minX && vert.x < maxX && vert.y > minY && vert.y < maxY && vert.z > minZ && vert.z < maxZ) within = true;
                        else allWithin = false;
                    }
                    if (within) segTris.push_back(tempTris[triIdx]);
                    if (allWithin) tempTris.erase(tempTris.begin() + triIdx);
                }
                // Calculate actual segment bounding box & associated variables
                triSegments.push_back(segTris);
                numTrisPerSegment.push_back(segTris.size());
                // Find the maximum and minimum displacements from origin for each coordinate axis
                for (int triIdx = 0; triIdx < segTris.size(); ++triIdx) {
                    for (int vertIdx = 0; vertIdx < 3; ++vertIdx) {
                        Point pt = segTris[triIdx].verts[vertIdx];
                        if (triIdx == 0 && vertIdx == 0) {
                            minX = pt.x;
                            maxX = pt.x;
                            minY = pt.y;
                            maxY = pt.y;
                            minZ = pt.z;
                            maxZ = pt.z;
                        }
                        if (pt.x < minX)
                            minX = pt.x;
                        if (pt.x > maxX)
                            maxX = pt.x;
                        if (pt.y < minY)
                            minY = pt.y;
                        if (pt.y > maxY)
                            maxY = pt.y;
                        if (pt.z < minZ)
                            minZ = pt.z;
                        if (pt.z > maxZ)
                            maxZ = pt.z;
                    }
                }
                // Center is exactly between each extreme
                ctr.x = (minX + maxX) / 2;
                ctr.y = (minY + maxY) / 2;
                ctr.z = (minZ + maxZ) / 2;
                // Extremes now need to be adjusted relative to the center
                bnds.I = abs(maxX - ctr.x);
                bnds.J = abs(maxY - ctr.y);
                bnds.K = abs(maxZ - ctr.z);
                segmentCenters.push_back(ctr);
                segmentBounds.push_back(bnds);
            }
        }
    }
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
        Vector originVector(origin, tris[triIdx].verts[0]);
        Vector normal = tris[triIdx].normal;
        Vector normRay = ray;
        normRay.normalize();

        double dist = (normal.dot(originVector)) / normal.dot(normRay);
        if (dist < rayDistance && tris[triIdx].checkWithin(ray, origin)) {
            rayDistance = dist;
            set = true;
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