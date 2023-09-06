#ifndef SEGMENTEDMESH_CPP	
#define SEGMENTEDMESH_CPP

#include <fstream>
#include <vector>
#include "SegmentedMesh.h"
#include "RenderUtils.h"

using namespace std;

SegmentedMesh::SegmentedMesh(void) {
    Mesh m;
    calcSegments(m);
}
SegmentedMesh::SegmentedMesh(MeshFile& mf) {
    Mesh m(mf);
    calcSegments(m);
}
SegmentedMesh::SegmentedMesh(Mesh m) {
    calcSegments(m);
}

void SegmentedMesh::calcSegments(Mesh m) {
    center = m.center;
    bounds = m.bounds;
    // Calculate split offsets
    double initialX = center.x - bounds.I;
    double stepX = (2 * bounds.I) / (NUMSPLITSX + 1);
    double initialY = center.y - bounds.J;
    double stepY = (2 * bounds.J) / (NUMSPLITSY + 1);
    double initialZ = center.z - bounds.K;
    double stepZ = (2 * bounds.K) / (NUMSPLITSZ + 1);
    // Move triangles into temporary vector to be able to drop them once processed
    std::vector<Triangle> tempTris;
    for (int idx = 0; idx < m.numTris; idx++) {
        tempTris.push_back(m.tris[idx]);
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
                numTris.push_back(segTris.size());
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

#endif