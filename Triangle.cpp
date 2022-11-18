#ifndef TRIANGLE_CPP
#define TRIANGLE_CPP

#include "Triangle.h"

Triangle::Triangle(void) {
    Point emptyPt = {};

    verts[0] = emptyPt;
    verts[1] = emptyPt;
    verts[2] = emptyPt;
}
Triangle::Triangle(const Point pts[3]) {
    verts[0] = pts[0];
    verts[1] = pts[1];
    verts[2] = pts[2];

    calculateNormal();
}

bool Triangle::checkWithin(Vector dir, const Point &origin) const {
    // Vectors from ray origin to each vertex as the bounds
    Vector limitA(origin, verts[0]);
    Vector limitB(origin, verts[1]);
    Vector limitC(origin, verts[2]);

    // Normalize everything for ease of calculation
    limitA.normalize();
    limitB.normalize();
    limitC.normalize();
    dir.normalize();

    // Create limiting planes using the bounding vectors
    Vector planeA = limitB.cross(limitC);
    Vector planeB = limitA.cross(limitC);
    Vector planeC = limitA.cross(limitB);

    // If the tested vector is on the same side of each plane as the one bounding vector not within the test plane
    // Therefore, the only way that it could be on the "inside" of each plane is if the tested vector is between the bounding vectors
    bool testA = limitA.dot(planeA) * dir.dot(planeA) > 0;
    bool testB = limitB.dot(planeB) * dir.dot(planeB) > 0;
    bool testC = limitC.dot(planeC) * dir.dot(planeC) > 0;

    return (testA && testB && testC);
}
void Triangle::calculateNormal(void) {
    Vector rAB(verts[1].x - verts[0].x, verts[1].y - verts[0].y, verts[1].z - verts[0].z);
    Vector rBC(verts[2].x - verts[1].x, verts[2].y - verts[1].y, verts[2].z - verts[1].z);
    Vector cross = rAB.cross(rBC);
    cross.normalize();
    normal = cross;
}

#endif