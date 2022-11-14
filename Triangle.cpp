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
    Vector limitA(origin, verts[0]);
    Vector limitB(origin, verts[1]);
    Vector limitC(origin, verts[2]);

    limitA.normalize();
    limitB.normalize();
    limitC.normalize();
    dir.normalize();

    Vector planeA = limitB.cross(limitC);
    Vector planeB = limitA.cross(limitC);
    Vector planeC = limitA.cross(limitB);

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