#ifndef TRIANGLE_CPP
#define TRIANGLE_CPP

#include "Triangle.h"

bool Triangle::checkWithin(const Point &pt) const {
    Vector edgeA(verts[1].x - verts[0].x, verts[1].y - verts[0].y, verts[1].z - verts[0].z);
    Vector edgeB(verts[2].x - verts[1].x, verts[2].y - verts[1].y, verts[2].z - verts[1].z);
    Vector edgeC(verts[0].x - verts[2].x, verts[0].y - verts[2].y, verts[0].z - verts[2].z);
    Vector cA(pt.x - verts[0].x, pt.y - verts[0].y, pt.z - verts[0].z);
    Vector cB(pt.x - verts[1].x, pt.y - verts[1].y, pt.z - verts[1].z);
    Vector cC(pt.x - verts[2].x, pt.y - verts[2].y, pt.z - verts[2].z);

    bool check1 = normal.dot(edgeA.cross(cA)) > 0.0;
    bool check2 = normal.dot(edgeB.cross(cB)) > 0.0;
    bool check3 = normal.dot(edgeC.cross(cC)) > 0.0;

    return (check1 && check2 && check3);
}
void Triangle::calculateNormal(void) {
    Vector rAB(verts[1].x - verts[0].x, verts[1].y - verts[0].y, verts[1].z - verts[0].z);
    Vector rBC(verts[2].x - verts[1].x, verts[2].y - verts[1].y, verts[2].z - verts[1].z);
    Vector cross = rAB.cross(rBC);
    cross.normalize();
    normal = cross;
}
void Triangle::calculateD(void) {
    Vector vertVec(verts[0].x, verts[0].y, verts[0].z);
    D = normal.dot(vertVec);
}
Triangle::Triangle(const Point pts[3]) {
    verts[0] = pts[0];
    verts[1] = pts[1];
    verts[2] = pts[2];

    calculateNormal();
    calculateD();
}

#endif