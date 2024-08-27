// Container Class
#ifndef SEGMENT_H
#define SEGMENT_H

#include <vector>
#include "Defines.h"
#include "Triangle.h"
#include "Vector.h"

class Segment {
public:
	const Segment* parent;
	std::vector<Segment> children;
	int numChildren;
	std::vector<Triangle> tris;
	int numTris;
	Point center;
	Vector bounds;

	Segment(const std::vector<Triangle> triangles, const Segment* parent);
	Segment(const std::vector<Segment> segments, const Segment* parent);

	// Splits a segment into multiple evenly, allocating triangles into the child segments
	void splitSegment();
	// Recursively splits segments until each child segment contains fewer triangles than a set target
	// Returns the total number of segments (including this one) after completing the split process
	int splitSegment(int maxTriCount);
	// Converts a tree structure of segments into a set of indexed segments
	// Returns whether or not the function succeeded
	bool indexSegments(IndexedSegment* segmentArr, Triangle* triArr, int maxSegments, int maxTris);
};

class IndexedSegment {
public:
	IndexedSegment* segmentArr;
	Triangle* triArr;
	int startChildrenIdx;
	int childCount;
	int startTriIdx;
	int triCount;
	Point center;
	Vector bounds;

	IndexedSegment();
	IndexedSegment(Triangle* triArr, int startTriIdx, int triCount, Point center, Vector bounds);
	IndexedSegment(IndexedSegment* segmentArr, int startSegmentIdx, int segmentCount, Point center, Vector bounds);
};

#endif