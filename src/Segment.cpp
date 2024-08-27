#ifndef SEGMENT_CPP
#define SEGMENT_CPP

#include "Segment.h"

Segment::Segment(const std::vector<Triangle> triangles, const Segment* parent) {
	Point min;
	Point max;

	numChildren = 0;
	for (int idx = 0; idx < triangles.size(); ++idx) {
		tris.push_back(triangles[idx]);
		for (int jdx = 0; jdx < 3; ++jdx) {
			if (idx != 0 || jdx != 0) {
				if (tris[idx].verts[jdx].x < min.x)
					min.x = tris[idx].verts[jdx].x;
				else if (tris[idx].verts[jdx].x > max.x)
					max.x = tris[idx].verts[jdx].x;
				if (tris[idx].verts[jdx].y < min.y)
					min.y = tris[idx].verts[jdx].y;
				else if (tris[idx].verts[jdx].y > max.y)
					max.y = tris[idx].verts[jdx].y;
				if (tris[idx].verts[jdx].z < min.z)
					min.z = tris[idx].verts[jdx].z;
				else if (tris[idx].verts[jdx].z > max.z)
					max.z = tris[idx].verts[jdx].z;
			} else {
				min.x = tris[idx].verts[jdx].x;
				max.x = tris[idx].verts[jdx].x;
				min.y = tris[idx].verts[jdx].y;
				max.y = tris[idx].verts[jdx].y;
				min.z = tris[idx].verts[jdx].z;
				max.z = tris[idx].verts[jdx].z;
			}
		}
	}
	numTris = tris.size();
	center = (max - min) / 2;
	bounds.I = abs(max.x - center.x);
	bounds.J = abs(max.y - center.y);
	bounds.K = abs(max.z - center.z);
	this->parent = parent;
}
Segment::Segment(const std::vector<Segment> segments, const Segment* parent) {
	Point min;
	Point max;
	Point tempMin;
	Point tempMax;

	numTris = 0;
	for (int idx = 0; idx < segments.size(); ++idx) {
		children.push_back(segments[idx]);
		tempMax = segments[idx].center + segments[idx].bounds;
		tempMin = segments[idx].center - segments[idx].bounds;
		if (idx != 0) {
			if (tempMax.x > max.x)
				max.x = tempMax.x;
			else if (tempMin.x < min.x)
				min.x = tempMin.x;
			if (tempMax.y > max.y)
				max.y = tempMax.y;
			else if (tempMin.y < min.y)
				min.y = tempMin.y;
			if (tempMax.z > max.z)
				max.z = tempMax.z;
			else if (tempMin.z < min.z)
				min.z = tempMin.z;
		} else {
			max = tempMax;
			min = tempMin;
		}
	}
	numChildren = children.size();
	center = (max - min) / 2;
	bounds.I = abs(max.x - center.x);
	bounds.J = abs(max.y - center.y);
	bounds.K = abs(max.z - center.z);
	this->parent = parent;
}

// Splits a segment into multiple evenly, allocating triangles into the child segments
void Segment::splitSegment() {
	if (numTris < 1)
		return; // We can't split a segment with no triangles

	// Split the longest dimension of the bounding box
	// Make sure to apply the buffer throughout this process, however internal splits can remain unbuffered
	int splitDim = bounds.I > bounds.J ? 0 : 1;
	splitDim = splitDim > bounds.K ? splitDim : 2;
	double dimLength = (splitDim == 0 ? 2 * bounds.I : splitDim == 1 ? 2 * bounds.J : 2 * bounds.K) + 2 * SEGMENTBUFFER;
	// The segment to assign a triangle to is the floor of (the distance from "zero" divided by the split size)
	double splitLength = dimLength / NUMCHILDSEGMENTS;
	// Loop through every triangle and assign it to all segments that it has a vertex in
	std::vector<Triangle> segments[NUMCHILDSEGMENTS];
	int idx, jdx, segmentIdx;
	double vertDim;
	double minDim = (splitDim == 0 ? center.x : splitDim == 1 ? center.y : center.z) - (splitDim == 0 ? bounds.I : splitDim == 1 ? bounds.J : bounds.K) - SEGMENTBUFFER;
	int assigned[2];
	for (idx = 0; idx < numTris; ++idx) {
		assigned[0] = -1;
		assigned[1] = -1;
		for (jdx = 0; jdx < 3; ++jdx) {
			// Grab the correct dimension relative to the calculated minimum
			vertDim = (splitDim == 0 ? tris[idx].verts[jdx].x : splitDim == 1 ? tris[idx].verts[jdx].y : tris[idx].verts[jdx].z) - minDim;
			// Calculate the segment index
			segmentIdx = (int)(vertDim / splitLength);
			// Assign the triangle to that index, if not done already
			if (assigned[0] != segmentIdx && assigned[1] != segmentIdx)
				segments[segmentIdx].push_back(tris[idx]);
		}
	}
	// Assign triangles to and append new child segments
	for (idx = 0; idx < NUMCHILDSEGMENTS; ++idx)
		children.push_back(Segment(segments[idx], this));
	// Update child count, and clear triangles on this segment
	numChildren = children.size();
	tris.clear();
	numTris = tris.size();
}
// Recursively splits segments until each child segment contains fewer triangles than a set target
// Returns the total number of segments (including this one) after completing the split process
int Segment::splitSegment(int maxTriCount) {
	std::vector<Segment*> unprocessed;
	int segmentCount = 0;
	Segment* head;
	unprocessed.push_back(this);
	while (unprocessed.size() > 0) {
		// Grab next unprocessed segment
		head = unprocessed.back();
		unprocessed.pop_back();
		++segmentCount;
		if (head->numTris < maxTriCount)
			continue; // No need to split if we're already under the target
		// Split current segment and add children to be processed
		head->splitSegment();
		for (int idx = 0; idx < head->numChildren; ++idx)
			unprocessed.push_back(&head->children[idx]);
	}
}
// Converts a tree structure of segments into a set of indexed segments
// This should be used on the root segment
// Returns whether or not the function succeeded
bool Segment::indexSegments(IndexedSegment* segmentArr, Triangle* triArr, int maxSegments, int maxTris) {
	

	std::vector<Segment*> unprocessed;
	Segment* head;
	unprocessed.push_back(this);
	int triIdx = 0;
	int segmentIdx = 0;
	while (unprocessed.size() > 0) {
		// Verify array sizes
		if (triIdx+1 > maxTris) {
			fprintf(stderr, "ERROR - Segment::indexSegments: Not enough space to populate triangles! %d -> %d\n", numTris, maxTris);
			return false;
		}
		if (segmentIdx+1 > maxSegments) {
			fprintf(stderr, "ERROR - Segment::indexSegments: Not enough space to populate child segments! %d -> %d\n", numChildren, maxSegments);
			return false;
		}
		// Grab next unprocessed segment
		head = unprocessed.back();
		unprocessed.pop_back();
		// Populate head into indexed segments
		segmentArr[segmentIdx].bounds = head->bounds;
		segmentArr[segmentIdx].center = head->center;
		segmentArr[segmentIdx].childCount = head->numChildren;
		segmentArr[segmentIdx].triCount = head->numTris;
		segmentArr[segmentIdx].segmentArr = segmentArr;
		segmentArr[segmentIdx].triArr = triArr;
		segmentArr[segmentIdx].startChildrenIdx = segmentIdx;
		segmentArr[segmentIdx].startTriIdx = triIdx;
		// Add any tris into the tri array




	}
}

IndexedSegment::IndexedSegment() {
	segmentArr = NULL;
	triArr = NULL;
	startChildrenIdx = -1;
	childCount = 0;
	startTriIdx = -1;
	triCount = 0;
}
IndexedSegment::IndexedSegment(Triangle* triArr, int startTriIdx, int triCount, Point center, Vector bounds) {
	this->segmentArr = NULL;
	this->triArr = triArr;
	this->startChildrenIdx = -1;
	this->childCount = 0;
	this->startTriIdx = startTriIdx;
	this->triCount = triCount;
	this->center = center;
	this->bounds = bounds;
}
IndexedSegment::IndexedSegment(IndexedSegment* segmentArr, int startSegmentIdx, int segmentCount, Point center, Vector bounds) {
	this->segmentArr = segmentArr;
	this->triArr = NULL;
	this->startChildrenIdx = startSegmentIdx;
	this->childCount = segmentCount;
	this->startTriIdx = -1;
	this->triCount = 0;
	this->center = center;
	this->bounds = bounds;
}

#endif