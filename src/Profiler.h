// Utility Class
#ifndef PROFILER_H
#define PROFILER_H

#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <ctime>

#include "Defines.h"
#include "RenderUtils.h"

struct Segment {
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point end;
	std::string name;
	bool ended;
};

struct SegmentStats {
	std::string name;
	double maxTime;
	double minTime;
	double totalTime;
	int count;
};

class Profiler {
protected:
	std::vector<Segment> segments;
	static std::vector<Segment> segmentArchive;
public:
	// Creates a timestamp to define the beginning of a segment
	void start(std::string);
	// Creates a timestamp to define the end of a segment
	void end();
	// Prints the name and elapsed time of all recorded segments
	void printSegments();
	// Archives segment list for future comparison
	void archiveSegments();
	// Calculates average, max, and min time taken for each name used within archived segments
	void printArchiveComparison();
	// Clears segment list
	void clear();
};

#endif