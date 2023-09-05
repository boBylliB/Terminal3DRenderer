#ifndef PROFILER_CPP
#define PROFILER_CPP

#include "Profiler.h"

std::vector<Segment> Profiler::segmentArchive;

// Creates a timestamp to define the beginning of a segment
void Profiler::start(std::string name = "_") {
	Segment segment;
	if (name != "_") segment.name = name;
	else segment.name = segments.size();
	segment.ended = false;
	segment.start = std::chrono::system_clock::now();
	segments.push_back(segment);
}
// Creates a timestamp to define the end of a segment
void Profiler::end() {
	if (!segments[segments.size() - 1].ended) {
		segments[segments.size() - 1].end = std::chrono::system_clock::now();
		segments[segments.size() - 1].ended = true;
	}
}
// Prints the name and elapsed time of all recorded segments
void Profiler::printSegments() {
	for (Segment segment : segments) {
		if (!segment.ended) continue;

		std::chrono::duration<double> elapsed_seconds = segment.end - segment.start;
		// Gets the end time in a readable format
		std::time_t end_time = std::chrono::system_clock::to_time_t(segment.end);
		char tmBuff[30];
		ctime_s(tmBuff, sizeof(tmBuff), &end_time);
		// Prints the statistics to the screen in a readable format
		std::cout << "Time segment " << segment.name << " completed at " << tmBuff
			<< "elapsed time: " << elapsed_seconds.count() << "s"
			<< std::endl;
	}
}
// Archives segment list for future comparison
void Profiler::archiveSegments() {
	for (Segment segment : segments) {
		if (segment.ended) segmentArchive.push_back(segment);
	}
}
// Calculates average, max, and min time taken for each name used within archived segments
void Profiler::printArchiveComparison() {
	std::vector<SegmentStats> statistics;

	for (Segment segment : segmentArchive) {
		bool foundStats = false;
		std::chrono::duration<double> elapsed_seconds = segment.end - segment.start;

		for (int idx = 0; idx < statistics.size(); idx++) {
			if (statistics[idx].name == segment.name) {
				foundStats = true;
				statistics[idx].count++;
				statistics[idx].totalTime += elapsed_seconds.count();
				if (elapsed_seconds.count() > statistics[idx].maxTime) statistics[idx].maxTime = elapsed_seconds.count();
				if (elapsed_seconds.count() < statistics[idx].minTime) statistics[idx].minTime = elapsed_seconds.count();
			}
		}

		if (!foundStats) {
			SegmentStats stats;
			stats.name = segment.name;
			stats.count = 1;
			stats.totalTime = elapsed_seconds.count();
			stats.maxTime = elapsed_seconds.count();
			stats.minTime = elapsed_seconds.count();
			statistics.push_back(stats);
		}
	}

	for (SegmentStats stats : statistics) {
		std::cout << "==================================================" << std::endl;
		std::cout << "Time segment: " << stats.name << std::endl;
		std::cout << "Max Time: " << stats.maxTime << std::endl;
		std::cout << "Min Time: " << stats.minTime << std::endl;
		std::cout << "Average Time: " << stats.totalTime / stats.count << std::endl;
	}
}
// Clears segment list
void Profiler::clear() {
	segments.clear();
}

#endif