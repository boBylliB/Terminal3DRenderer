// Container Class
#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <string>
#include <iostream>

#include "Defines.h"
#include "RenderUtils.h"

class Frame {
public:
	std::vector<int> pixels;
	std::string grayscale;
	int height;
	int width;

	Frame(const std::vector<int>, const int, const int, const std::string = GRAYSCALE);
	Frame(const std::vector<double>, const int, const int, const bool = false, const std::string = GRAYSCALE);

	// Prints the frame to screen
	void print(void);
	// Applies a simple version of Floyd-Steinberg Dithering to the given vector of doubles and applies it as the new frame data
	void dither(std::vector<double>, const int, const int);
	// Deletes any blank rows from a given vector (of doubles) of pixel brightness, before putting one blank row along the top and bottom for spacing
	std::vector<double> trimPixels(const std::vector<double>&, int&, const int);
	std::vector<int> trimPixels(const std::vector<int>&, int&, const int);
	void trimPixels(void);
};

#endif