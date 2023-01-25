#ifndef FRAME_CPP
#define FRAME_CPP

#include "Frame.h"

Frame::Frame(const std::vector<int> pixels, const int height, const int width) {
	this->pixels = pixels;
	this->height = height;
	this->width = width;
}
Frame::Frame(const std::vector<double> distances, const int height, const int width, const bool willDither) {
	this->height = height;
	this->width = width;
	// Calculate the minimum distance for brightness falloff
	double minDist = DBL_MAX;
	double maxDist = 0;
	for (int idx = 0; idx < distances.size(); idx++) {
		if (distances[idx] < minDist && distances[idx] > 0)
			minDist = distances[idx];
		if (distances[idx] > maxDist)
			maxDist = distances[idx];
	}
	if (maxDist <= minDist)
		maxDist = DBL_MAX;
	double falloff = maxDist - minDist;
	// For each pixel, the "brightness" is the number of rays in that pixel that have an intersection, scaled linearly by distance
	std::cout << "Calculating brightness values" << std::endl;
	if (willDither) {
		std::vector<double> brightness;
		for (int idx = 0; idx < distances.size(); idx++) {
			double brightDouble = 0;
			if (distances[idx] > 0) {
				brightDouble = 1.0 - (distances[idx] - minDist) / falloff;
				// Make sure value is between FALLOFFMIN and 1
				brightDouble = (brightDouble >= 1) ? 1 : brightDouble;
				brightDouble = (brightDouble <= FALLOFFMIN) ? FALLOFFMIN : brightDouble;
			}
			brightness.push_back(brightDouble);
		}
		dither(brightness, height, width);
	}
	else {
		std::vector<int> pixelBrightness(height * width, 0);
		for (int idx = 0; idx < distances.size(); idx++) {
			double brightness = 0;
			if (distances[idx] > 0) {
				brightness = 1.0 - (distances[idx] - minDist) / falloff;
				// Make sure value is between FALLOFFMIN and 1
				brightness = (brightness >= 1) ? 1 : brightness;
				brightness = (brightness <= FALLOFFMIN) ? FALLOFFMIN : brightness;

				brightness *= 10;
				std::string grayscale = GRAYSCALE;
				if (brightness - intPart(brightness) >= 0.5)
					brightness += 1;
				int brightInt = brightness;
				if (brightInt > grayscale.length() - 1)
					brightInt = grayscale.length() - 1;
				if (brightInt < 0)
					brightInt = 0;

				pixelBrightness[idx] = brightInt;
			}
		}
		this->pixels = pixelBrightness;
		trimPixels();
	}
}

void Frame::print(void) {
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			std::string grayscale = GRAYSCALE;
			std::cout << grayscale[pixels[row * width + col]];
		}
		std::cout << std::endl;
	}
}
void Frame::dither(std::vector<double> brightness, const int height, const int width) {
	this->height = height;
	this->width = width;
	std::string grayscale = GRAYSCALE;
	std::vector<int> pixelData(height * width, 0);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			double oldPixel = 10 * brightness[row * width + col];
			double newPixel = oldPixel;
			if (newPixel - intPart(newPixel) >= 0.5)
				newPixel += 1;
			newPixel = (int)newPixel;
			if (newPixel > grayscale.length() - 1)
				newPixel = grayscale.length() - 1;
			if (newPixel < 0)
				newPixel = 0;
			if (col > 0 && row < height - 1 && col < width - 1) {
				double quantError = oldPixel - newPixel;
				brightness[row * width + (col + 1)] += (quantError * 7.0 / 16.0) / 10.0;
				brightness[(row + 1) * width + (col - 1)] += (quantError * 3.0 / 16.0) / 10.0;
				brightness[(row + 1) * width + col] += (quantError * 5.0 / 16.0) / 10.0;
				brightness[(row + 1) * width + (col + 1)] += (quantError * 1.0 / 16.0) / 10.0;
			}
			pixelData[row * width + col] = newPixel;
		}
	}
	pixels = pixelData;
	trimPixels();
}
// Deletes any blank rows from a given vector (of doubles) of pixel brightness, before putting one blank row along the top and bottom for spacing
std::vector<double> Frame::trimPixels(const std::vector<double>& pixels, int& height, const int width) {
	int removedRows = 0;
	std::vector<double> output;

	// Insert blank row
	for (int col = 0; col < width; col++) {
		output.push_back(0);
	}

	// Insert any non-zero rows
	for (int row = 0; row < height; row++) {
		bool isEmpty = true;
		for (int col = 0; col < width; col++) {
			if (pixels[row * width + col] > 0)
				isEmpty = false;
		}

		if (!isEmpty) {
			for (int col = 0; col < width; col++) {
				output.push_back(pixels[row * width + col]);
			}
		}
		else
			removedRows++;
	}

	// Insert blank row
	for (int col = 0; col < width; col++) {
		output.push_back(0);
	}

	removedRows -= 2; // Account for 2 added blank rows
	height -= removedRows;

	return output;
}
std::vector<int> Frame::trimPixels(const std::vector<int>& pixels, int& height, const int width) {
	int removedRows = 0;
	std::vector<int> output;

	// Insert blank row
	for (int col = 0; col < width; col++) {
		output.push_back(0);
	}

	// Insert any non-zero rows
	for (int row = 0; row < height; row++) {
		bool isEmpty = true;
		for (int col = 0; col < width; col++) {
			if (pixels[row * width + col] > 0)
				isEmpty = false;
		}

		if (!isEmpty) {
			for (int col = 0; col < width; col++) {
				output.push_back(pixels[row * width + col]);
			}
		}
		else
			removedRows++;
	}

	// Insert blank row
	for (int col = 0; col < width; col++) {
		output.push_back(0);
	}

	removedRows -= 2; // Account for 2 added blank rows
	height -= removedRows;

	return output;
}
void Frame::trimPixels(void) {
	pixels = trimPixels(pixels, height, width);
}

#endif