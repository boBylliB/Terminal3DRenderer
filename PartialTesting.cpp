#include <iostream>
#include <cstdlib>
#include <string>

#include "Angle.h"
#include "Camera.h"
#include "Defines.h"
#include "Mesh.h"
#include "MeshFile.h"
#include "Point.h"
#include "RenderUtils.h"
#include "Triangle.h"
#include "Vector.h"

using namespace std;

int main(void) {
	int outputHeight = 10;
	int outputWidth = 10;
	vector<double> intersectDistances(outputHeight * outputWidth * 9, 0.0);
	for (int i = 50; i < 100; i++) {
		intersectDistances[i] = 1;
	}

	std::vector<double> pixelBrightness(outputHeight * outputWidth, 0.0);
	for (int row = 0; row < outputHeight * 3; row++) {
		for (int col = 0; col < outputWidth * 3; col++) {
			if (intersectDistances[row*outputWidth + col] > 0)
				pixelBrightness[(row / 3) * outputWidth + (col / 3)] += 1.0;
		}
	}
	// Display the calculated image to the screen
	for (int row = 0; row < outputHeight; row++) {
		for (int col = 0; col < outputWidth; col++) {
			if (pixelBrightness[row*outputWidth + col] > 0)
				std::cout << "@";
			else
				std::cout << " ";
		}
		std::cout << std::endl;
	}

	return 0;
}