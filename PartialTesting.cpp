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
/*
int main(void) {
	int outputHeight = 10;
	int outputWidth = 10;
	vector<double> intersectDistances(outputHeight * outputWidth * 9, 0.0);
	for (int i = 0; i < outputHeight * outputWidth * 9; i++) {
		if ((i / (outputWidth*3)) % 2 == 0) {
			intersectDistances[i] = 1;
		}
	}

	std::vector<double> pixelBrightness(outputHeight * outputWidth, 0.0);
	for (int idx = 0; idx < outputHeight * outputWidth * 3; idx++) {
		int interIdx = idx;
		int pixIdx = idx / 3;
		cout << pixIdx << endl;
		if (intersectDistances[interIdx] > 0)
			pixelBrightness[pixIdx] += 1.0;
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
}*/