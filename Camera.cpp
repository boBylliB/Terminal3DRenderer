#ifndef CAMERA_CPP
#define CAMERA_CPP

#include "Camera.h"

// Getter Functions
Point Camera::getPosition(void) {
	return position;
}
Vector Camera::getDirection(void) {
	return direction;
}
Angle Camera::getFOV(void) {
	return fieldOfView;
}
double Camera::getRoll(void) {
	return roll;
}
// Returns a 2 element array, height in index 0 and width in index 1
int* Camera::getOutputSize(void) {
	int size[] = { outputHeight, outputWidth };
	return size;
}

// Setter Functions
Camera::Camera(const Point& position, const Vector& direction, const double fieldOfView, const double roll, const int height, const int width) {
	this->position = position;
	this->roll = roll;
	this->outputHeight = height;
	this->outputWidth = width;

	this->direction = direction;
	this->direction.normalize();

	// CHARRATIO attempts to even out the distortion from chars being taller than their width
	this->fieldOfView.phi = fieldOfView * CHARRATIO;
	this->fieldOfView.theta = fieldOfView;
}
void Camera::setPosition(const Point& pos) {
	position = pos;
}
void Camera::setDirection(const Vector& dir) {
	direction = dir;
	direction.normalize();
}
void Camera::setFOV(const double fieldOfView) {
	this->fieldOfView.phi = fieldOfView * CHARRATIO;
	this->fieldOfView.theta = fieldOfView;
}
void Camera::setRoll(const double r) {
	roll = r;
}
void Camera::setOutput(const int height, const int width) {
	if (width < 1)
		outputWidth = height;
	else
		outputWidth = width;
	outputHeight = height;
}

// Basic Utility Functions
// Adjusts the position by the given vector
void Camera::move(const Vector& diff) {
	position.x += diff.I;
	position.y += diff.J;
	position.z += diff.K;
}
// Adjusts the direction by rotating the current view vector by the given angle
void Camera::rotate(const Angle& a) {
	direction.rotate(a);
}
// Adjusts the FOV by the direct inverse of the given double (so that positive zoom is "zoom in")
void Camera::zoom(const double z) {
	double fov = fieldOfView.theta;
	if (z > 0)
		fov /= z;
	else if (z < 0)
		fov *= z;
	fieldOfView.theta = fov;
	fieldOfView.phi = fov;
}
// Adjusts the roll by the given double
void Camera::rollAdjust(const double r) {
	roll += r;
	while (roll > degToRad(360))
		roll -= degToRad(360);
	while (roll < 0)
		roll += degToRad(360);
}
// Rotates the camera around the given point while keeping the radius the same, sets the view to point at the orbit center
void Camera::orbit(const Angle& a, const Point& origin) {
	direction.difference(position, origin);
	rotate(a);

	Vector diff(direction.toPoint(), origin);
	move(diff);
}
// Runs the orbit function with a point generated from the current view direction and the given radius
void Camera::orbitCurrent(const Angle& a, const double radius) {
	direction.normalize();
	direction.scale(radius);
	orbit(a, direction.toPoint());
}

// Core functions (and any functions that are too complex to be considered "utility")
// Calculates a snapshot of the mesh from this camera and displays it to the screen
void Camera::display(const Mesh& m) {
	// Calculate the angle between pixels
	Angle angleBetween(fieldOfView.theta / outputWidth, fieldOfView.phi / outputHeight);
	// Create 9 rays, evenly spaced, per on-screen "pixel"
	angleBetween /= 3;
	
	Angle startingAngle((angleBetween.theta * (3.0 * outputWidth / 2.0) * -1.0), (angleBetween.phi * (3.0 * outputHeight / 2.0) * -1.0));
	startingAngle += direction.toAngle();

	std::vector<Vector> rays;

	for (int row = 0; row < outputHeight; row++) {
		for (int col = 0; col < outputWidth; col++) {
			for (int subrow = row*3; subrow < row*3 + 3; subrow++) {
				for (int subcol = col * 3; subcol < col * 3 + 3; subcol++) {
					Angle rayAngle = startingAngle;
					rayAngle.theta += subcol * angleBetween.theta;
					rayAngle.phi += subrow * angleBetween.phi;

					Vector ray(rayAngle);
					rays.push_back(ray);
				}
			}
		}
	}
	std::cout << "Rays created, calculating intersects" << std::endl;
	// Calculate the intersection distances for each ray
	std::vector<double> intersectDistances = m.calculateIntersectDistances(position, rays);
	// For each pixel, the "brightness" is the number of rays in that pixel that have an intersection, scaled linearly by distance
	std::cout << "Calculating brightness values" << std::endl;
	std::vector<double> pixelBrightness(outputHeight * outputWidth, 0.0);
	for (int idx = 0; idx < outputHeight * outputWidth * 9; idx++) {
		int interIdx = idx;
		int pixIdx = idx / 9;
		if (intersectDistances[interIdx] > 0)
			pixelBrightness[pixIdx] += 1.0 / (intersectDistances[interIdx] * FALLOFF);
	}
	// Display the calculated image to the screen
	for (int row = 0; row < outputHeight; row++) {
		for (int col = 0; col < outputWidth; col++) {
			double brightness = pixelBrightness[row*outputWidth + col];
			std::string grayscale = GRAYSCALE;
			if (brightness - intPart(brightness) >= 0.5)
				brightness += 1;
			int brightInt = brightness;
			if (brightInt > grayscale.length() - 1)
				brightInt = grayscale.length() - 1;
			if (brightInt < 0)
				brightInt = 0;
			std::cout << grayscale[brightInt];
		}
		std::cout << std::endl;
	}
}

#endif