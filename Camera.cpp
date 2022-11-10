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
Camera::Camera(const Point& position, const Vector& direction, const double fieldOfView = FOV, const double roll = 0, const int height = HEIGHT, const int width = WIDTH) {
	this->position = position;
	this->roll = roll;
	this->outputHeight = height;
	this->outputWidth = width;

	this->direction = direction;
	this->direction.normalize();

	this->fieldOfView.phi = fieldOfView;
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
	this->fieldOfView.phi = fieldOfView;
	this->fieldOfView.theta = fieldOfView;
}
void Camera::setRoll(const double r) {
	roll = r;
}
void Camera::setOutput(const int height, const int width = -1) {
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

	Vector diff;
	diff.difference(direction.toPoint(), origin);
	move(diff);
}
// Runs the orbit function with a point generated from the current view direction and the given radius
void Camera::orbitCurrent(const Angle& a, const double radius) {
	direction.normalize();
	direction.scale(radius);
	orbit(a, direction.toPoint());
}

// Core functions (and any functions that are too complex to be considered "utility")
// Displays 3 "compasses" that show the current view direction and roll
void Camera::visualizeAngle(void) {

}
// Calculates a snapshot of the mesh from this camera and displays it to the screen
void Camera::display(const Mesh& m) {
	// Create 9 rays per pixel
}

#endif