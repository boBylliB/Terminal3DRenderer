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
Camera::Camera(Point position, Vector direction, double fieldOfView= FOV, double roll = 0, int height = HEIGHT, int width = WIDTH) {
	this->position = position;
	this->direction = direction;
	this->roll = roll;
	this->outputHeight = height;
	this->outputWidth = width;

	this->fieldOfView.phi = fieldOfView;
	this->fieldOfView.theta = fieldOfView;
}
void Camera::setPosition(Point) {

}
void Camera::setDirection(Vector) {

}
void Camera::setFOV(double) {

}
void Camera::setRoll(double) {

}
void Camera::setOutput(double, double = -1) {

}

// Basic Utility Functions
// Adjusts the position by the given vector
void Camera::move(Vector) {

}
// Adjusts the direction by rotating the current view vector by the given angle
void Camera::rotate(Angle) {

}
// Adjusts the FOV by the direct inverse of the given double (so that positive zoom is "zoom in")
void Camera::zoom(double) {

}
// Adjusts the roll by the given double
void Camera::rollAdjust(double) {

}
// Rotates the camera around the given point while keeping the radius the same, sets the view to point at the orbit center
void Camera::orbit(Angle, Point) {

}
// Runs the orbit function with a point generated from the current view direction and the given radius
void Camera::orbitCurrent(Angle, double) {

}

// Core functions (and any functions that are too complex to be considered "utility")
// Displays 3 "compasses" that show the current view direction and roll
void Camera::visualizeAngle(void) {

}
// Calculates a snapshot of the mesh from this camera and displays it to the screen
void Camera::display(Mesh) {

}

#endif