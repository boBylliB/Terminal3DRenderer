#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include "Defines.h"
#include "Point.h"
#include "Angle.h"
#include "Vector.h"
#include "Mesh.h"

class Camera {
private:
	Point position;
	Vector direction;
	Angle fieldOfView;
	double roll;
	int outputHeight;
	int outputWidth;
public:
	// Getter functions
	Point getPosition(void);
	Vector getDirection(void);
	Angle getFOV(void);
	double getRoll(void);
	int* getOutputSize(void); // Returns a 2 element array, height in index 0 and width in index 1

	// Setter functions
	Camera(Point, Vector, double = FOV, double = 0, int = HEIGHT, int = WIDTH);
	void setPosition(Point);
	void setDirection(Vector);
	void setFOV(double);
	void setRoll(double);
	void setOutput(double, double=-1);

	// Basic utility functions
	void move(Vector); // Adjusts the position by the given vector
	void rotate(Angle); // Adjusts the direction by rotating the current view vector by the given angle
	void zoom(double); // Adjusts the FOV by the direct inverse of the given double (so that positive zoom is "zoom in")
	void rollAdjust(double); // Adjusts the roll by the given double
	void orbit(Angle, Point); // Rotates the camera around the given point while keeping the radius the same, sets the view to point at the orbit center
	void orbitCurrent(Angle, double); // Runs the orbit function with a point generated from the current view direction and the given radius

	// Core functions (and any functions that are too complex to be considered "utility")
	void visualizeAngle(void); // Displays 3 "compasses" that show the current view direction and roll
	void display(Mesh); // Calculates a snapshot of the mesh from this camera and displays it to the screen
};

#endif