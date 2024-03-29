// Control Class
#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include "Defines.h"
#include "Point.h"
#include "Angle.h"
#include "Vector.h"
#include "Mesh.h"
#include "RenderUtils.h"
#include "Frame.h"

class Camera {
protected:
	Point position;
	Vector direction;
	Angle fieldOfView;
	double roll;
	int outputHeight;
	int outputWidth;
public:
	Camera(const Point&, const Vector&, const double = FOV, const double = 0, const int = HEIGHT, const int = WIDTH);
	
	// Getter functions
	Point getPosition(void);
	Vector getDirection(void);
	Angle getFOV(void);
	double getRoll(void);
	int* getOutputSize(void); // Returns a 2 element array, height in index 0 and width in index 1

	// Setter functions
	void setPosition(const Point&);
	void setDirection(const Vector&);
	void setFOV(const double);
	void setRoll(const double);
	void setOutput(const int, const int=-1);

	// Basic utility functions
	// Adjusts the position by the given vector
	void move(const Vector&);
	// Adjusts the direction by rotating the current view vector by the given angle
	void rotate(const Angle&); 
	// Adjusts the FOV by the direct inverse of the given double (so that positive zoom is "zoom in")
	void zoom(const double); 
	// Adjusts the roll by the given double
	void rollAdjust(const double); 
	// Rotates the camera around the given point while keeping the radius the same, sets the view to point at the orbit center
	void orbit(const Angle&, const Point&); 
	// Runs the orbit function with a point generated from the current view direction and the given radius
	void orbitCurrent(const Angle&, const double);

	// Core functions (and any functions that are too complex to be considered "utility")
	// Calculates a snapshot of the mesh from this camera and displays it to the screen
	void display(const Mesh&, const bool = false);
	// Just the math of the display function above, outputting to a Frame to be displayed later
	Frame displayMath(const Mesh&, const bool = false);
};

#endif