// The values in this file are either defaults, advanced settings, or other things
// Changing the values in here is intended, but could affect many different things
#ifndef DEFINES_H
#define DEFINES_H

// Default output height
#define HEIGHT 300
// Default output width
#define WIDTH 300
// Default field of view
#define FOV 40
// The ratio of height to width for characters in the terminal
// Used to adjust the raycast angles to try and counteract distortion, so it might not match the actual ratio exactly
#define CHARRATIO 2
// The character scale from "black" to "white"
#define GRAYSCALE " .:-=+*%@#&"
// The value to minimize falloff to, to make pixels at the back more visible
#define FALLOFFMIN 0.1
// The number of threads to open in multithreaded rendering
// More threads usually means faster rendering at the cost of higher load on the computer
// However keep in mind once you hit 100% CPU usage it VERY RAPIDLY becomes diminishing returns
#define NUMTHREADS 24
// The number of GPU threads to use within each block
// The number of threads per block should ALWAYS be a multiple of 32, with a maximum of 1024 (by standard, check your GPU specs to see if it can take more)
// Be careful when increasing the number of threads per block as it can cause major issues
#define NUMTHREADSPERBLOCK 640

#endif