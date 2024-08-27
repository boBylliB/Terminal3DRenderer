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
#define LOWRESGRAYSCALE " .:-=+*&#%@"
// The value to minimize falloff to, to make pixels at the back more visible
#define FALLOFFMIN 0.1
// The number of threads to open in multithreaded rendering
// More threads usually means faster rendering at the cost of higher load on the computer
// However keep in mind once you hit 100% CPU usage it VERY RAPIDLY becomes diminishing returns
#define NUMTHREADS 24
// The number of GPU threads to use within each block
// The number of threads per block should ALWAYS be a multiple of 32, with a maximum of 1024 (by standard, check your GPU specs to see if it can take more)
// Be careful when increasing the number of threads per block as it can cause major issues
#define NUMTHREADSPERBLOCK 512
// The number of chunks to split segments into
#define NUMCHILDSEGMENTS 2
// The number of splits to perform when segmenting a mesh
// All three directions are defined, so for example 1 split in each direction would lead to 8 segments
// The direction is the normal of the plane to use, so for instance the X direction defines the number of split planes parallel to the YZ plane
#define NUMSPLITSX 1
#define NUMSPLITSY 1
#define NUMSPLITSZ 1
// The number of splits to perform when chunking rays
// Two directions are defined, in relation to screen space
// The direction is the direction perpendicular to the split line, so for instance the X direction defines the number of splits parallel to the Y axis
#define NUMCHUNKSX 1
#define NUMCHUNKSY 1
// The buffer surrounding a segment bounding box
// This should be balanced to be as small as possible while avoiding any possible missed triangles
#define SEGMENTBUFFER 0.00001

#endif