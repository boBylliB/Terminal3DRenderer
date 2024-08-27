#ifndef CUDACAMERA_CU
#define CUDACAMERA_CU

#include <thread>

#include "CUDACamera.cuh"

using namespace std;

// CUDA kernel functions to run in parallel on the GPU (cannot be a member function)
__global__ void CUDACalculateIntersectDistances(int* numRays, double* distances, Point* pos, Triangle* tris, int* numTris, Vector* rays) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < *numRays; idx += stride) {
		distances[idx] = CUDACalculateIntersectDistance(tris, *numTris, *pos, rays[idx]);
	}
}
__global__ void CUDACalculateIntersectDistancesSegmented(int* numRays, double* distances, Point* pos, Triangle* tris, Triangle** triSegments, int* numTris, int* numTrisPerSegment, Vector* rays, int* numSegments, bool** segmentAssignment) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < *numRays; idx += stride) {
		// Get the chunk index that the current ray resides in
		// Grab the assigned segments based on that chunk index
		// Calculate intersect distances for each segment
		// Set the resulting distance to the closest calculated distance
		distances[idx] = CUDACalculateIntersectDistance(tris, *numTris, *pos, rays[idx]);
	}
}
__global__ void CUDACalculateRays(int* outDim, Angle* outAngles, int* numRays, Vector* rays) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < *numRays; idx += stride) {
		Angle rayAngle = outAngles[0];
		rayAngle.theta += (idx % outDim[1]) * outAngles[1].theta;
		rayAngle.phi += (idx / outDim[0]) * outAngles[1].phi;

		Vector ray = CUDAAngleVector(rays[0], rayAngle);
		rays[idx] = ray;
	}
}

// CUDA device functions
__device__ double CUDACalculateIntersectDistance(Triangle* tris, int numTris, Point origin, Vector ray) {
	// Get the normal of the triangle
	// Get the vector from any point on the triangle to the origin
	// Distance = (normal dot (triangle to origin)) / (normal dot normalized ray)
	// If the distance is negative, it is behind the origin of the ray
	// Find the point of intersection by adding distance * normalized ray to the origin coordinates
	// Check for intersection within triangle
	double rayDistance = NPP_MAXABS_64F;
	bool set = false;
	Vector normRay = CUDANormalize(ray);

	for (int triIdx = 0; triIdx < numTris; triIdx++) {
		Vector originVector = CUDADifferenceVector(ray, origin, tris[triIdx].verts[0]);
		Vector normal = tris[triIdx].normal;

		double dist = CUDADot(normal, originVector) / CUDADot(normal, normRay);
		if (dist > 0 && dist < rayDistance && CUDACheckWithin(tris[triIdx], normRay, origin)) {
			set = true;
			rayDistance = dist;
		}
	}
	if (set) {
		return rayDistance;
	}
	else {
		return -1;
	}
}
__device__ bool CUDACheckWithin(Triangle tri, Vector dir, Point origin) {
	// printf("%d: I=%lf, J=%lf, K=%lf, I=%lf, J=%lf, K=%lf\n", triIdx, dir.I, dir.J, dir.K, tri.normal.I, tri.normal.J, tri.normal.K);

	// Vectors from ray origin to each vertex as the bounds
	Vector limitA = CUDADifferenceVector(dir, origin, tri.verts[0]);
	Vector limitB = CUDADifferenceVector(dir, origin, tri.verts[1]);
	Vector limitC = CUDADifferenceVector(dir, origin, tri.verts[2]);

	// Create limiting planes using the bounding vectors
	Vector planeA = CUDACross(limitB, limitC);
	// If the tested vector is on the same side of each plane as the one bounding vector not within the test plane
	// Therefore, the only way that it could be on the "inside" of each plane is if the tested vector is between the bounding vectors
	if (CUDADot(limitA, planeA) * CUDADot(dir, planeA) <= 0) return false;

	Vector planeB = CUDACross(limitA, limitC);
	if (CUDADot(limitB, planeB) * CUDADot(dir, planeB) <= 0) return false;

	Vector planeC = CUDACross(limitA, limitB);
	if (CUDADot(limitC, planeC) * CUDADot(dir, planeC) <= 0) return false;

	// If we get here, then all tests passed
	return true;
}
__device__ Vector CUDADifferenceVector(Vector vec, Point a, Point b) {
	Vector out = vec;
	out.I = b.x - a.x;
	out.J = b.y - a.y;
	out.K = b.z - a.z;
	return out;
}
__device__ Vector CUDAAngleVector(Vector vec, Angle ang) {
	Vector out = vec;
	double degToRadCoeff = (1 / 180.0)* atan(1.0) * 4.0;
	out.I = cos(degToRadCoeff*ang.theta) * sin(degToRadCoeff*ang.phi);
	out.J = cos(degToRadCoeff*ang.phi);
	out.K = sin(degToRadCoeff*ang.theta) * sin(degToRadCoeff*ang.phi);

	// If there's no way to tell how large a vector should be, I just normalize it so that it's easier to scale later
	out = CUDANormalize(out);
}
__device__ Vector CUDANormalize(Vector vec) {
	double mag = sqrt(vec.I * vec.I + vec.J * vec.J + vec.K * vec.K);
	vec.I = vec.I / mag;
	vec.J = vec.J / mag;
	vec.K = vec.K / mag;
	return vec;
}
__device__ double CUDADot(Vector a, Vector b) {
	double output = a.I * b.I + a.J * b.J + a.K * b.K;
	return output;
}
__device__ Vector CUDACross(Vector a, Vector b) {
	Vector out = a;
	out.I = a.J * b.K - a.K * b.J;
	out.J = a.K * b.I - a.I * b.K;
	out.K = a.I * b.J - a.J * b.I;
	return out;
}

CUDACamera::CUDACamera(const Point& position, const Vector& direction, const double fov = FOV, const double roll = 0, const int outHeight = HEIGHT, const int outWidth = WIDTH) : Camera(position, direction, fov, roll, outHeight, outWidth) {

}
CUDACamera::CUDACamera(Camera& cam) : Camera(cam.getPosition(), cam.getDirection(), cam.getFOV().theta, cam.getRoll(), cam.getOutputSize()[0], cam.getOutputSize()[1]) {

}

// Core Functions
// Functions the same as the standard camera display, but does the math in parallel on the GPU for speed
void CUDACamera::CUDADisplay(const Mesh& m, const bool dither) {
	CUDADisplayMath(m, dither).print();
}
// Just the math of the display function above, outputting to a Frame to be displayed later
Frame CUDACamera::CUDADisplayMath(const Mesh& m, const bool dither) {
	// Initializing profiler
	Profiler profiler;

	// Display a line for the output width for verification that the whole display fits on screen
	for (int idx = 0; idx < outputWidth; idx++) {
		cout << "@";
	}
	cout << endl << endl;
	// Calculate the angle between pixels
	Angle angleBetween(fieldOfView.theta / outputWidth, fieldOfView.phi / outputHeight);
	Angle startingAngle((angleBetween.theta * (outputWidth / 2.0) * -1.0), (angleBetween.phi * (outputHeight / 2.0) * -1.0));
	startingAngle += direction.toAngle();
	// Generate the angles to split at for each ray chunk
	Angle splitAngle(fieldOfView.theta / (NUMCHUNKSX + 1), fieldOfView.phi / (NUMCHUNKSY + 1));
	// Create Vectors to form the bounding planes
	vector<vector<Vector>> boundingVectors;
	for (int idx = 0; idx < NUMCHUNKSX + 2; idx++) {
		vector<Vector> currentCol;
		for (int jdx = 0; jdx < NUMCHUNKSY + 2; jdx++) {
			Angle currentAngle(startingAngle.theta + idx * splitAngle.theta, startingAngle.phi + jdx * splitAngle.phi);
			Vector currentVec(currentAngle);
			currentCol.push_back(currentVec);
		}
		boundingVectors.push_back(currentCol);
	}
	profiler.start("Segment Allocation");
	// Segments may lie on both sides of a bounding plane, in which case it should be assigned to all affected chunks
	vector<vector<bool>> segmentsPerChunk; // each boolean represents a segment, each vector of booleans represents the segments assigned to a chunk of rays
	// We have to initialize these to zero for future math to work, even though we won't be directly using these on the first pass
	Vector boundingPlaneY[2] = {};
	Vector boundingPlaneX[2] = {};
	for (int idx = 0; idx < NUMCHUNKSX + 1; idx++) {
		for (int jdx = 0; jdx < NUMCHUNKSY + 1; jdx++) {
			// Generate bounding planes
			boundingPlaneX[1] = boundingPlaneX[0];
			boundingPlaneY[0] = boundingVectors[idx + 1][jdx + 1].cross(boundingVectors[idx + 2][jdx + 1]);
			boundingPlaneX[1] = boundingPlaneX[0];
			boundingPlaneX[0] = boundingVectors[idx + 1][jdx + 1].cross(boundingVectors[idx + 1][jdx]);
			// We move from low angle to high angle, so each iteration should pull chunks from the lower side of the plane
			// If we're at an iteration greater than 0, we should also take into account the previous split, ignoring any chunks completely covered by previous planes
			vector<bool> pickedSegments;
			for (int kdx = 0; kdx < m.numSegments; kdx++) {
				// Compare the segment center against the bounding planes using a Vector to the top left corner as reference
				Vector toCenter(position, m.segmentCenters[kdx]);
				if (boundingVectors[0][0].dot(boundingPlaneX[0]) * toCenter.dot(boundingPlaneX[0]) >= 0 && boundingVectors[0][0].dot(boundingPlaneY[0]) * toCenter.dot(boundingPlaneY[0]) >= 0) {
					// Make sure at least one of the vertices of the segment bounding box is on the same side of the previous bounding planes as the bottom right corner
					bool notExcluded = false;
					for (int xShift = -1; xShift < 2; xShift += 2) {
						for (int yShift = -1; yShift < 2; yShift += 2) {
							for (int zShift = -1; zShift < 2; zShift += 2) {
								Point corner = m.segmentCenters[kdx];
								corner.x += m.segmentBounds[kdx].I * xShift;
								corner.y += m.segmentBounds[kdx].J * yShift;
								corner.z += m.segmentBounds[kdx].K * zShift;
								Vector toCorner(position, corner);
								// If this is iteration 0, there are no previous bounding planes, so it cannot have been excluded
								if ((boundingVectors[NUMCHUNKSX + 1][NUMCHUNKSY + 1].dot(boundingPlaneX[1]) * toCorner.dot(boundingPlaneX[1]) >= 0 || idx == 0)
									&& (boundingVectors[NUMCHUNKSX + 1][NUMCHUNKSY + 1].dot(boundingPlaneY[1]) * toCenter.dot(boundingPlaneY[1]) >= 0 || jdx == 0)) {
									notExcluded = true;
								}
							}
						}
					}
					// If true, add it to be picked and continue
					if (notExcluded) {
						pickedSegments.push_back(true);
						continue;
					}
				}
				else {
					// Check if at least one of the vertices of the segment bounding box is on the same side of the current bounding planes as the top left corner
					bool notExcluded = false;
					for (int xShift = -1; xShift < 2; xShift += 2) {
						for (int yShift = -1; yShift < 2; yShift += 2) {
							for (int zShift = -1; zShift < 2; zShift += 2) {
								Point corner = m.segmentCenters[kdx];
								corner.x += m.segmentBounds[kdx].I * xShift;
								corner.y += m.segmentBounds[kdx].J * yShift;
								corner.z += m.segmentBounds[kdx].K * zShift;
								Vector toCorner(position, corner);
								if (boundingVectors[0][0].dot(boundingPlaneX[0]) * toCorner.dot(boundingPlaneX[0]) >= 0
									&& boundingVectors[0][0].dot(boundingPlaneY[0]) * toCenter.dot(boundingPlaneY[0]) >= 0) {
									notExcluded = true;
								}
							}
						}
					}
					// If true, add it to be picked and continue
					if (notExcluded) {
						pickedSegments.push_back(true);
						continue;
					}
				}
				// If nothing passes, add false to the vector
				pickedSegments.push_back(false);
			}
			segmentsPerChunk.push_back(pickedSegments);
		}
	}
	profiler.end();
	profiler.start("Memory Allocation");
	// Create required arrays & data on shared GPU & CPU memory
	int* numRays;
	cudaMallocManaged(&numRays, sizeof(int));
	*numRays = outputHeight * outputWidth;
	Point* pos;
	cudaMallocManaged(&pos, sizeof(Point));
	*pos = position;
	int* numTris;
	cudaMallocManaged(&numTris, sizeof(int));
	*numTris = m.tris.size();
	double* intersectDistances;
	cudaMallocManaged(&intersectDistances, *numRays * sizeof(double));
	Triangle* triArr;
	cudaMallocManaged(&triArr, *numTris * sizeof(Triangle));
	for (int idx = 0; idx < *numTris; idx++) {
		triArr[idx] = m.tris[idx];
	}
	int* outDim;
	cudaMallocManaged(&outDim, 2 * sizeof(int));
	outDim[0] = outputHeight;
	outDim[1] = outputWidth;
	Angle* outAngles;
	cudaMallocManaged(&outAngles, 2 * sizeof(Angle));
	outAngles[0] = startingAngle;
	outAngles[1] = angleBetween;
	int* numSegments;
	cudaMallocManaged(&numSegments, sizeof(int));
	*numSegments = m.numSegments;
	// Create rays array on shared CPU & GPU memory
	vector<Vector> rayVector;
	Vector* rays;
	cudaMallocManaged(&rays, *numRays * sizeof(Vector));
	// Segment assignment is an array of booleans for each ray that is read for each ray to determine which tri segments to check for intersections
	bool** segmentAssignment;
	cudaMallocManaged(&segmentAssignment, *numRays * (*numSegments * sizeof(bool)));
	profiler.end();
	profiler.start("Ray Calculation");
	vector<vector<bool>> segmentAssignmentVector;
	for (int row = 0; row < outputHeight; row++) {
		for (int col = 0; col < outputWidth; col++) {
			Angle rayAngle = startingAngle;
			rayAngle.theta += col * angleBetween.theta;
			rayAngle.phi += row * angleBetween.phi;

			Vector ray(rayAngle);
			rayVector.push_back(ray);

			// Determine which chunk this ray belongs to
			Vector boundingPlaneY[2] = {};
			Vector boundingPlaneX[2] = {};
			for (int idx = 0; idx < NUMCHUNKSX + 1; idx++) {
				for (int jdx = 0; jdx < NUMCHUNKSY + 1; jdx++) {
					// Generate bounding planes
					boundingPlaneX[1] = boundingPlaneX[0];
					boundingPlaneY[0] = boundingVectors[idx + 1][jdx + 1].cross(boundingVectors[idx + 2][jdx + 1]);
					boundingPlaneX[1] = boundingPlaneX[0];
					boundingPlaneX[0] = boundingVectors[idx + 1][jdx + 1].cross(boundingVectors[idx + 1][jdx]);
					// Test that this vector lies on the same side of the current bounding planes as the top left corner
					// And that this vector lies on the same side of the previous bounding planes as the bottom right corner
					// (If this is the first bounding plane for either index, this must be true but would normally return false)
					if (boundingVectors[0][0].dot(boundingPlaneX[0]) * ray.dot(boundingPlaneX[0]) >= 0
						&& boundingVectors[0][0].dot(boundingPlaneY[0]) * ray.dot(boundingPlaneY[0]) >= 0
						&& (boundingVectors[NUMCHUNKSX + 1][NUMCHUNKSY + 1].dot(boundingPlaneX[1]) * ray.dot(boundingPlaneX[1]) >= 0 || idx == 0)
						&& (boundingVectors[NUMCHUNKSX + 1][NUMCHUNKSY + 1].dot(boundingPlaneY[1]) * ray.dot(boundingPlaneY[1]) >= 0 || jdx == 0)) {
						// If all of these tests pass, this ray exists within this chunk, and we assign it the segments that its chunk needs to test
						segmentAssignmentVector.push_back(segmentsPerChunk[idx * (NUMCHUNKSY + 1) + jdx]);
					}
				}
			}
		}
	}
	profiler.end();
	for (int idx = 0; idx < *numRays; idx++) {
		rays[idx] = rayVector[idx];
		for (int jdx = 0; jdx < *numSegments; jdx++) {
			segmentAssignment[idx][jdx] = segmentAssignmentVector[idx][jdx];
		}
	}
	cout << "Rays created, calculating intersects" << endl;
	// Calculates the number of thread blocks needed, making sure to round up if needed
	int numBlocks = (*numRays + NUMTHREADSPERBLOCK - 1) / NUMTHREADSPERBLOCK;
	// Prefetch the needed data from the CPU onto the GPU before running the relevant GPU code
	int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(numRays, sizeof(int), device, NULL);
	cudaMemPrefetchAsync(intersectDistances, *numRays * sizeof(double), device, NULL);
	cudaMemPrefetchAsync(pos, sizeof(Point), device, NULL);
	cudaMemPrefetchAsync(triArr, *numTris * sizeof(Triangle), device, NULL);
	cudaMemPrefetchAsync(numTris, sizeof(int), device, NULL);
	cudaMemPrefetchAsync(rays, *numRays * sizeof(Vector), device, NULL);
	//// Calculate rays on the GPU
	//profiler.start("GPU Ray Calculation");
	//CUDACalculateRays<<<numBlocks, NUMTHREADSPERBLOCK>>>(outDim, outAngles, numRays, rays);
	//// Now we wait for all threads to finish and collect the results
	//cudaDeviceSynchronize();
	//profiler.end();
	//int raysMatch = 0;
	//for (int idx = 0; idx < *numRays; idx++) {
	//	if (rays[idx] == rayVector[idx]) raysMatch++;
	//	rays[idx] = rayVector[idx];
	//}
	//cout << raysMatch << " rays matched out of " << *numRays << endl;
	// Calculate the intersection distances for each ray in parallel
	cout << "Beginning GPU Distance Calculations" << endl;
	profiler.start("GPU Distance Calculations");
	CUDACalculateIntersectDistances<<<numBlocks, NUMTHREADSPERBLOCK>>>(numRays, intersectDistances, pos, triArr, numTris, rays);
	// Now we wait for all threads to finish and collect the results
	cudaDeviceSynchronize();
	//cout << "I=" << rays[0].I << ", J=" << rays[0].J << ", K=" << rays[0].K << ", I=" << triArr[0].normal.I << ", J=" << triArr[0].normal.J << ", K=" << triArr[0].normal.K << endl;
	profiler.end();
	cout << "Calculated Distances" << endl;

	profiler.start("Brightness falloff calculations");
	vector<double> distances;
	for (int idx = 0; idx < *numRays; idx++) {
		distances.push_back(intersectDistances[idx]);
	}
	// Calculate the minimum distance for brightness falloff
	Frame frame(distances, outputHeight, outputWidth, dither);
	frame.trimPixels();
	profiler.end();

	profiler.start("Freeing memory");
	// Free memory
	for (int idx = 0; idx < NUMTHREADS; idx++) {
		cudaFree(numRays);
		cudaFree(intersectDistances);
		cudaFree(pos);
		cudaFree(triArr);
		cudaFree(numTris);
		cudaFree(rays);
	}
	profiler.end();

	profiler.printSegments();
	profiler.archiveSegments();

	return frame;
}

#endif