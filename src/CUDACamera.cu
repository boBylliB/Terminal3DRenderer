#ifndef CUDACAMERA_CU
#define CUDACAMERA_CU

#include <thread>

#include "CUDACamera.cuh"

using namespace std;

// CUDA kernel function to run in parallel on the GPU (cannot be a member function)
__global__ void CUDACalculateIntersectDistances(int* numRays, double* distances, Point* pos, Triangle* tris, int* numTris, Vector* rays) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < *numRays; idx += stride) {
		distances[idx] = CUDACalculateIntersectDistance(tris, *numTris, *pos, rays[idx]);
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

	for (int triIdx = 0; triIdx < numTris; triIdx++) {
		if (CUDACheckWithin(tris[triIdx], ray, origin)) {
			Vector originVector = CUDADifferenceVector(ray, origin, tris[triIdx].verts[0]);
			Vector normal = tris[triIdx].normal;
			Vector normRay = CUDANormalize(ray);

			double dist = CUDADot(normal, originVector) / CUDADot(normal, normRay);
			if (dist < rayDistance) {
				rayDistance = dist;
				set = true;
			}
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

	// Normalize everything for ease of calculation
	limitA = CUDANormalize(limitA);
	limitB = CUDANormalize(limitB);
	limitC = CUDANormalize(limitC);
	dir = CUDANormalize(dir);

	// Create limiting planes using the bounding vectors
	Vector planeA = CUDACross(limitB, limitC);
	Vector planeB = CUDACross(limitA, limitC);
	Vector planeC = CUDACross(limitA, limitB);

	// If the tested vector is on the same side of each plane as the one bounding vector not within the test plane
	// Therefore, the only way that it could be on the "inside" of each plane is if the tested vector is between the bounding vectors
	bool testA = CUDADot(limitA, planeA) * CUDADot(dir, planeA) > 0;
	bool testB = CUDADot(limitB, planeB) * CUDADot(dir, planeB) > 0;
	bool testC = CUDADot(limitC, planeC) * CUDADot(dir, planeC) > 0;

	return (testA && testB && testC);
}
__device__ Vector CUDADifferenceVector(Vector vec, Point a, Point b) {
	Vector out = vec;
	out.I = b.x - a.x;
	out.J = b.y - a.y;
	out.K = b.z - a.z;
	return out;
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
	// Display a line for the output width for verification that the whole display fits on screen
	for (int idx = 0; idx < outputWidth; idx++) {
		cout << "@";
	}
	cout << endl << endl;
	// Calculate the angle between pixels
	Angle angleBetween(fieldOfView.theta / outputWidth, fieldOfView.phi / outputHeight);

	Angle startingAngle((angleBetween.theta * (outputWidth / 2.0) * -1.0), (angleBetween.phi * (outputHeight / 2.0) * -1.0));
	startingAngle += direction.toAngle();

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
	// Create rays array on shared CPU & GPU memory
	vector<Vector> rayVector;
	Vector* rays;
	cudaMallocManaged(&rays, *numRays * sizeof(Vector));

	for (int row = 0; row < outputHeight; row++) {
		for (int col = 0; col < outputWidth; col++) {
			Angle rayAngle = startingAngle;
			rayAngle.theta += col * angleBetween.theta;
			rayAngle.phi += row * angleBetween.phi;

			Vector ray(rayAngle);
			rayVector.push_back(ray);
		}
	}
	for (int idx = 0; idx < *numRays; idx++) {
		rays[idx] = rayVector[idx];
	}
	cout << "Rays created, calculating intersects" << std::endl;
	// Calculates the number of thread blocks needed, making sure to round up if needed
	int numBlocks = (*numRays + NUMTHREADSPERBLOCK - 1) / NUMTHREADSPERBLOCK;
	// Prefetch the data from the CPU onto the GPU, since we're about to run the GPU code
	int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(numRays, sizeof(int), device, NULL);
	cudaMemPrefetchAsync(intersectDistances, *numRays * sizeof(double), device, NULL);
	cudaMemPrefetchAsync(pos, sizeof(Point), device, NULL);
	cudaMemPrefetchAsync(triArr, *numTris * sizeof(Triangle), device, NULL);
	cudaMemPrefetchAsync(numTris, sizeof(int), device, NULL);
	cudaMemPrefetchAsync(rays, *numRays * sizeof(Vector), device, NULL);
	// Calculate the intersection distances for each ray in parallel
	cout << "Beginning GPU Distance Calculations" << endl;
	CUDACalculateIntersectDistances << <numBlocks, NUMTHREADSPERBLOCK >> > (numRays, intersectDistances, pos, triArr, numTris, rays);
	// Now we wait for all threads to finish and collect the results
	cudaDeviceSynchronize();
	//cout << "I=" << rays[0].I << ", J=" << rays[0].J << ", K=" << rays[0].K << ", I=" << triArr[0].normal.I << ", J=" << triArr[0].normal.J << ", K=" << triArr[0].normal.K << endl;
	cout << "Calculated Distances" << endl;
	vector<double> distances;
	for (int idx = 0; idx < *numRays; idx++) {
		distances.push_back(intersectDistances[idx]);
	}
	// Calculate the minimum distance for brightness falloff
	Frame frame(distances, outputHeight, outputWidth, dither);
	frame.trimPixels();

	// Free memory
	for (int idx = 0; idx < NUMTHREADS; idx++) {
		cudaFree(numRays);
		cudaFree(intersectDistances);
		cudaFree(pos);
		cudaFree(triArr);
		cudaFree(numTris);
		cudaFree(rays);
	}

	return frame;
}

#endif