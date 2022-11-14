#ifndef QUICKSORT_CPP
#define QUICKSORT_CPP

#include "QuickSort.h"

void QuickSort::sort(std::vector<double> vec) {
	this->vec = vec;

	quickSort(0, vec.size() - 1);
}

int QuickSort::partition(int start, int end) {
	int pivot = vec[start];

	int count = 0;
	for (int idx = start + 1; idx <= end; idx++) {
		if (vec[idx] <= pivot)
			count++;
	}

	// Give pivot element the correct index
	int pivotIdx = start + count;
	swap(pivotIdx, start);

	// Sort the left and right parts of the pivot element
	int i = start, j = end;
	while (i < pivotIdx && j > pivotIdx) {
		while (vec[i] <= pivot)
			i++;
		while (vec[j] > pivot)
			j--;
		if (i < pivotIdx && j > pivotIdx)
			swap(i, j);
	}

	return pivotIdx;
}
void QuickSort::swap(int idxA, int idxB) {
	double temp = vec[idxA];
	vec[idxA] = vec[idxB];
	vec[idxB] = temp;
}
void QuickSort::quickSort(int start, int end) {
	// Test for base case
	if (start < end) {
		// Partition the vector
		int p = partition(start, end);
		// Sort the left part recursively
		quickSort(start, p - 1);
		// Sort the right part recursively
		quickSort(p + 1, end);
	}
}

#endif