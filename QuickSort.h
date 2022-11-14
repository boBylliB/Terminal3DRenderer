#ifndef QUICKSORT_H
#define QUICKSORT_H

#include <vector>

class QuickSort {
private:
	std::vector<double> vec;

	int partition(int, int);
	void swap(int, int);
	void quickSort(int, int);
public:
	void sort(std::vector<double>);
};

#endif