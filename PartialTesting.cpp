#include <iostream>
#include <cstdlib>
#include <string>

#include "RenderUtils.h"

using namespace std;

int main(void) {
	cout << "Enter num: ";
	string input;
	cin >> input;

	cout.precision(1000);
	cout << pow(10, input.length() - 1) << endl;
	cout << stringToFloat(input) << endl;
	cout << stringToDouble(input) << endl;
	cout << stringToInt(input) << endl;

	return 0;
}