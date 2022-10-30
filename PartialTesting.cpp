#include <iostream>
#include <cstdlib>
#include <string>

#include "RenderUtils.h"

using namespace std;

int main(void) {
	cout << "Enter num: ";
	string input;
	cin >> input;

	cout << stringToFloat(input);

	return 0;
}