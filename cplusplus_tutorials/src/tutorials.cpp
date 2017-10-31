//============================================================================
// Name        : tutorials.cpp
// Author      : testing
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "functions.hpp"
using namespace std;

int main() {
	cout << "!!!Hello World!!!\n" << endl; // prints !!!Hello World!!!

	int m1 = 4;
	int m2 = 5;
	int product;

	product = m1 * m2;

	//call the function
	printProduct(m1, m2, product);

	printFarenheitToCelsius(53.0);
	return 0;
}

