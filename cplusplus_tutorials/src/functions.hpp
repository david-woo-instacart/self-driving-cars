/*
 * functions.hpp
 *
 *  Created on: Oct 31, 2017
 *      Author: davidwoo
 */

 #include <iostream>

 /*The function declaration can be omitted in the header file.
 **As long as the function is defined before it is used,
 **the declaration is optional.
**It is often considered good practice to list the declarations
**at the top of the header file.
*/
 void printProduct(int m1, int m2, int product);
 void printFarenheitToCelsius(float farenheit, float celsius);


 void printProduct(int m1, int m2, int product)
 {
     std::cout << m1 <<"*"<< m2 <<" = "<<product << " \n";
 }

 void printFarenheitToCelsius(float farenheit)
 {
	 std::cout << "Farenheit = " << farenheit << " is Celsius = " << (farenheit -32) / 1.8;
 }


