#include <iostream>
#include <time.h>
#include <gmp.h>
#include <mpfr.h>
#include <fplll.h>
#include <fplll/nr/nr.h>
//#include <fpll/defs.h>
#include "fplll/util.h"
#include <fplll/wrapper.h>
#include <fplll/gso.h>
#include<fplll/gso_gram.h>
#include <fplll/gso_interface.h>
#include <math.h>
#include <fplll/nr/numvect.h>
using namespace std;
using namespace fplll;

//Compile using these flags: g++ -std=c++11 -O3 -march=native Babai.cpp -lfplll -lmpfr -lgmp  -o Babai
//Run with ./Babai

/** 
	@brief: Function that initializes a vector of given dimension with random numbers
	@param NumVect<FP_NR<T>> vector: Vector to be randomized, called by reference
	@return: vector, randomized by rand()

*/

template <class T, class U> NumVect<FP_NR<T>> addRow (MatrixRow<Z_NR<U>> &&vector, FP_NR<T> num){
	
	NumVect<FP_NR<T>> result(vector.size());
	for (int i = 0; i < vector.size(); i++) {

		result[i].add(num, vector[i].get_ld(), MPFR_RNDN);// This is FP_NR class's function to implement multiplication,
														 // allowing us to avoid confusing mpfr ang mpz functions in case
														 // of usage of MPFR or GMP Libraries.
	}
	return result;

}

template <class T> NumVect<FP_NR<T>> randomSet (NumVect<FP_NR<T>> &vector) {
	
	for (int i = 0; i < vector.size(); i++) {

		vector[i] = rand();// Instantiate each element by a random number, generated here

	}
	return vector;

}

/** 
	@brief: This function multiplies the coefficients of a given vector vector by a number num.
	@param: MatrixRow<Z_NR<U>> vector: This is a vector of integers, in NPA the initial integer lattice base.
	@param FP_NR<T>> num: Number by which to multiply coefficients
	@return: result, vector multiplied by num

*/

template <class T, class U> NumVect<FP_NR<T>> mult (MatrixRow<Z_NR<U>> &&vector, FP_NR<T> num){
	
	NumVect<FP_NR<T>> result(vector.size());
	for (int i = 0; i < vector.size(); i++) {

		result[i].mul(num, vector[i].get_ld(), MPFR_RNDN);// This is FP_NR class's function to implement multiplication,
														 // allowing us to avoid confusing mpfr ang mpz functions in case
														 // of usage of MPFR or GMP Libraries.
	}
	return result;

}



/** 
	@brief: This function multiplies the coefficients of a given vector vector by a number num.
	@param: MatrixRow<FP_NR<T>> vector: This is a vector of floating point numbers, in NPA the GSO-ed lattice base.
	@param: FP_NR<T>> num: Number by which to multiply coefficients
	@return: result, vector multiplied by num

*/

template <class T> NumVect<FP_NR<T>> multRow (NumVect<FP_NR<T>> &row, FP_NR<T> num) {
	NumVect<FP_NR<T>> toReturn(row.size());
	for (int i = 0; i < row.size(); i++) {

		toReturn[i].mul(row[i], num, MPFR_RNDN);// See comments of mult() function above
	}
	return toReturn;
}

template <class T> NumVect<FP_NR<T>> addRow (MatrixRow<FP_NR<T>> &&row, FP_NR<T> num) {
	NumVect<FP_NR<T>> toReturn(row.size());
	for (int i = 0; i < row.size(); i++) {

		toReturn[i].add(row[i], num, MPFR_RNDN);// See comments of mult() function above
	}
	return toReturn;
}

/** 
	@brief Function that calculates the dot product of two vectors (must be generic type as data may be of many data types, either from fplll's data or standard data)
	
	@param NumVect<FP_NR<T>> vector1: First vector, called by explicit reference (bound to temporary object). This is a vector of type NumVect.
	@param MatrixRow<FP_NR<T>> vector2: Second vector, same as vector1. This is a vector of type, MatrixRow, fplll's data type to represent vectors
	of lattice bases and GSO-ed bases
	@param int length1: length of first vector
	@param: length of second vector
	@return: Sum, result of dot product of two vectors, of type equal to that presented

*/

template <class T> FP_NR<T> dotProduct (NumVect<FP_NR<T>> &vector1, MatrixRow<FP_NR<T>> &&vector2, int length1, int length2) {
	
	FP_NR<T> sum = FP_NR<T> (0.0);
	int i = max(length1, length2);
	for (int j = 0; j <= i - 1; j++) {

		sum = sum + vector1[j]*vector2[j];

	}
	return sum;

}

/** 
	@brief Function that calculates the dot product of two vectors (must be generic type as data may be of many data types, either from fplll's data or standard data)
	
	@param MatrixRow<FP_NR<T>> vector1: First vector, called by explicit reference (bound to temporary object). This is a vector of type
	MatrixRow, fplll's data type to represent vectors of lattice bases and GSO-ed bases.
	@param MatrixRow<FP_NR<T>> vector2: Second vector, same as vector1. This is a vector of type, MatrixRow, fplll's data type to represent vectors
	of lattice bases and GSO-ed bases.
	@param int length1: length of first vector
	@param: length of second vector
	@return: Sum, result of dot product of two vectors, of type equal to that presented

*/

template <class T> FP_NR<T> dotProduct (MatrixRow<FP_NR<T>> vector1, MatrixRow<FP_NR<T>> vector2, int length1, int length2) {
	
	FP_NR<T> sum = 0.0;
	int i = max(length1, length2);
	for (int j = 0; j <= i - 1; j++) {

		sum = sum + vector1[j]*vector2[j];

	}

	return sum;

}

template <class T, class U> FP_NR<T> dotProduct (MatrixRow<Z_NR<U>> &&vector1, NumVect<FP_NR<T>> &vector2, int length1, int length2) {
	
	FP_NR<T> sum = FP_NR<T> (0.0);
	NumVect<FP_NR<T>> vect(1);
	vect.fill(0.0);
	int i = max(length1, length2);
	for (int j = 0; j <= i - 1; j++) {
		vect[0].mul(vector2[j], vector1[j].get_ld(), MPFR_RNDN);
		sum = sum + vect[0];

	}
	return sum;

}

template <class T> FP_NR<T> dotProduct (NumVect<FP_NR<T>> &vector1, NumVect<FP_NR<T>> &vector2, int length1, int length2) {
	
	FP_NR<T> sum = FP_NR<T> (0.0);
	int i = max(length1, length2);
	for (int j = 0; j <= i - 1; j++) {

		sum = sum + vector1[j]*vector2[j];

	}
	return sum;

}


/**
	@brief Function that generates base for Lattice of given dimension, with or without random seed
	
	@param ZZ_mat<mpz_t> &base: The base of our Lattice, a Matrix of integers (we use the FPLLL's Matrix struct as the lattice base's data type). This is called by reference.
	@param int dim: The dimension of the Lattice base
	@param bool isRandom: If set to true, we will generate the Lattice with a random seed based on current time 

*/



ZZ_mat<mpz_t> latticeGen(ZZ_mat<mpz_t> &base, int dim, bool isRandom) {

	if (isRandom) {

		RandGen:: init_with_time();//If isRandom == TRUE, RandGen method initializes random seed with current clock time

	}
	base.resize(dim, dim);//Base is resized according to dimensions required
	base.gen_uniform(dim);// A uniform (dim * dim) lattice base is generated
	return base;// Return generated Lattice base


}

/** 
	@brief Function that generates orthogonized base for the lattice as per the Gram-Schmidt orthogonization method, as well as storing m coeffs
	
	@param ZZ_mat<mpz_t> base: The base of our Lattice, a Matrix of integers (we use the FPLLL's Matrix struct as the lattice base's data type)
	@param Matrix<FP_NR<double>> &gramBase: The GSO-ed base of the Lattice, called by reference. We use the FP_NR<mpfr_t> data type to handle arbitrarily large numbers

*/

/** 
	@brief Function that generates orthogonized base for the lattice as per the Gram-Schmidt orthogonization method
	
	@param ZZ_mat<mpz_t> base: The base of our Lattice, a Matrix of integers (we use the FPLLL's Matrix struct as the lattice base's data type)
	@param Matrix<FP_NR<double>> &gramBase: The GSO-ed base of the Lattice, called by reference. We use the FP_NR<double> data type to handle arbitrarily large numbers

*/

NumVect<NumVect<FP_NR<mpfr_t>>> gSO (ZZ_mat<mpz_t> &base, NumVect<NumVect<FP_NR<mpfr_t>>> &gramBase) {
	ZZ_mat<mpz_t> identity;//Identity matrix of integers, used in LLL
	ZZ_mat<mpz_t> idTrans;//Transposed version of identity
	int dimension = base.get_cols();//Retrieve dimension of lattice
	FP_NR<mpfr_t> l1 = 0.0;
	FP_NR<mpfr_t> l2 = 0.0;
	FP_NR<mpfr_t> l = 0.0;
	FP_NR<mpfr_t> zero = FP_NR<mpfr_t> (0.0);
	NumVect<FP_NR<mpfr_t>> vect(dimension);
	NumVect<FP_NR<mpfr_t>> muVect(dimension);
	muVect.fill(0.0);
	vect.fill(0.0);
	vect = addRow(base[0], zero);
	gramBase[0] = vect;
	for (int i = 1; i < dimension; i++) {
		vect = addRow(base[i], zero);
		for (int j = i - 1; j >= 0; j--) {
			l1 = dotProduct(base[i], gramBase[j], base[i].size(), gramBase[j].size());
			l2 = dotProduct(gramBase[j], gramBase[j], gramBase[j].size(),gramBase[j].size());
			l = l1 / l2;
			muVect.mul(gramBase[j], l);
			vect.sub(muVect);
		}
		gramBase[i] = vect;
	}
	return gramBase;// Return GSO-ed base

}

/** 
	@brief This function implements default LLL reduction on base of lattice
	
	@param ZZ_mat<mpz_t> base: The base of the lattice, a Matrix of intgers, called by reference


*/

void reduceLLL (ZZ_mat<mpz_t> &base) {
	ZZ_mat<mpz_t> identity;//Identity matrix, used in the DEFAULT_GSO method
	ZZ_mat<mpz_t> idTrans;//Transposed ID matrix;
	Wrapper *wrapper = new Wrapper (base, identity, idTrans, 0.75, 0.51, LLL_DEFAULT);
	bool status = wrapper -> lll();
	cout << status << endl;

}

/** 
	@brief Function that implements Babai's Nearest Plane Algorithm for given target vector. Preprocessing is considered done.
	
	@param ZZ_mat<mpz_t> base: Base of integer lattice, reduced either by LLL or BKZ (choice to be implemented)
	@param Matrix<FP_NR<mpfr_t>> gramBase: GSO-ed Lattice base of above lattice.
	@param NumVect<FP_NR<mpfr_t>> target_vector: Target vector for which NP vector will be calculated
	@return: toReturn, NumVect<FP_NR<mpfr_t>> type, Nearest Plane vector
*/

NumVect<FP_NR<mpfr_t>> babai (ZZ_mat<mpz_t> &base, NumVect<NumVect<FP_NR<mpfr_t>>> &gramBase, NumVect<FP_NR<mpfr_t>> target_vector, int dim) {
	// See pseudocode of NPA presented in "The (R)LWE problem on cryptography" master thesis by Michael Anastasiadis, pp. 29-30
	//link here: https://ikee.lib.auth.gr/record/300429/?ln=el
	NumVect<NumVect<FP_NR<mpfr_t>>> w(dim);
	NumVect<NumVect<FP_NR<mpfr_t>>> u(dim);
	u.resize(dim, dim);
	NumVect<FP_NR<mpfr_t>> toReturn(dim);
	toReturn.fill(0.0);
	FP_NR<mpfr_t> l1 = 0.0;
	FP_NR<mpfr_t> l2 = 0.0;
	FP_NR<mpfr_t> l = 0.0;
	FP_NR<mpfr_t> l_unRND = 0.0;
	NumVect<FP_NR<mpfr_t>> gSOMult(dim);
	w[dim - 1] = target_vector;
	for (int i = dim - 1; i > 0; i--) {
		//l[i] =  <w[i], b[i]*> / <b[i]*, b[i]*>
		l1 = dotProduct(w[i], gramBase[i], w[i].size(), gramBase[i].size());
		l2 = dotProduct(gramBase[i], gramBase[i], gramBase[i].size(),gramBase[i].size());
		l = l1 / l2;
		l_unRND = l;// Unrounded l[i]
		l.rnd(l + 0.6);// round l[i], function is void, so affects object itself, hence the existence of l_unRND[i]
		u[i] = mult(base[i], l);// y[i] = round(l[i]) * b[i];
		toReturn.add(u[i]);
		gSOMult = multRow(gramBase[i], (l_unRND - l)); // this equals (l[i] - round(l[i]) * b[i]*)
		w[i - 1] = w[i];
		w[i - 1].sub (gSOMult);
		w[i - 1].sub(u[i]);
		//Above four operations mean w[i -1] = w[i] - (l[i] - round(l[i]) * b[i]* - l[i]b[i]

	}
	return toReturn;

}



int main() {
	ZZ_mat<mpz_t> base;
	NumVect<NumVect<FP_NR<mpfr_t>>> gramBase(5);
	NumVect<FP_NR<mpfr_t>> target(5);
	NumVect<FP_NR<mpfr_t>> test(5);
	target.fill(0.0);
	test.fill(0.0);
	base.resize(5, 5);
	base.fill(0);
	base = latticeGen(base, 5, 1);
	/*base[0][0] = 2;
	base[0][1] = 1;
	base[0][2] = 1;
	base[0][3] = -1;
	base[0][4] = -1;
	base[1][0] = -1;
	base[1][1] = -3;
	base[1][2] = 0;
	base[1][3] = -3;
	base[1][4] = 0;
	base[2][0] = -2;
	base[2][1] = 0;
	base[2][2] = -1;
	base[2][3] = 0;
	base[2][4] = -6;
	base[3][0] = 2;
	base[3][1] = 4;
	base[3][2] = -7;
	base[3][3] = -2;
	base[3][4] = 0;
	base[4][0] = -5;
	base[4][1] = 5;
	base[4][2] = 5;
	base[4][3] = -2;
	base[4][4] = 2;*/
	target[0] = 0.0;
	target[1] = 3.0;
	target[2] = 1.0;
	target[3] = 4.0;
	target[4] = 1.0;
	cout << "Lattice Base" << endl;
	cout << endl;
	cout << base << endl;
	cout << endl;
	cout << "Target Vector" << endl;
	cout << endl;
	cout << target << endl;
	cout << endl;
	reduceLLL(base);
	cout << "Lattice Base, LLL-reduced" << endl;
	cout << endl;
	cout << base << endl;
	cout << endl;
	gramBase = gSO (base, gramBase);
	cout << "Gram-Schmidt Lattice Base" << endl;
	cout << endl;
	cout << gramBase << endl;
	cout << endl;
	test = babai(base, gramBase, target, 5);
	cout << "Babai's output:" << endl;
	cout << endl;
	cout << test << endl;
	cout << endl;
}