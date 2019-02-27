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

/** Function that initializes a vector of given dimension with random numbers
	
	@param NumVect<FP_NR<T>> vector: Vector to be randomized, called by reference
	@return: vector, randomized by rand()

*/

template <class T> NumVect<FP_NR<T>> randomSet (NumVect<FP_NR<T>> &vector) {
	
	for (int i = 0; i < vector.size(); i++) {

		vector[i] = rand();// Instantiate each element by a random number, generated here

	}
	return vector;

}

/** This function multiplies the coefficients of a given vector vector by a number num.
	@param: MatrixRow<Z_NR<U>> vector: This is a vector of integers, in NPA the initial integer lattice base.
	@param FP_NR<T>> num: Number by which to multiply coefficients
	@return: result, vector multiplied by num

*/

template <class T, class U> NumVect<FP_NR<T>> mult (MatrixRow<Z_NR<U>> &&vector, FP_NR<T> num){
	
	NumVect<FP_NR<T>> result(vector.size());
	for (int i = 0; i < vector.size(); i++) {

		result[i].mul(num, vector[i].get_ld(), GMP_RNDN);// This is FP_NR class's function to implement multiplication,
														 // allowing us to avoid confusing mpfr ang mpz functions in case
														 // of usage of MPFR or GMP Libraries.
	}
	return result;

}

/** This function multiplies the coefficients of a given vector vector by a number num.
	@param: MatrixRow<FP_NR<T>> vector: This is a vector of floating point numbers, in NPA the GSO-ed lattice base.
	@param FP_NR<T>> num: Number by which to multiply coefficients
	@return: result, vector multiplied by num

*/


template <class T> NumVect<FP_NR<T>> multRow (MatrixRow<FP_NR<T>> &&row, FP_NR<T> num) {
	NumVect<FP_NR<T>> toReturn(row.size());
	for (int i = 0; i < row.size(); i++) {

		toReturn[i].mul(row[i], num, GMP_RNDN);// See comments of mult() function above
	}
	return toReturn;
}

/** Function that calculates the dot product of two vectors (must be generic type as data may be of many data types, either from fplll's data or standard data)
	
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

/** Function that calculates the dot product of two vectors (must be generic type as data may be of many data types, either from fplll's data or standard data)
	
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






/* Function that generates base for Lattice of given dimension, with or without random seed
	
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

/* Function that generates orthogonized base for the lattice as per the Gram-Schmidt orthogonization method
	
	@param ZZ_mat<mpz_t> base: The base of our Lattice, a Matrix of integers (we use the FPLLL's Matrix struct as the lattice base's data type)
	@param Matrix<FP_NR<double>> &gramBase: The GSO-ed base of the Lattice, called by reference. We use the FP_NR<double> data type to handle arbitrarily large numbers

*/

Matrix<FP_NR<mpfr_t>> gSO (ZZ_mat<mpz_t> base, Matrix<FP_NR<mpfr_t>> &gramBase) {

	int dimension = base.get_cols();//Retrieve dimension of lattice
	gramBase.resize(dimension, dimension);
	ZZ_mat<mpz_t> identity;//Identity matrix, used in the DEFAULT_GSO method
	ZZ_mat<mpz_t> idTrans;
	idTrans.transpose();//Transposed ID matrix
	MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> wrapper (base, identity, idTrans, GSO_INT_GRAM);// This wrapper is a class that allows us to work with all GSO-related algorithms and data
	 																					//conversions required
	wrapper.update_gso();// Run GSO algorithm with the method selected in constructor (See Documentation for more information)
	gramBase = wrapper.get_r_matrix();// Get the Gram-Schmidt orthogonized base here
	for (int i = 0; i < dimension -1; i++) {
		for (int j = dimension - 1; j > i; j--){
			gramBase[i][j] = 0.0;//Clear @NaN@ flag of MPFR and replace with zero (when the data type is double, this is unnecessary-possible fplll bug?)
		}
	}
	return gramBase;// Return GSO-ed base

}

/** This function implements default LLL reduction on base of lattice
	
	@param ZZ_mat<mpz_t> base: The base of the lattice, a Matrix of intgers, called by reference


*/

void reduceLLL (ZZ_mat<mpz_t> &base) {

	ZZ_mat<mpz_t> identity;//Identity matrix of integers, used in LLL
	ZZ_mat<mpz_t> idTrans;//Transposed version of identity
	Wrapper *wrapper = new Wrapper(base, identity, idTrans, 0.99, 0.51, LLL_DEFAULT);//This class implements lll-reduction, with a plethora of types
																					 // based on data type of lattice base and other requirements
	bool status = wrapper->lll();// LLL-reductiuon is called here, parameters already given to the wrapper: δ = 0.99, ε = 0.51
	//cout << status << endl;//if TRUE (1) reduction was successful
	//cout << endl;

}

/** Function that implements Babai's Nearest Plane Algorithm for given target vector. Preprocessing is considered done.
	
	@param ZZ_mat<mpz_t> base: Base of integer lattice, reduced either by LLl or BKZ (choice to be implemented)
	@param Matrix<FP_NR<mpfr_t>> gramBase: GSO-ed Lattice base of above lattice.
	@param NumVect<FP_NR<mpfr_t>> target_vector: Target vecto for which NP vector will be calculated
	@return: toReturn, NumVect<FP_NR<mpfr_t>> type, Nearest Plane vector
*/

NumVect<FP_NR<mpfr_t>> babai (ZZ_mat<mpz_t> &base, Matrix<FP_NR<mpfr_t>> &gramBase, NumVect<FP_NR<mpfr_t>> target_vector, int dim) {
	// See pseudocode of NPA presented in "The (R)LWE problem on cryptography" master thesis by Michael Anastasiadis, pp. 29-30
	//link here: https://ikee.lib.auth.gr/record/300429/?ln=el
	NumVect<NumVect<FP_NR<mpfr_t>>> w(dim);
	NumVect<NumVect<FP_NR<mpfr_t>>> u(dim);
	u.resize(dim, dim);
	NumVect<FP_NR<mpfr_t>> toReturn(dim);
	toReturn.gen_zero(dim);
	NumVect<FP_NR<mpfr_t>> l(dim);
	FP_NR<mpfr_t> l_unRND = 0.0;
	NumVect<FP_NR<mpfr_t>> gSOMult(dim);
	w[dim - 1] = target_vector;
	for (int i = dim - 1; i > 0; i--) {
		//l[i] =  <b[i], b[i]*> / <b[i]*, b[i]*>
		l[i] = dotProduct(w[dim - 1], gramBase[i], w[dim - 1].size(), gramBase[i].size()) / dotProduct(gramBase[i], gramBase[i], gramBase[i].size(),gramBase[i].size());
		l_unRND = l[i];// Unrounded l[i]
		l[i].rnd(l[i]);// round l[i], function is void, so affects object, hence the existence of l_unRND[i]
		u[i] = mult(base[i], l[i]);// y[i] = round(l[i]) * b[i]
		toReturn.add(u[i]);
		gSOMult = multRow(gramBase[i], (l_unRND - l[i])); // this equals (l[i] - round(l[i]) * b[i]*)
		w[i - 1] = w[i];
		w[i - 1].sub (gSOMult);
		w[i - 1].sub(u[i]);
		//Above four operations mean w[i -1] = w[i] - (l[i] - round(l[i]) * b[i]* - l[i]b[i]

	}
	return toReturn;

}



int main() {

	ZZ_mat<mpz_t> nRandBase;//Non-rand base
	Matrix<FP_NR<mpfr_t>> gramBase;// Soon - to - be GSO-ed base of above base
	gramBase.resize(5, 5);// Resize matrix according to dimension given
	gramBase.fill(0.0);// Initialize with zeroes
	NumVect<FP_NR<mpfr_t>> target (5);// Target vector for NPA
	target = randomSet (target);// Randomize target vector
	NumVect<FP_NR<mpfr_t>> test (5);// This is the return vector of NPA
	test.fill(0.0); // Instantiate with zeroes
	nRandBase = latticeGen(nRandBase, 5, 0);//Generate Lattice here
	gramBase = gSO(nRandBase, gramBase); // GSO base here
	cout << "Non-Random Lattice base of dimension 5" << endl;
	cout << endl;
	cout << nRandBase << endl;// Print lattice base
	cout << endl;
	cout << "Base orthogonized by Gram-Schmidt orthogonization" << endl;
	cout << gramBase << endl;// Print GSO-ed base
	cout << endl;
 	cout << "LLL-reduction on base " << endl;
	cout << endl;
	reduceLLL(nRandBase);// Reduce base here
	cout << nRandBase << endl;// Print LLL-ed base here
	cout << endl;
	cout << "Random Target Vector on which NPA will be performed" << endl;
	cout << endl;
	cout << target << endl;// Print target vector
	cout << endl;
	cout << "Nearest Plane Vector for random target vector" << endl;
	cout << endl; 
	test = babai (nRandBase, gramBase, target, 5);// NPA is executed here
	cout << test << endl;// Print Nearest Plane vector
	cout << endl;
}