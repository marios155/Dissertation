#include <iostream>
//#include "mp2mpfr.h"
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
#include <fplll/wrapper.h>
#include <math.h>
#include <fplll/nr/numvect.h>

using namespace std;
using namespace fplll;

//Compile using these flags: g++ -std=c++11 -O3 -march=native Babai.cpp -lfplll -lmpfr -lgmp  -o Babai
//Run with ./Babai

/* Function that calculates the dot product of two vectors (must be generic type as data may be of many data types, either from fplll's data or standard data)
	
	@param DataType1 vector1: First vector, called by explicit reference (bound to temporary object)
	@param Datatype 2 vector2: Second vector, same as vector1
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
			gramBase[i][j] = 0.0;
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



NumVect<FP_NR<mpfr_t>> babai (ZZ_mat<mpz_t> &base, Matrix<FP_NR<mpfr_t>> &gramBase, NumVect<FP_NR<mpfr_t>> target_vector, int dim) {

	NumVect<NumVect<FP_NR<mpfr_t>>> w(dim);
	ZZ_mat<mpz_t> u;
	u.resize(dim, dim);
	NumVect<FP_NR<mpfr_t>> toReturn(dim);
	NumVect<FP_NR<mpfr_t>> l(dim);
	w[dim - 1] = target_vector;
	for (int i = dim - 1; i >= 0; i--) {
		l[i] = dotProduct(w[dim - 1], gramBase[i], w[dim - 1].size(), gramBase[i].size()) / dotProduct(gramBase[i], gramBase[i], gramBase[i].size(),gramBase[i].size());
		//w[i - 1] = w[i] - (l[i] - (mpz_t) round(l[i])) * gramBase[i] - u[i];

	}
	return l;

}



int main() {

	ZZ_mat<mpz_t> nRandBase;//Non-rand base
	Matrix<FP_NR<mpfr_t>> gramBase;// Soon - to - be GSO-ed base of above base
	NumVect<FP_NR<mpfr_t>> target (5, 3.0);
	NumVect<FP_NR<mpfr_t>> test (5);
	nRandBase = latticeGen(nRandBase, 5, 0);//Generate Lattice here
	gramBase = gSO(nRandBase, gramBase); // GSO base here
	cout << "Non-Random Lattice base of dimension 5" << endl;
	cout << endl;
	cout << nRandBase << endl;//Print lattice base
	cout << endl;
	cout << "Base orthogonized by Gram-Schmidt orthogonization" << endl;
	cout << gramBase << endl;
	cout << endl;
 	cout << "LLL-reduction on base " << endl;
	cout << endl;
	reduceLLL(nRandBase);
	cout << nRandBase << endl;
	cout << endl;
	test = babai (nRandBase, gramBase, target, 5);
	cout << test << endl;
	cout << endl;
}