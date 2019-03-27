#include <iostream>
#include <math.h>
#include <time.h>
#include <gmp.h>
#include <mpfr.h>
#include <fplll.h>
#include "fplll/defs.h"
#include "fplll/util.h"
#include <fplll/nr/nr.h>
#include <fplll/nr/numvect.h>
#include <fplll/wrapper.h>
#include <fplll/gso.h>
#include<fplll/gso_gram.h>
#include <fplll/gso_interface.h>
using namespace std;
using namespace fplll;

#ifndef TESTDATADIR
#define TESTDATADIR ".."
#endif

// Compile using these flags: g++ -std=c++11 -O3 -march=native Lindner.cpp -lfplll -lmpfr -lgmp  -o Lindner
// Run with ./
// Argument 1: "test" if you wish to test specific lattice, otherwise "0" for non-random, "1" for random
// Argument 2: dimension of lattice
// Argument 3: "0" for non-random vector (change values in lines 378-382), "1" for random, generated here

/**
   @brief Read T from `input_filename`.
   @param X T (T is usually a ZZ_mat<ZT> or a vector<Z_NR<ZT>>
   @param input_filename
   @return zero if the file is correctly read, 1 otherwise.
*/

template <class T> int read_file(T &X, const char *input_filename) {
  int status = 0;
  ifstream is;
  is.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    is.open(input_filename);
    is >> X;
    is.close();
  }
  catch (const ifstream::failure&) {
    status = 1;
    cerr << "Error by reading " << input_filename << "." << endl;
    cout << is.rdstate() << endl;
  }

  return status;
}

template <class T, class U> NumVect<FP_NR<T>> addRow (MatrixRow<Z_NR<U>> &&vector, FP_NR<T> num){
	
	NumVect<FP_NR<T>> result(vector.size());
	for (int i = 0; i < vector.size(); i++) {

		result[i].add(num, vector[i].get_ld(), MPFR_RNDN);// This is FP_NR class's function to implement multiplication,
														 // allowing us to avoid confusing mpfr ang mpz functions in case
														 // of usage of MPFR or GMP Libraries.
	}
	return result;

}

/** 
	@brief: Function that initializes a vector of given dimension with random numbers
	@param NumVect<FP_NR<T>> vector: Vector to be randomized, called by reference
	@return: vector, randomized by rand()

*/

template <class T> ZZ_mat<T> randomSet (ZZ_mat<T> &target) {
	
	for (int i = 0; i < target.get_cols(); i++) {
		for (int j = 0; j < target.get_rows(); j++) {
			
			target[i][j] = rand() % 100;// Instantiate each element by a random number, generated here

		}
		
	}
	return target;

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

template <class T> FP_NR<T> dotProduct (Matrix<FP_NR<T>> matrix, NumVect<FP_NR<T>> vector) {
	FP_NR<T> res = 0.0;
	for (int i = 0; i < matrix.get_cols(); i++) {
		res = res + vector[i] * matrix[0][i];
	}
	return res;
}

template <class T, class U> FP_NR<T> dotProduct (ZZ_mat<U> matrix, NumVect<FP_NR<T>> vector) {
	FP_NR<T> res = 0.0;
	for (int i = 0; i < matrix.get_cols(); i++) {
		res = res + vector[i].get_d() * matrix[0][i].get_d();
	}
	return res;
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

template <class T, class U> ZZ_mat<U> mul (MatrixRow<Z_NR<U>> &&vector, FP_NR<T> num){
	
	ZZ_mat<U> result(1, vector.size());
	for (int i = 0; i < vector.size(); i++) {

		result[0][i].mul_si(vector[i], num.get_d());// This is FP_NR class's function to implement multiplication,
														 // allowing us to avoid confusing mpfr ang mpz functions in case
														 // of usage of MPFR or GMP Libraries.
	}
	return result;

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

NumVect<NumVect<FP_NR<mpfr_t>>> gSO (ZZ_mat<mpz_t> & base, NumVect<NumVect<FP_NR<mpfr_t>>> & gramBase) {
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

void reduceLLL (ZZ_mat<mpz_t> & base) {
	int dim = base.get_cols();
	ZZ_mat<mpz_t> identity;//Identity matrix, used in the DEFAULT_GSO method
	ZZ_mat<mpz_t> idTrans;//Transposed ID matrix;
	ZZ_mat<mpz_t> temp (2, dim);
	temp.fill(0);
	Wrapper *wrapper = new Wrapper (base, identity, idTrans, 0.75, 0.51, LLL_DEFAULT);
	bool status = wrapper -> lll();
	temp[0].add(base[dim - 2]);
	temp[1].add(base[dim - 1]);
	base[dim - 2].fill(0);
	base[dim - 2].add(temp[1]);
	base[dim - 1].fill(0);
	base[dim -1].add (temp[0]);
	//cout << status << endl;
	/*int status = lll_reduction(base, 0.75, 0.51, LM_PROVED, FT_MPFR, 0, LLL_DEFAULT);
	MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> M (base, identity, idTrans, 0);
	status = is_lll_reduced<Z_NR<mpz_t>, FP_NR<mpfr_t>>(M, 0.75, 0.51);*/

}

/** 
	@brief Function that implements Lindner-Peikert's generalization of Babai's Nearest Plane Algorithm for given target vector. Preprocessing is considered done.
	
	@param ZZ_mat<mpz_t> base: Base of integer lattice, reduced either by LLL or BKZ (choice to be implemented)
	@param Matrix<FP_NR<mpfr_t>> gramBase: GSO-ed Lattice base of above lattice.
	@param NumVect<FP_NR<mpfr_t>> target_vector: Target vector for which NP vector will be calculated
	@param NumVect<Z_NR<mpz_t>> buffer: Initial distance vector of n elements, where n is base dimension. Usually filled with same integer number. 
	@return: toReturn, NumVect<NumVect<FP_NR<mpfr_t>>> type, Nearest Plane vectors
*/

ZZ_mat<mpz_t> Lindner (ZZ_mat<mpz_t> &base, NumVect<NumVect<FP_NR<mpfr_t>>> &gramBase, ZZ_mat<mpz_t> target, NumVect<int> buffer) {
	ZZ_mat<mpz_t> toReturn;
	ZZ_mat<mpz_t> toCompute;
	Matrix<FP_NR<mpfr_t>> buff;
	ZZ_mat<mpz_t> temp;
	ZZ_mat<mpz_t> temp1;
	int dimension = base.get_cols();
	FP_NR<mpfr_t> c1 = 0.0;
	FP_NR<mpfr_t> c2 = 0.0;
	FP_NR<mpfr_t> c = 0.0;
	FP_NR<mpfr_t> c_unRND = 0.0;
	FP_NR<mpfr_t> l = 0.0;
	int count = 0;
	toReturn.resize(1, dimension);
	toReturn.fill(0);
	toCompute.resize(1, dimension);
	temp.resize(target.get_rows(), target.get_cols());
	temp.fill(0);
	for (int i = dimension - 1; i >= 0; i--) {
		toCompute.resize(toReturn.get_rows(), dimension);
		toCompute.fill(0);
		for (int j = 0; j < toReturn.get_rows(); j++) {
			count++;
			temp[0].add(target[0]);
			temp[0].sub(toReturn[j]);
			c1 = dotProduct(temp, gramBase[i]);
			c2 = dotProduct(gramBase[i], gramBase[i], dimension, dimension);
			c_unRND = c1 / c2;
			temp.fill(0);
			toCompute.resize(toCompute.get_rows() + buffer[i], dimension);
			for (int k = 1; k <= buffer[i]; k++) {
				if (k % 2 == 0){
					l.floor(c_unRND);
				}
				else{
					l.floor(c_unRND + 1.0);
				}
				temp1 = mul (base[i], l);
				temp1[0].add(toReturn[j]);
				toCompute[j + k - 1].add(temp1[0]);
				temp1.fill(0);
			}	
		}
		toReturn.resize(toReturn.get_rows() + count, dimension);
		toReturn[count].fill(0);
		toReturn[count].add(toCompute[0]);
		for (int u = count + 1; u < toReturn.get_rows(); u++) {
			toReturn[u].fill(0);
			toReturn[u].add(toCompute[u - count]);
		}
		count = 0;
	}
	return toReturn;
}



int main(int argc, char** argv) {
	cout << endl;
	ZZ_mat<mpz_t> base;
	int status = 0;
	if (argc == 4) {
		int dim = atoi(argv[2]);
		NumVect<NumVect<FP_NR<mpfr_t>>> gramBase(dim);
		ZZ_mat<mpz_t> target(1, dim);
		NumVect<int> buffer(dim);
		ZZ_mat<mpz_t> res;
		buffer.fill(2);
		base.resize(dim, dim);
		target.fill(0.0);
		res.resize(dim, dim);
		if (strcmp(argv[1], "test") == 0) {
			status = read_file(base, "lattice");
		}
		else {
			if (strcmp(argv[1], "0") == 0){
				base = latticeGen(base, dim, 0);
			}
			else {
				base = latticeGen(base, dim, 1);
			}
		}
		if (strcmp(argv[3], "1") ==0) {
			target = randomSet(target);
		}
		else {
			target[0][0] = 1;
			target[0][1] = 0;
			target[0][2] = 2;
			target[0][3] = 1;
			target[0][4] = 0;
		}
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
		res = Lindner(base, gramBase, target, buffer);
		cout << "Lindner's output:" << endl;
		cout << endl;
		cout << res << endl;
		cout << endl;
	}
	else {
		cerr << "Expected 4 arguments,"<< " " << argc << " " << "provided" << endl;
	}
}