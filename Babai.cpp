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
#include <fplll/gso_interface.h>

using namespace std;
using namespace fplll;

//Compile using these flags: g++ -std=c++1 -O3 -march=native Babai.cpp -lfpll -lgm -lmpfr -o Babai
//Run with ./Babai


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

Matrix<FP_NR<double>> gSO (ZZ_mat<mpz_t> base, Matrix<FP_NR<double>> &gramBase) {

	int dimension = base.get_cols();//Since base is uniform, both dimensions equal its columns. These we get here.
	ZZ_mat<mpz_t> identity;//Identity matrix, used in the DEFAULT_GSO method
	ZZ_mat<mpz_t> idTrans;//Transposed ID matrix
	MatGSO<Z_NR<mpz_t>, FP_NR<double>> wrapper (base, identity, idTrans, GSO_DEFAULT);// This wrapper is a class that allows us to work with all GSO-related algorithms and data
	 																					//conversions required
	gramBase.resize(dimension, dimension);//Resize gramBase according to dimensions of base
	wrapper.update_gso();// Run GSO algorithm with the method selected in constructor (See Documentation for more information)
	gramBase = wrapper.get_r_matrix();// Get the Gram-Schmidt orthogonized base here
	return gramBase;// Return GSO-ed base

}

/*
	
	@param:


*/



int main() {

	ZZ_mat<mpz_t> nRandBase;//Non-rand base
	Matrix<FP_NR<double>> gramBase;// Soon - to - be GSO-ed base of above base
	nRandBase = latticeGen(nRandBase, 5, 0);//Generate Lattice here
	gramBase = gSO(nRandBase, gramBase); // GSO base here
	cout << "Non-Random Lattice base of dimension 5" << endl;
	cout << endl;
	cout << nRandBase << endl;//Print lattice base
	cout << endl;
	cout << "Base orthogonized by Gram-Schmidt orthogonization" << endl;
	cout << endl;
	cout << gramBase << endl;//Print GSO-ed base
	cout << endl;

}