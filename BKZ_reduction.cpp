#include <iostream>
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

/**
   @brief Write T to `output_filename`.
   @param X T (T is usually a ZZ_mat<ZT> or a vector<Z_NR<ZT>>
   @param output_filename
   @return zero if the file is correctly written to, 1 otherwise.
*/

template <class T> int write_to_file(T &X, const char *output_filename) {
  int status = 0;
  ofstream os;
  os.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  try {
    os.open(output_filename);
    os << X;
    os.close();
  }
  catch (const ofstream::failure&) {
    status = 1;
    cerr << "Error by writing to " << output_filename << "." << endl;
    cout << os.rdstate() << endl;
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
	@brief Function that generates base for Lattice of given dimension, with or without random seed
	
	@param ZZ_mat<mpz_t> &base: The base of our Lattice, a Matrix of integers (we use the FPLLL's Matrix struct as the lattice base's data type). This is called by reference.
	@param int dim: The dimension of the Lattice base
	@param bool isRandom: If set to true, we will generate the Lattice with a random seed based on current time 

*/

void reduceBKZ (ZZ_mat<mpz_t> &base, int blockSize) {

	bool status = bkz_reduction(base, blockSize, BKZ_DEFAULT, FT_MPFR, 100);
	cout << "Process finished with status: " << status << endl;
}

int main(int argc, char** argv) {
	cout << endl;
	ZZ_mat<mpz_t> base;
	int status = 0;
	if (argc == 3) {
		int dim = atoi(argv[1]);
		base.resize(dim, dim);
		if (strcmp(argv[2], "0") == 0) {
			int j = read_file(base, "lattice");
		}
		else {
			latticeGen(base, dim, 1);
		}
		cout << "Lattice base:" << base << endl;
		cout << endl;
		reduceBKZ(base, 5);
		cout << "BKZ - reduced base:" << base << endl;
		cout << endl;
		int i = write_to_file(base, "storage");
		return i;
	}
	else {
		cout << "Error: expected 2 arguments, " << argc << " provided" << endl;
		return 1;
	}
	
}