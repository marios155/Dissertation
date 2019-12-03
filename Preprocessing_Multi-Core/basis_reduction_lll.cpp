#include <stdio.h>
#include <iostream>
#include <math.h>
#include <mpfr.h>
#include <gmp.h>
#include <fplll.h>
#include <fplll/nr/nr.h>
#include <fplll/wrapper.h>
#include "fplll/defs.h"
#include "fplll/util.h"

using namespace std;
using namespace fplll;

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

template <class T> int write_to_file (T &X, string input_filename) {
  int status = 0;
  ofstream os;
  os.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    os.open(input_filename);
    os << X;
    os.close();
  }
  catch (const ifstream::failure&) {
    status = 1;
    cerr << "Error by reading " << input_filename << "." << endl;
    cout << os.rdstate() << endl;
  }

  return status;
}

void lll_reduction(ZZ_mat<mpz_t> &base, string method, double delta, double eta) 
{
	int status = 0;
	ZZ_mat<mpz_t> identity_matrix;
	ZZ_mat<mpz_t> identity_matrix_transposed;
	if (method == "PROVED")
	{
		status = lll_reduction(base, identity_matrix, identity_matrix_transposed, delta, eta, LM_PROVED, FT_MPFR, 0, LLL_DEFAULT);
	}
	else
	{
		if (method == "FAST")
		{
			status = lll_reduction(base, identity_matrix, identity_matrix_transposed, delta, eta, LM_FAST, FT_DOUBLE, 0, LLL_DEFAULT);
		}
		else
		{
			status = lll_reduction(base, identity_matrix, identity_matrix_transposed, delta, eta, LM_HEURISTIC, FT_MPFR, 0, LLL_DEFAULT);
		}
	}
}

int main (int argc, char** argv)
{
	ZZ_mat<mpz_t> basis;
	int columns = 0;
	int rows = 0;
	double delta = 0.0;
	double eta = 0.0;
	int status = 0;
	string method;
	string output_filename;
	if (argc == 7)
	{
		columns = atoi(argv[1]);
		rows = atoi(argv[2]);
		delta = atof(argv[3]);
		eta = atof(argv[4]);
		basis.resize(columns, rows);
		status = read_file (basis, argv[5]);
		basis.transpose();
		method = argv[6];
		lll_reduction(basis, method, delta, eta);
		output_filename = "lll_reduced_basis_" + method;
		status = write_to_file (basis, output_filename);
	}
	else 
	{
		cout << "Expected 6 arguments, " << argc - 1 << " provided." << endl;
		return 1;
	}
	return 0;
}