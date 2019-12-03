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

/**
   @brief Write T to `input_filename`.
   @param X T (T is usually a ZZ_mat<ZT> or a vector<Z_NR<ZT>>
   @param input_filename
   @return zero if the file is correctly written to, 1 otherwise.
*/

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

void strategize(int block_size, vector<Strategy> &strategies)
{
	for (int i = 0; i <= block_size; i++) 
	{
		Strategy strategy = Strategy::EmptyStrategy(i);
		if ((i == 10) || (i == 20) || (i == 30)) 
		{
			strategy.preprocessing_block_sizes.emplace_back(i / 2);
		}
		strategies.emplace_back(move(strategy));
	}
}

void bkz_reduction (ZZ_mat<mpz_t> &basis, int block_size, string method)
{
	int status = 0;
	if (method == "DEFAULT")
	{
		status = bkz_reduction(basis, block_size, BKZ_DEFAULT, FT_MPFR, 128);
	}
	else
	{
		vector<Strategy> strategies;
		strategize(block_size, strategies);
		/* This class generates the BKZ parameters */
		BKZParam parameters (block_size, strategies);
		/**
	 	 * The flag for the parameter is set to BKZ_DEFAULT. 
	 	 */
		parameters.flags = BKZ_DEFAULT; 
		status = bkz_reduction(&basis, NULL, parameters, FT_DEFAULT, 53);
	}
}

int main (int argc, char** argv)
{
	ZZ_mat<mpz_t> basis;
	int columns = 0, rows = 0, block_size = 0, status = 0;
	string output = "bkz_reduced_basis_";
	string method;
	if (argc == 6)
	{
		columns = atoi(argv[1]);
		rows = atoi(argv[2]);
		block_size = atoi(argv[3]);
		basis.resize(columns, rows);
		status = read_file(basis, argv[4]);
		basis.transpose();
		bkz_reduction(basis, block_size, argv[5]);
		method = argv[5];
		output = output + method;
		status = write_to_file(basis, output);
	}
	else
	{
		cout << "Expected 6 arguments, " << argv - 1 << "provided." << endl;
		return 1;
	}
	return 0;
}