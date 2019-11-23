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
			status = lll_reduction(base, identity_matrix, identity_matrix_transposed, delta, eta, LM_FAST, FT_MPFR, 0, LLL_DEFAULT);
		}
		else
		{
			status = lll_reduction(base, identity_matrix, identity_matrix_transposed, delta, eta, LM_HEURISTIC, FT_MPFR, 0, LLL_DEFAULT);
		}
	}
}

void bkz_reduction (ZZ_mat &base, string method)
{

}

int main (int argc, char** argv)
{
	return 0;
}