#include <iostream>
//#include "mp2mpfr.h"
#include <gmp.h>
#include <mpfr.h>
#include <fplll.h>
#include <fplll/nr/nr.h>
//#include <fpll/defs.h>
#include "fplll/util.h"

using namespace std;
using namespace fplll;


ZZ_mat<mpz_t> latticeGen(ZZ_mat<mpz_t> &base, int dim, bool isRandom) {

	if (isRandom) {

		RandGen:: init_with_time();

	}

	base.resize(dim, dim);
	base.gen_uniform(dim);

	return base;


}



int main() {

	ZZ_mat<mpz_t> nRandBase;

	nRandBase = latticeGen(nRandBase, 5, 0);

	cout << nRandBase << endl;


}