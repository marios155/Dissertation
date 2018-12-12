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


ZZ_mat<mpz_t> latticeGen(ZZ_mat<mpz_t> &base, int dim, bool isRandom) {

	if (isRandom) {

		RandGen:: init_with_time();

	}

	base.resize(dim, dim);
	base.gen_uniform(dim);

	return base;


}

Matrix<FP_NR<double>> gSO (ZZ_mat<mpz_t> base, Matrix<FP_NR<double>> &gramBase) {

	int dimension = base.get_cols();
	ZZ_mat<mpz_t> identity;
	ZZ_mat<mpz_t> idTrans;
	MatGSO<Z_NR<mpz_t>, FP_NR<double>> wrapper (base, identity, idTrans, GSO_DEFAULT);
	gramBase.resize(dimension, dimension);
	wrapper.update_gso();
	gramBase = wrapper.get_r_matrix();
	return gramBase;

}



int main() {

	ZZ_mat<mpz_t> nRandBase;
	Matrix<FP_NR<double>> gramBase;
	nRandBase = latticeGen(nRandBase, 5, 0);
	gramBase = gSO(nRandBase, gramBase);
	cout << "Non-Random Lattice base of dimension 5" << endl;
	cout << endl;
	cout << nRandBase << endl;
	cout << endl;
	cout << "Base orthogonized by Gram-Schmidt orthogonization" << endl;
	cout << endl;
	cout << gramBase << endl;
	cout << endl;

}