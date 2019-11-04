#include <stdlib.h>
#include<stdio.h>
#include <iostream>
#include <string>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gmp.h>
#include <mpfr.h>
#include <fplll.h>
#include "fplll/defs.h"
#include "fplll/util.h"
#include <fplll/nr/nr.h>
#include <fplll/nr/numvect.h>
using namespace std;
using namespace fplll;

#ifndef TESTDATADIR
#define TESTDATADIR ".."
#endif
#define N  1600
#define THREADS_PER_BLOCK 40
#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_

#include <cuda.h>

  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

  #else
  static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }


  #endif
#endif

/**
   @brief Read T from `input_filename`.
   @param X T (T is usually a ZZ_mat<ZT> or a vector<Z_NR<ZT>>
   @param input_filename
   @return zero if the file is correctly read, 1 otherwise.
*/

__host__
void read_file(ZZ_mat<mpz_t> &X, const char *input_filename) {
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
    cout << status << endl;
    cout << is.rdstate() << endl;
  }

}

void read_vector(vector<double> &vector, const char *filename) {
	ifstream is;
	string string;
	is.open(filename);
	while (!is.eof()){
		while (getline(is, string)){
		if (string.c_str() !=" ") {
				vector.push_back(atof(string.c_str()));
			}
		}
	}
	is.close();
}


/*int read_dimension (const char *input_filename) {
	int dim;
	string input;
	ifstream is;
	is.open(input_filename);
	try {
		getline(is, input);
		if (input.size() > 0) {
			dim = atoi(input.c_str());
			return dim;
		}
		return 0;
	}
	catch (const ifstream::failure&) {
		cerr << "Error reading " << input_filename << "." << endl;
		return 0;
	}
}*/

template <class T, class U> NumVect<FP_NR<T>> addRow (MatrixRow<Z_NR<U>> &&vector, FP_NR<T> num){
	
	NumVect<FP_NR<T>> result(vector.size());
	for (int i = 0; i < vector.size(); i++) {

		result[i].add(num, vector[i].get_ld(), MPFR_RNDN);// This is FP_NR class's function to implement multiplication,
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

template <class T, class U> FP_NR<T> dotProduct (MatrixRow<Z_NR<U>> &&vector1, NumVect<FP_NR<T>> &vector2, int length1, int length2) {
	
	FP_NR<T> sum = FP_NR<T> (0.0);
	NumVect<FP_NR<T>> vect(1);
	vect.fill(0.0);
	int i = max(length1, length2);
	for (int j = 0; j <= i - 1; j++) {
		vect[0].mul(vector2[j], vector1[j].get_ld(), MPFR_RNDN);
		sum = sum + vect[0];
		vect.fill(0.0);

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

void get_gram(ZZ_mat<mpz_t> &base, FP_mat<mpfr_t> &gram) {
	int dimension = base.get_cols();
	NumVect<NumVect<FP_NR<mpfr_t>>> gramBase(dimension);
	gramBase = gSO(base, gramBase);
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			gram[i][j] = gramBase[i][j];
		}
	}
}

void preprocess(ZZ_mat<mpz_t> &lattice, FP_mat<mpfr_t> &gram, vector<vector<double>> &doubleLattice,  vector<vector<double>> &doubleGram) {
	int rows = lattice.get_rows();
	int cols = lattice.get_cols();
	for (int i = 0; i < rows; i++) {
		doubleLattice[i].resize(cols);
		doubleGram[i].resize(cols);
		for (int j = 0; j < cols; j++) {
			doubleLattice[i][j] = lattice[i][j].get_si();
			doubleGram[i][j] = gram[i][j].get_d();		
		}
	}
}

//==========================================================================================================



__global__ void dotProduct(double *vector1, double *vector2, double *total, int dim) {
  __shared__ double temp[THREADS_PER_BLOCK];
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dim) {
  	 temp[index] =  vector1[index] * vector2[index];
  }
  __syncthreads();
  if (0 == threadIdx.x) {
  		double sum = 0;
  		for (int i = 0; i < dim; i++) {
  			sum = sum + temp[i];
  		}
  	atomicAdd(total, sum);
  }
}

__global__ void vectorAdd (double *vector1, double *vector2, double *result) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	result[threadIdx.x] = vector1[index] + vector2[index];
}

__global__ void vectorSub (double *vector1, double *vector2, double *result) {
	int index = threadIdx.x +blockIdx.x * blockDim.x;
	result[threadIdx.x] = vector1[index] - vector2[index];
}    



double dot(vector<double> &vector1, vector<double> &vector2, int dim) {
	double * realVector1;
	double * realVector2;
	double * total;
	double * result = new double;
	cudaMallocManaged (&realVector1, dim * sizeof(double));
	cudaMallocManaged(&realVector2, dim * sizeof(double));
	cudaMallocManaged(&total, 1 * sizeof(double));
	*total = 0.0;
	for (int i = 0; i < dim; i++) {
		realVector1[i] = vector1[i];
		realVector2[i] = vector2[i];
	}
	dotProduct<<<1,  dim>>> (realVector1, realVector2, total, dim);
	cudaDeviceSynchronize(); 
	*result = *total;
	cudaFree(realVector1);
	cudaFree(realVector2);
	cudaFree(total);
	return *result;
}

double integer_production (vector <double *> &list, vector<vector<double>> &gramBase, double* cuda_target, int index, int dim, int position) {
	double * c1;
	double * list_element;
	double * gram_temp;
	double * unrounded_integer;
	double   result = 0.0;
	cudaMallocManaged (&c1, dim * sizeof(double));
	cudaMallocManaged (&list_element, dim * sizeof(double));
	cudaMallocManaged (&unrounded_integer, dim * sizeof(double));
	cudaMallocManaged (&gram_temp, dim * sizeof(double));
	for (int i = 0; i < dim; i++) {
		list_element[i] = list[index][i];
		gram_temp[i] = gramBase[position][i];
		c1[i] = 0.0;
	}
	*unrounded_integer = 0.0;
	vectorSub<<<1, dim>>> (cuda_target, list_element, c1);
	cudaDeviceSynchronize();
	dotProduct<<<1, dim>>> (c1, gram_temp, unrounded_integer, dim);
	cudaDeviceSynchronize();
	result = *unrounded_integer;
	cudaFree (list_element);
	cudaFree(c1);
	cudaFree(gram_temp);
	cudaFree (unrounded_integer);
	return result;
}

void lindner (ZZ_mat<mpz_t> lattice, FP_mat<mpfr_t> gram, double* target, vector<int> buffer) {
	int dim = lattice.get_rows();
	vector<vector<double>> base (dim);
	vector<vector<double>> gramBase (dim);
	vector<double *> list (dim);
	list[0] = new double [dim];
	int k = 1;
	double result = 0.0;
	preprocess(lattice, gram, base, gramBase);
	double *gramDP = new double[dim];
	double *cuda_target;
	cudaMallocManaged(&cuda_target, dim * sizeof(double));
	lattice.clear();
	gram.clear();
	for (int i = 0; i < dim; i++) {
		cuda_target[i] = target[i];
		list[0][i] = 0.0;
		gramDP[i] = dot(gramBase[i], gramBase[i], dim);
	}
	delete[] target;
	for (int i = dim -1; i >= 0; i--) {
		double ** tempList = new double*[dim];
		for (int j = 0; j < k; j++) {
			result = integer_production(list, gramBase, cuda_target, j, dim, i);
			cout << result << endl;
		}
		delete[] tempList;
	}
}



int main(int argc, char** argv) {
	//int dim = atoi(argv[1]);
	int dim = 5;
	ZZ_mat<mpz_t> lattice;
	FP_mat<mpfr_t> gramLattice;
	vector<double> target;
	vector<int> buffer (dim, 2);
	if (dim > 0) {
		lattice.resize(dim, dim);
		read_file(lattice, "storage");
		gramLattice.resize(dim, dim);
		get_gram(lattice, gramLattice);
		read_vector(target, "vector");
	}
	else {
		cout << "No dimension entered, program will exit." << endl;
	}
	double *targetReal = new double [dim];
	for (int i = 0; i < dim; i++) {
		targetReal[i] = target[i];
	}
	cout << endl;
	lindner (lattice, gramLattice, targetReal, buffer);
	return 0;

}