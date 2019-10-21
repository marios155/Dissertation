#include <stdlib.h>
#include<stdio.h>
#include <iostream>
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
#include <fplll/wrapper.h>
#include <fplll/gso.h>
#include<fplll/gso_gram.h>
#include <fplll/gso_interface.h>
using namespace std;
using namespace fplll;

#ifndef TESTDATADIR
#define TESTDATADIR ".."
#endif
#define N  25
#define THREADS_PER_BLOCK 5
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

__global__ void dotProduct(double *realLattice, double *realVector, double *total) {
  
  __shared__ double temp[THREADS_PER_BLOCK];
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  temp[threadIdx.x] =  realLattice[index] * realVector[index];
  printf("%d \n", realLattice[index]);
  printf("%d \n", temp[threadIdx.x]);
  __syncthreads();
  if (0 == threadIdx.x) {
  		double sum = 0;
  		for (int i = 0; i < THREADS_PER_BLOCK; i++) {
  			sum = sum + temp[i];
  		}
  	atomicAdd(total, sum);
  }
}  

/**
   @brief Read T from `input_filename`.
   @param X T (T is usually a ZZ_mat<ZT> or a vector<Z_NR<ZT>>
   @param input_filename
   @return zero if the file is correctly read, 1 otherwise.
*/

__host__
int read_file(ZZ_mat<mpz_t> &X, const char *input_filename) {
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

int main(int argc, char** argv) {
	ZZ_mat<mpz_t> lattice;
	FP_mat<mpfr_t> vector;
	vector.resize(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
	vector.fill(1.0);
	lattice.resize(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
	int i =read_file(lattice, "lattice");
	double * realLattice = new double [N];
	double * realVector = new double [N];
	double * total = new double;
	cudaMallocManaged (&realLattice, N * sizeof(double));
	cudaMallocManaged(&realVector, N * sizeof(double));
	cudaMallocManaged(&total, 1 * sizeof(double));
	int k = 0;
	for (int i = 0; i < THREADS_PER_BLOCK; i++) {
		for (int j = 0; j < THREADS_PER_BLOCK; j++) {
			realLattice[k] = lattice[i][j].get_si();
			realVector[k] = vector[i][j].get_si();
			k++;
		}
	}
	*total = 0.0;
	dotProduct<<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (realLattice, realVector, total);
	cudaDeviceSynchronize();
	cout << "Dot Product of Vectors is " << *total << endl;
	cudaFree(realLattice);
	cudaFree(realVector);
	cudaFree(total);
	return 0;

}