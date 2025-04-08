/*
 * Generation of single particle density matrix 
 * using the Matrix Mixer algorithm. 
 *
 * Author: Mikolaj M. Paraniak <mikolajp@protonmail.ch>
 *
 */
#ifndef RHO1P_MATRIX_MIXER
#define RHO1P_MATRIX_MIXER

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <vector>
#include <algorithm>
#include <numeric>

/*
 * Generate the single particle density matrix 
 * using the Matrix Mixer algorithm based 
 * on the method by B.-G. Englert. 
 *
 * rho: matrix
 * L: size of the matrix 
 * occnum: vector of occupation numbers 
 * N: cumulative sum of the occupation numbers (must be even)
 *
 * assumes L >= N/2 (at most 2 particles per state)
 * assumes L >= M
 * assumes num > 0.0
 *
 */
int GenerateRho1pMixer(std::vector<std::vector<double>>& rho, size_t L, std::vector<double> &occnum, double num);

extern "C" int generate_rho1p_mixer(size_t L, double* rho, size_t M, double* occnum, double num);

#endif /*RHO1P_MATRIX_MIXER*/

