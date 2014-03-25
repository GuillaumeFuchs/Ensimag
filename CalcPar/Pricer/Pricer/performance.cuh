#ifndef PERFORMANCE_CUH
#define PERFORMANCE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#pragma comment(lib, "cuda.lib") 
#pragma comment(lib, "cudart.lib") 

// PAYOFF D'UNE OPTION PERFORMANCE
//int N: nombre de pas de temps
//int tid: numero thread dans la grille
//int size: taille de l'option
//float* d_coeff: proportion de chaque actif dans l'option
//float* d_path: ensemble des chemins
__device__ float payoff_performance(
	int N,
	int tid,
	int size,
	float* d_coeff,
	float* d_path)
{
	float pay = 0.;
	float num, den;

	for (int i = 1; i < N; i++){
		num = 0.;
		den = 0.;

		for (int d = 0; d < size; d++){
			num += d_path[tid*(N+1)*size + d*(N+1) + i]*d_coeff[d]; 
			den += d_path[tid*(N+1)*size + d*(N+1) + i-1]*d_coeff[d];
		}
		pay += num/den;
	}

	return 1+MIN(MAX(pay/(float)N-1, 0.0), 0.1);
}

// ITERATION DE MONTE_CARLO
//int N: nombre de pas de temps
//int size: taille de l'option
//int samples: nb échantillon de MC
//float* d_coeff: proportion de chaque actif dans l'option
//float* d_path: ensemble des chemins des iterations de Monte Carlo
//float* per_block_results_price: resultat du payoff pour le calcul du prix
//float* per_block_results_ic: resultat du payoff pour le calcul de l'ic
__global__ void mc_performance(
	int N,
	int size,
	int samples,
	float* d_coeff,
	float* d_path,
	float* per_block_results_price,
	float* per_block_results_ic)
{
	extern __shared__ float s_data[];
	float *s_data_price = s_data;
	float *s_data_ic = &s_data[blockDim.x];

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < samples){
		//Calcul du payoff
		float payoff = payoff_performance(N, tid, size, d_coeff, d_path);
		s_data_price[threadIdx.x] = payoff;
		s_data_ic[threadIdx.x] = payoff*payoff;

		//On charge le r�sultat du payoff dans la m�moire partag�
		//On attend que tous les threads aient calcul�s le payoff
		__syncthreads();

		//R�duction partielle pour chaque thread
		for (int offset = blockDim.x/2; offset > 0; offset >>= 1){
			if (threadIdx.x < offset){
				s_data_price[threadIdx.x] += s_data_price[threadIdx.x + offset];
				s_data_ic[threadIdx.x] += s_data_ic[threadIdx.x + offset];
			}

			//On attend que tous les threads aient effectu�s leur somme partielle
			__syncthreads();
		}

		//Le thread 0 charge le r�sultat
		if (threadIdx.x == 0){
			per_block_results_price[blockIdx.x] = s_data_price[0];
			per_block_results_ic[blockIdx.x] = s_data_ic[0];
		}
	}
}
#endif
