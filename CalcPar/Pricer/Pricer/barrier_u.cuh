#ifndef BARRIER_U_CUH
#define BARRIER_U_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#pragma comment(lib, "cuda.lib") 
#pragma comment(lib, "cudart.lib") 

// PAYOFF D'UNE OPTION BARRIER_U
//int N: nombre de pas de temps
//int tid: numero thread dans la grille
//int size: taille de l'option
//float K: strike de l'option
//float* d_coeff: proportion de chaque actif dans l'option
//float* d_bu: barrière désactivante up de chaque actif
//float* d_path: ensemble des chemins
__device__ float payoff_barrier_u(
	int N,
	int tid,
	int size,
	float K,
	float* d_coeff,
	float* d_bu,
	float* d_path)
{
	float pay = 0.;
	for (int d = 0; d < size; d++){
		float s = d_path[tid*(N+1)*size+ (d+1)*(N+1)-1]; 
		if (s > d_bu[d])
			return 0;
		else
			pay+=s*d_coeff[d];
	}

	for (int d = 0; d < size; d++){
		for (int i = 0; i < N; i++){
			float s = d_path[tid*(N+1)*size+d*(N+1)+i];
			if (s > d_bu[d])
				return 0;
		}
	}
	
	return max(pay - K, 0.0);
}

// ITERATION DE MONTE_CARLO
//int N: nombre de pas de temps
//int size: taille de l'option
//int samples: nb échantillon de MC
//float K: strike de l'option
//float* d_coeff: proportion de chaque actif dans l'option
//float* d_bu: barrière désactivante up de chaque actif
//float* d_path: ensemble des chemins des iterations de Monte Carlo
//float* per_block_results_price: resultat du payoff pour le calcul du prix
//float* per_block_results_ic: resultat du payoff pour le calcul de l'ic
__global__ void mc_barrier_u(
	int N,
	int size,
	int samples,
	float K,
	float* d_coeff,
	float* d_bu,
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
			float payoff = payoff_barrier_u(N, tid, size, K, d_coeff, d_bu, d_path);
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
