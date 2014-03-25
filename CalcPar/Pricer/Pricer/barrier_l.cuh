#ifndef BARRIER_L_CUH
#define BARRIER_L_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#pragma comment(lib, "cuda.lib") 
#pragma comment(lib, "cudart.lib") 

// PAYOFF D'UNE OPTION BARRIER_L
//int N: nombre de pas de temps
//int tid: numero thread dans la grille
//int size: taille de l'option
//float K: strike de l'option
//float* d_coeff: proportion de chaque actif dans l'option
//float* d_bl: barrière désactivante down de chaque actif
//float* d_path: ensemble des chemins
__device__ float payoff_barrier_l(
	int N,
	int tid,
	int size,
	float K,
	float* d_coeff,
	float* d_bl,
	float* d_path)
{
	float pay = 0.;
	for (int d = 0; d < size; d++){
		float s = d_path[tid*(N+1)*size+ (d+1)*(N+1)-1]; 
		if (s < d_bl[d])
			return 0;
		else
			pay+=s*d_coeff[d];
	}

	for (int d = 0; d < size; d++){
		for (int i = 0; i < N; i++){
			float s = d_path[tid*(N+1)*size+d*(N+1)+i];
			if (s < d_bl[d])
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
//float* d_bl: barrière désactivante down de chaque actif
//float* d_path: ensemble des chemins des iterations de Monte Carlo
//float* per_block_results_price: resultat du payoff pour le calcul du prix
//float* per_block_results_ic: resultat du payoff pour le calcul de l'ic
__global__ void mc_barrier_l(
	int N,
	int size,
	int samples,
	float K,
	float* d_coeff,
	float* d_bl,
	float* d_path,
	float* per_block_results_price)
{
		extern __shared__ float sdata_price[];
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < samples){
			//Calcul du payoff
			sdata_price[threadIdx.x] = payoff_barrier_l(N, tid, size, K, d_coeff, d_bl, d_path);

			//On charge le r�sultat du payoff dans la m�moire partag�
			//On attend que tous les threads aient calcul�s le payoff
			__syncthreads();

			//R�duction partielle pour chaque thread
			for (int offset = blockDim.x/2; offset > 0; offset >>= 1){
				if (threadIdx.x < offset)
					sdata_price[threadIdx.x] += sdata_price[threadIdx.x + offset];

				//On attend que tous les threads aient effectu�s leur somme partielle
				__syncthreads();
			}

			//Le thread 0 charge le r�sultat
			if (threadIdx.x == 0){
				per_block_results_price[blockIdx.x] = sdata_price[0];
			}
		}
}
#endif
