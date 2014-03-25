#ifndef BASKET_CUH
#define BASKET_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#pragma comment(lib, "cuda.lib") 
#pragma comment(lib, "cudart.lib") 

// PAYOFF D'UNE OPTION BASKET
//int N: nombre de pas de temps
//int tid: numero thread dans la grille
//int size: taille de l'option
//float K: strike de l'option
//float* d_coeff: proportion de chaque actif dans l'option
//float* d_path: ensemble des chemins
__device__ float payoff_basket(
	int N,
	int tid,
	int size,
	float K,
	float* d_coeff,
	float* d_path)
{
	float pay = 0.;
	for (int d = 0; d < size; d++)
		pay+=d_path[(d+1)*(N+1)-1+tid*(N+1)*size]*d_coeff[d];
	
	return max(pay - K, 0.0);
}

// ITERATION DE MONTE_CARLO
//int N: nombre de pas de temps
//int size: taille de l'option
//int nUseThreads: nb de chemin calculés par option
//float K: strike de l'option
//float* d_coeff: proportion de chaque actif dans l'option
//float* d_path: ensemble des chemins des iterations de Monte Carlo
//float* per_block_results_price: resultat du payoff pour le calcul du prix
//float* per_block_results_ic: resultat du payoff pour le calcul de l'ic
__global__ void mc_basket(
	int N,
	int size,
	int nUseThreads,
	float K,
	float* d_coeff,
	float* d_path,
	float* per_block_results_price,
	float* per_block_results_ic)
{
		extern __shared__ float s_data[];
		float *s_data_price = s_data;
		float *s_data_ic = &s_data[blockDim.x];
		
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (tid < nUseThreads){
			//Calcul du payoff
			float payoff = payoff_basket(N, tid, size, K, d_coeff, d_path);
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
