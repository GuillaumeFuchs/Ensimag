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
//double K: strike de l'option
//float* d_coeff: proportion de chaque actif dans l'option
//float* d_path: ensemble des chemins
__device__ float payoff_basket(
	int N,
	int tid,
	int size,
	double K,
	float* d_coeff,
	float* d_path)
{
	double pay = 0.;
	for (int d = 0; d < size; d++)
		pay+=d_path[(d+1)*(N+1)-1+tid*(N+1)*size]*d_coeff[d];
	
	return max(pay - K, 0.0);
}

// ITERATION DE MONTE_CARLO
//int N: nombre de pas de temps
//int size: taille de l'option
//double K: strike de l'option
//float* d_coeff: proportion de chaque actif dans l'option
//float* d_path: ensemble des chemins des iterations de Monte Carlo
//float* per_block_results_price: resultat du payoff pour le calcul du prix
//float* per_block_results_ic: resultat du payoff pour le calcul de l'ic
__global__ void mc_basket(
	int N,
	int size,
	double K,
	float* d_coeff,
	float* d_path,
	float* per_block_results_price)
{
		extern __shared__ float sdata_price[];
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

		//Calcul du payoff
		sdata_price[threadIdx.x] = payoff_basket(N, tid, size, K, d_coeff, d_path);

		//On charge le résultat du payoff dans la mémoire partagé
		//On attend que tous les threads aient calculés le payoff
		__syncthreads();

		//Réduction partielle pour chaque thread
		for (int offset = blockDim.x/2; offset > 0; offset >>= 1){
			if (threadIdx.x < offset)
				sdata_price[threadIdx.x] += sdata_price[threadIdx.x + offset];
			
			//On attend que tous les threads aient effectués leur somme partielle
			__syncthreads();
		}

		//Le thread 0 charge le résultat
		if (threadIdx.x == 0){
			per_block_results_price[blockIdx.x] = sdata_price[0];
		}
}
#endif