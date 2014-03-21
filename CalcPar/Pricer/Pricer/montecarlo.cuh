#ifndef MONTECARLO_CUH
#define MONTECARLO_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#pragma comment(lib, "cuda.lib") 
#pragma comment(lib, "cudart.lib") 


// INITIALISE LE GENERATEUR DE NOMBRE ALEATOIRE
//int samples: donne le nombre d'itération de Monte-Carlo
//int N: nombre de pas de temps
//int seed: noyau du générateur
//curandState* state: donne un randState à chaque thread
__global__ void init_stuff(
	int samples,
	int N,
	int seed,
	curandState *state)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < samples)
		curand_init(seed, tid, 0, &state[tid]);
}

// GENERATION DU NOMBRE ALEATOIRE
//int samples: donne le nombre d'itération de Monte-Carlo
//int N: nombre de pas de temps
//curandState* state: donne un randState à chaque thread
//float* rand: tableau où est stocké les nombres aléatoires
__global__ void make_rand(
	int samples,
	int N,
	curandState * state,
	float *d_rand)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < samples){
		for (int i = 0; i < N; i++)
			d_rand[tid*N+i] = curand_normal(&state[tid]);
	}
}

// ITERATION DE MONTE_CARLO
//int N: nombre de pas de temps
//float* d_path: ensemble des chemins des iterations de Monte Carlo
//float* per_block_results_price: resultat du payoff pour le calcul du prix
//float* per_block_results_ic: resultat du payoff pour le calcul de l'ic
__global__ void monte_carlo_iteration(
	int N,
	float *d_path,
	float *per_block_results_price)
{
		extern __shared__ float sdata_price[];
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		float payoff;

		const int endPath = (N+1)*(tid+1)-1;
		payoff = max(d_path[endPath] - 100., 0.0);
		sdata_price[threadIdx.x] = payoff;

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