#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#pragma comment(lib, "cuda.lib") 
#pragma comment(lib, "cudart.lib") 

// INITIALISE LE GENERATEUR DE NOMBRE ALEATOIRE
//int nGenerator: donne le nombre de générateur
//int seed: noyau du générateur
//curandState* state: donne un randState à chaque thread
__global__ void init_stuff(
	int nGenerator,
	int seed,
	curandState *state)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < nGenerator)
		curand_init(seed, tid, 0, &state[tid]);
}

// GENERATION DU NOMBRE ALEATOIRE
//int nGenerator: donne le nombre de générateur
//curandState* state: donne un randState à chaque thread
//float* rand: tableau où est stocké les nombres aléatoires
__global__ void make_rand(
	int nGenerator,
	curandState * state,
	float *d_rand)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < nGenerator)
		d_rand[tid] = curand_normal(&state[tid]);
}