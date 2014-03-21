#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#pragma comment(lib, "cuda.lib") 
#pragma comment(lib, "cudart.lib") 

// CALCUL DU CHEMIN DES ACTIFS
//int N: nombre de pas de temps
//int samples: donne le nombre d'itération de Monte-Carlo
//float spot: spot des actifs
//float sigma: volatilité des actifs
//float r: taux sans risque
//float dt: pas de temps
//float* d_cho: matrice de Cholesky
//float* d_path: chemin de l'ensemble des actifs
//float* d_rand: tableau contenant les alea
__global__ void asset_compute(
	int N,
	int samples,
	float spot,
	float sigma,
	float r,
	float dt,
	float* cho,
	float* d_path,
	float* d_rand)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < samples){
		const int beginPath = (N+1)*tid;
		d_path[beginPath] = spot;
		for (int i = 0; i < N; i++)
			d_path[beginPath+i+1] = d_path[beginPath+i]*exp((r-sigma*sigma/2)*dt + sigma*sqrt(dt)*d_rand[tid*N+i]);
	}
}
