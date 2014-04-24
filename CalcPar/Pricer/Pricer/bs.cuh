#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#pragma comment(lib, "cuda.lib") 
#pragma comment(lib, "cudart.lib") 


// CALCUL DU CHEMIN DES ACTIFS
//int N: nombre de pas de temps
//int size: nombre d'actif
//int samples: nombre de samples
//float spot: spot des actifs
//float sigma: volatilité des actifs
//float r: taux sans risque
//float dt: pas de temps
//float* d_cho: matrice de Cholesky
//float* d_path: chemin de l'ensemble des actifs
//float* d_rand: tableau contenant les alea
__global__ void asset_compute(
	int N,
	int size,
	int samples,
	float* spot,
	float* sigma,
	float r,
	float dt,
	float* d_cho,
	float* d_path,
	float* d_rand)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int beginPath;
	double rho;
	int indice_d;
	int size_rand = N*size;

	//On contrôle que le thread correspond à un sample
	if (tid < samples){
		//Parcours de tous les sous-jacents
		for (int d = 0; d < size; d++){
			//Recherche du spot de chaque sous-jacent
			//Ainsi que de l'indice du sous-jacent pour la matrice de correlation
			beginPath = d*(N+1)+size*(N+1)*tid;
			d_path[beginPath] = spot[d];
			indice_d = d*(d+1)/2;

			//Calcul de l'ensemble des pas de temps du sous-jacent
			for (int i = 0; i < N; i++){
				rho = 0.;
				//Calcul de la corrélation d'un sous-jacent avec les autres sous-jacent de l'option
				for (int k = 0; k < d+1; k++)
					rho += d_cho[k+indice_d]*
					d_rand[tid*size_rand + i*size + k];

				d_path[beginPath+i+1] = d_path[beginPath+i]*
				exp((r-sigma[d]*sigma[d]/2)*dt + sigma[d]*sqrt(dt)*rho);
			}
		}
	}
}
