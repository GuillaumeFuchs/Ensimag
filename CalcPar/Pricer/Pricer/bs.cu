/*!
*  \file	bs.cpp
*  \brief	Implémentation de la classe Black-SCholes
*  \author Equipe 11
*/

#include "bs.h"
#include <math.h>
#include <cstdio>
#include "montecarlo.h"
#include "bs.cuh"

using namespace std;
BS::BS(){
	size_ = 0;
	r_ = 0.0;
	rho_ = 0.0;
	sigma_ = pnl_vect_new();
	spot_ = pnl_vect_new();
	Cho_ = pnl_mat_new();
	Gi_ = pnl_vect_new();
	Ld_ = pnl_vect_new();
}

BS::BS(Parser &pars)
{
	(*this).size_ = pars.getInt("option size");
	(*this).r_ = pars.getDouble("interest rate");
	(*this).rho_ = pars.getDouble("correlation");
	(*this).sigma_ = pnl_vect_copy(pars.getVect("volatility"));
	(*this).spot_ = pnl_vect_copy(pars.getVect("spot"));
	(*this).Cho_ = pnl_mat_create_from_double(size_, size_, rho_);
	for (int j=0; j<size_; j++){
		pnl_mat_set_diag(Cho_, 1, j);
	}
	pnl_mat_chol(Cho_);

	Gi_ = pnl_vect_create(size_);
	Ld_ = pnl_vect_create(size_);

	float* sigma_gpu = (float*)malloc(size_*sizeof(float));
	float* spot_gpu = (float*)malloc(size_*sizeof(float));
	float* Cho_gpu = (float*)malloc(size_*(size_+1)/2*sizeof(float));

	for (int i = 0; i < size_; i++){
		sigma_gpu[i] = GET(sigma_, i);
		spot_gpu[i] = GET(spot_, i);
	}
	int k = 0;
	for (int i = 0; i < size_; i++){
		for (int j = 0; j < i+1; j++){
			Cho_gpu[k] = MGET(Cho_, i, j);
			k++;
		}
	}
	cudaMalloc((float**)&d_cho , size_*(size_+1)/2*sizeof(float));
	cudaMemcpy(d_cho, Cho_gpu, size_*(size_+1)/2*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((float**)&d_spot, size_*sizeof(float));
	cudaMemcpy(d_spot, spot_gpu, size_*sizeof(float), cudaMemcpyHostToDevice);
		
	cudaMalloc((float**)&d_sigma, size_*sizeof(float));
	cudaMemcpy(d_sigma, sigma_gpu, size_*sizeof(float), cudaMemcpyHostToDevice);
}

BS::~BS()
{
	pnl_vect_free(&sigma_);
	pnl_vect_free(&spot_);
	pnl_vect_free(&Gi_);
	pnl_vect_free(&Ld_);
	pnl_mat_free(&Cho_);

	cudaFree(d_cho);
	cudaFree(d_spot);
	cudaFree(d_sigma);
}

int BS::get_size(){	return size_; }
double BS::get_r(){ return r_; }
double BS::get_rho(){ return rho_; }
PnlVect * BS::get_sigma(){ return sigma_; }
PnlVect * BS::get_spot(){ return spot_; }
PnlMat * BS::get_cho(){	return Cho_; }
PnlVect * BS::get_gi(){	return Gi_; }
PnlVect * BS::get_ld(){ return Ld_; }
void BS::set_size(int size){ size_ = size; }
void BS::set_r(double r){ r_ = r; }
void BS::set_rho(double rho){ rho_ = rho; }
void BS::set_sigma(PnlVect *sigma){ sigma_ = sigma; }
void BS::set_spot(PnlVect *spot){ spot_ = spot; }
void BS::set_cho(PnlMat *Cho){ Cho_ = pnl_mat_copy(Cho); }
void BS::set_gi(PnlVect *Gi){ Gi_ = Gi; }
void BS::set_ld(PnlVect *Ld){ Ld_ = Ld; }


void BS::asset(
	PnlMat *path, 
	double T, 
	int N, 
	PnlRng *rng, 
	PnlMat* G, 
	PnlVect* grid)
{
	//s: double pour la valeur du sous-jacent à la date t_{i+1}
	double s;
	//diff: double t_{i+1}-t_{i}
	double diff;

	//Simulation de la suite (G_i)i>=1 iid de vecteurs gaussiens centrés de matrice de covariance identité
	pnl_mat_rng_normal(G, N, size_, rng);

	//Calcul de l'évolution du sous-jacent de chaque actif pour t=t_1 à t=t_N;
	for (int d=0; d<size_; d++){
		for (int i=0; i<N; i++){
			//Sélection de la ligne de la matrice de Cholesky associé à l'actif sur lequel on travaille
			pnl_mat_get_row(Ld_, Cho_, d);
			//Sélection de la ligne de la matrice de vecteurs gaussiens associé au temps sur lequel on travaille
			pnl_mat_get_row(Gi_, G, i);
			//Calcul de la différence de pas de temps
			diff = pnl_vect_get(grid, i+1)-pnl_vect_get(grid, i);
			//Calcul de l'évolution du sous-jacent à l'aide de la formule du modèle de BS
			s = pnl_mat_get(path, d, i)*
				exp((r_-pow(pnl_vect_get(sigma_, d),2.0)/2)*diff +
				pnl_vect_get(sigma_, d)*sqrt(diff)*pnl_vect_scalar_prod(Ld_, Gi_));
			//Ajout du résultat dans path
			pnl_mat_set(path, d, i+1, s);
		}
	}
}

void BS::assetGPU(
	dim3 dimGrid,
	dim3 dimBlock,
	int samples,
	int N,
	float T,
	float* d_path,
	float* d_rand)
{
		//Calcul du chemin des sous-jacents de l'option
		float dt = T/(float)N;

		asset_compute<<<dimGrid, dimBlock>>>(N, size_, samples, d_spot, d_sigma, (float)r_, dt, d_cho, d_path, d_rand);
		cudaThreadSynchronize();

		//TEST PATH
		//printf("PATH:\n");
		//float *path = (float*)malloc(samples*(N+1)*size_*sizeof(float));
		//cudaMemcpy(path, d_path, samples*(N+1)*size_*sizeof(float), cudaMemcpyDeviceToHost);
		//for (int m = 0; m < samples; m++){
		//for (int d = 0; d < size_; d++){
		//for (int i = 0; i < N+1; i++)
		//printf("%d: %f ", i+d*(N+1)+size_*(N+1)*m, path[i+d*(N+1)+size_*(N+1)*m]);
		//printf("\n");
		//}
		//printf("\n");
		//}
}