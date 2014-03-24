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

#define N_THREADS 16;

using namespace std;
BS::BS(){
	size_ = 0;
	r_ = 0.0;
	rho_ = 0.0;
	sigma_ = pnl_vect_new();
	spot_ = pnl_vect_new();
	trend_ = pnl_vect_new();
	Cho_ = pnl_mat_new();
	Gi_ = pnl_vect_new();
	Ld_ = pnl_vect_new();
	
	r_gpu = 0.;
	rho_gpu = 0.;
}

BS::BS(Parser &pars){
	(*this).size_ = pars.getInt("option size");
	(*this).r_ = pars.getDouble("interest rate");
	(*this).rho_ = pars.getDouble("correlation");
	(*this).sigma_ = pnl_vect_copy(pars.getVect("volatility"));
	(*this).spot_ = pnl_vect_copy(pars.getVect("spot"));
	(*this).trend_ = pars.getVect("trend");
	(*this).Cho_ = pnl_mat_create_from_double(size_, size_, rho_);
	for (int j=0; j<size_; j++){
		pnl_mat_set_diag(Cho_, 1, j);
	}
	pnl_mat_chol(Cho_);

	Gi_ = pnl_vect_create(size_);
	Ld_ = pnl_vect_create(size_);

	r_gpu = (float)r_;
	rho_gpu = (float)rho_;
	sigma_gpu = (float*)malloc(size_*sizeof(float));
	spot_gpu = (float*)malloc(size_*sizeof(float));
	trend_gpu = (float*)malloc(size_*sizeof(float));
	Cho_gpu = (float*)malloc(size_*(size_+1)/2*sizeof(float));

	for (int i = 0; i < size_; i++){
		sigma_gpu[i] = GET(sigma_, i);
		spot_gpu[i] = GET(spot_, i);
		trend_gpu[i] = GET(trend_, i);
	}
	int k = 0;
	for (int i = 0; i < size_; i++){
		for (int j = 0; j < i+1; j++){
			Cho_gpu[k] = MGET(Cho_, i, j);
			k++;
		}
	}
}

BS::~BS(){
	pnl_vect_free(&sigma_);
	pnl_vect_free(&spot_);
	pnl_vect_free(&trend_);
	pnl_vect_free(&Gi_);
	pnl_vect_free(&Ld_);
	pnl_mat_free(&Cho_);
}

int BS::get_size(){
	return size_;
}

double BS::get_r(){
	return r_;
}

double BS::get_rho(){
	return rho_;
}

PnlVect * BS::get_sigma(){
	return sigma_;
}

PnlVect * BS::get_spot(){
	return spot_;
}

PnlVect * BS::get_trend(){
	return trend_;
}

PnlMat * BS::get_cho(){
	return Cho_;
}

PnlVect * BS::get_gi(){
	return Gi_;
}

PnlVect * BS::get_ld(){
	return Ld_;
}

void BS::set_size(int size){
	size_ = size;
}

void BS::set_r(double r){
	r_ = r;
}

void BS::set_rho(double rho){
	rho_ = rho;
}

void BS::set_sigma(PnlVect *sigma){
	sigma_ = sigma;
}

void BS::set_spot(PnlVect *spot){
	spot_ = spot;
}

void BS::set_trend(PnlVect *trend){
	trend_ = trend;
}

void BS::set_cho(PnlMat *Cho){
	Cho_ = pnl_mat_copy(Cho);
}

void BS::set_gi(PnlVect *Gi){
	Gi_ = Gi;
}

void BS::set_ld(PnlVect *Ld){
	Ld_ = Ld;
}

void BS::asset(PnlMat *path, double T, int N, PnlRng *rng, PnlMat* G, PnlVect* grid){
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

float* BS::assetGPU(
	int &nBlocks,
	int &nThreads,
	int samples,
	int N,
	double T)
{

	//Property grid & blocks
	nThreads = N_THREADS;
	nBlocks = ceil((double)samples/(double)nThreads);
	dim3 dimGrid(nBlocks, 1, 1);
	dim3 dimBlock(nThreads, 1, 1);

	//Génération de nombre aléatoire
	curandState *d_state;
	float *d_rand;

	cudaMalloc((float**)&d_rand, samples*N*size_*sizeof(float));
	cudaMalloc(&d_state, samples*sizeof(curandState));

	init_stuff<<<nBlocks, nThreads>>>(samples, time(NULL), d_state);
	cudaThreadSynchronize();
	make_rand<<<nBlocks, nThreads>>>(samples, N, size_, d_state, d_rand);
	cudaThreadSynchronize();

	float *rand = (float*)malloc(samples*N*size_*sizeof(float));
	cudaMemcpy(rand, d_rand, N*samples*size_*sizeof(float), cudaMemcpyDeviceToHost);
	
	//printf("RAND:\n");
	//for (int m = 0; m < samples; m++){
	//for (int i = 0; i < N; i++){
	//for (int d = 0; d < size_; d++)
	//printf("%f ", rand[d+i*size_+(N*size_)*m]);
	//printf("\n");
	//}
	//printf("\n");
	//}

	//Compute asset
	float *d_cho;
	float *d_path;
	float *d_spot;
	float *d_sigma;

	cudaMalloc((float**)&d_cho , size_*(size_+1)/2*sizeof(float));
	cudaMemcpy(d_cho, Cho_gpu, size_*(size_+1)/2*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((float**)&d_path, samples*(N+1)*size_*sizeof(float));
	cudaMalloc((float**)&d_spot, size_*sizeof(float));
	cudaMemcpy(d_spot, spot_gpu, size_*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((float**)&d_sigma, size_*sizeof(float));
	cudaMemcpy(d_sigma, sigma_gpu, size_*sizeof(float), cudaMemcpyHostToDevice);

	double dt = T/(double)N;
	asset_compute<<<nBlocks, nThreads>>>(N, size_, samples, d_spot, d_sigma, r_, dt, d_cho, d_path, d_rand);
	cudaThreadSynchronize();
	
	//printf("PATH:\n");
	//float *path = (float*)malloc(samples*(N+1)*size_*sizeof(float));
	//cudaMemcpy(path, d_path, samples*(N+1)*size_*sizeof(float), cudaMemcpyDeviceToHost);
	//for (int m = 0; m < samples; m++){
	//for (int d = 0; d < size_; d++){
	//for (int i = 0; i < N+1; i++)
	//printf("%f ", path[i+d*(N+1)+size_*(N+1)*m]);
	//printf("\n");
	//}
	//printf("\n");
	//}

	cudaFree(d_rand);
	cudaFree(d_state);
	return d_path;
}


void BS::asset(PnlMat *path, double t, int N, double T, PnlRng *rng, const PnlMat *past, int taille, PnlMat* G, PnlVect* grid){
	//s: double pour la valeur du sous-jacent à la date t_{i+1}
	double s;
	//diff: double t_{i+1}-t_{i}
	double diff;

	//Simulation de la suite (G_i)i>=1 iid de vecteurs gaussiens centés de matrice de covariance identité
	pnl_mat_rng_normal(G, N-taille, size_, rng);
	//Calcul de l'évolution du sous-jacent de chaque actif pour t=t_{i+1} à t=t_N;
	for (int d=0; d<size_; d++){
		for (int i=0; i<N-taille; i++){
			//Sélection de la ligne de la matrice de Cholesky associé à l'actif sur lequel on travaille
			pnl_mat_get_row(Ld_, Cho_, d);
			//Sélection de la ligne de la matrice de vecteurs gaussiens associé au temps sur lequel on travaille
			pnl_mat_get_row(Gi_, G, i);
			//Calcul de la différence de pas
			diff = pnl_vect_get(grid, i+1)-pnl_vect_get(grid,i);
			//Si calcul du temps après t
			//alors on selectionne la valeur du sous-jacent à t dans la matrice past
			//sinon on sélectionne à la valeur du sous-jacent à t_{i} dans path
			if (i==0)
				s = pnl_mat_get(past, d, past->n-1);
			else
				s = pnl_mat_get(path, d, i+taille);
			//Calcul de l'évolution du sous-jacent à l'aide de la formule du modèle de BS
			s = s*exp((r_-pow(pnl_vect_get(sigma_, d),2.0)/2)*diff + pnl_vect_get(sigma_, d) * sqrt(diff) * pnl_vect_scalar_prod(Ld_, Gi_));
			//Ajout du résultat dans path
			pnl_mat_set(path, d, i+taille+1, s);
		}
	}
}


void BS:: shift_asset (PnlMat *_shift_path, const PnlMat *path,
	int d, double h, double t, double timestep){
		pnl_mat_clone(_shift_path, path);
		for (int i=0; i<timestep+1; i++){
			if (i>t){
				pnl_mat_set(_shift_path, d,i, (1+h)*pnl_mat_get(path, d,i));
			}
		}
}

void BS:: simul_market (PnlMat* past, int H, double T, PnlRng *rng){
	//Temps: incrémentation pour chaque date de constation
	double temps = T/H;
	//s: valeur du sous-jacent à la date t_{i+1}
	double s;
	//diff: calcul de la différence de temps entre t_{i+1} et t_{i}
	double diff;
	//G: matrice de dimension H*d pour générer une suite iid selon la loi normale centrée réduite
	PnlMat *G = pnl_mat_create(H, size_);
	//Grid: vecteur de taille H+1 pour générer la grille de temps (t_0=0, ..., t_N)
	PnlVect *grid = pnl_vect_create(H+1);

	//Calcul des dates de constatation;
	for (int t=0; t<H+1; t++){
		pnl_vect_set(grid, t, temps*t);
	}
	//Ajout de la trajectoire du modèle dans past
	//Ajout du prix spot dans la première colonne de path
	pnl_mat_set_col(past, spot_, 0);
	//Simulation de la suite (G_i)i>=1 iid de vecteurs gaussiens centés de matrice de covariance identité
	pnl_mat_rng_normal(G, H, size_, rng);
	//Calcul de l'évolution du sous-jacent de chaque actif pour t=tau_1 à t=tau_H;
	//Idem à asset pour t>0 mais on utilise le taux d'intérêt sous la probabilité historique
	//et on ajoute les résultats dans la matrice past
	for (int d=0; d<size_; d++){
		for (int i=0; i<H; i++){
			pnl_mat_get_row(Ld_, Cho_, d);
			pnl_mat_get_row(Gi_, G, i);
			diff = pnl_vect_get(grid, i+1)-pnl_vect_get(grid, i);
			s = pnl_mat_get(past, d, i)*
				exp((pnl_vect_get(trend_, d)-pow(pnl_vect_get(sigma_, d),2.0)/2)*diff +
				pnl_vect_get(sigma_, d)*sqrt(diff)*pnl_vect_scalar_prod(Ld_, Gi_));
			pnl_mat_set(past, d, i+1, s);
		}
	}

	pnl_mat_free(&G);
	pnl_vect_free(&grid);
}
