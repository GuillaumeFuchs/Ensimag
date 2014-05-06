/**
*  @file	montecarlo.cpp
*  @brief	Implémentation de la classe Monte Carlo
*
*  \author 
*  Equipe 11 - Julia DITISHEIM, Guillaume FUCHS, Barthelemy LONGUET DE LA GIRAUDIERE, Mailys RENEAUME, Alexis SCIAU
*
*/
#include "montecarlo.h"
#include <math.h>
#include <pnl/pnl_vector.h>
#include <pnl/pnl_matrix.h>
#include <pnl/pnl_mathtools.h>
#include <cstdio>
#include <ctime>

#include "montecarlo.cuh"

#define N_THREADS 512;

using namespace std;

MonteCarlo::MonteCarlo(PnlRng * rng){
	(*this).rng = rng;
	h_ = 0.0;
	samples_ = 0;
}

MonteCarlo::MonteCarlo(BS *mod_, Option *opt_, PnlRng *rng, double h_, int samples_){
	(*this).mod_ = mod_;
	(*this).opt_ = opt_;
	(*this).rng = rng;
	(*this).h_ = h_;
	(*this).samples_ = samples_;
}

/**
* Accesseurs
*
*/
MonteCarlo::~MonteCarlo(){
}

BS* MonteCarlo::get_mod(){
	return mod_;
}

Option* MonteCarlo::get_opt(){
	return opt_;
}

PnlRng* MonteCarlo::get_rng(){
	return rng;
}

double MonteCarlo::get_h(){
	return h_;
}

int MonteCarlo::get_samples(){
	return samples_;
}

/**
* Mutateurs
*
*/
void MonteCarlo::set_mod(BS* mod){
	mod_ = mod;
}

void MonteCarlo::set_opt(Option* opt){
	opt_ = opt;
}

void MonteCarlo::set_rng(PnlRng* rng){
	(*this).rng = rng;
}

void MonteCarlo::set_h(double h){
	h_ = h;
}

void MonteCarlo::set_samples(int samples){
	samples_ = samples;
}

/**
* Price
*
*/
void MonteCarlo::priceCPU(
	double &prix_cpu, 
	double &ic_cpu, 
	double &time_cpu)
{
	//Récupération des paramètres de l'option
	int size = opt_->get_size();
	int N = opt_->get_timesteps();
	double r = mod_->get_r();
	double T = opt_->get_T();

	double dt = T/N; //incrémentation pour chaque date de constation
	double payoff;	//valeur du payoff de l'option

	PnlMat *path = pnl_mat_create(size, N+1); //matrice de dimension d x (N+1) pour stocker le chemin de l'option
	PnlMat *G = pnl_mat_create(N, size); //matrice de dimension N*d pour générer une suite iid selon la loi normale centrée réduite
	PnlVect *grid = pnl_vect_create(N+1); //vecteur de taille N pour générer la grille de temps (t_0=0, ..., t_N)
	PnlVect *tirages = pnl_vect_create(samples_); //vecteur de taille samples_ contenant les valeurs des M payoff
	clock_t tbegin, tend; //variables pour calculer le temps d'exécution du pricer

	prix_cpu = 0.;
	ic_cpu = 0.;

	//Calcul de chaque date de constatation;
	for (int t=0; t<N+1; t++)
		pnl_vect_set(grid, t, dt*t);

	//Ajout du prix spot dans la première colonne de path
	pnl_mat_set_col(path, mod_->get_spot(), 0);

	tbegin = clock();
	for (int j=0; j<samples_; j++){
		//Génération de la trajectoire du modèle de Black Scholes
		mod_->asset(path, T, N, rng, G, grid);

		//Calcul du payoff et stockage dans le prix et la variance
		payoff = opt_->payoff(path);
		prix_cpu += payoff;
		ic_cpu += payoff*payoff;
	}
	tend = clock();

	//Calcul du prix à l'aide de la formule de MC
	prix_cpu = exp(-r*T)*(1/(double)samples_)*prix_cpu;
	time_cpu = (double)(tend-tbegin)/CLOCKS_PER_SEC;

	//Calcul de la variance de l'estimateur pour avoir l'intervalle de confiance
	ic_cpu = exp(-2*r*T) * ic_cpu/(double)samples_ - prix_cpu * prix_cpu;
	ic_cpu = 1.96*sqrt(ic_cpu/(double)samples_);

	pnl_vect_free(&grid);
	pnl_vect_free(&tirages);
	pnl_mat_free(&path);
	pnl_mat_free(&G);
}

void MonteCarlo::priceGPU(
	double &prix_gpu, 
	double &ic_gpu, 
	double &time_gpu, 
	const double ic_target)
{
	//
	//Récupération des paramètres de l'option
	//
	int size = opt_->get_size();
	int N = opt_->get_timesteps();
	double r = mod_->get_r();
	double T = opt_->get_T();

	clock_t tbegin, tend; //variables pour le calcul du temps d'exécution du pricer
	int samples = 0; //compte le nombre de samples total calculé
	double limit = (ic_target/1.96)*(ic_target/1.96)*exp(2*r*T); //limit à dépasser pour obtenir l'interval de confiance demandé
	int optimalSamples = 10240/(N*size); //nombre de samples pour chaque tour de boucle
	int nBreaks = ceil((double)samples_/(double)optimalSamples); //nombre de division du samples total

	//si le nombre de samples exécuté est différent du nombre de samples demandé par l'utilisateur alors on le signale et on change le nombre demandé
	if (!(fabs(ic_target) > 0.00001) && (samples_ != nBreaks * optimalSamples)){
		printf("ATTENTION: nombre d'iterations modifie. %d -> %d\n", samples_, nBreaks*optimalSamples);
		set_samples(nBreaks*optimalSamples);
	}

	//
	//Propriétés de la grille et des blocs
	//
	int nThreads = N_THREADS;
	int nBlocks_rand = ceil(optimalSamples * N * size / (double)nThreads);
	int nBlocks = ceil(optimalSamples / (double)nThreads);
	int nAll = optimalSamples * N * size;
	dim3 dimGrid_rand(nBlocks_rand, 1, 1);
	dim3 dimGrid(nBlocks, 1, 1);
	dim3 dimBlock(nThreads, 1, 1);

	//
	//Variables pour la génération aléatoire
	//
	curandState *d_state;
	float *d_rand;

	//
	//Variables de l'option
	//
	float *d_path;
	double sum_prix_gpu = 0.;
	double sum_ic_gpu = 0.;

	//Allocations mémoires
	cudaMalloc(&d_state, nAll*sizeof(curandState));
	cudaMalloc((float**)&d_rand, nAll*sizeof(float));
	cudaMalloc((float**)&d_path, optimalSamples *(N+1)*size*sizeof(float));

	tbegin = clock();
	//Initialisation des générateurs de nombre aléatoire
	init_stuff<<<dimGrid_rand, dimBlock>>>(nAll, time(NULL), d_state);
	cudaThreadSynchronize();

	for (int k = 0; k < nBreaks; k++){
		//Génération des nombre aléatoires
		make_rand<<<dimGrid_rand, dimBlock>>>(nAll, d_state, d_rand);
		cudaThreadSynchronize();
		//Calcul du chemin des sous-jacents
		mod_->assetGPU(dimGrid, dimBlock, optimalSamples, N, T, d_path, d_rand);

		//Calcul du payoff et de l'ic
		opt_->priceMC(dimGrid, dimBlock, prix_gpu, ic_gpu, N, optimalSamples, d_path);

		//Stockage du payoff et de l'ic
		sum_prix_gpu += prix_gpu;
		sum_ic_gpu += ic_gpu;
		samples += optimalSamples;

		//Si un intervalle de confiance minimal est demandé alors on calcul l'intervalle de confiance actuelle et si celui si est en dessous de la limite alors on stop l'itération
		if (fabs(ic_target) > 0.00001){
			if ( (sum_ic_gpu/(double)samples - (sum_prix_gpu/(double)samples)*(sum_prix_gpu/(double)samples))/(double)samples < limit){
				break;
			}else{
				nBreaks++;
			}
		}
	}
	tend = clock();

	//Calcul du prix et de l'intervalle de confiance de l'option
	prix_gpu = exp(-r*T)/(double)samples * sum_prix_gpu;
	ic_gpu = exp(-2*r*T)*sum_ic_gpu/(double)samples - prix_gpu * prix_gpu;
	ic_gpu = 1.96 *sqrt(ic_gpu/(double)samples);

	set_samples(samples);

	//Libération mémoire
	cudaFree(d_state);
	cudaFree(d_rand);
	cudaFree(d_path);
	time_gpu = (double)(tend-tbegin)/CLOCKS_PER_SEC;
}
