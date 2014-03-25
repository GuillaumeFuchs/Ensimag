# include "option.h"
# include "basket.h"
# include <pnl/pnl_mathtools.h>

#include "basket.cuh"

/*!
* \file basket.cpp
* \brief Option panier
*/

Basket :: Basket() : Option() {
	Strike_ = 0;
	Coeff_ = pnl_vect_new();
}

Basket :: Basket(Parser& pars):Option(pars){
	Strike_ = pars.getDouble("strike");
	Coeff_ = pnl_vect_copy(pars.getVect("payoff coefficients"));

	Coeff_gpu = (float*)malloc(size_*sizeof(float));

	for (int i = 0; i < size_; i++)
		Coeff_gpu[i] = GET(Coeff_, i);
}

Basket :: ~Basket(){
	pnl_vect_free(&Coeff_);
}

double Basket :: get_Strike() {
	return Strike_;
}

PnlVect * Basket :: get_Coeff() {
	return Coeff_;
}

void Basket :: set_Strike(double Strike) {
	Strike_ = Strike;
}

void Basket :: set_Coeff(PnlVect *Coeff) {
	Coeff_ = Coeff;
}

double Basket :: payoff (const PnlMat *path) {
	double sum;
	PnlVect* final = pnl_vect_create(path->m);

	//On met dans final la dernière colonne de Path correspond à la valeur à maturité des sous-jacents.
	pnl_mat_get_col(final, path, (path->n-1));
	sum = pnl_vect_scalar_prod(final, Coeff_) - Strike_;
	pnl_vect_free(&final);
	//On retourne le max entre le résultat de la somme et 0
	return MAX(sum, 0);
}

void Basket::price_mc(
	double &prix,
	int nBlocks,
	int nThreads,
	int N,
	int samples,
	float* d_path) 
{
	//Compute price
	float* d_per_block_results_price;
	cudaMalloc((float**)&d_per_block_results_price, nBlocks*sizeof(float));
	float* d_coeff;
	cudaMalloc((float**)&d_coeff, size_*sizeof(float));
	cudaMemcpy(d_coeff, Coeff_gpu, size_*sizeof(float), cudaMemcpyHostToDevice);

	mc_basket<<<nBlocks, nThreads, nBlocks*sizeof(float)>>>(N, size_, samples, (float)Strike_, d_coeff, d_path, d_per_block_results_price);
	cudaThreadSynchronize();

	float* per_block_results_price = (float*)malloc(nBlocks*sizeof(float));
	cudaMemcpy(per_block_results_price, d_per_block_results_price, nBlocks*sizeof(float), cudaMemcpyDeviceToHost);

	prix = 0.0;
	for (int i = 0; i < nBlocks; i++){
		prix += per_block_results_price[i];
		
	}
	cudaFree(d_per_block_results_price);
}
