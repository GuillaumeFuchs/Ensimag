# include "option.h"
# include "asian.h"
# include <pnl/pnl_mathtools.h>
#include <pnl/pnl_vector.h>

#include "asian.cuh"

/*!
 * \file asian.cpp
 * \brief Implémentation de la classe fille d'Option: Asian 
 */

Asian :: Asian() : Option() {
  Strike_ = 0.;
}

Asian::Asian(Parser &pars) : Option(pars){
  Strike_ = pars.getDouble("strike");
}

Asian :: ~Asian(){
}

double Asian :: get_Strike() {
  return Strike_;
}

void Asian :: set_Strike(double Strike){
  Strike_=Strike;
}

double Asian :: payoff (const PnlMat *path) {
  double sum;
  //Vecteur pour mette les valeurs des S_{ti}
  //Dimension D=1 donc path ne contient qu'une seule ligne (indice 0)
  PnlVect* final = pnl_vect_create(TimeSteps_+1);
  pnl_mat_get_row(final ,path, 0);
  //Calcul d'une option asiatique discrète
  sum = (1/(double)(TimeSteps_))*pnl_vect_sum(final) - Strike_;
  pnl_vect_free(&final);
  return MAX(sum, 0);
}

void Asian::price_mc(
	dim3 dimGrid,
	dim3 dimBlock,
	double &prix,
	double &ic,
	int N,
	int samples,
	float* d_path) 
{
	//Compute price
	float* d_per_block_results_price;
	cudaMalloc((float**)&d_per_block_results_price, (dimGrid.x)*sizeof(float));
	float* d_per_block_results_ic;
	cudaMalloc((float**)&d_per_block_results_ic, (dimGrid.x)*sizeof(float));

	int nUseThreads = samples/10;
	mc_asian<<<dimGrid, dimBlock, 2*(dimBlock.x)*sizeof(float)>>>(N, size_, nUseThreads, (float)Strike_, d_path, d_per_block_results_price, d_per_block_results_ic);
	cudaThreadSynchronize();

	float* per_block_results_price = (float*)malloc((dimGrid.x)*sizeof(float));
	cudaMemcpy(per_block_results_price, d_per_block_results_price, (dimGrid.x)*sizeof(float), cudaMemcpyDeviceToHost);
	float* per_block_results_ic = (float*)malloc((dimGrid.x)*sizeof(float));
	cudaMemcpy(per_block_results_ic, d_per_block_results_ic, (dimGrid.x)*sizeof(float), cudaMemcpyDeviceToHost);

	prix = 0.;
	ic = 0.;
	for (int i = 0; i < dimGrid.x; i++){
		prix += per_block_results_price[i];
		ic += per_block_results_ic[i];
	}
	cudaFree(d_per_block_results_price);
	cudaFree(d_per_block_results_ic);
}
