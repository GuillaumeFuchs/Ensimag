# include "option.h"
# include "barrier_u.h"
# include <pnl/pnl_mathtools.h>

#include "barrier_u.cuh"

/*!
 * \file barrier_u.cpp
 * \brief Option barrière haute
 * \author Equipe 11
 */

Barrier_u :: Barrier_u() : Option() {
  Strike_ = 0.;
  Coeff_ = pnl_vect_new();
  Bu_ = pnl_vect_new();
}

Barrier_u ::Barrier_u(Parser& pars): Option(pars){
  Strike_ = pars.getDouble("strike");
  Coeff_ = pnl_vect_copy(pars.getVect("payoff coefficients"));
  Bu_ = pnl_vect_copy(pars.getVect("upper barrier"));

  Coeff_gpu = (float*)malloc(size_*sizeof(float));
  Bu_gpu = (float*)malloc(size_*sizeof(float));

  for (int i = 0; i < size_; i++){
	  Coeff_gpu[i] = GET(Coeff_, i);
	  Bu_gpu[i] = GET(Bu_, i);
  }
}

Barrier_u :: ~Barrier_u(){
  pnl_vect_free(&Coeff_);
  pnl_vect_free(&Bu_);
}
double Barrier_u :: get_Strike(){
  return Strike_;
}

PnlVect* Barrier_u :: get_Coeff(){
  return Coeff_;
}

PnlVect* Barrier_u :: get_Bu(){
  return Bu_;
}

void Barrier_u :: set_Strike(double Strike) {
  Strike_ = Strike;
}

void Barrier_u :: set_Coeff(PnlVect *Coeff) {
  Coeff_ = Coeff;
}

void Barrier_u :: set_Bu(PnlVect *Bu) {
  Bu_ = Bu;
}

double Barrier_u :: payoff (const PnlMat *path) {
  double sum ;
  //Vecteur utilisé pour effectuer la somme de chaque actif à maturité
  PnlVect* final = pnl_vect_create(size_);

  //On met dans final la dernière colonne de Path correspond à la valeur à maturité des sous-jacents
  pnl_mat_get_col(final, path, TimeSteps_);
  sum = pnl_vect_scalar_prod(final, Coeff_) - Strike_;
  //On vérifie que toutes les valeurs des sous-jacents soient en dessous de la barrière
  //Si on en trouve une alors le prix de l'option est de 0
  for (int i=0; i<TimeSteps_+1; i++){
	for (int d=0; d<size_; d++){
	  if (pnl_mat_get(path,d,i) > pnl_vect_get(Bu_,d)){
		pnl_vect_free(&final);
		return 0;
	  }
	}
  }
  return MAX(sum, 0);
}

void Barrier_u::priceMC(
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
	float* d_coeff;
	cudaMalloc((float**)&d_coeff, size_*sizeof(float));
	cudaMemcpy(d_coeff, Coeff_gpu, size_*sizeof(float), cudaMemcpyHostToDevice);
	float* d_bu;
	cudaMalloc((float**)&d_bu, size_*sizeof(float));
	cudaMemcpy(d_bu, Bu_gpu, size_*sizeof(float), cudaMemcpyHostToDevice);

	mc_barrier_u<<<dimGrid, dimBlock, 2*(dimBlock.x)*sizeof(float)>>>(N, size_, samples, (float)Strike_, d_coeff, d_bu, d_path, d_per_block_results_price, d_per_block_results_ic);
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
