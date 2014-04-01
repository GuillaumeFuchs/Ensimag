# include "option.h"
# include "performance.h"
# include <pnl/pnl_mathtools.h>

#include "performance.cuh"
/*!
 * \file performance.cpp
 * \brief option performance 
 */

Performance :: Performance() : Option() {
  Coeff_ = pnl_vect_new();
}

Performance :: Performance(Parser & pars) : Option(pars){
  Coeff_ = pnl_vect_copy(pars.getVect("payoff coefficients"));

  Coeff_gpu = (float*)malloc(size_*sizeof(float));

  for (int i = 0; i < size_; i++)
	  Coeff_gpu[i] = GET(Coeff_, i);
}
 
Performance :: ~Performance(){
}

PnlVect* Performance :: get_Coeff(){
  return Coeff_;
}

void Performance :: set_Coeff(PnlVect *Coeff) {
  Coeff_ = Coeff;
}

double Performance :: payoff (const PnlMat *path) {
  double sum = 0.0;
  double temp_num;
  double temp_deno;

  //Numerateur: vecteur contenant la somme des d actifs au temps t_i
  //Denominateur: vecteur contenant la somme des d actifs au temps t_{i-1}
  PnlVect* numerateur = pnl_vect_create(size_);
  PnlVect* denominateur = pnl_vect_create(size_);

  for (int i=1; i<TimeSteps_+1; i++){
	  //On met les d actif au temps t_i dans numerateur
	  //et ceux au temps t_{i-1} dans denominateur
	  pnl_mat_get_col(numerateur, path, i);
	  pnl_mat_get_col(denominateur, path, i-1);
	  temp_num = pnl_vect_scalar_prod(numerateur, Coeff_);
	  temp_deno = pnl_vect_scalar_prod(denominateur, Coeff_);
	  sum = sum + temp_num/temp_deno;
  }
  sum = sum/(double)(TimeSteps_) - 1;
  pnl_vect_free(&numerateur);
  pnl_vect_free(&denominateur);
  return 1+MIN(MAX(sum,0), 0.1);
}

void Performance::price_mc(
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

	mc_performance<<<dimGrid, dimBlock, 2*(dimBlock.x)*sizeof(float)>>>(N, size_, samples, d_coeff, d_path, d_per_block_results_price, d_per_block_results_ic);
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