# include "option.h"
# include "barrier_u.h"
# include <pnl/pnl_mathtools.h>

/*!
 * \file barrier_u.cpp
 * \brief Option barrière haute
 * \author Equipe 11
 */

Barrier_u :: Barrier_u() : Option() {
  Strike_ = 0;
  Coeff_ = pnl_vect_new();
  Bu_ = pnl_vect_new();
  Strike_gpu = Strike_;
}

Barrier_u ::Barrier_u(Parser& pars): Option(pars){
  Strike_ = pars.getDouble("strike");
  Coeff_ = pnl_vect_copy(pars.getVect("payoff coefficients"));
  Bu_ = pnl_vect_copy(pars.getVect("upper barrier"));

  Strike_gpu = (float)Strike_;
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

void Barrier_u::price_mc(double &prix, int nBlocks, int nThreads, int N, float* d_path) 
{
}
