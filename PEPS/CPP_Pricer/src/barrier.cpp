# include "option.h"
# include "barrier.h"
# include <pnl/pnl_mathtools.h>

/*!
 * \file barrier.cpp
 * \brief Option barrière
 * \author Equipe 11
 */

Barrier :: Barrier() : Option() {
  Strike_ = 0;
  Coeff_ = pnl_vect_new();
  Bl_ = pnl_vect_new();
  Bu_ = pnl_vect_new();
}

Barrier ::Barrier(Parser& pars) : Option(pars){
  Strike_ = pars.getDouble("strike");
  Coeff_ = pnl_vect_copy(pars.getVect("payoff coefficients"));
  Bu_ = pnl_vect_copy(pars.getVect("upper barrier"));
  Bl_ = pnl_vect_copy(pars.getVect("lower barrier"));
}

Barrier :: ~Barrier(){
  pnl_vect_free(&Coeff_);
  pnl_vect_free(&Bu_);
  pnl_vect_free(&Bl_);
}

double Barrier :: get_Strike(){
  return Strike_;
}

PnlVect* Barrier :: get_Coeff(){
  return Coeff_;
}

PnlVect* Barrier :: get_Bl(){
  return Bl_;
}

PnlVect* Barrier :: get_Bu(){
  return Bu_;
}

void Barrier :: set_Strike(double Strike) {
  Strike_ = Strike;
}

void Barrier :: set_Coeff(PnlVect *Coeff) {
  Coeff_ = Coeff;
}

void Barrier :: set_Bl(PnlVect *Bl) {
  Bl_ = Bl;
}

void Barrier :: set_Bu(PnlVect *Bu) {
  Bu_ = Bu;
}

double Barrier :: payoff (const PnlMat *path) {
  double sum ;
  PnlVect* final = pnl_vect_create(size_);

  //On met dans final la dernière colonne de Path correspond à la valeur à maturité des sous-jacents.
  pnl_mat_get_col(final, path, TimeSteps_);
  sum = pnl_vect_scalar_prod(final, Coeff_) - Strike_;
  //On vérifie que toutes les valeurs des sous-jacents soient entre les 2 barrières
  //Si on en trouve une alors le prix de l'option est de 0
  for (int i=0; i<TimeSteps_+1; i++){
	for (int d=0; d<size_; d++){
	  if ((pnl_mat_get(path,d,i) < pnl_vect_get(Bl_,d)) || (pnl_mat_get(path,d,i) > pnl_vect_get(Bu_, d))){
		pnl_vect_free(&final);
		return 0;
	  }
	}
  }
  return MAX(sum, 0);
}
