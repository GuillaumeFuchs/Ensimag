# include "Option.h"
# include "barrier_l.h"
# include <pnl/pnl_mathtools.h>

/*!
 * \file barrier_l.cpp
 * \brief Option barrière basse
 * \author Equipe 11
 */

Barrier_l :: Barrier_l() : Option() {
  strike_ = 0;
  Coeff_ = pnl_vect_new();
  Bl_ = pnl_vect_new();
}

Barrier_l::Barrier_l(double strike, double* coeff, double *bl, double T, int timeStep, int size) : Option(T, timeStep, size){
	strike_ = strike;
	Coeff_ = pnl_vect_create(size_);
	Bl_ = pnl_vect_create(size_);
	for (int i = 0; i < size_; i++){
		LET(Coeff_, i) = coeff[i];
		LET(Bl_, i) = bl[i];
	}
}

Barrier_l :: ~Barrier_l(){
  pnl_vect_free(&Coeff_);
  pnl_vect_free(&Bl_);
}

double Barrier_l :: get_Strike() const{
  return strike_;
}

PnlVect* Barrier_l :: get_Coeff() const{
  return Coeff_;
}

PnlVect* Barrier_l :: get_Bl() const{
  return Bl_;
}

void Barrier_l :: set_Strike(double Strike) {
  strike_ = Strike;
}

void Barrier_l :: set_Coeff(PnlVect *Coeff) {
  Coeff_ = Coeff;
}

void Barrier_l :: set_Bl(PnlVect *Bl) {
  Bl_ = Bl;
}

double Barrier_l :: payoff (const PnlMat *path) const{
  double sum ;
  PnlVect* final = pnl_vect_create(size_);

  //On met dans final la dernière colonne de Path correspond à la valeur à maturité des sous-jacents.
  pnl_mat_get_col(final, path, timeStep_);
  sum = pnl_vect_scalar_prod(final, Coeff_) - strike_;
  //On vérifie que toutes les valeurs des sous-jacents soient au dessus de la barrière
  //Si on en trouve une alors le prix de l'option est de 0
  for (int i=0; i<timeStep_+1; i++){
	for (int d=0; d<size_; d++){
	  if (pnl_mat_get(path,d,i) < pnl_vect_get(Bl_,d)){
		pnl_vect_free(&final);
		return 0;
	  }
	}
  }
  return MAX(sum, 0);
}
