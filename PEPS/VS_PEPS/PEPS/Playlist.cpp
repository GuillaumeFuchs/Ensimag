#include "Option.h"
#include "Playlist.h"
#include <math.h>
#include <iostream>
using namespace std;
/*!
 * \file Playlist.cpp
 * \brief Option panier
 * \author Equipe 11
 */

Playlist :: Playlist() : Option() {
  strike_ = 0;
  Coeff_ = pnl_vect_new();
  r_ = 0;
}

Playlist::Playlist(double strike, double* coeff, double T, int timeStep, int size, double r) : Option(T, timeStep, size){
	strike_ = strike;
	Coeff_ = pnl_vect_create(size_);
	for (int i = 0; i < size_; i++){
		LET(Coeff_, i) = coeff[i];
	}
	r_ = r;
}

Playlist :: ~Playlist(){
  pnl_vect_free(&Coeff_);
}

double Playlist :: get_Strike() const{
  return strike_;
}

PnlVect * Playlist :: get_Coeff() const{
  return Coeff_;
}

double Playlist :: get_r() const{
	return r_;
}

void Playlist :: set_Strike(double strike) {
  strike_ = strike;
}

void Playlist :: set_Coeff(PnlVect *Coeff) {
  Coeff_ = Coeff;
}

void Playlist :: set_r(double r) {
	r_ = r;
}

double Playlist :: payoff (const PnlMat *path) const{
	if (T_ == 1){
		return exp(-r_*T_)*150*1.045;
	}

	double gain = 0;
	int compteur = 0;

	//matrice des termes S0
	PnlVect* S0 = pnl_vect_create(size_);
	pnl_mat_get_col(S0, path, 0);
	for (int i = 1 ; i < T_*(timeStep_/6)+1; i++){
		for (int d = 0; d < size_; d++){
			if ( (fabs((MGET(path, d, i)/GET(S0, d)-1)*100 - 10) < 0.00001) || ((MGET(path, d, i)/GET(S0, d)-1)*100 >= 10) ){
				compteur++;
				if (compteur > 2){
					gain += 0.045;
					break;
				}
			}
		}
		compteur = 0;
	}
	pnl_vect_free(&S0);
	return exp(-r_*T_)*150*(1+gain);
}
