# include "Playlist.h"
# include <pnl/pnl_mathtools.h>

/*!
 * \file playlist.cpp
 * \brief Produit structuré
 * \author Equipe 11
 */

Playlist :: Playlist(){
  size_ = 4;
  T_ = 1;
  timeStep_ = 0;
  strike_ = 0;
}

Playlist :: Playlist(double T, int timeStep, int size, double Strike){
	T_ = T;
	timeStep_ = timeStep;
	size_ = size;
	strike_ = Strike;
}

Playlist :: ~Playlist(){
}

double Playlist :: get_T() {
  return T_;
}

int Playlist :: get_timeStep() {
  return timeStep_;
}

int Playlist:: get_size(){
	return size_;
}

double Playlist :: get_strike() {
  return strike_;
}

void Playlist :: set_T(double T) {
  T_ = T;
}

void Playlist :: set_timeStep(int timeStep) {
  timeStep_ = timeStep;
}

void Playlist:: set_size(int size){
	size_ = size;
}

void Playlist :: set_strike(double Strike) {
  strike_ = Strike;
}

double Playlist :: payoff (const PnlMat *path) {
  //On retourne le max entre le résultat de la somme et 0
  return MAX(pnl_mat_get(path,0,50)-strike_, 0);
}