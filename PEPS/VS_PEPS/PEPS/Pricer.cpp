#include "pricer.h"
using namespace std;

/*!
* \file pricer.cpp
* \brief Fichier de test pour le price d'option à t=0
* \author equipe 11
*/

void Engine::price(double &prix, double &ic, char *type, int size, double* spot, double strike, double maturity, double* sigma, double r, double* rho, double *coeff, int timeStep, int samples){

	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));

	Bs bs(size, r, rho, sigma, spot, NULL);
	//if (!strcmp("basket", type)){
	Basket opt(strike, coeff, maturity, timeStep, size);
	//Playlist play(1,52,size,strike);
	MonteCarlo mc(&bs, &opt, rng, 0.1,samples);
	mc.price(prix, ic);
	pnl_rng_free(&rng);
}
