#include "Computations.h"
#include "pnl/pnl_vector.h"
#include "pnl/pnl_mathtools.h"
#include "pnl/pnl_random.h"
#include <ctime>
#include <math.h>

using namespace std;

double payoff(double S0, double K, double mean_term, double var_term, double r, double T, int J, const PnlVect* G)
{
	double s = S0;
	double sum = 0;
	for(int i = 1; i < J ; i++)
	{
		s = s * exp( mean_term + var_term * GET(G, i-1));
		sum += s;
	}
	s =  s * exp( mean_term + var_term * GET(G, J-1));
	return exp( -r * T) * MAX( (T/(double)J) * ((S0 + s)/2 + sum) - K, 0)  ;
}

void Computations::monteCarlo (double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic)
{
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));
	PnlVect *G = pnl_vect_create(J);

	double sum = 0;
	double variance = 0;
	double pay = 0;
	double mean_term = (r - pow(sigma, 2.0) / 2 ) * T/(double)J;
	double var_term = sigma * sqrt(T/(double)J);

	for (int i= 0; i<M; i++){
		pnl_vect_rng_normal(G, J, rng);
		pay= payoff(S0, K, mean_term, var_term, r, T, J, G);
		prix = payoff(S0, K, mean_term, var_term, r, T, J, G);
		sum += pay;
		variance += pay * pay;
	}

	prix = exp(-r * T) * sum / (double)M;

	variance = exp(-2 * r * T) * variance/(double)M - prix * prix;
	ic = 1.96 * sqrt(variance/M);
	pnl_vect_free(&G);
	pnl_rng_free(&rng);
}
/*
void Computations::monteCarloAnti (double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic)
{
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));

	PnlVect *G_p = pnl_vect_create(J);
	PnlVect *G_m = pnl_vect_create(J);

	double payoff_p, payoff_m;
	double sum = 0;
	double variance_p = 0;
	double variance_m = 0;
	double cov = 0;
	double variance = 0;
	for (int i= 0; i<M; i++){
		pnl_vect_rng_normal(G_p, J, rng);
		payoff_p = payoff(S0, sigma, r, T, K, J, G_p);
		G_m = pnl_vect_copy(G_p);
		pnl_vect_mult_double(G_m, -1);
		payoff_m = payoff(S0, sigma, r, T, K, J, G_m);
		sum += 0.5 * (payoff_p + payoff_m);
		variance_p += payoff_p * payoff_p;
		variance_m += payoff_m * payoff_m;
		cov += payoff_p * payoff_m;
	}

	prix = exp(-r*T) * sum / (double)M;	
	variance = (variance_p + variance_m + 2 * cov) / 4;
	variance = exp (-2. * r * T) * variance / (double)M - prix*prix;
	ic = 1.96 * sqrt(variance / (double) M);
	pnl_vect_free(&G_p);
	pnl_vect_free(&G_m);
	pnl_rng_free(&rng);
}

double var_control(double S0, double sigma, double r, double T, double K, int J, const PnlVect* G) {
	double s;
	double max = 0;
	double timeStep = T/(double)J;
	double sT;

	PnlVect *path = pnl_vect_create(J+1);
	LET(path, 0) = S0;

	for(int i = 0; i < J; i++)
	{
		s = GET(path, i) * 
			exp((r - pow(sigma, 2.0)/2) * timeStep +
			sigma * sqrt(timeStep) * GET(G, i));
		LET(path, i+1) = s;
	}
	sT = GET(path, J);
	pnl_vect_free(&path);
	return (sT-S0*exp(r*T));
}

void Computations::monteCarloControl (double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic)
{
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));
	PnlVect *G = pnl_vect_create(J);
	double sum = 0;
	double sum_control = 0;
	double pay;
	double control = 0;
	double lambda = 0;
	double cov = 0;
	double variance_control = 0;
	double var_pay = 0;
	for (int i= 0; i<M; i++){
		pnl_vect_rng_normal(G, J, rng);
		pay = payoff(S0, sigma, r, T, K, J, G);
		control = var_control(S0, sigma, r, T, K, J, G);
		sum += pay;
		sum_control += control;
		cov += control * pay;
		variance_control += control * control;
		var_pay += pay * pay;
	}
	lambda = cov/variance_control;
	sum = sum - lambda * sum_control;
	prix = exp(-r * T) * sum / (double) M;
	double variance = var_pay + lambda * lambda * variance_control - 2 * lambda * cov;
	variance = exp(-2. * r * T) * variance / (double)M - prix * prix;
	ic = 1.96 * sqrt(variance / M);	
	pnl_vect_free(&G);
	pnl_rng_free(&rng);
}
*/