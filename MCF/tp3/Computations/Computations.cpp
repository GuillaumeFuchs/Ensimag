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
	//Calcul S_1 à S_{J-1}
	for(int i = 1; i < J ; i++)
	{
		s = s * exp( mean_term + var_term * GET(G, i-1));
		sum += s;
	}
	//Calcul S_T
	s =  s * exp( mean_term + var_term * GET(G, J-1));
	return exp( -r * T) * MAX( (1/(double)J)*((S0 + s)/2 + sum) - K, 0)  ;
}
double payoff(double S0, double K, double mean_term, double var_term, double r, double T, int J, const PnlVect* G, PnlVect* path)
{
	LET(path, 0) = S0;
	double s = S0;
	double sum = 0;
	//Calcul S_1 à S_{J-1}
	for(int i = 1; i < J ; i++)
	{
		s = s * exp( mean_term + var_term * GET(G, i-1));
		sum += s;
		LET(path, i) = s;
	}
	//Calcul S_T
	s =  s * exp( mean_term + var_term * GET(G, J-1));
	LET(path, J) = s;
	return exp( -r * T) * MAX( (1/(double)J)*((S0 + s)/2 + sum) - K, 0)  ;
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
		sum += pay;
		variance += pay * pay;
	}

	prix = exp(-r * T) * sum / (double)M;

	variance = exp(-2 * r * T) * variance/(double)M - prix * prix;
	ic = 1.96 * sqrt(variance/M);
	pnl_vect_free(&G);
	pnl_rng_free(&rng);
}

void Computations::monteCarloControl (double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic)
{
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));
	PnlVect *G = pnl_vect_create(J);
	PnlVect *path = pnl_vect_create(J+1);

	double variance;
	double sum_pay = 0;
	double var_pay = 0;
	double sum_control = 0;
	double var_control = 0;
	double cov = 0;

	double pay = 0;
	double control = 0;

	double mean_term = (r - pow(sigma, 2.0) / 2 ) * T/(double)J;
	double var_term = sigma * sqrt(T/(double)J);
	
	for (int i= 0; i<M; i++){
		pnl_vect_rng_normal(G, J, rng);
		pay = payoff(S0, K, mean_term, var_term, r, T, J, G, path);
		sum_pay += pay;
		var_pay += pay * pay;

		control = (1/(double)J) * ((log(S0) + log(GET(path, J)))/2 + pnl_vect_sum(path) - S0 - GET(path, J));
		sum_control += control;
		var_control += control * control;

		cov += control * pay;
	}
	double esp_control = S0 * exp(r * T/2 - pow(sigma, 2.0)*T/12);
	
	double c = (var_pay - pow(sum_pay/(double)M, 2.0)) * (cov - sum_pay/(double)M * sum_control/(double)M) / (var_control - pow(esp_control, 2.0));
	
	sum_pay = sum_pay - c * (sum_control - (double)M*esp_control);
	
	prix = exp(-r * T) * sum_pay / (double)M;

	//variance = exp(-2 * r * T) * variance/(double)M - prix * prix;
	//ic = 1.96 * sqrt(variance/M);
	pnl_vect_free(&G);
	pnl_rng_free(&rng);
	pnl_vect_free(&path);
	ic = 1;
}
