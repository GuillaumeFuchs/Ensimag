#include "Computations.h"
#include "iostream"
#include "ctime"
#include "pnl/pnl_random.h"

using namespace std;
 
double payoff(double S0, double K, double L, int J, double drift, double sigma, double sqrt_dt, PnlRng* rng)
{
	double s = S0;
	for (int i = 1; i < J+1; i++){
		if (s < L){
			s = 0;
			break;
		}
		s = s * exp(drift + sigma * sqrt_dt * pnl_rng_normal(rng));
	}

	return MAX(s - K, 0.);
}

void Computations::price (double &ic, double &px, double S0, double K, double r, double sigma, double T, double L, int J, int M)
{
	double const drift = (r - sigma*sigma) * T/J;
	double const sqrt_dt = sqrt(T/J);
	double sum = 0, var = 0;

	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));

	double pay;

	for (int i = 0; i < M; i++){
		pay= payoff(S0, K, L, J, drift, sigma, sqrt_dt, rng);
		sum += pay;
		var += pay*pay;
	}

	px = exp(-r*T) * sum/M;
	var = exp(-2*r*T) * var/M - px*px;
	ic = 1.96 * sqrt(var/M);
	pnl_rng_free(&rng);
}