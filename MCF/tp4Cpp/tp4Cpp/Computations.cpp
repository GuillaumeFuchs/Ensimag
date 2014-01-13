#include "Computations.h"
#include "iostream"
#include "ctime"
#include "pnl/pnl_random.h"

using namespace std;
 
double payoff(double S0, double K, double L, int J, double drift, double sigma_sqrt_dt, PnlVect* G)
{
	double s = S0;
	for (int k = 1; k < J+1; k++){
		if (s < L)
			return 0;
		s = s * exp(drift + sigma_sqrt_dt * GET(G, k-1));
	}
	return MAX(s - K, 0.);
}

void price (double &ic, double &px, double S0, double K, double r, double sigma, double T, double L, int J, int M)
{
	double const drift = (r - sigma*sigma/2) * T/(double)J;
	double const sigma_sqrt_dt = sigma*sqrt(T/(double)J);
	double sum = 0, var = 0;

	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	PnlVect *G = pnl_vect_create(J);

	pnl_rng_sseed(rng, time(NULL));

	double pay;
	for (int i = 0; i < M; i++){
		pnl_vect_rng_normal(G, J, rng);
		pay = payoff(S0, K, L, J, drift, sigma_sqrt_dt, G);
		sum += pay;
		var += pay * pay;
	}

	px = exp(-r*T) * sum/(double)M;
	var = exp(-2*r*T) * var/(double)M - px*px;
	ic = 1.96 * sqrt(var/(double)M);
	pnl_rng_free(&rng);
	pnl_vect_free(&G);
}

int main(){
	double px;
	double ic;
	
	double S0 = 100;
	double K = 110;
	double r= 0.05;
	double sigma = 0.2;
	double T = 2;
	double L = 80; 
	int J = 24;
	int M = 50000;

	price(ic, px, S0, K, r, sigma, T, L, J, M);

	printf("px: %f\nic: %f\n", px, ic);
	system("pause");
	return 0;
}
