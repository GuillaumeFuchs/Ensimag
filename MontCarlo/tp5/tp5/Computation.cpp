#include <math.h>
#include <pnl/pnl_mathtools.h>
#include <pnl/pnl_random.h>
#include <ctime>

double payoff (double S0, double drift, double var, double K, double g){
	double s = S0 * exp(drift + var*g);

	return MAX(s-K, 0);
}
void price(int M, double &prix, double &ic, PnlRng* rng, double S0, double drift, double var, double K, double r, double T){
	double pay;
	double sum = 0.;
	double variance = 0.;
	double g;

	for (int i = 0; i < M; i++){
		g = pnl_rng_normal(rng);
		pay = payoff(S0, drift, var, K, g);
		sum += pay;
		variance += pay * pay;
	}
	prix = exp(-r*T) * sum/M;
	variance = exp(-2.*r*T) * variance/M - prix*prix;
	ic = 1.96*sqrt(variance/M);
}

void delta_df(int M, double h, double &delta, PnlRng* rng, double S0, double drift, double var, double K, double r, double T){
	double pay1, pay2;
	double sum = 0.;
	double g;

	for(int i = 0; i < M; i++){
		g = pnl_rng_normal(rng);
		pay1 = payoff(S0+h, drift, var, K, g);
		pay2 = payoff(S0-h, drift, var, K, g);
		sum += (pay1 - pay2)/(2*h);
	}

	delta = exp(-r*T) * sum/M;
}

void delta_likelihood(){

}

void delta_pathwise(int M, double &delta, PnlRng* rng, double S0, double drift, double var, double K, double r, double T){
	double pay;
	double sum = 0.;
	double g;

	for (int i = 0; i < M; i++){
		g = pnl_rng_normal(rng);
		pay = payoff(S0, drift, var, K, g);
		if (pay != 0){
			sum += (pay+K)/S0;
		}
	}

	delta = exp(-r*T) * sum/M;
}

int main(){
	double prix, ic, delta;

	double S0 = 100.;
	double sigma = .2;
	double r = .05;
	double T = 1.;
	double K = 100.;
	int M = 50000;
	double h = .01;

	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));

	double drift = (r - sigma*sigma/2.)*T;
	double var = sigma*sqrt(T);

	price(M, prix, ic, rng, S0, drift, var, K, r, T);
	printf("%f\n%f\n", prix, ic);

	delta_df(M, h, delta, rng, S0, drift, var, K, r, T);
	printf("%f\n", delta);

	delta_pathwise(M, delta, rng, S0, drift, var, K, r, T);
	printf("%f\n", delta);

	pnl_rng_free(&rng);
	system("pause");
	return 0;
}