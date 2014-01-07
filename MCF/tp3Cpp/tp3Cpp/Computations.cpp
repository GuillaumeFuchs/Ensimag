#include "Computations.h"
#include "pnl/pnl_vector.h"
#include "pnl/pnl_mathtools.h"
#include "pnl/pnl_random.h"
#include "pnl/pnl_cdf.h"
#include <ctime>
#include <math.h>
#include <iostream>

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
	return  MAX( (1/(double)J)*((S0 + s)/2 + sum) - K, 0)  ;
}
void monteCarlo (double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic)
{
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));
	PnlVect *G = pnl_vect_create(J);

	double sum = 0;
	double variance = 0;
	double pay = 0;
	double mean_term = (r- pow(sigma, 2.0)/2 ) * T/J;
	double var_term = sigma*sqrt(T/J);

	for (int i= 0; i<M; i++){
		pnl_vect_rng_normal(G, J, rng);
		pay= payoff(S0, K, mean_term, var_term, r, T, J, G);
		sum += pay;
		variance += pay * pay;
	}

	prix = exp(-r * T) * sum / (double)M;
	variance = exp(-2 * r * T) * variance/(double)M - pow(prix, 2.0);
	ic = 1.96 * sqrt(variance/M);
	pnl_vect_free(&G);
	pnl_rng_free(&rng);
}

double payoffControl(double S0, double K, double mean_term, double var_term, double r, double T, int J, const PnlVect* G, double &control)
{
	double s = S0;
	double sum = 0.;
	double sumC = 0.;
	//Calcul S_1 à S_{J-1}
	for(int i = 1; i < J ; i++)
	{
		s = s * exp( mean_term + var_term * GET(G, i-1));
		sum += s;
		sumC += log(s);
	}
	//Calcul S_T
	s =  s * exp( mean_term + var_term * GET(G, J-1));
	sumC += (log(S0) + log(s))/2.;
	control = exp(1/(double)J * sumC);

	return MAX( (T/(double)J)*( (S0 + s)/2. + sum ) - K, 0)  ;
}
void monteCarloControl (double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic)
{
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));
	PnlVect *G = pnl_vect_create(J);

	double sum_pay = 0., var_pay = 0., sum_control = 0., var_control = 0., pay = 0., cov = 0., control = 0.;
	
	double mean_term = (r - pow(sigma, 2.0)/2. ) * T/J;
	double var_term = sigma * sqrt(T/J);


	for (int i= 0; i<M; i++){
		pnl_vect_rng_normal(G, J, rng);
		pay = payoffControl(S0, K, mean_term, var_term, r, T, J, G, control);
		sum_pay += pay;
		var_pay += pay * pay;

		sum_control += control;
		var_control += control * control;

		cov += control * pay;
	}
	double esp_control = S0*exp(r * T/2 - pow(sigma, 2.0)*T/12);

	double c = (cov - esp_control * sum_pay/M) / (var_control - pow(esp_control, 2.0));

	sum_pay = exp(-r*T)*sum_pay/M - c * (sum_control/M - esp_control);

	prix = sum_pay;
	double variance = exp(-2 * r * T) * (var_pay - 2*c*cov  + c*c*var_control)/M - pow(prix, 2.0);
	ic = 1.96 * sqrt(variance/M);
	pnl_vect_free(&G);
	pnl_rng_free(&rng);
}

double payoffControlBis(double S0, double K, double mean_term, double var_term, double r, double T, int J, const PnlVect* G, double &control, double exp_r_t)
{
	double s = S0;
	double sum = 0.;
	double sumC = 0.;
	//Calcul S_1 à S_{J-1}
	for(int i = 1; i < J ; i++){
		s = s * exp( mean_term + var_term * GET(G, i-1));
		sum += s;
		sumC += log(s);
	}
	//Calcul S_T
	s =  s * exp( mean_term + var_term * GET(G, J-1));
	sumC += (log(S0) + log(s))/2.;
	control = exp_r_t * MAX(exp(1/(double)J * sumC) - K, 0);

	return MAX( (T/(double)J)*( (S0 + s)/2. + sum ) - K, 0)  ;
}
void monteCarloControlBis(double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic)
{
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));
	PnlVect *G = pnl_vect_create(J);
	PnlVect *path = pnl_vect_create(J+1);

	double sum_pay = 0., var_pay = 0., sum_control = 0., var_control = 0., pay = 0., cov = 0., control = 0.;


	double exp_r_t = exp(-r*T);
	double mean_term = (r - pow(sigma, 2.0) / 2 ) * T/J;
	double var_term = sigma * sqrt(T/J);

	for (int i= 0; i<M; i++){
		pnl_vect_rng_normal(G, J, rng);
		pay = payoffControlBis(S0, K, mean_term, var_term, r, T, J, G, control, exp_r_t);
		sum_pay += pay;
		var_pay += pay * pay;

		sum_control += control;
		var_control += control * control;

		cov += control * pay;
	}

	double d2= 1/sigma*sqrt(3/T) * (log(K/S0)- (r-pow(sigma, 2.0)/2) *T/2);
	double d1 = -d2 + sigma * sqrt(T/3);

	double esp_control = exp(-r*T) * (-K * pnl_cdfnor(-d2) + S0*exp((r - pow(sigma, 2.0)/6)*T/2) * pnl_cdfnor(d1) );

	double c = (cov - esp_control * sum_pay/M) / (var_control - pow(esp_control, 2.0));
	sum_pay = sum_pay/M - c * (sum_control/M - esp_control);
	
	prix = exp(-r * T) * sum_pay;

	double variance = exp(-2 * r * T) * (var_pay - 2*c*cov  + c*c*var_control)/M - pow(prix, 2.0);
	ic = 1.96 * sqrt(variance/M);

	pnl_vect_free(&G);
	pnl_rng_free(&rng);
	pnl_vect_free(&path);
}

double esperanceZ(double S0, double K, double T, double sigma, double r, int J) {
	double sqrtT3 = sqrt(T/3.);
	double spot = S0*exp((r - sigma*sigma/6.)*T/2.);
	double d = -1.*(1./sigma) * (sqrt(3/T)) * (log(K/S0) - (r-sigma*sigma/2.)*T/2.);
	double d2 = d + sigma * sqrtT3;
	double q, bound;
	int status;
	int which = 1;
	double p2, p1;
	double mean = 0.;
	double sd = 1.;
	pnl_cdf_nor(&which, &p2, &q, &d2, &mean, &sd, &status, &bound);
	pnl_cdf_nor(&which, &p1, &q, &d, &mean, &sd, &status, &bound);
	double espZ = spot * p2 - K*p1;
	espZ *= exp(-r*T);
	return espZ;
}
double payoffAsianControlZ(double S0, double K, double T, double sigma, double r, int J, PnlRng* rng
	, double &y) {
		double timeSteps = (double)(T/J);
		double drift = (r - sigma*sigma/2.);
		double sum = 0;
		double sum_y = 0;
		double s = S0;
		for (int i = 0; i < J; i++) {
			s = s*exp(drift*timeSteps + sigma*sqrt(timeSteps)*pnl_rng_normal(rng));
			if (i < J-1 ) {
				sum += s;
				sum_y += log(s);
			}
		}
		sum += (S0 + s)/2;
		sum_y += (log(S0) + log(s))/2;
		sum*=T/J;
		sum_y *= T/J;
		double y1 = exp((1/T)*sum_y);
		y = MAX(y1 - K, 0);
		return MAX(sum - K, 0);
}
void asianCallControlZ(double &prix, double &ic, double S0, double K, 
	double T, double sigma, double r, int J, int M) {

		double px = 0;
		double var = 0;
		double sum = 0;
		double sum_z = 0;
		double z = 0;
		double e_z = esperanceZ(S0, K, T, sigma, r, J);
		double sum_square = 0;
		PnlRng* rng = pnl_rng_create(PNL_RNG_MERSENNE);
		pnl_rng_sseed(rng, time(NULL));
		for (int i = 1; i <= M; i++) {
			px = payoffAsianControlZ(S0, K, T, sigma, r, J, rng, z);
			sum += px;
			sum_z += z;
			sum_square += (px - z)*(px-z);
		}
		sum /= M;
		sum_z /= M;
		prix = exp(-r * T) * (sum - sum_z) + e_z;
		var = exp(-2 * r * T)*((sum_square/M) - (sum - sum_z)*(sum - sum_z));

		//var = var * exp(-2*r*T)/M - prix*prix;
		ic = 1.96*sqrt(var/M);
}
void monteCarloBar(double S0, double K, double r, double sigma, double T, double L, int J, int M, double &prix, double& ic){
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));
	PnlVect *G = pnl_vect_create(J);

	double mean_term = (r - pow(sigma, 2.0) / 2 ) * T/J;
	double var_term = sigma * sqrt(T/J);

	double s = 0;
	double pay;
	double sum = 0;
	double variance = 0;

	for (int i = 0; i < M; i++){
		pnl_vect_rng_normal(G, J, rng);

		s = S0;
		for (int j = 1; j < J+1; j++){
			if (s<L){
				s = 0;
				break;
			}else{
				s = s * exp(mean_term + var_term*GET(G, j-1));
			}
		}

		pay = MAX(s-K, 0);
		sum += pay;
		variance += pay*pay;
	}
	prix = exp(-r * T) * sum / (double)M;

	variance = exp(-2 * r * T) * variance/(double)M - pow(prix, 2.0);
	ic = 1.96 * sqrt(variance/M);
	pnl_vect_free(&G);
	pnl_rng_free(&rng);
}

int main(){
	double px;
	double ic;
	/*
	monteCarlo(100, 0.2, 0.095, 1, 100, 52, 50000, px, ic);
	printf("prix: %f\n", px);
	printf("ic: %f\n", ic);

	monteCarloControl(100, 0.2, 0.095, 1, 100, 52, 50000, px, ic);
	printf("prix: %f\n", px);
	printf("ic: %f\n", ic);

	*/
	monteCarloControlBis(100, 0.2, 0.095, 1, 100, 52, 50000, px, ic);
	printf("prix: %f\n", px);
	printf("ic: %f\n", ic);
	asianCallControlZ(px, ic, 100, 100, 1, 0.2, 0.095, 52, 50000);
	printf("prix: %f\n", px);
	printf("ic: %f\n", ic);
	/*
	monteCarloBar(100, 105, 0.02, 0.25, 1, 90, 100, 100000, px, ic);
	printf("prix: %f\n", px);
	printf("ic: %f\n", ic);*/

	system("pause");
	return 0;
}
