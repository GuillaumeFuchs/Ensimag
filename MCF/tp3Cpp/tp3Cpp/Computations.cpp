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
	return MAX( (T/(double)J)*((S0 + s)/2 + sum) - K, 0)  ;
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

double varControl(double T, int J, PnlVect* path)
{
	double sum = 0;
	for (int i = 1; i < J; i++){
		sum += log(GET(path, i));
	}
	return exp((1/(double)J) * ((log(GET(path, 0)) + log(GET(path, J)))/2 + sum));
}
void monteCarloControl (double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic)
{
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));
	PnlVect *G = pnl_vect_create(J);
	PnlVect *path = pnl_vect_create(J+1);

	double sum_pay = 0;
	double var_pay = 0;
	double sum_control = 0;
	double var_control = 0;
	double cov = 0;

	double pay = 0;
	double control = 0;

	double mean_term = (r - pow(sigma, 2.0) / 2 ) * T/J;
	double var_term = sigma * sqrt(T/J);


	for (int i= 0; i<M; i++){
		pnl_vect_rng_normal(G, J, rng);
		pay = payoff(S0, K, mean_term, var_term, r, T, J, G, path);
		sum_pay += pay;
		var_pay += pay * pay;

		control = varControl(T, J, path);

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
	pnl_vect_free(&path);
}

double varControlBis(double T, int J, double K, double r, PnlVect* path)
{
	double sum = 0;
	for (int i = 1; i < J; i++){
		sum += log(GET(path, i));
	}
	double integral = exp((1/(double)J) * ((log(GET(path, 0)) + log(GET(path, J)))/2 + sum));
	return exp(-r*T)*MAX(integral - K, 0);

}
void monteCarloControlBis(double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic)
{
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));
	PnlVect *G = pnl_vect_create(J);
	PnlVect *path = pnl_vect_create(J+1);

	double sum_pay = 0;
	double var_pay = 0;
	double sum_control = 0;
	double var_control = 0;
	double cov = 0;

	double pay = 0;
	double control = 0;

	double mean_term = (r - pow(sigma, 2.0) / 2 ) * T/J;
	double var_term = sigma * sqrt(T/J);

	for (int i= 0; i<M; i++){
		pnl_vect_rng_normal(G, J, rng);
		pay = payoff(S0, K, mean_term, var_term, r, T, J, G, path);
		sum_pay += pay;
		var_pay += pay * pay;

		control = varControlBis(T, J, K, r, path);

		sum_control += control;
		var_control += control * control;

		cov += control * pay;
	}

	double d2= -(1/sigma*sqrt(3/T) * (log(K/S0)- (r-pow(sigma, 2.0)/2) *T/2) );
	double d1 = d2 + sigma * sqrt(T/3);

	double esp_control = exp(-r*T) * (-K * pnl_cdfnor(d2) + S0*exp((r - pow(sigma, 2.0)/6)*T/2) * pnl_cdfnor(d1) );

	double c = (cov - esp_control * sum_pay/M) / (var_control - pow(esp_control, 2.0));
	sum_pay = sum_pay/M - c * (sum_control/M - esp_control);

	prix = exp(-r * T) * sum_pay;

	double variance = exp(-2 * r * T) * (var_pay - 2*c*cov  + c*c*var_control)/M - pow(prix, 2.0);
	ic = 1.96 * sqrt(variance/M);
	pnl_vect_free(&G);
	pnl_rng_free(&rng);
	pnl_vect_free(&path);
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




double payoffAsian(double S0, double K, double T, double sigma, double r, int J, PnlRng* rng) {
	double timeSteps = (double)(T/J);
	double drift = (r - sigma*sigma/2.);
	double sum = 0;

	double s = S0;
	for (int i = 0; i < J; i++) {
		s = s*exp(drift*timeSteps + sigma*sqrt(timeSteps)*pnl_rng_normal(rng));
		if (i < J-1 ) {
			sum += s;
		}
	}
	sum += (S0 + s)/2;
	sum*=T/J;
	return MAX(sum - K, 0);
}

double payoffAsianControlY(double S0, double K, double T, double sigma, double r, int J, PnlRng* rng
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
		y = exp((1/T)*sum_y);
		return MAX(sum - K, 0);
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

void asianCall(double &prix, double &ic, double S0, 
	double K, double T, double sigma, double r, int J, int  M) {

		double px = 0;
		double var = 0;
		double sum = 0;
		PnlRng* rng = pnl_rng_create(PNL_RNG_MERSENNE);
		pnl_rng_sseed(rng, time(NULL));
		for (int i = 1; i <= M; i++) {
			px = payoffAsian(S0, K, T, sigma, r, J, rng);
			sum += px;
			var += px*px;
		}
		prix = exp(-r*T)*sum/M;
		var = var * exp(-2*r*T)/M - prix*prix;
		ic = 1.96*sqrt(var/M);

}

void asianCallControlY(double &prix, double &ic, double S0, double K, 
	double T, double sigma, double r, int J, int M) {

		double px = 0;
		double var = 0;
		double sum = 0;
		double sum_y = 0;
		double e_y = S0 * exp(r*(T/2) - (sigma * sigma * T / 12));
		double y = 0;
		double sum_square = 0;
		double exp_rT = exp(- r * T);
		PnlRng* rng = pnl_rng_create(PNL_RNG_MERSENNE);
		pnl_rng_sseed(rng, time(NULL));
		for (int i = 1; i <= M; i++) {
			px = payoffAsianControlY(S0, K, T, sigma, r, J, rng, y);
			sum += px;
			sum_y += y;
			sum_square += (exp_rT * px - y) * (exp_rT * px - y);
		}
		sum /= M;
		sum_y /= M;
		prix = exp_rT*sum - sum_y + e_y;
		var = sum_square/M - (sum * exp_rT - sum_y) * (sum * exp_rT - sum_y); 
		ic = 1.96*sqrt(var/M);
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
int main(){
	double px;
	double ic;
	/*
	monteCarlo(100, 0.2, 0.095, 1, 100, 52, 50000, px, ic);
	printf("prix: %f\n", px);
	printf("ic: %f\n", ic);

	asianCall(px, ic, 100, 100, 1, 0.2, 0.095, 52, 50000);
	printf("prix: %f\n", px);
	printf("ic: %f\n\n", ic);
	*/
	monteCarloControl(100, 0.2, 0.1, 1, 100, 52, 50000, px, ic);
	printf("prix: %f\n", px);
	printf("ic: %f\n", ic);
	asianCallControlY(px, ic, 100, 100, 1, 0.2, 0.095, 52, 50000);
	printf("prix: %f\n", px);
	printf("ic: %f\n\n", ic);
	/*
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
