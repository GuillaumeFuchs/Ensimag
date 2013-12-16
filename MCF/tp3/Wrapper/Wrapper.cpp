#include "stdafx.h"
 
#include "Wrapper.h"
 
using namespace Computations;
namespace Wrapper {
	void WrapperClass::getPriceOptionMC(double S0, double sigma, double r, double T, double K, int J, int M) {
		double ic, px;
		
		monteCarlo (S0, sigma, r, T, K, J, M, px, ic);
		
		this->intConfiance = ic;
		this->prix = px;
	}
	/*
	void WrapperClass::getPriceOptionMCA(double S0, double sigma, double r, double T, double K, int J, int M) {
		double ic, px;
		
		monteCarloAnti (S0, sigma, r, T, K, J, M, px, ic);

		this->intConfiance = ic;
		this->prix = px;
	}

	void WrapperClass::getPriceOptionMCC(double S0, double sigma, double r, double T, double K, int J, int M) {
		double ic, px;
		
		monteCarloControl (S0, sigma, r, T, K, J, M, px, ic);

		this->intConfiance = ic;
		this->prix = px;
	}
	*/
}