#include "stdafx.h"
 
#include "Wrapper.h"
 
using namespace Computations;
namespace Wrapper {
	void WrapperClass::getPriceOption(int M, double T, double S0, double K, double L, double sigma, double r, int J) {
		double ic, px;
		price(ic, px, S0, K, r, sigma, T, L, J, M);
		this->intConfiance = ic;
		this->prix = px;
	}
}