// Il s'agit du fichier DLL principal.

#include "stdafx.h"

#include "Wrapper2.h"

using namespace Engine;

namespace Wrapper
{
	void WrapperClass::getPriceOption(int M, double T, double S0, double K, double sigma, double r, int size, int timeStep){
		double ic, px;

		priceOption(ic, px, M, T, S0, K, sigma, r, size, timeStep);

		this->intConfiance = ic;
		this->prix = px;
	}
}