#include "stdafx.h"
#include "Wrapper.h"

using namespace Computations;

namespace Wrapper {
	/*
	void WrapperClass::getPriceOption(String ^type_sharp, int size, array<double> ^spot_sharp, 
double strike, double maturity, array<double> ^sigma_sharp, double r, array<double> ^rho_sharp, array<double> ^coeff_sharp, int timeStep, int samples){
		double ic = 0;
		double px = 0;
		int i = 0;

		char* type = static_cast<char *>(System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(type_sharp).ToPointer());
		System::Runtime::InteropServices::Marshal::FreeHGlobal(safe_cast<IntPtr>(type));
		
		double* spot = new double[size];
		for (i = 0; i < size; i++)
			spot[i] = (double)spot_sharp->GetValue(i);

		double* sigma = new double[size];
		for (i = 0; i < size; i++)
			sigma[i] = (double)sigma_sharp->GetValue(i);
		
		int rho_size = (size-1)*size/2;
		double* rho = new double[rho_size];
		for (i = 0; i < rho_size; i++)
			rho[i] = (double)rho_sharp->GetValue(i);
		
		double* coeff = new double[size];
		for (i = 0; i < size; i++)
			coeff[i] = (double)coeff_sharp->GetValue(i);

		price(px, ic, type, size, spot, strike, maturity, sigma, r, rho, coeff, timeStep, samples);

		this->intConfiance = ic;
		this->prix = px;
	}
	*/
	void WrapperClass::getPriceOption(int M, double T, double S0, double K, double sigma, double r, int size, int timeStep)
	{
		double ic = 0;
		double px = 0;

		priceOption(ic, px, M, T, S0, K, sigma, r, size, timeStep);

		this->intConfiance = ic;
		this->prix = px;
	}
}