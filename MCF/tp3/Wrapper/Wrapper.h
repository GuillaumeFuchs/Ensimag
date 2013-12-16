#pragma once
#include "Computations.h"
using namespace System;
 
namespace Wrapper {
 
	public ref class WrapperClass
	{
	private:
		double intConfiance;
		double prix;
	public:
		WrapperClass() {intConfiance = prix = 0;};
		void getPriceOptionMC(double S0, double sigma, double r, double T, double K, int J, int M);
		/*void getPriceOptionMCA(double S0, double sigma, double r, double T, double K, int J, int M);
		void getPriceOptionMCC(double S0, double sigma, double r, double T, double K, int J, int M);*/
		double getPrice() {return prix;};
		double getIC() {return intConfiance;};
	};
}