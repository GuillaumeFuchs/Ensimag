#pragma once
//#include "Basket.h"
//#include "Bs.h"
//#include "Montecarlo.h"
//#include "Option.h"
#include "Pricer.h"
#include "pnl/pnl_matrix.h"
#include "pnl/pnl_vector.h"
using namespace System;
using namespace std;
 
namespace Wrapper {
 
	public ref class WrapperClass
	{
	private:
		double intConfiance;
		double prix;
	public:
		WrapperClass() {intConfiance = prix = 0;};
		/*void createBasket(double strike, PnlVect* coeff, Basket *bask);
		void createBs(int size, double r, double rho, PnlVect *sigma, PnlVect *spot, PnlVect *trend, BS *Bsmod);
		void createMontecarlo(BS *Bs, Option *Opt, PnlRng* rng, double h, int samples, MonteCarlo *Mc);
	    void getPriceOption(double strike, int size, double r, double rho, double h, int samples);*/
		
		void getPriceOption(String ^type_sharp, int size, array<double> ^spot_sharp, double strike, double maturity, array<double> ^sigma_sharp, double r, array<double> ^rho_sharp, array<double> ^coeff_sharp, int timeStep, int samples);
		double getPrice() {return prix;};
		double getIC() {return intConfiance;};
	};
}
