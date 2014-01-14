#pragma once
#define DLLEXP __declspec ( dllexport )

namespace Computations{
	DLLEXP void monteCarlo (double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic);
	DLLEXP void monteCarloAnti (double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic);
	DLLEXP void monteCarloControl (double S0, double sigma, double r, double T, double K, int J, int M, double &prix, double &ic);
}