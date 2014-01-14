#pragma once
#define DLLEXP __declspec ( dllexport ) 
namespace Computations{
	DLLEXP void priceOption(double &ic, double &prix, int M, double T, double S0, double K, double sigma, double r, int size, int timeStep);
}