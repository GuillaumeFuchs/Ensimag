#pragma once
#define DLLEXP   __declspec( dllexport )
namespace Computations{
   DLLEXP void price(double &ic, double &px, double S0, double K, double r, double sigma, double T, double L, int J, int M); 
}