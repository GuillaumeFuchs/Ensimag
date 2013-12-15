#include "pnl/pnl_vector.h"
#include "basket.h"
#include "playlist.h"
#include "performance.h"
#include "montecarlo.h"
#include "bs.h"
#include "pnl/pnl_random.h"
#include <cstdio>
#include <ctime>
#include <iostream>
#pragma once
#define DLLEXP   __declspec( dllexport )
namespace Engine{
   DLLEXP void price(double &prix, double &ic, char *type, int size, double* spot, double strike, double maturity, double* sigma, double r, double* rho, double *coeff, int timeStep, int samples);
}