#include "basket.h"
#include "asian.h"
#include "barrier_l.h"
#include "barrier_u.h"
#include "barrier.h"
#include "performance.h"
#include "montecarlo.h"
#include "bs.h"
#include "pnl/pnl_random.h"
#include "pnl/pnl_vector.h"
#include <cstdio>
#include "parser.h"
#include <assert.h>
#include <ctime>
#include <iostream>

/*!
* \file pricer.cpp
* \brief Fichier de test pour le price d'option à t=0
*/

void price_compute(char *file, PnlRng* rng)
{
	int m = 0;
	double prix_cpu, ic_cpu, time_cpu;
	double prix_gpu, ic_gpu, time_gpu;
	double ic_target = 0.01;

	Parser mon_parser = Parser(file);
	BS *bs = new BS(mon_parser);
	Option *opt;
	const char* type = mon_parser.getString("option type");

	if (!strcmp("asian", type)){
		opt = new Asian(mon_parser);
	}else if (!strcmp("barrier", type)){
		opt = new Barrier(mon_parser);
	}else if (!strcmp("barrier_l", type)){
		opt = new Barrier_l(mon_parser);
	}else if (!strcmp("barrier_u", type)){
		opt = new Barrier_u(mon_parser);
	}else if (!strcmp("basket", type)){
		opt = new Basket(mon_parser);
	}else{
		opt = new Performance(mon_parser);
		
	}
	MonteCarlo mc(bs, opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.priceCPU(prix_cpu, ic_cpu, time_cpu);
	mc.priceGPU(prix_gpu, ic_gpu, time_gpu, 0.01);
	if (fabs(ic_target)>0.00001)
		m = mc.get_samples();

	printf("\n");
	printf("%s\n", file);
	printf("\nCPU:\n");
	printf("  Prix: %f \n", prix_cpu);
	printf("  Ic: %f \n", ic_cpu);
	printf("  Time: %f \n", time_cpu);
	printf("\nGPU:\n");
	printf("  Prix: %f \n", prix_gpu);
	printf("  Ic: %f \n", ic_gpu);
	printf("  Time: %f \n", time_gpu);
	if (fabs(ic_target)>0.00001)
		printf("  Iteration: %d \n", m);
	printf("\n");
}

int main(int argc, char **argv)
{
	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
	pnl_rng_sseed(rng, time(NULL));

	printf("*********************************************\n");
	printf("**               RESULTATS                 **\n"); 
	printf("*********************************************\n");

	/*
	price_compute("exemples/call.dat", rng);
	price_compute("exemples/basket_5d.dat", rng);
	price_compute("exemples/asian.dat", rng);
	price_compute("exemples/barrier.dat", rng);
	price_compute("exemples/barrier_l.dat", rng);
	price_compute("exemples/barrier_l2.dat", rng);
	price_compute("exemples/barrier_u.dat", rng);
	price_compute("exemples/barrier_u2.dat", rng);
	price_compute("exemples/basket_1.dat", rng);
	price_compute("exemples/basket_2.dat", rng);
	price_compute("exemples/perf.dat", rng);
	price_compute("exemples/put.dat", rng);*/

	assert(argc > 1);

	int m = 0;
	double prix_cpu, ic_cpu, time_cpu;
	double prix_gpu, ic_gpu, time_gpu;
	double ic_target = 0.0;
	if (argc == 3)
		ic_target = atof(argv[2]);

	Parser mon_parser = Parser(argv[1]);
	BS *bs = new BS(mon_parser);
	Option *opt;
	const char* type = mon_parser.getString("option type");

	if (!strcmp("asian", type)){
		opt = new Asian(mon_parser);
	}else if (!strcmp("barrier", type)){
		opt = new Barrier(mon_parser);
	}else if (!strcmp("barrier_l", type)){
		opt = new Barrier_l(mon_parser);
	}else if (!strcmp("barrier_u", type)){
		opt = new Barrier_u(mon_parser);
	}else if (!strcmp("basket", type)){
		opt = new Basket(mon_parser);
	}else{
		opt = new Performance(mon_parser);
		
	}
	MonteCarlo mc(bs, opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.priceCPU(prix_cpu, ic_cpu, time_cpu);
	mc.priceGPU(prix_gpu, ic_gpu, time_gpu, ic_target);
	if (fabs(ic_target)>0.00001)
		m = mc.get_samples();

	printf("\n");
	printf("%s\n", argv[1]);
	printf("\nCPU:\n");
	printf("  Prix: %f \n", prix_cpu);
	printf("  Ic: %f \n", ic_cpu);
	printf("  Time: %f \n", time_cpu);
	printf("\nGPU:\n");
	printf("  Prix: %f \n", prix_gpu);
	printf("  Ic: %f \n", ic_gpu);
	printf("  Time: %f \n", time_gpu);
	if (argc == 3)
		printf("  Iteration: %d \n", m);
	printf("\n");

	pnl_rng_free(&rng);
	return 0;
}
