//#include "basket.h"
//#include "asian.h"
//#include "barrier_l.h"
//#include "barrier_u.h"
//#include "barrier.h"
//#include "performance.h"
//#include "montecarlo.h"
//#include "bs.h"
//#include "pnl/pnl_random.h"
//#include "pnl/pnl_vector.h"
//#include <cstdio>
//#include "parser.h"
//#include <assert.h>
//#include <ctime>
//#include <iostream>
//
///*!
//* \file pricer.cpp
//* \brief Fichier de test pour le price d'option à t=0
//*/
//
//int main(int argc, char **argv)
//{
//	PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
//	pnl_rng_sseed(rng, time(NULL));
//
//	printf("*********************************************\n");
//	printf("**               RESULTATS                 **\n"); 
//	printf("*********************************************\n");
//	
//	assert(argc > 1);
//
//	double prix_cpu, ic_cpu, time_cpu;
//
//	Parser mon_parser = Parser(argv[1]);
//	BS *bs = new BS(mon_parser);
//	Option *opt;
//	const char* type = mon_parser.getString("option type");
//
//	if (!strcmp("asian", type)){
//		opt = new Asian(mon_parser);
//	}else if (!strcmp("barrier", type)){
//		opt = new Barrier(mon_parser);
//	}else if (!strcmp("barrier_l", type)){
//		opt = new Barrier_l(mon_parser);
//	}else if (!strcmp("barrier_u", type)){
//		opt = new Barrier_u(mon_parser);
//	}else if (!strcmp("basket", type)){
//		opt = new Basket(mon_parser);
//	}else{
//		opt = new Performance(mon_parser);
//		
//	}
//	MonteCarlo mc(bs, opt, rng, 0.1,mon_parser.getInt("sample number") );
//	mc.priceCPU(prix_cpu, ic_cpu, time_cpu);
//
//	printf("\n");
//	printf("%s\n", argv[1]);
//	printf("\nCPU:\n");
//	printf("  Prix: %f \n", prix_cpu);
//	printf("  Ic: %f \n", ic_cpu);
//	printf("  Time: %f \n", time_cpu);
//	printf("\n");
//	
//	pnl_rng_free(&rng);
//	return 0;
//}
