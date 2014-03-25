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
#include <ctime>
#include <iostream>

/*!
 * \file pricer.cpp
 * \brief Fichier de test pour le price d'option à t=0
 */

void price_compute(char *file, PnlRng* rng)
{
  Parser mon_parser = Parser(file);
  const char* type = mon_parser.getString("option type");
  BS bs(mon_parser);

  double prix, ic, time_cpu;
  double prix_gpu, ic_gpu, time_gpu;

  if (!strcmp("basket", type)){
	Basket opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic, time_cpu, prix_gpu, ic_gpu, time_gpu);
  }else if (!strcmp("asian", type)){
	Asian opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic, time_cpu, prix_gpu, ic_gpu, time_gpu);
  }else if (!strcmp("barrier_l", type)){
	Barrier_l opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic, time_cpu, prix_gpu, ic_gpu, time_gpu);
  }else if (!strcmp("barrier_u", type)){
	Barrier_u opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic, time_cpu, prix_gpu, ic_gpu, time_gpu);
  }else if (!strcmp("barrier", type)){
	Barrier opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic, time_cpu, prix_gpu, ic_gpu, time_gpu);
  }else{
	Performance opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic, time_cpu, prix_gpu, ic_gpu, time_gpu);
  }
  printf("\n");
  printf("%s\n", file);
  printf("\nCPU:\n");
  printf("  Prix: %f \n", prix);
  printf("  Ic: %f \n", ic);
  printf("  Time: %f \n", time_cpu);
  printf("\nGPU:\n");
  printf("  Prix: %f \n", prix_gpu);
  printf("  Ic: %f \n", ic_gpu);
  printf("  Time: %f \n", time_gpu);
  printf("\n");

}

int main(int argc, char **argv)
{
  PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
  pnl_rng_sseed(rng, time(NULL));

  printf("*********************************************\n");
  printf("**               RESULTATS                 **\n"); 
  printf("*********************************************\n");

  //price_compute("exemples/call.dat", rng);
  //price_compute("exemples/basket_5d.dat", rng);
  //price_compute("exemples/asian.dat", rng);
  //price_compute("exemples/barrier.dat", rng);
  price_compute("exemples/barrier_l.dat", rng);
  //price_compute("exemples/barrier_l2.dat", rng);
  //price_compute("exemples/barrier_u.dat", rng);
  //price_compute("exemples/barrier_u2.dat", rng);
  //price_compute("exemples/basket_1.dat", rng);
  //price_compute("exemples/basket_2.dat", rng);
  //price_compute("exemples/perf.dat", rng);
  //price_compute("exemples/put.dat", rng);

  pnl_rng_free(&rng);
  return 0;
}
