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
 * \author equipe 11
 */

int main(int argc, char **argv)
{
  PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
  pnl_rng_sseed(rng, time(NULL));

  Parser mon_parser = Parser("exemples/call.dat");
  const char* type = mon_parser.getString("option type");
  BS bs(mon_parser);

  double prix, ic;
  clock_t tbegin, tend;
  if (!strcmp("basket", type)){
	Basket opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	tbegin = clock();
	mc.price(prix, ic);
	tend = clock();
  }else if (!strcmp("asian", type)){
	Asian opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic);
  }else if (!strcmp("barrier_l", type)){
	Barrier_l opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic);
  }else if (!strcmp("barrier_u", type)){
	Barrier_u opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic);
  }else if (!strcmp("barrier", type)){
	Barrier opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic);
  }else{
	Performance opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic);
  }
  printf("\n");
  printf("*********************************************\n");
  printf("**               RESULTATS                 **\n"); 
  printf("*********************************************\n");
  printf("%s\n", "exemples/call.dat");
  printf("  Prix: %f \n", prix);
  printf("  Ic: %f \n", ic);
  printf("  Time: %f \n", (float)(tend-tbegin)/CLOCKS_PER_SEC);  
  printf("\n");

  system("pause");
  pnl_rng_free(&rng);
  return 0;
}
