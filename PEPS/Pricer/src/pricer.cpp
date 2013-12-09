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
  if (argc < 2){
	printf("Argument manquant\n");
	return 0;
  }
  PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
  pnl_rng_sseed(rng, time(NULL));

  Parser mon_parser = Parser(argv[1]);
  const char* type = mon_parser.getString("option type");
  BS bs(mon_parser);
  double prix;
  double ic;
  if (!strcmp("basket", type)){
	Basket opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(prix, ic);
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
  printf("%s\n", argv[1]);
  printf("  Prix: %f \n", prix);
  printf("  ic: %f \n", ic);
  printf("\n");

  pnl_rng_free(&rng);
  return 0;
}
