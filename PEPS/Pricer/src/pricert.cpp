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
#include "parser.h"
#include <cstdio>
#include <stdlib.h>
#include <ctime>

/*!
 * \file pricert.cpp
 * \brief Fichier de test pour le price d'option à t>=0
 * \author equipe 11
 */

int main(int argc, char **argv)
{
  if (argc < 3){
	printf("Arguments manquants\n");
	return 0;
  }
  Parser mon_parser = Parser(argv[1]);
  char* type = mon_parser.getString("option type");
  BS bs(mon_parser);
  PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
  pnl_rng_sseed(rng, time(NULL));

  double prix;
  double ic;
  double T = mon_parser.getDouble("maturity");
  int N = mon_parser.getInt("timestep number");
  int size = mon_parser.getInt("option size");
  double t = atof(argv[2]);
  double indice = 0.0;

  if (fabs(fmod(t, T/N)) < 0.00001){
	indice = t/(T/(double)N);
  }else{
	printf("Veuillez donner un temps correspond à un pas d'incrémentation\n Pas d'incrémenation: %f \n", T/(double)N);
	return 0;
  }

  PnlMat* past = pnl_mat_create(size, N+1);
  bs.simul_market(past, N, T, rng);

  PnlMat* past_sub = pnl_mat_create(size, int(indice)+1);
  pnl_mat_extract_subblock(past_sub, past, 0, size, 0, int(indice)+1);

  if (!strcmp("basket", type)){
	Basket opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1, mon_parser.getInt("sample number") );
	mc.price(past_sub, t, prix, ic);

  }else if (!strcmp("asian", type)){
	Asian opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng,1.0,mon_parser.getInt("sample number") );
	mc.price(past_sub, t, prix, ic);

  }else if (!strcmp("barrier_l", type)){
	Barrier_l opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng,1.0,mon_parser.getInt("sample number") );
	mc.price(past_sub, t, prix, ic);

  }else if (!strcmp("barrier_u", type)){
	Barrier_u opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng,1.0,mon_parser.getInt("sample number") );
	mc.price(past_sub, t, prix, ic);

  }else if (!strcmp("barrier", type)){
	Barrier opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );
	mc.price(past_sub, t, prix, ic);

  }else{
	Performance opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng,1.0,mon_parser.getInt("sample number") );	
	mc.price(past_sub, t, prix, ic);


  }
  printf("\n");
  printf("*********************************************\n");
  printf("**               RESULTATS                 **\n"); 
  printf("*********************************************\n");
  printf("%s\n", argv[1]);
  printf("  Prix(%f): %f \n", t, prix);
  printf("  ic: %f \n", ic);
  printf("\n");

  pnl_mat_free(&past);
  pnl_mat_free(&past_sub);
  pnl_rng_free(&rng);
  return 0;
}
