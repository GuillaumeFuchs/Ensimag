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
#include <ctime>
#include "parser.h"

/*!
 * \file couv.cpp
 * \brief Fichier de test pour la couverture
 * \author equipe 11
 */

int main(int argc, char **argv)
{
  if (argc < 4){
	printf("Arguments manquants\n");
	return 0;
  }
  Parser mon_parser = Parser(argv[1]);
  char* type = mon_parser.getString("option type");
  BS bs(mon_parser);
  PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
  pnl_rng_sseed(rng, time(NULL));
  int H = atoi(argv[2]);
  double h = atof(argv[3]);
  double pl;
  PnlMat *past = pnl_mat_create(mon_parser.getInt("option size"), H+1);
  if (!strcmp("basket", type)){
	Basket opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, h, mon_parser.getInt("sample number") );
	mc.couv(past, pl, H, opt.get_T());
  }else if (!strcmp("asian", type)){
	Asian opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, h, mon_parser.getInt("sample number") );
	mc.couv(past, pl, H, opt.get_T());
  }else if (!strcmp("barrier_l", type)){
	Barrier_l opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, h, mon_parser.getInt("sample number") );
	mc.couv(past, pl, H, opt.get_T());
  }else if (!strcmp("barrier_u", type)){
	Barrier_u opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, h, mon_parser.getInt("sample number") );
	mc.couv(past, pl, H, opt.get_T());
  }else if (!strcmp("barrier", type)){
	Barrier opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, h, mon_parser.getInt("sample number") );
	mc.couv(past, pl, H, opt.get_T());
  }else{
	Performance opt(mon_parser);
	MonteCarlo mc(&bs, &opt, rng, h, mon_parser.getInt("sample number") );
	mc.couv(past, pl, H, opt.get_T());
  }
  printf("\n");
  printf("*********************************************\n");
  printf("**               RESULTAT                  **\n"); 
  printf("*********************************************\n");
  printf("%s\n", argv[1]);
  printf("  Couverture: %f \n", pl);
  printf("\n");

  pnl_mat_free(&past);
  pnl_rng_free(&rng);
  return 0;
}
