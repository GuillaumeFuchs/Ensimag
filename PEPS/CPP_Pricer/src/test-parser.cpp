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

/*!
 * \file parser.cpp
 * \brief Fichier de test pour le parser
 * \author equipe 11
 */

int main(int argc, char **argv)
{
  if (argc < 2){
	printf("Argument manquant\n");
	return 0;
  }
  Parser mon_parser = Parser(argv[1]);
  char* type = mon_parser.getString("option type");
  BS bs(mon_parser);
  PnlRng *rng = pnl_rng_create(PNL_RNG_MERSENNE);
  pnl_rng_sseed(rng, time(NULL));

  printf("RESULTAT: %s \n", argv[1]);
  if (!strcmp("basket", type)){
    
    Basket opt(mon_parser);
    MonteCarlo mc(&bs, &opt, rng, 0.1, mon_parser.getInt("sample number") );

    printf("\n**************** OPTION BAKSET *********************\n");
    printf("*** Maturité: %f \n*** Nombre de pas de temps de discretisation (TimeSteps): %d \n*** Dimension du modele (Size): %d \n*** Strike du sous-jacent (Strike): %f\n",opt.get_T(), opt.get_timesteps(), opt.get_size(), opt.get_Strike());
    printf("*** Coefficient des payoff du sous-jacent (Coeff):\n");
    pnl_vect_print(opt.get_Coeff());
    printf("\n************* MODELE DE BLACK-SCHOLES **************\n");
    printf("*** Taille du sous-jacent (Size): %d\n *** Taux d'interet du sous-jacent: %f\n *** Parametre de correlation du sous-jacent: %f\n",bs.get_size(), bs.get_r(), bs.get_rho()); 
    printf("*** Vecteur de volatilite sigma:\n");
    pnl_vect_print(bs.get_sigma());
    printf("*** Valeurs initiales du sous-jacent (Spot):\n");
    pnl_vect_print(bs.get_spot());
    printf("*** Taux d'interet sous probabilite historique (Trend): \n");
    pnl_vect_print(bs.get_trend());
    printf("*** Matrice de Cholesky associee a la correlation du sous-jacent:\n");
    pnl_mat_print(bs.get_cho());
    printf("\n************ MODELE DE MONTE-CARLO ****************\n");
    printf("*** Nombre de tirages de MonteCarlo (Samples): %d\n", mc.get_samples());
    printf("*** Pas de difference finie (h): %f\n", mc.get_h());

  }else if (!strcmp("asian", type)){
    Asian opt(mon_parser);
    MonteCarlo mc(&bs, &opt, rng,1.0,mon_parser.getInt("sample number") );

    printf("\n***************** OPTION ASIAN *********************\n");
    printf("*** Maturité: %f \n *** Nombre de pas de temps de discretisation (TimeSteps): %d \n *** Dimension du modele (Size): %d\n *** Strike du sous_jacent: %f\n",opt.get_T(), opt.get_timesteps(), opt.get_size(), opt.get_Strike());
    
    printf("\n************* MODELE DE BLACK-SCHOLES **************\n");
    printf("*** Taille du sous-jacent (Size): %d\n *** Taux d'interet du sous-jacent: %f\n *** Parametre de correlation du sous-jacent: %f\n",bs.get_size(), bs.get_r(), bs.get_rho()); 
    printf("*** Vecteur de volatilite sigma:\n");
    pnl_vect_print(bs.get_sigma());
    printf("*** Valeurs initiales du sous-jacent (Spot):\n");
    pnl_vect_print(bs.get_spot());
    //printf("*** Taux d'interet sous probabilite historique (Trend): \n");
    //pnl_vect_print(bs.get_trend());
    printf("*** Matrice de Cholesky associee a la correlation du sous-jacent:\n");
    pnl_mat_print(bs.get_cho());
    printf("\n************ MODELE DE MONTE-CARLO ****************\n");
    printf("*** Nombre de tirages de MonteCarlo (Samples): %d\n", mc.get_samples());
    printf("*** Pas de difference finie (h): %f\n", mc.get_h());

  }else if (!strcmp("barrier_l", type)){
    
    Barrier_l opt(mon_parser);
    MonteCarlo mc(&bs, &opt, rng,1.0,mon_parser.getInt("sample number") );
    mc.set_h(1.0);

    printf("\n************* OPTION BARRIERE BASSE *****************\n");
    printf("*** Maturité: %f \n *** Nombre de pas de temps de discretisation (TimeSteps): %d \n *** Dimension du modele (Size): %d\n *** Strike du sous_jacent: %f\n",opt.get_T(), opt.get_timesteps(), opt.get_size(), opt.get_Strike());
    printf("*** Coefficient des payoff du sous-jacent: \n");
    pnl_vect_print(opt.get_Coeff());
    printf("*** Barriere basse: \n");
    pnl_vect_print(opt.get_Bl());
    
    printf("\n************* MODELE DE BLACK-SCHOLES **************\n");
    printf("*** Taille du sous-jacent (Size): %d\n *** Taux d'interet du sous-jacent: %f\n *** Parametre de correlation du sous-jacent: %f\n",bs.get_size(), bs.get_r(), bs.get_rho()); 
    printf("*** Vecteur de volatilite sigma:\n");
    pnl_vect_print(bs.get_sigma());
    printf("*** Valeurs initiales du sous-jacent (Spot):\n");
    pnl_vect_print(bs.get_spot());
    //printf("*** Taux d'interet sous probabilite historique (Trend): \n");
    //pnl_vect_print(bs.get_trend());
    printf("*** Matrice de Cholesky associee a la correlation du sous-jacent:\n");
    pnl_mat_print(bs.get_cho());
    printf("\n************ MODELE DE MONTE-CARLO ****************\n");
    printf("*** Nombre de tirages de MonteCarlo (Samples): %d\n", mc.get_samples());
    printf("*** Pas de difference finie (h): %f\n", mc.get_h());


  }else if (!strcmp("barrier_u", type)){
    
    Barrier_u opt(mon_parser);
    MonteCarlo mc(&bs, &opt, rng,1.0,mon_parser.getInt("sample number") );

    printf("\n************* OPTION BARRIERE HAUTE *****************\n");
    printf("*** Maturité: %f \n *** Nombre de pas de temps de discretisation (TimeSteps): %d \n *** Dimension du modele (Size): %d\n *** Strike du sous_jacent: %f\n",opt.get_T(), opt.get_timesteps(), opt.get_size(), opt.get_Strike());
    printf("*** Coefficient des payoff du sous-jacent: \n");
    pnl_vect_print(opt.get_Coeff());
    printf("*** Barriere haute: \n");
    pnl_vect_print(opt.get_Bu());
    
    printf("\n************* MODELE DE BLACK-SCHOLES **************\n");
    printf("*** Taille du sous-jacent (Size): %d\n *** Taux d'interet du sous-jacent: %f\n *** Parametre de correlation du sous-jacent: %f\n",bs.get_size(), bs.get_r(), bs.get_rho()); 
    printf("*** Vecteur de volatilite sigma:\n");
    pnl_vect_print(bs.get_sigma());
    printf("*** Valeurs initiales du sous-jacent (Spot):\n");
    pnl_vect_print(bs.get_spot());
    //printf("*** Taux d'interet sous probabilite historique (Trend): \n");
    //pnl_vect_print(bs.get_trend());
    printf("*** Matrice de Cholesky associee a la correlation du sous-jacent:\n");
    pnl_mat_print(bs.get_cho());
    printf("\n************ MODELE DE MONTE-CARLO ****************\n"); 
    printf("*** Nombre de tirages de MonteCarlo (Samples): %d\n", mc.get_samples());
    printf("*** Pas de difference finie (h): %f\n", mc.get_h());

  }else if (!strcmp("barrier_l", type)){
    Barrier opt(mon_parser);
    MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );

    printf("\n************* OPTION BARRIERE BASSE *****************\n");
    printf("*** Maturité: %f \n *** Nombre de pas de temps de discretisation (TimeSteps): %d \n *** Dimension du modele (Size): %d\n *** Strike du sous_jacent: %f\n",opt.get_T(), opt.get_timesteps(), opt.get_size(), opt.get_Strike());
    printf("*** Coefficient des payoff du sous-jacent: \n");
    pnl_vect_print(opt.get_Coeff());
    printf("*** Barriere basse: \n");
    pnl_vect_print(opt.get_Bl());
    
    printf("\n************* MODELE DE BLACK-SCHOLES **************\n");
    printf("*** Taille du sous-jacent (Size): %d\n *** Taux d'interet du sous-jacent: %f\n *** Parametre de correlation du sous-jacent: %f\n",bs.get_size(), bs.get_r(), bs.get_rho()); 
    printf("*** Vecteur de volatilite sigma:\n");
    pnl_vect_print(bs.get_sigma());
    printf("*** Valeurs initiales du sous-jacent (Spot):\n");
    pnl_vect_print(bs.get_spot());
    //printf("*** Taux d'interet sous probabilite historique (Trend): \n");
    //pnl_vect_print(bs.get_trend());
    printf("*** Matrice de Cholesky associee a la correlation du sous-jacent:\n");
    pnl_mat_print(bs.get_cho());
    printf("\n************ MODELE DE MONTE-CARLO ****************\n");
    printf("*** Nombre de tirages de MonteCarlo (Samples): %d\n", mc.get_samples());
    printf("*** Pas de difference finie (h): %f\n", mc.get_h());


  }else if (!strcmp("barrier", type)){
    printf("**************** OPTION BARRIERE *********************\n");
    
    Barrier opt(mon_parser);
    MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );

    printf("*** Maturité: %f \n *** Nombre de pas de temps de discretisation (TimeSteps): %d \n *** Dimension du modele (Size): %d\n *** Strike du sous_jacent: %f\n",opt.get_T(), opt.get_timesteps(), opt.get_size(), opt.get_Strike());
    printf("*** Coefficient des payoff du sous-jacent: \n");
    pnl_vect_print(opt.get_Coeff());
    printf("*** Barriere basse: \n");
    pnl_vect_print(opt.get_Bl());
    printf("*** Barriere haute: \n");
    pnl_vect_print(opt.get_Bu());
    
    printf("\n************* MODELE DE BLACK-SCHOLES **************\n");
    printf("*** Taille du sous-jacent (Size): %d\n *** Taux d'interet du sous-jacent: %f\n *** Parametre de correlation du sous-jacent: %f\n",bs.get_size(), bs.get_r(), bs.get_rho()); 
    printf("*** Vecteur de volatilite sigma:\n");
    pnl_vect_print(bs.get_sigma());
    printf("*** Valeurs initiales du sous-jacent (Spot):\n");
    pnl_vect_print(bs.get_spot());
    //printf("*** Taux d'interet sous probabilite historique (Trend): \n");
    //pnl_vect_print(bs.get_trend());
    printf("*** Matrice de Cholesky associee a la correlation du sous-jacent:\n");
    pnl_mat_print(bs.get_cho());
    printf("\n************ MODELE DE MONTE-CARLO ****************\n");
    printf("*** Nombre de tirages de MonteCarlo (Samples): %d\n", mc.get_samples());
    printf("*** Pas de difference finie (h): %f\n", mc.get_h());


    
  }else{
    Performance opt(mon_parser);
    MonteCarlo mc(&bs, &opt, rng, 0.1,mon_parser.getInt("sample number") );

    printf("\n************* OPTION PERFORMANCE *****************\n");
    printf("*** Maturité: %f \n *** Nombre de pas de temps de discretisation (TimeSteps): %d \n *** Dimension du modele (Size): %d\n",opt.get_T(), opt.get_timesteps(), opt.get_size());
    printf("*** Coefficient des payoff du sous-jacent: \n");
    pnl_vect_print(opt.get_Coeff());
    
    printf("\n************* MODELE DE BLACK-SCHOLES **************\n");
    printf("*** Taille du sous-jacent (Size): %d\n *** Taux d'interet du sous-jacent: %f\n *** Parametre de correlation du sous-jacent: %f\n",bs.get_size(), bs.get_r(), bs.get_rho()); 
    printf("*** Vecteur de volatilite sigma:\n");
    pnl_vect_print(bs.get_sigma());
    printf("*** Valeurs initiales du sous-jacent (Spot):\n");
    pnl_vect_print(bs.get_spot());
    //printf("*** Taux d'interet sous probabilite historique (Trend): \n");
    //pnl_vect_print(bs.get_trend());
    printf("*** Matrice de Cholesky associee a la correlation du sous-jacent:\n");
    pnl_mat_print(bs.get_cho());
    printf("\n************ MODELE DE MONTE-CARLO ****************\n");
    printf("*** Nombre de tirages de MonteCarlo (Samples): %d\n", mc.get_samples());
    printf("*** Pas de difference finie (h): %f\n", mc.get_h());

  }

  //pnl_rng_free(&rng);
  //  free(bs);
  return 0;
}
