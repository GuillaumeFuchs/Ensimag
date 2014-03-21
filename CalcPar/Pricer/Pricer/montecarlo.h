#ifndef MonteCarloH
#define MonteCarloH
#include "parser.h"
#include "bs.h"
#include "option.h"
#include <pnl/pnl_random.h>

/*!
 *  \file	montecarlo.h
 *  \brief	Header de la classe MonteCarlo
 *  \author Equipe 11
 */

/*!
 * \class MonteCarlo
 * \brief Classe representant le pricer Monte Carlo
 */

class MonteCarlo {
  private:
	BS *mod_; /*!< pointeur vers le modele */
	Option *opt_; /*!< pointeur sur l’option */
	PnlRng *rng; /*!< pointeur sur le generateur */
	double h_; /*!< pas de difference finie */
	int samples_; /*!< nombre de tirages Monte Carlo */

  public:

	/*!
	 * \brief Constructeur par defaut
	 *
	 * Constructeur par defaut de la classe montecarlo
	 * \param rng: generateur aleatoire
	 */
	MonteCarlo(PnlRng *rng);


	/*!
	 * \brief Constructeur avec argument
	 *
	 Constructeur avec argument de la classe montecarlo
	 *
	 * \param mod_: pointeur vers le modele 
	 * \param opt_: pointeur sur l'option 
	 * \param rng: pointeur sur le generateur
	 * \param h_: pas de difference finie
	 * \param samples_: nombre de tirages Monte Carlo
	 */
	MonteCarlo(BS *mod_, Option *opt_, PnlRng *rng, double h_, int samples_);


	/*!
	 * \brief Destructeur
	 *
	 * Destructeur de la classe bs
	 */
	~MonteCarlo();


	/*!
	 * \brief Accesseur de mod_
	 *
	 *  Acceder au modele du sous-jacent
	 *
	 *  \return le modele du sous-jacent
	 */
	BS* get_mod();

	/*!
	 * \brief Accesseur de opt_
	 *
	 *  Acceder à l'option
	 *
	 * \return l'option
	 */
	Option* get_opt();

	/*!
	 * \brief Accesseur de rng_
	 *
	 *  Acceder au generateur
	 *
	 * \return le generateur
	 */
	PnlRng* get_rng();

	/*!
	 * \brief Accesseur de h_
	 *
	 *  Acceder au pas de difference finie
	 *
	 * \return le pas de difference finie
	 */
	double get_h();

	/*!
	 * \brief Accesseur de samples_
	 *
	 *  Acceder au nombre de tirages Monte Carlo
	 *
	 * \return le nombre de tirage Monte Carlo
	 */
	int get_samples();



	/*!
	 * \brief Mutateur de mod_
	 *
	 *  Modifier le modele
	 *
	 * \param le nouveau modele
	 */
	void set_mod(BS* mod);

	/*!
	 * \brief Mutateur de opt_
	 *
	 *  Modifier l'option
	 *
	 * \param la nouvelle option
	 */
	void set_opt(Option* opt);

	/*!
	 * \brief Mutateur de rng_
	 *
	 *  Modifier le generateur
	 *
	 * \param le nouveau generateur
	 */
	void set_rng(PnlRng* rng);

	/*!
	 * \brief Mutateur de h_
	 *
	 *  Modifier le pas de difference finie
	 *
	 * \param le nouveau pas de difference finie
	 */
	void set_h(double h);

	/*!
	 * \brief Mutateur de samples_
	 *
	 *  Modifier le nombre de tirages Monte Carlo
	 *
	 * \param le nouveau nombre de tirages Monte Carlo
	 */
	void set_samples(int samples);

	/*!
	 * \brief Calcule du prix de l’option a la date 0
	 *
	 * \param prix (ouptut) contient le prix
	 * \param ic (ouptut) contient la largeur de l’intervalle de confiance sur le calcul du prix
	 */
	void price (double &prix, double &ic);	

	/*!
	 * \brief Calcule le prix de l’option a la date t
	 *
	 * \param past (input) contient la trajectoire du sous-jacent jusqu’a l’instant t
	 * \param t (input) date a laquelle le calcul est fait
	 * \param prix (ouptut) contient le prix
	 * \param ic (ouptut) contient la largeur de l’intervalle de confiance sur le calcul du prix
	 */
	void price (const PnlMat *past, double t, double &prix, double &ic);

	/*!
	 * \brief Calcul du delta de l’option a la date t
	 *
	 * \param past (input) contient la trajectoire du sous-jacent jusqu’a l’instant t
	 * \param t (input) date a laquelle le calcul est fait
	 * \param delta (ouptut) contient le vecteur de delta
	 * \param ic (ouptut) contient la largeur de l’intervalle de confiance sur le calcul du delta
	 */
	void delta (const PnlMat *past, double t, PnlVect *delta, PnlVect *ic); 

	/*!
	 * \brief Calcul de l'erreur de couverture
	 *
	 * \param past contient une trajectoire du modele 
	 * \param pl erreur de courverture du portefeuille
	 * \param H nombre de date dans la simulation
	 * \param T maturite du portefeuille
	 */
	void couv (PnlMat *past, double &pl, int H, double T);

};
#endif
