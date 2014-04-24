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
	 * \brief Calcule du prix de l’option a la date 0 en utilisant uniquement le CPU
	 *
	 * \param prix_cpu (ouptut) contient le prix
	 * \param ic_cpu (ouptut) contient la largeur au centre de l’intervalle de confiance sur le calcul du prix
	 * \param time_cpu (ouptut) temps d'exécution via le calcul uniquement par CPU
	 */
	void priceCPU (double &prix_cpu, double &ic_cpu, double &time_cpu);	
	/*!
	 * \brief Calcule du prix de l’option a la date 0 en utilisant le CPU et le GPU
	 *
	 * \param prix_gpu (ouptut) contient le prix
	 * \param ic_gpu (ouptut) contient la largeur au centre de l’intervalle de confiance sur le calcul du prix
	 * \param time_gpu (ouptut) temps d'exécution via le calcul uniquement par CPU & GPU
	 * \param ic_target (input) si différent de 0.0 alors la fonction calcul le prix en ayant un interval de confiance inférieur à celui renseigné par ic_target
	 */
	void priceGPU (double &prix_gpu, double &ic_gpu, double &time_gpu, const double ic_target);
};
#endif
