#include "option.h"
#include "pnl/pnl_matrix.h"
#include "pnl/pnl_vector.h"
#ifndef barrier_ulH
#define barrier_ulH

/*!
 *  \file	barrier_u.h
 *  \brief	Header de la classe fille d'Option: barrier_u
 *  \author Equipe 11
 */

/*!
 * \class Barrier_u
 * \brief Classe representant l'option barriere haute
 */
class Barrier_u : public Option {

  private:
	double Strike_; /*!< strike du sous-jacent */
	PnlVect *Coeff_; /*!< payoff coefficients */
	PnlVect *Bu_; /*!< barriere haute */

  public:

	/*!
	 * \brief Constructeur par defaut
	 *
	 * Constructeur par defaut de la classe barrier_u
	 */
	Barrier_u();

	/*!
	 * \brief Constructeur avec argument
	 *
	 * Constructeur avec argument de la classe barrier_u
	 *
	 * \param pars: Liste des donnees relatives à l'option du Parser
	 */
	Barrier_u(Parser &pars);

	/*!
	 * \brief Destructeur
	 *
	 * Destructeur de la classe barrier_u
	 */
	virtual ~Barrier_u();

	/*!
	 * \brief Accesseur de Strike_
	 *
	 *  Acceder au strike du sous-jacent
	 *
	 * \return le strike du sous-jacent 
	 */
	double get_Strike();

	/*!
	 * \brief Accesseur de Coeff_
	 *
	 *  Acceder au vecteur des coefficients des payoff du sous-jacent
	 *
	 * \return le vecteur des coefficients des payoff
	 */
	PnlVect* get_Coeff();

	/*!
	 * \brief Accesseur de Bu_
	 *
	 *  Acceder au vecteur de la barriere haute basse du sous-jacent
	 *
	 * \return le vecteur de la barriere haute basse du sous-jacent
	 */
	PnlVect* get_Bu();

	/*!
	 * \brief Mutateur de Strike_
	 *
	 * Modifie la valeur du strike du sous-jacent
	 *
	 * \param Strike: nouveau strike
	 */
	void set_Strike(double Strike);

	/*!
	 * \brief Mutateur de Coeff_
	 *
	 * Modifie le vecteur des coefficients des payoff du sous-jacent 
	 *
	 * \param Coeff: nouveau vecteur des coefficients des payoff
	 */
	void set_Coeff(PnlVect *Coeff);

	/*!
	 * \brief Mutateur de Bu_
	 *
	 * Modifie la vecteur de la barriere haute basse du sous-jacent
	 *
	 * \param Bu: nouveau vecteur de la barriere haute basse du sous-jacent
	 */
	void set_Bu(PnlVect *Bl);

	/*!
	 * \brief Payoff option barriere haute
	 *
	 * Calcul la valeur du payoff du sous-jacent barriere haute sur la trajectoire passee en parametre
	 *
	 * \param path: matrice de taille d x (N+1) contenant une trajectoire du modele telle que creee par la fonction asset
	 * \return payoff du sous-jacent barriere haute
	 */
	double payoff (const PnlMat *path) ;
};
#endif
