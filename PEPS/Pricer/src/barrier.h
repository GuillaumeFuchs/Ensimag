#include "option.h"
#include "pnl/pnl_matrix.h"
#include "pnl/pnl_vector.h"
#ifndef BarrierH
#define BarrierH

/*!
 *  \file	barrier.h
 *  \brief	Header de la classe fille d'Option: Barrier
 *  \author Equipe 11
 */

/*!
 * \class Barrier
 * \brief Classe representant l'option barriere
 */
class Barrier : public Option {

  private:
	double Strike_; /*!< strike du sous-jacent */
	PnlVect *Coeff_; /*!< payoff coefficients */
	PnlVect *Bu_; /*!< upper barrier */
	PnlVect *Bl_; /*!< lower barrier */

  public:

	/*!
	 * \brief Constructeur par defaut
	 *
	 * Constructeur par defaut de la classe barrier
	 */
	Barrier();

	/*!
	 * \brief Constructeur avec argument
	 *
	 * Constructeur avec argument de la classe barrier
	 *
	 * \param pars: Liste des donnees relatives à l'option du Parser
	 */
	Barrier(Parser& pars);

	/*!
	 * \brief Destructeur
	 *
	 * Destructeur de la classe barrier
	 */
	virtual ~Barrier();


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
	 * \brief Accesseur de Bl_
	 *
	 *  Acceder au vecteur de la barriere basse du sous-jacent
	 *
	 * \return le vecteur de la barriere basse du sous-jacent
	 */
	PnlVect* get_Bl();

	/*!
	 * \brief Accesseur de Bu_
	 *
	 *  Acceder au vecteur de la barriere haute du sous-jacent
	 *
	 * \return le vecteur de la barriere haute du sous-jacent
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
	 * \brief Mutateur de Bl_
	 *
	 * Modifie la vecteur de la barriere basse du sous-jacent
	 *
	 * \param Bl: nouveau vecteur de la barriere basse du sous-jacent
	 */
	void set_Bl(PnlVect *Bl);

	/*!
	 * \brief Mutateur de Bu_
	 *
	 * Modifie le vecteur de la barriere haute du sous-jacent
	 *
	 * \param Bu: nouveau vecteur de la barriere haute du sous-jacent
	 */
	void set_Bu(PnlVect *Bu);

	/*!
	 * \brief Payoff option barriere
	 *
	 * Calcul la valeur du payoff du sous-jacent barriere sur la trajectoire passee en parametre
	 *
	 * \param path: matrice de taille d x (N+1) contenant une trajectoire du modele telle que creee par la fonction asset
	 * \return payoff du sous-jacent barriere
	 */
	double payoff (const PnlMat *path);
};
#endif
