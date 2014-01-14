#include "Option.h"
#include "pnl/pnl_matrix.h"
#ifndef AsianH
#define AsianH

/*!
 *  \file	asian.h
 *  \brief	Header de la classe fille d'Option: Asian
 *  \author Equipe 11
 */

/*!
 * \class Asian
 * \brief Classe representant l'option asiatique discrete en dimension 1
 */
class Asian : public Option {

  private:
	double strike_; /*!< strike du sous-jacent */

  public:

	/*!
	 * \brief Constructeur par defaut
	 *
	 * Constructeur par defaut de la classe Asian
	 */
	Asian();
	Asian(double strike, double T, int timeStep, int size);


	/*!
	 * \brief Destructeur
	 *
	 * Destructeur de la classe Asian
	 */
	virtual ~Asian();


	/*!
	 * \brief Accesseur de strike_
	 *
	 *  Acceder au strike du sous-jacent
	 *
	 * \return le strike du sous-jacent 
	 */
	double get_Strike() const;


	/*!
	 * \brief Mutateur de strike_
	 *
	 * Modifie la valeur du strike du sous-jacent
	 *
	 * \param Strike: nouveau strike
	 */
	void set_Strike(double Strike);


	/*!
	 * \brief Payoff option asiatique
	 *
	 * Calcul la valeur du payoff du sous-jacent asiatique sur la trajectoire passee en parametre
	 *
	 * \param path: matrice de taille d x (N+1) contenant une trajectoire du modele telle que creee par la fonction asset
	 * \return payoff du sous-jacent asiatique
	 */
	double payoff (const PnlMat *path) const;
};

#endif
