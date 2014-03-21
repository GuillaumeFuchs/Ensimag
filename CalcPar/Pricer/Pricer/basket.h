#include "option.h"
#include "pnl/pnl_matrix.h"
#include "pnl/pnl_vector.h"
#ifndef BasketH
#define BasketH


/*!
 *  \file	basket.h
 *  \brief	Header de la classe fille d'Option: basket
 *  \author Equipe 11
 */

/*!
 * \class Basket
 * \brief Classe representant l'option panier
 */
class Basket : public Option {

  private:
	double Strike_; /*! strike de l'option */ 
	PnlVect *Coeff_; /*! payoff coefficients */

  public:

	/*!
	 * \brief Constructeur par defaut
	 *
	 * Constructeur par defaut de la classe basket
	 */
	Basket();

	/*!
	 * \brief Constructeur avec argument
	 *
	 * Constructeur avec argument de la classe basket
	 *
	 * \param pars: Liste des donnees relatives a l'option du Parser
	 */
	Basket(Parser& pars);

	/*!
	 * \brief Destructeur
	 *
	 * Destructeur de la classe basket
	 */
	virtual ~Basket();

	/*!
	 * \brief Accesseur de Strike_
	 *
	 *  Acceder au strike de l'option
	 *
	 * \return le strike de l'option 
	 */
	double get_Strike();

	/*!
	 * \brief Accesseur de Coeff_
	 *
	 *  Acceder au vecteur des coefficients des payoff de l'option
	 *
	 * \return le vecteur des coefficients des payoff
	 */
	PnlVect * get_Coeff();

	/*!
	 * \brief Mutateur de Strike_
	 *
	 * Modifie la valeur du strike de l'option
	 *
	 * \param Strike: nouveau strike
	 */
	void set_Strike(double Strike);

	/*!
	 * \brief Mutateur de Coeff_
	 *
	 * Modifie le vecteur des coefficients des payoff de l'option 
	 *
	 * \param Coeff: nouveau vecteur des coefficients des payoff
	 */
	void set_Coeff(PnlVect *Coeff);

	/*!
	 * \brief Payoff option panier
	 *
	 * Calcul la valeur du payoff de l'option panier sur la trajectoire passee en parametre
	 *
	 * \param path: matrice de taille d x (N+1) contenant une trajectoire du modele telle que creee par la fonction asset
	 * \return payoff de l'option panier
	 */
	double payoff (const PnlMat * path);
}; 

#endif 

