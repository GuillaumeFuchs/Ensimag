#include "Option.h"
#include "pnl/pnl_matrix.h"
#include "pnl/pnl_vector.h"
#ifndef PerformanceH
#define PerformanceH

/*!
 *  \file	performance.h
 *  \brief	Header de la classe fille d'Option: performance
 *  \author Equipe 11
 */

/*!
 * \class Performance
 * \brief Classe representant l'option performance
 */
class Performance : public Option {

  private:
	PnlVect *Coeff_; /*! payoff coefficients */

  public:

	/*!
	 * \brief Constructeur par defaut
	 *
	 * Constructeur par defaut de la classe performance
	 */
	Performance();
	Performance(double* coeff, double T, int timeStep, int size);

	/*!
	 * \brief Destructeur
	 *
	 * Destructeur de la classe performance
	 */
	~ Performance();

	/*!
	 * \brief Accesseur de Coeff_
	 *
	 *  Acceder au vecteur des coefficients des payoff du sous-jacent
	 *
	 * \return le vecteur des coefficients des payoff
	 */
	PnlVect* get_Coeff() const;

	/*!
	 * \brief Mutateur de Coeff_
	 *
	 * Modifie le vecteur des coefficients des payoff du sous-jacent 
	 *
	 * \param Coeff: nouveau vecteur des coefficients des payoff
	 */
	void set_Coeff(PnlVect *Coeff);

	/*!
	 * \brief Payoff option performance
	 *
	 * Calcul la valeur du payoff du sous-jacent performance sur la trajectoire passee en parametre
	 *
	 * \param path: matrice de taille d x (N+1) contenant une trajectoire du modele telle que creee par la fonction asset
	 * \return payoff du sous-jacent performance
	 */
	double payoff (const PnlMat *path) const;
};
#endif
