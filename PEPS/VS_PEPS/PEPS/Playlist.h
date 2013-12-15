#include "pnl/pnl_matrix.h"
#include "pnl/pnl_vector.h"
#ifndef PlaylistH
#define PlaylistH

/*!
 *  \file	playlist.h
 *  \brief	Header de la classe fille d'Option: playlist
 *  \author Equipe 11
 */

/*!
 * \class Playlist
 * \brief Classe representant le produit structuré Playlist
 */
class Playlist{

  private:
	double T_; /*!< maturite fixé à 1an par défault */
	int timeStep_; /*!< nombre de pas de temps de discretisation pour 1 année */
	int size_; /*!< dimension du modele, redondant avec option::size_, fixé à 4 */
	double strike_; /*!<Valeur d'une part normalement soit 150€*/

  public:

	/*!
	 * \brief Constructeur par defaut
	 *
	 * Constructeur par defaut de la classe basket
	 */
	Playlist();

	/*!
	 * \brief Constructeur avec argument
	 *
	 * Constructeur avec argument de la classe basket
	 *
	 * \param pars: Liste des donnees relatives a l'option du Parser
	 */
	Playlist(double T, int timeStep, int size, double strike);
	/*!
	 * \brief Destructeur
	 *
	 * Destructeur de la classe basket
	 */
	virtual ~Playlist();

	/*!
	 * \brief Accesseur de T__
	 *
	 *  Acceder à la maturité de l'option
	 *
	 * \return la maturité de l'option 
	 */
	double get_T();

	/*!
	 * \brief Accesseur de timeStep_
	 *
	 *  Acceder au nombre de pas de discrétisation par an de l'option
	 *
	 * \return le nombre de pas de discrétisation de l'option 
	 */
	int get_timeStep();

	/*!
	 * \brief Accesseur de size_
	 *
	 *  Acceder au nombre de sous ajcents de l'option
	 *
	 * \return le nombre de sous jacents de l'option 
	 */
	int get_size();

	/*!
	 * \brief Accesseur de strike_
	 *
	 *  Acceder au strike de l'option
	 *
	 * \return le strike de l'option 
	 */
	double get_strike();

	/*!
	 * \brief Mutateur de T_
	 *
	 * Modifie la valeur de la maturité de l'option
	 *
	 * \param T: nouvelle maturité
	 */
	void set_T(double T);

	/*!
	 * \brief Mutateur de timeStep
	 *
	 * Modifie la valeur du nombre de discrétisation de l'option
	 *
	 * \param timeStep : nouveau nombre de dsicrétisation
	 */
	void set_timeStep(int timeStep);

	/*!
	 * \brief Mutateur de size_
	 *
	 * Modifie la valeur du nombre de sous jacents de l'option
	 *
	 * \param Size: nouveau nombre de sous jacent
	 */
	void set_size(int size);

	/*!
	 * \brief Mutateur de strike_
	 *
	 * Modifie la valeur du strike de l'option
	 *
	 * \param Strike: nouveau strike
	 */
	void set_strike(double strike);

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