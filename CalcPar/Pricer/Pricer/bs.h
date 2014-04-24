#ifndef BSH
#define BSH
#include <pnl/pnl_vector.h>
#include <pnl/pnl_random.h>
#include "parser.h"
#include "device_launch_parameters.h"

/*!
 *  \file	bs.h
 *  \brief	Header de la classe BS
 */

/*!
 * \class BS
 * \brief Classe representant le modele de Black Scholes
 */
class BS {
  private:
	int size_; /*!< nombre d’actifs du modele */
	double r_; /*!< taux d’interet */
	double rho_; /*!< parametre de correlation */
	PnlVect *sigma_; /*!< vecteur de volatilites */
	PnlVect *spot_; /*!< valeurs initiales du sous-jacent */
	PnlMat *Cho_; /*!< Matrice de Cholesky utilise pour la correlation*/
	PnlVect *Gi_; /*!< Vecteur gaussien centré du modele de BS multidimensionnel*/
	PnlVect *Ld_; /*!< Ligne d de la matrice de Cholesky Cho_*/

	float *d_sigma; /*!< vecteur de volatilites */
	float *d_spot; /*!< valeurs initiales du sous-jacent */
	float *d_cho; /*!< Matrice de Cholesky utilise pour la correlation*/

  public:

	/*!
	 * \brief Constructeur par defaut
	 *
	 * Constructeur par defaut de la classe bs
	 */
	BS();

	/*!
	 * \brief Constructeur avec argument
	 *
	 * Constructeur avec argument de la classe bs
	 *
	 * \param pars: Liste des donnees relatives à l'option du Parser
	 */
	BS(Parser &pars);

	/*!
	 * \brief Destructeur
	 *
	 * Destructeur de la classe bs
	 */
	~BS();

	/*!
	 * \brief Accesseur de size_
	 *
	 *  Acceder à la taille du sous-jacent
	 *
	 * \return la taille du sous-jacent
	 */
	int get_size();

	/*!
	 * \brief Accesseur de r_
	 *
	 *  Acceder au taux d'interet du sous-jacent
	 *
	 * \return le taux d'interet du sous-jacent
	 */
	double get_r();

	/*!
	 * \brief Accesseur de rho_
	 *
	 *  Acceder au parametre de correlation du sous-jacent
	 *
	 * \return le parametre de correlation du sous-jacent 
	 */
	double get_rho();

	/*!
	 * \brief Accesseur de Sigma_
	 *
	 *  Acceder au vecteur de volatilites
	 *
	 * \return le vecteur de volatilites
	 */
	PnlVect *get_sigma();

	/*!
	 * \brief Accesseur de Spot_
	 *
	 *  Acceder aux valeurs initiales du sous-jacent
	 *
	 * \return la valeurs initiales du sous-jacent
	 */
	PnlVect *get_spot();

	/*!
	 * \brief Accesseur de Cho_
	 *
	 *  Acceder a la matrice de cholesky associee a la correlation du sous-jacent
	 *
	 *  \return la matrice de cholesky associee a la correlation du sous-jacent
	 */
	PnlMat *get_cho();

	/*!
	 * \brief Accesseur de Gi_
	 *
	 *  Acceder au vecteur gaussien centre du modele de BS
	 *
	 *  \return le vecteur gaussien centré du modele de BS
	 */
	PnlVect *get_gi();

	/*!
	 * \brief Accesseur de Ld_
	 *
	 *  Acceder a la ligne d de la matrice de Cholesky Cho_
	 *
	 *  \return la ligne d de la matrice de Cholesky Cho_
	 */
	PnlVect *get_ld();

	/*!
	 * \brief Mutateur de size_
	 *
	 *  Modifier la taille du sous-jacent
	 *
	 * \param la nouvelle taille du sous-jacent
	 */
	void set_size(int size);

	/*!
	 * \brief Mutateur de r_
	 *
	 *  Modifier le taux d'interet du sous-jacent
	 *
	 * \param le nouveau taux d'interet du sous-jacent
	 */
	void set_r(double r);

	/*!
	 * \brief Mutateur de rho_
	 *
	 *  Modifier au parametre de correlation du sous-jacent
	 *
	 * \param le nouveau parametre de correlation du sous-jacent 
	 */
	void set_rho(double rho);

	/*!
	 * \brief Mutateur de Sigma_
	 *
	 *  Modifier au vecteur de volatilites
	 *
	 * \param le nouveau vecteur de volatilites
	 */
	void set_sigma(PnlVect *sigma);

	/*!
	 * \brief Mutateur de Spot_
	 *
	 *  Modifier aux valeurs initiales du sous-jacent
	 *
	 * \param la nouvelle valeurs initiales du sous-jacent
	 */
	void set_spot(PnlVect *spot);

	/*!
	 * \brief Mutateur de Cho_
	 *
	 *  Modifier la matrice de cholesky associee a la correlation du sous-jacent
	 *
	 *  \param la nouvelle matrice de cholesky associee a la correlation du sous-jacent
	 */
	void set_cho(PnlMat *Cho);

	/*!
	 * \brief Mutateur de Gi_
	 *
	 *  Modifier le vecteur gaussien centre du modele de BS
	 *
	 *  \param le nouveau vecteur gaussien centre du modele de BS
	 */
	PnlVect *set_gi();
	void set_gi(PnlVect *Gi);

	/*!
	 * \brief Mutateur de Ld_
	 *
	 *  Modifier la ligne d de la matrice de Cholesky Cho_
	 *
	 *  \param la nouvelle ligne d de la matrice de Cholesky Cho_
	 */
	void set_ld(PnlVect *Ld);

	/*!
	 * \brief Génère une trajectoire du modele et la stocke dans path
	 *
	 * \param path contient une trajectoire du modele. C’est une matrice de dimension d x (N+1)
	 * \param T  maturite
	 * \param N nombre de dates de constatation
	 * \param rng pointeur sur le generateur de nombre aleatoire
	 * \param G contient N vecteurs gaussiens centres de matrice de covariance identite
	 * \param grid contient les indices de temps utilises pour l'evolution du sous-jacent
	 */
	void asset(PnlMat *path, double T,  int N, PnlRng *rng, PnlMat* G, PnlVect* grid) ;
	void assetGPU(dim3 dimGrid, dim3 dimBlock, int samples, int N, float T, float* d_path, float* d_rand);
};
#endif
