#ifndef BsH
#define BsH
#include <pnl/pnl_vector.h>
#include <pnl/pnl_random.h>

/*!
 * \class Bs
 * \brief Classe representant le modele de Black Scholes
 */
class Bs {
  private:
	int size_; /*!< nombre d’actifs du modele */
	double r_; /*!< taux d’interet */
	double* rho_; /*!< parametre de correlation */
	PnlVect *sigma_; /*!< vecteur de volatilites */
	PnlVect *spot_; /*!< valeurs initiales du sous-jacent */
	PnlVect *trend_; /*!< taux utilise sous la probabilite historique */
	PnlMat *Cho_; /*!< Matrice de Cholesky utilise pour la correlation*/
	PnlVect *Gi_; /*!< Vecteur gaussien centré du modele de Bs multidimensionnel*/
	PnlVect *Ld_; /*!< Ligne d de la matrice de Cholesky Cho_*/

  public:

	/*!
	 * \brief Constructeur par defaut
	 *
	 * Constructeur par defaut de la classe Bs
	 */
	Bs();
	Bs(int size, double r, double* rho, double* sigma, double* spot, double* trend);


	/*!
	 * \brief Destructeur
	 *
	 * Destructeur de la classe Bs
	 */
	~Bs();

	/*!
	 * \brief Accesseur de size_
	 *
	 *  Acceder à la taille du sous-jacent
	 *
	 * \return la taille du sous-jacent
	 */
	int get_size() const;

	/*!
	 * \brief Accesseur de r_
	 *
	 *  Acceder au taux d'interet du sous-jacent
	 *
	 * \return le taux d'interet du sous-jacent
	 */
	double get_r() const;

	/*!
	 * \brief Accesseur de rho_
	 *
	 *  Acceder au parametre de correlation du sous-jacent
	 *
	 * \return le parametre de correlation du sous-jacent 
	 */
	double* get_rho() const;
 
	/*!
	 * \brief Accesseur de Sigma_
	 *
	 *  Acceder au vecteur de volatilites
	 *
	 * \return le vecteur de volatilites
	 */
	PnlVect *get_sigma() const;

	/*!
	 * \brief Accesseur de Spot_
	 *
	 *  Acceder aux valeurs initiales du sous-jacent
	 *
	 * \return la valeurs initiales du sous-jacent
	 */
	PnlVect *get_spot() const;

	/*!
	 * \brief Accesseur de Trend_
	 *
	 *  Acceder au vecteur de taux sous la probabilite historique
	 *
	 * \return le vecteur de taux sous la probabilite historique
	 */
	PnlVect *get_trend() const;

	/*!
	 * \brief Accesseur de Cho_
	 *
	 *  Acceder a la matrice de cholesky associee a la correlation du sous-jacent
	 *
	 *  \return la matrice de cholesky associee a la correlation du sous-jacent
	 */
	PnlMat *get_cho() const;

	/*!
	 * \brief Accesseur de Gi_
	 *
	 *  Acceder au vecteur gaussien centre du modele de Bs
	 *
	 *  \return le vecteur gaussien centré du modele de Bs
	 */
	PnlVect *get_gi() const;

	/*!
	 * \brief Accesseur de Ld_
	 *
	 *  Acceder a la ligne d de la matrice de Cholesky Cho_
	 *
	 *  \return la ligne d de la matrice de Cholesky Cho_
	 */
	PnlVect *get_ld() const;

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
	void set_rho(double* rho);

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
	 * \brief Mutateur de Trend_
	 *
	 *  Modifier au vecteur de taux sous la probabilite historique
	 *
	 * \param le nouveau vecteur de taux sous la probabilite historique
	 */
	void set_trend(PnlVect *trend);

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
	 *  Modifier le vecteur gaussien centre du modele de Bs
	 *
	 *  \param le nouveau vecteur gaussien centre du modele de Bs
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

	/*!
	 * \brief Calcule une trajectoire du sous-jacent connaissant le passe jusqu'a la date t
	 *
	 * \param path contient une trajectoire du sous-jacent donnee jusqu'a l'instant par la matrice past
	 * \param t date jusqu'a laquelle on connait la trajectoire t n'est pas forcement une date de discretisation
	 * \param N nombre de pas de constatation
	 * \param T date jusqu'a laquelle on simule la trajectoire (maturite)
	 * \param rng pointeur sur le generateur de nombre aleatoire
	 * \param past trajectoire realisee jusqu'a la date t
	 * \param taille contient le nombre d'evolution du sous-jacent jusqu'a la date t
	 * \param G contient N-taille vecteurs gaussiens centrés de matrice de covariance identite
	 * \param grid contient les indices de temps utilises pour l'evolution du sous-jacent
	 */
	void asset(PnlMat *path, double t, int N, double T, PnlRng *rng, const PnlMat *past, int taille, PnlMat* G, PnlVect* grid);

	/*!
	 * \brief Shift d’une trajectoire du sous-jacent
	 *
	 * \param path (input) contient en input la trajectoire du sous-jacent
	 * \param shift_path (output) contient la trajectoire path dont la composante d a ete shiftee par (1+h) a partir de la date t.
	 * \param t (input) date a partir de laquelle on shift
	 * \param h (input) pas de differences finies
	 * \param d (input) indice du sous-jacent à shifter
	 * \param timestep (input) pas de constatation du sous-jacent
	 */
	void shift_asset (PnlMat *_shift_path, const PnlMat *path,
		int d, double h, double t, double timestep);

	/*!
	 *
	 * \brief Simulation du marche
	 *
	 * \param past (output) trajectoire du sous-jacent sous la probabilite historique
	 * \param H (input) nombre de date
	 * \param T (input) date jusqu'a laquelle on simule la trajectoire (maturite)
	 * \param PL (output) erreur de couverture
	 */
	void simul_market (PnlMat* past, int H, double T, PnlRng *rng); 

};
#endif
