/**
 *  @file	montecarlo.cpp
 *  @brief	Implémentation de la classe Monte Carlo
 *
 *  \author 
 *  Equipe 11 - Julia DITISHEIM, Guillaume FUCHS, Barthelemy LONGUET DE LA GIRAUDIERE, Mailys RENEAUME, Alexis SCIAU
 *
 */
 
#include "montecarlo.h"
#include <math.h>
#include <pnl/pnl_vector.h>
#include <pnl/pnl_matrix.h>
#include <pnl/pnl_mathtools.h>
#include <cstdio>

MonteCarlo::MonteCarlo(PnlRng * rng){
  (*this).rng = rng;
  h_ = 0.0;
  samples_ = 0;
}

MonteCarlo::MonteCarlo(BS *mod_, Option *opt_, PnlRng *rng, double h_, int samples_){
  (*this).mod_ = mod_;
  (*this).opt_ = opt_;
  (*this).rng = rng;
  (*this).h_ = h_;
  (*this).samples_ = samples_;
}

/**
 * Accesseurs
 *
 */
MonteCarlo::~MonteCarlo(){
}

BS* MonteCarlo::get_mod(){
  return mod_;
}

Option* MonteCarlo::get_opt(){
  return opt_;
}

PnlRng* MonteCarlo::get_rng(){
  return rng;
}

double MonteCarlo::get_h(){
  return h_;
}

int MonteCarlo::get_samples(){
  return samples_;
}

/**
 * Mutateurs
 *
 */
void MonteCarlo::set_mod(BS* mod){
  mod_ = mod;
}

void MonteCarlo::set_opt(Option* opt){
  opt_ = opt;
}

void MonteCarlo::set_rng(PnlRng* rng){
  (*this).rng = rng;
}

void MonteCarlo::set_h(double h){
  h_ = h;
}

void MonteCarlo::set_samples(int samples){
  samples_ = samples;
}

/**
 * Price
 *
 */
void MonteCarlo::price(double &prix, double &ic){
  int size = opt_->get_size();
  int TimeSteps = opt_->get_timesteps();
  double r = mod_->get_r();
  double T = opt_->get_T();
  //Initialisation de path comme une matrice de dimension d x (N+1)
  PnlMat *path = pnl_mat_create(size, TimeSteps+1);
  //G: matrice de dimension N*d pour générer une suite iid selon la loi normale centrée réduite
  PnlMat *G = pnl_mat_create(TimeSteps, size);
  //grid: vecteur de taille N pour générer la grille de temps (t_0=0, ..., t_N)
  PnlVect *grid = pnl_vect_create(TimeSteps+1);
  //tirages: vecteur de taille samples_ contenant les valeurs des M payoff
  PnlVect *tirages = pnl_vect_create(samples_);
  //temps: incrémentation pour chaque date de constation
  double temps = T/TimeSteps;

  //Calcule de chaque date de constatation;
  for (int t=0; t<TimeSteps+1; t++){
	pnl_vect_set(grid, t, temps*t);
  }

  //Ajout du prix spot dans la première colonne de path
  pnl_mat_set_col(path, mod_->get_spot(), 0);
  for (int j=0; j<samples_; j++){
	//Génération de la trajectoire du modèle de Black Scholes
	mod_->asset(path, T, TimeSteps, rng, G, grid);
	//Ajout du payoff dans tirages
	pnl_vect_set(tirages, j, opt_->payoff(path));
  }

  //Calcul du prix à l'aide de la formule de MC
  prix = exp(-r*T)*(1/(double)samples_)*pnl_vect_sum(tirages);
  //Calcul de la variance de l'estimateur pour avoir l'intervalle de confiance
  ic = (3.92/sqrt((double)samples_) )* sqrt(exp(-2*r*T)*((1/(double)samples_)*SQR(pnl_vect_norm_two(tirages))-SQR((1/(double)samples_)*pnl_vect_sum(tirages))));

  pnl_vect_free(&grid);
  pnl_vect_free(&tirages);
  pnl_mat_free(&path);
  pnl_mat_free(&G);
}

void MonteCarlo::price(const PnlMat *past, double t, double &prix, double &ic){
  double T = opt_->get_T();
  //Cas où on price à maturité
  //Toutes les informations sont déjà déterminés à l'aide de past
  //donc on a besoin d'aucune simulation
  //et l'intervalle de confiance est de 0 car on est dans un calcul du prix déterministe
  if (t == T){
	prix = opt_->payoff(past);
	//Calcul de la variance
	ic = 0;
	return;
  }

  int size = opt_->get_size();
  int TimeSteps = opt_->get_timesteps();
  double r = mod_->get_r();
  //Initialisation de path comme une matrice de dimension d x (N+1)
  PnlMat *path = pnl_mat_create(size, TimeSteps+1);
  //tirages: vecteur de taille samples_ contenant la valeur des payoff de l'option
  PnlVect *tirages = pnl_vect_create(samples_);
  //temps: incrémentation pour chaque date de constatation
  double temps = T/TimeSteps;
  //extract: vecteur de taille size servant à extraire de la matrice past les colonnes pour les insérer dans path
  PnlVect *extract = pnl_vect_create(size);

  //taille: entier servant à déterminer le nombre de d'évolution du sous jacent (TimeSteps-taille termes à simuler pour chaque actif)
  //Si t est un pas d'incrémentation du temps alors s_{t_i} et s_t sont confondus donc l'indice i est égal à past->n-1
  //Sinon i=past->n-2
  int taille;
  if (fabs(t-temps*(past->n-1))<0.00001)
	taille = past->n-1;
  else
	taille = past->n-2;
  //G: matrice de dimension (N-taille)*d pour générer une suite iid selon la loi normale centrée réduite
  PnlMat *G = pnl_mat_create(TimeSteps-taille, size);
  //Grid: vecteur de taille N-taille+1 pour générer la grille de temps (t, t_{i+1}, ..., t_{N})
  PnlVect *grid = pnl_vect_create(TimeSteps-taille+1);
  pnl_vect_set(grid, 0, t);

  //Calcul de chaque date de constatation;
  for (int i=1; i<TimeSteps-taille+1; i++){
	pnl_vect_set(grid, i, temps*(i+taille));
  }

  //Ajout de la trajectoire du modèle dans path
  //Ajout des prix constaté jusqu'à la date t dans les taille+1 premières colonnes (cela correspond en fait au nombre de colonne de past)
  for (int i=0; i<taille+1; i++){
	//Extraction des colonnes de past dans extract pour les insérer dans path
	pnl_mat_get_col(extract, past, i);
	pnl_mat_set_col(path, extract, i);
  }

  for (int j=0; j<samples_; j++){
	//Si on travaille à maturité
	//alors on clone la matrice past dans path
	//sinon on génère l'évolution du sous-jacent par le modèle de BS
	if (t==T)
	  pnl_mat_clone(path,past);
	else  
	  mod_->asset(path, t, TimeSteps , T, rng, past, taille, G, grid);
	//Ajout dans tirages
	pnl_vect_set(tirages, j, opt_->payoff(path));
  }
  //Calcul du prix
  prix = exp(-r*(T-t))*(1/(double)samples_)*pnl_vect_sum(tirages);
  //Calcul de la variance
  ic = (3.92/sqrt((double)samples_)) * sqrt(exp(-2*r*(T-t))*((1/(double)samples_)*SQR(pnl_vect_norm_two(tirages))-SQR((1/(double)samples_)*pnl_vect_sum(tirages))));

  pnl_mat_free(&path);
  pnl_mat_free(&G);
  pnl_vect_free(&tirages);
  pnl_vect_free(&extract);
  pnl_vect_free(&grid);
}


void MonteCarlo::delta (const PnlMat *past, double t, PnlVect *delta, PnlVect *ic){
  int size = opt_->get_size();
  int TimeSteps = opt_->get_timesteps();
  double r = mod_->get_r();
  double T = opt_->get_T();
  double temps1 = 0;
  double temps2 = 0;
  double sum, sum2;
  double facteur;
  double result, resultic;
  //Temps: incrémentation pour chaque date de constatation
  double temps = T/TimeSteps;
  //Extract: vecteur de taille size servant à extraire de la matrice past les colonnes pour les insérer dans path
  PnlVect *extract = pnl_vect_create(size);
  //On initialise path, _shift_path_plus et _shift_path_moins comme des matrices de dimension d x (N+1)
  PnlMat *path = pnl_mat_create(size, TimeSteps+1);  
  PnlMat *_shift_path_plus = pnl_mat_create(size, TimeSteps+1);
  PnlMat *_shift_path_moins = pnl_mat_create(size, TimeSteps+1);
  PnlVect *tirages = pnl_vect_create(samples_);
  PnlVect *tirages2 =  pnl_vect_create(samples_);

  //Taille: entier servant à déterminer le nombre de d'évolution du sous jacent (TimeSteps-taille termes à simuler pour chaque actif)
  //Si t est un pas d'incrémentation du temps alors s_{t_i} et s_t sont confondus donc l'indice i est égal à past->n-1
  //Sinon i=past->n-2
  int taille;
  if (fabs(t-temps*(past->n-1))<0.00001)
	taille = past->n-1;
  else
	taille = past->n-2;

  //G: matrice de dimension (N-taille)*d pour générer une suite iid selon la loi normale centrée réduite
  PnlMat *G = pnl_mat_create(TimeSteps-taille, size);
  //Grid: vecteur de taille N-taille+1 pour générer la grille de temps (t, t_{i+1}, ..., t_{N})
  PnlVect *grid = pnl_vect_create(TimeSteps-taille+1);
  pnl_vect_set(grid, 0, t);
  for (int i=1; i<TimeSteps-taille+1; i++){
	//On calcule chaque date de constatation;
	pnl_vect_set(grid, i, temps*(i+taille));
  }

  //On met la trajectoire du modèle dans path
  //On met les prix constaté jusqu'à la date t dans les taille+2 premières colonnes (cela correspond en fait au nombre de colonne de past)
  for (int i=0; i<taille+1; i++){
	//Extraction des colonnes de past dans extract pour les insérer dans path
	pnl_mat_get_col(extract, past, i);
	pnl_mat_set_col(path, extract, i);
  }

  for (int d=0; d<size; d++){
	//printf("%d/%d\n",d, size-1);
	for (int j=0; j<samples_; j++){
	  if (t==T)
		pnl_mat_clone(path,past);
	  else {
		mod_->asset(path, t, TimeSteps, T, rng, past, taille, G, grid);
	  }
	  // On récupère la trajectoire shiftée
	  mod_->shift_asset(_shift_path_plus, path, d, h_, t,TimeSteps);
	  mod_->shift_asset (_shift_path_moins, path, d, -h_, t,TimeSteps);

	  //On calcul la valeur du payoff et on retiens la différence
	  temps1 = opt_->payoff(_shift_path_plus);
	  temps2 = opt_->payoff(_shift_path_moins);
	  pnl_vect_set(tirages, j, temps1-temps2);
	  // recuperation des (phi(plus)-pi(moins))²
	  pnl_vect_set(tirages2, j, (temps1-temps2)*(temps1-temps2));
	}
	// somme des phi
	sum = pnl_vect_sum(tirages);
	// somme des phi²
	sum2 = pnl_vect_sum(tirages2);

	// calcul de delta
	facteur =  1/(2*(double)h_*(double)samples_*(double)pnl_mat_get(past, d, past->n-1));
	result =  exp(-r*(T-t))* facteur * sum;
	pnl_vect_set(delta, d, result);

	// calcul de l'intervalle de confiance
	sum = (sum*facteur)*(sum*facteur);
	resultic = exp(-r*(T-t))/2 * (facteur*sum2 - sum);
	resultic = 3.92*sqrt(resultic * 1/ (double)samples_);
	pnl_vect_set(ic, d, resultic);
  }

  pnl_mat_free(&G);
  pnl_mat_free(&path);
  pnl_mat_free(&_shift_path_plus);
  pnl_mat_free(&_shift_path_moins);
  pnl_vect_free(&tirages);
  pnl_vect_free(&tirages2);
  pnl_vect_free(&extract);
  pnl_vect_free(&grid);
}

void MonteCarlo::couv(PnlMat *past, double &pl, int H, double T)
{
  //Simulation du modèle sous la probabilité historique
  mod_->simul_market(past, H, T, rng);

  double r = mod_->get_r();
  int size = opt_->get_size();
  int TimeSteps = opt_->get_timesteps();
  //temps_tau: pas de discrétisation du temps avec H comme nombre de dates
  double temps_tau = T/H;
  //mult: coefficient multiplicateur de H et TimeSteps
  int mult = (int)(H/TimeSteps);
  //compteur: permet de savoir à quel length correspond les colonnes de past avec H dates et les colonnes dont on a besoin avec N dates
  int compteur = 0;
  int length =0;

  //prix: prix du sous-jacent à l'instant 0
  double prix;
  double ic;
  price(prix, ic);

  //delta1: vecteur contenant le delta à un instant donné
  PnlVect* delta1 = pnl_vect_create(size);
  PnlVect* ic_vect = pnl_vect_create(size);

  //past_sub: matrice contenant la matrice past jusqu'à un instant donné
  PnlMat* past_sub = pnl_mat_create(size,1);
  pnl_mat_extract_subblock(past_sub, past, 0, size, 0, 1);

  //extract: vecteur utilisé pour extraire des colonne de past et les insérer dans past_sub
  PnlVect* extract = pnl_vect_create(size);

  //S: vecteur contenant le prix dusous-jacent à un instant donné
  PnlVect* S = pnl_vect_create(size);
  //result: vecteur contenant le résultat de la soustraction entre delta1 et delta2
  PnlVect* result;

  //Calcul à l'instant 0 de l'évolution de l'évolution de la part investie
  delta(past_sub, 0, delta1, ic_vect);
  pnl_mat_get_col(S, past, 0);

  PnlVect* V = pnl_vect_create(H);
  pnl_vect_set(V, 0, prix - pnl_vect_scalar_prod(delta1, S));
  compteur=mult;

  //delta2: vecteur contenant le delta à une date de constation précédente de delta1
  PnlVect* hist;
  for (int i=1; i<H+1; i++)
  {
	//On met dans delta2 la valeur du delta à l'instant i-1
	PnlVect* delta2 = pnl_vect_copy(delta1);
	  
	//On cherche l'indice k tel que tau{i} < t_{k} < tau_{i+1}
	//On met dans past_sub les colonne correspondant à
	//past[,j*mult] tel que j = 0..i
	//past_sub est une matrice de dimension size * (i+1)
	if (compteur == mult){
	  length++;
	  pnl_mat_resize(past_sub, size, length+1);
	  compteur=0;
	}
	compteur++;
	for (int j=0; j<length; j++){
	  //On extrait la colonne d'length j*mult de past 
	  //et on l'insère dans la colonne d'length j de past
	  pnl_mat_get_col(extract, past, j*mult);
	  pnl_mat_set_col(past_sub, extract, j);
	}
	pnl_mat_get_col(extract, past, i);
	pnl_mat_set_col(past_sub, extract, length);

	if (i == H){
	  hist = pnl_vect_copy(delta2);
	  price(past_sub, temps_tau*i, prix, ic);
	}else{

	  //Calcul du delta à l'instant i
	  delta(past_sub, temps_tau*i, delta1, ic_vect);
	  price(past_sub, temps_tau*i, prix, ic);

	  //On met delta1 dans result pour pouvoir faire la soustraction de delta1 avec delta2 dans result car on aura besoin de delta1 dans la boucle suivant
	  result = pnl_vect_copy(delta1);
	  pnl_vect_minus_vect(result, delta2);

	  pnl_mat_get_col(S, past, i);
	  pnl_vect_set(V, i, pnl_vect_get(V,i-1) * exp(r*T/H) - pnl_vect_scalar_prod(result , S)); 
	  pnl_vect_free(&result);
	}
	pnl_vect_free(&delta2);
  }
  //Calcul à l'instant H du delta et du prix des actifs
  pnl_mat_get_col(S, past, H);

  prix = opt_->payoff(past_sub);

  //Calcul de l'erreur de couverture
  pl = pnl_vect_get(V, H-1) * exp(r*T/H) + pnl_vect_scalar_prod(hist, S) - prix;

  pnl_vect_free(&V);
  pnl_vect_free(&S);
  pnl_vect_free(&hist);
  pnl_vect_free(&extract);
  pnl_mat_free(&past_sub);
  pnl_vect_free(&delta1);
  pnl_vect_free(&ic_vect);
}
