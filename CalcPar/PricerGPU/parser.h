#include <pnl/pnl_vector.h>
#include <string.h>
#include "typeval.h"
#include <map>

#ifndef PARSERH
#define PARSERH

/*!
 *  \fileparser.h
 *  \briefHeader de la classe Parser
 *  \author Equipe 11
 */


// structure de comparaison pour la map
struct ltstr
{
  bool operator()(const char* s1, const char* s2) const
  {
    return strcmp(s1, s2) < 0;
  }
};

typedef std::map <const char *, TypeVal, ltstr> HashParam;
// Initialisation de la map< ... >
HashParam create_hash_map ();

/*!
 * \class Parser
 * \brief Classe representant le Parser
 */
class Parser {
 private:
  HashParam table; /*!< Table de hachage */ 
  char **data; /*!<  Tableau de chaînes de caractères*/
  int size; /*!<  Taille du tableau*/
  
  /*!
   * \brief Méthode de lecture du fichier
   *
   * Lit le fichier et stock les données contenues dans l'attribut data
   * \param nomFichier: Nom du fichier qui contient les données
   */ 
  void fichierToData(const char *nomFichier); 

  /*!
   * \brief MaJ de l'attribut map en fonction de l'attribut data
   *
   * Méthode permettant de récupérer les valeurs des différentes clés et de remplir la table à partir de data
   * \param nomFichier: Nom du fichier qui contient les données
   */
  void dataToMap();

 public:
  /*!
   * \brief Constructeur par défaut
   *
   * Constructeur par défaut de la parser
   */
  Parser();
  
  /*!
   * \brief Constructeur à partir d'un fichier
   *
   * Constructeur du parser à partit d'un fichier de donées décrivant l'action
   * \param nomFichier: Nom du fichier qui contient les données
   */
  Parser(const char *nomFichier); 
  
  /*!
   * \brief Destructeur
   *
   * Destructeur de la classe parser
   */
  ~Parser();

  /*!
   * \brief Accesseur de table
   *
   *  Acceder à la table de hachage
   *
   *  \return la table de hachage
   */
  HashParam get_Table();

  /*!
   * \brief Accesseur de size
   *
   *  Acceder à la taille du table
   *
   *  \return la taille du tableau
   */ 
  int get_Size();

  /*!
   * \brief Accesseur de data
   *
   *  Acceder au tableau de données
   *
   *  \return le tableau de données
   */
  char** get_Data();
  
  /*!
   * \brief Accesseur pout récupérer un entier dans la table
   *
   *  Acceder à la valeur d'une clé qui a une valeur entière
   *
   *  \param cle: cle à laquelle on souhaite accéder
   *  \return valeur de la clé
   */
  int getInt(char *cle);
  
  /*!
   * \brief Accesseur pout récupérer un double dans la table
   *
   *  Acceder à la valeur d'une clé qui a une valeur réelle
   *
   *  \param cle: cle à laquelle on souhaite accéder
   *  \return valeur de la clé
   */
  double getDouble(char *cle);

  /*!
   * \brief Accesseur pout récupérer une chaîne de caractères
   *
   *  Acceder à la valeur d'une clé qui a une valeur de chaîne de caractères
   *
   *  \param cle: cle à laquelle on souhaite accéder
   *  \return valeur de la clé
   */
  char *getString(char *cle);

  /*!
   * \brief Accesseur pout récupérer un vecteur
   *
   *  Acceder à la valeur d'une clé qui a une valeur vectorielle
   *
   *  \param cle: cle à laquelle on souhaite accéder
   *  \return valeur de la clé
   */
  PnlVect *getVect(char *cle);

};

/* void set_from_map (const HashParam &table, const char *cle, int &valeur); */
/* void set_from_map (const HashParam &table, const char *cle, double &valeur); */
/* void set_from_map (const HashParam &table, const char *cle, PnlVect **valeur, int size); */
/* void set_from_map (const HashParam &table, const char *cle, char const **valeur); */


#endif
