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
  char **data; /*!<  Tableau de cha�nes de caract�res*/
  int size; /*!<  Taille du tableau*/
  
  /*!
   * \brief M�thode de lecture du fichier
   *
   * Lit le fichier et stock les donn�es contenues dans l'attribut data
   * \param nomFichier: Nom du fichier qui contient les donn�es
   */ 
  void fichierToData(const char *nomFichier); 

  /*!
   * \brief MaJ de l'attribut map en fonction de l'attribut data
   *
   * M�thode permettant de r�cup�rer les valeurs des diff�rentes cl�s et de remplir la table � partir de data
   * \param nomFichier: Nom du fichier qui contient les donn�es
   */
  void dataToMap();

 public:
  /*!
   * \brief Constructeur par d�faut
   *
   * Constructeur par d�faut de la parser
   */
  Parser();
  
  /*!
   * \brief Constructeur � partir d'un fichier
   *
   * Constructeur du parser � partit d'un fichier de don�es d�crivant l'action
   * \param nomFichier: Nom du fichier qui contient les donn�es
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
   *  Acceder � la table de hachage
   *
   *  \return la table de hachage
   */
  HashParam get_Table();

  /*!
   * \brief Accesseur de size
   *
   *  Acceder � la taille du table
   *
   *  \return la taille du tableau
   */ 
  int get_Size();

  /*!
   * \brief Accesseur de data
   *
   *  Acceder au tableau de donn�es
   *
   *  \return le tableau de donn�es
   */
  char** get_Data();
  
  /*!
   * \brief Accesseur pout r�cup�rer un entier dans la table
   *
   *  Acceder � la valeur d'une cl� qui a une valeur enti�re
   *
   *  \param cle: cle � laquelle on souhaite acc�der
   *  \return valeur de la cl�
   */
  int getInt(char *cle);
  
  /*!
   * \brief Accesseur pout r�cup�rer un double dans la table
   *
   *  Acceder � la valeur d'une cl� qui a une valeur r�elle
   *
   *  \param cle: cle � laquelle on souhaite acc�der
   *  \return valeur de la cl�
   */
  double getDouble(char *cle);

  /*!
   * \brief Accesseur pout r�cup�rer une cha�ne de caract�res
   *
   *  Acceder � la valeur d'une cl� qui a une valeur de cha�ne de caract�res
   *
   *  \param cle: cle � laquelle on souhaite acc�der
   *  \return valeur de la cl�
   */
  char *getString(char *cle);

  /*!
   * \brief Accesseur pout r�cup�rer un vecteur
   *
   *  Acceder � la valeur d'une cl� qui a une valeur vectorielle
   *
   *  \param cle: cle � laquelle on souhaite acc�der
   *  \return valeur de la cl�
   */
  PnlVect *getVect(char *cle);

};

/* void set_from_map (const HashParam &table, const char *cle, int &valeur); */
/* void set_from_map (const HashParam &table, const char *cle, double &valeur); */
/* void set_from_map (const HashParam &table, const char *cle, PnlVect **valeur, int size); */
/* void set_from_map (const HashParam &table, const char *cle, char const **valeur); */


#endif
