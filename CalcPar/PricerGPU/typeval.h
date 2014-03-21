#include <pnl/pnl_vector.h>
#include <string.h>

#ifndef TYPEVALH
#define TYPEVALH

/*!
 *  \filetypeval.h
 *  \brief Header de la structure TypeVal
 *  \author Equipe 11
 */



// liste des types
typedef enum {
  T_NULL,
  T_INT,
  T_DOUBLE,
  T_VECTOR,
  T_STRING
} T_type;

/*!
 * \struct TypeVal
 * \brief Structure des types utilisés dans le parser
 */
struct TypeVal
{
  bool is_set; /*!< La valeur de la clé existe */
  T_type type; /*!< Type de la valeur de la clé */
  union {
    double V_double; /*!< Valeur réelle de la clé */
    int V_int; /*!< Valeur entière de la clé */
    char *V_string; /*!< Valeur d'une chaîne de caractères de la clé */
    PnlVect *V_vector; /*!< Valeur vectorielle de la clé */
  }; // value
  
  /*!
   * \brief Constructeur par défaut
   *
   * Constructeur par défaut d'un objet TypeVal
   */
  TypeVal ();

  /*!
   * \brief Destructeur par défaut
   *
   * Destructeur par défaut d'un objet TypeVal
   */
  ~TypeVal ();
};


#endif
