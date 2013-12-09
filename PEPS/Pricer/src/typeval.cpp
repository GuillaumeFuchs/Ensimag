#include "typeval.h"
#include <typeinfo>

TypeVal::TypeVal(){
  is_set = false;
}

TypeVal::~TypeVal(){ // On enlève le destructeur
  // if(typeid(type) == typeid(T_VECTOR)){
  //   delete [] V_vector;
  // } else if(typeid(type) == typeid(T_STRING)){
  //   delete [] V_string;
  // }
}
