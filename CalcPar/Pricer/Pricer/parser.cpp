#include <fstream>
#include "parser.h"
#include <iostream>
#include <cstdlib>
#include <stdio.h> //pour le strtok
#include <string>

using namespace std;

// Initialisation de la map< ... >
HashParam create_hash_map ()
{
  HashParam M;
  TypeVal T;
  T.type = T_INT;
  M["option size"] = T;
  M["sample number"] = T;
  M["timestep number"] = T;
  T.type = T_DOUBLE;
  M["interest rate"] = T;
  M["correlation"] = T;
  M["maturity"] = T;
  M["strike"] = T;
  T.type = T_VECTOR;
  M["spot"] = T;
  M["payoff coefficients"] = T;
  M["upper barrier"] = T;
  M["lower barrier"] = T;
  M["volatility"] = T;
  M["trend"] = T;
  T.type = T_STRING;
  M["option type"] = T;
  return M;
}

Parser::Parser(){
  table = create_hash_map();
  size=0;
}

Parser::Parser(const char *nomFichier){
  table = create_hash_map();
  data = new char*[15];
  fichierToData(nomFichier);
  dataToMap();
}

Parser::~Parser(){
}

HashParam Parser::get_Table(){
  return table;
}

int Parser::get_Size(){
  return size;
}

char** Parser::get_Data(){
  return data;
}

void Parser::fichierToData(const char *nomFichier){
  ifstream fichier(nomFichier, ios::in); //ouverture fichier
  if(fichier){
	string ligne;
	int i =0;
	//On remplit data :
	while(getline(fichier, ligne)){
	  //On vérifie d'abord que la ligne ne soit ni un commentaire, ni une ligne vide
	  if(!(ligne.empty()) && ligne[0]!='#') {
	    //Dans ce cas on la stocke dans data:
		data[i] = new char[ligne.length()+1]; 
		string tmp;
		strcpy(data[i], ligne.c_str()); 
		strcat(data[i], "\0");
		i++;
	  }
	}
	size=i; //Mise à jour de size = nombre de lignes lues (utile dans les autres fonctions)
	fichier.close(); //fermeture fichier
  } else {
	cerr << "Echec ouverture fichier" << endl;
  }
}

//Petite procédure permettant d'obtenir la valeur au bout d'une ligne
//lorsqu'elle est numérique 
void getVal(char *ligne, char *val){
  char *P=ligne,*Q=val;
  while(*P!='\0'){
	if(!(*P>='a' && *P<='z')){
	  *Q=*P;
	  Q++;
	}
	P++;
  }
  *Q=0;
}

void Parser::dataToMap(){
  //Prérequis : fichier valide (informations cohérentes etc)

  char val[100];
  int option_size, compteur=1, i=1;
  string aux;
  size_t length, taille;
  char val_aux[20];

  //On traite la première ligne de data à part pour initialiser option_size
  table["option size"].is_set = true;
  getVal(data[0], val); 
  option_size =  atoi(val);
  table["option size"].V_int = option_size;  

  while(compteur <size){

	bool cle_trouve=false;
	std::map <const char *, TypeVal, ltstr>::iterator it = table.begin();

	//Puis on boucle sur toutes les lignes
	while(!cle_trouve && it!=table.end()){
	  if(!strncmp(data[compteur], it->first, strlen(it->first))){
		cle_trouve = true;
	  }else{
		it++;
	  }
	}
	char cle[40];
	strcpy(cle,it->first);

	//Puis on traite la ligne selon le type correspondant à la clé trouvée
	switch(table[cle].type){

	  case T_INT: getVal(data[compteur], val);
				  table[cle].V_int = atoi(val);
				  break;

	  case T_DOUBLE: getVal(data[compteur], val);
					 table[cle].V_double = atof(val);
					 break;

	  case T_VECTOR: getVal(data[compteur], val);
					 {
					   char *val_def = (char*)malloc(sizeof(char)*20);
					   val_def = strtok(val, " ");
					   //Lorsqu'il s'agit d'un vecteur, on le crée avec la première valeur trouvée comme valeur par défaut
					   table[cle].V_vector = pnl_vect_create_from_double(option_size, atof(val_def));

					   //Puis on boucle sur les autres valeurs si elles existent
					   val_def = strtok(NULL, " ");
					   while(val_def!=NULL && i<option_size){
						 pnl_vect_set(table[cle].V_vector,i,atof(val_def));
						 val_def = strtok(NULL, " ");
						 i++;
					   }
					   i=1;
					 }
					 break;

	  case T_STRING:
					 {
					 length = strlen(data[compteur])-strlen(it->first)-1;
					 aux = string(data[compteur]);
					 //Dans le cas d'un T_STRING, on recopie la partie de la ligne correspondante dans la map
					 taille = aux.copy(val_aux, length,strlen(it->first)+1);
					 val_aux[taille]='\0';
					 table[cle].V_string = (char*)malloc(sizeof(char)*(strlen(val_aux)+1));
					 strcpy(table[cle].V_string , val_aux);
					 }
					 break;

	  case T_NULL: cout << "T_NULL!!-"<<cle<<"-"<< endl;
				   break;
	}
	//Dans tous les cas, la case de la map a maintenant une valeur :
	table[cle].is_set = true;
	compteur++;
  }
}


int Parser::getInt(char *cle){
  return table[cle].V_int;
}
double Parser::getDouble(char *cle){
  return table[cle].V_double;
}
char *Parser::getString(char *cle){
  return table[cle].V_string;
}
PnlVect *Parser::getVect(char *cle){
  return table[cle].V_vector;
}