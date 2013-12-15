# include "option.h"
#include "pnl/pnl_matrix.h"
#include "pnl/pnl_vector.h"

/*!
 * \file option.cpp
 * \brief classe option 
 * \author equipe 11
 */

Option :: Option(){
  T_ = 0;
  timeStep_ = 0;
  size_ = 0;
}

//Option::Option(Parser pars){
//  T_ = pars.getDouble("maturity");
//  timeStep_ = pars.getInt("timestep number");
//  size_ = pars.getInt("option size");
//}

Option::Option(double T, int timeStep, int size){
	T_ = T;
    timeStep_ = timeStep;
    size_ = size;
}

Option :: ~Option(){
}

double Option :: get_T(){
  return T_;
}

int Option :: get_timeStep(){
  return timeStep_;
}

int Option :: get_size(){
  return size_;
}

void Option :: set_T(double T){
  T_ = T;
}

void Option :: set_timeStep(int timeStep){
  timeStep_ = timeStep;
}

void Option :: set_size(int size){
  size_ = size;
}
