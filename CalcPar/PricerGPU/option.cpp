# include "option.h"

/*!
 * \file option.cpp
 * \brief classe option 
 * \author equipe 11
 */

Option :: Option(){
  T_ = 0;
  TimeSteps_ = 0;
  size_ = 0;
}

Option::Option(Parser pars){
  T_ = pars.getDouble("maturity");
  TimeSteps_ = pars.getInt("timestep number");
  size_ = pars.getInt("option size");
}

Option :: ~Option(){
}

double Option :: get_T(){
  return T_;
}

int Option :: get_timesteps(){
  return TimeSteps_;
}

int Option :: get_size(){
  return size_;
}

void Option :: set_T(double T){
  T_ = T;
}

void Option :: set_TimeSteps(int TimeSteps){
  TimeSteps_ = TimeSteps;
}

void Option :: set_size(int size){
  size_ = size;
}
