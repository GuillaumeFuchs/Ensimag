# include "Option.h"

Option :: Option(){
  T_ = 0;
  timeStep_ = 0;
  size_ = 0;
}

Option::Option(double T, int timeStep, int size){
	T_ = T;
	timeStep_ = timeStep;
	size_ = size;
}

Option :: ~Option(){
}

double Option :: get_T() const{
  return T_;
}

int Option :: get_timeStep() const{
  return timeStep_;
}

int Option :: get_size() const{
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
