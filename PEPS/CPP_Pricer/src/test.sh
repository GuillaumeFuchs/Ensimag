#!/bin/bash

for i in `seq 1 500`;
do
  ./couv exemples/basket_5d.dat 50 0.1
done
