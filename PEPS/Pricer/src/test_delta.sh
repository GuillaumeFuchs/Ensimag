#!/bin/bash

for f in exemples/*
do
  ./delta $f 0 0.1
done
