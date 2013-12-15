#!/bin/bash

for f in exemples/*
do
  ./test-parser $f
done
