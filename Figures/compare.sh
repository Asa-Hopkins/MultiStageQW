#!/bin/bash

#Make sure that MultiQW is compiled with -DVERBOSE

N=8
FILENAME="../data/Adam/SK_8n"
START=1780
PROBLEMS=2

for m in 1 2 5 10 20 50 200 400 800 1600; do
    ../MultiQW $N $m $FILENAME $START $PROBLEMS >> walks.txt
done