#!/bin/bash

for t in 2 4 6 8 10
do
  for ffo in 1 1.3 1.7 2
  do
    for ffr in 1 1.3 1.7 2
    do
     echo -e "\n\e[43m\e[30m Next Iteration\n\e[0m"
     ./main_old $t $3 $1 $2 $ffo $ffr $4
    done
  done
done
