#!/bin/bash
version=$5

echo -e "R\tC\tT\tSi\tOp_FF\tRes_FF\tSt\tPt\tiPt\tSU\tbSize\tCuTilet\tCuSU\tSeed:${3}\n" > "$3_$1_$2_$4.txt"
sleep 2
for t in 2 4 6 8 10
do
  for ffo in 1 1.3 1.7 2
  do
    for ffr in 1 1.3 1.7 2
    do
     echo -e "\n\e[43m\e[30m Next Script Iteration    \n\e[0m"
     ./$version $t $3 $1 $2 $ffo $ffr $4
    done
  done
done
