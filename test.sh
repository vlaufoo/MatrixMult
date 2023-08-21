#!/bin/bash
#
for i in {2, 4, 6, 8, 10, 12, 14, 16}
do 
  make threads=$i main_old
  ./main_old
done
