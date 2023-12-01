#include<iostream>
#include<cstdio>
#include"Classes.h"
#include<thread>
#include<vector>
#include<mutex>
#include<ctime>
#include<cmath>

int main(int argc, char** argv){

  Matrix<float> A(12, 100);
  Matrix<float> B(100, 20);

  Matrix<float> C(12, 20);

  A.RandomMatrix(0, 60, 94234852);
  B.RandomMatrix(0, 60, 12334564);
  C.RandomMatrix(0, 60, 12424235);

  

  return 0;
}
