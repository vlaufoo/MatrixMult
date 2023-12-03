#include"Functions.hpp"

int main(int argc, char** argv){

  Matrix<float> A(90, 100);
  Matrix<float> B(100, 200);

  Matrix<float> C(12, 20);

  A.RandomMatrix(0, 60, 94234852);
  B.RandomMatrix(0, 60, 12334564);
  C.RandomMatrix(0, 60, 12424235);

  CudaMult(A, B, C, 16);


  return 0;
}
