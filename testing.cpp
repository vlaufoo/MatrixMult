#include"Functions.hpp"

int main(int argc, char** argv){

  Matrix A(10, 4);
  Matrix B(4, 15);

  Matrix C(A.Rows(), B.Columns());

  A.RandomMatrix(0, 60, 94234852);
  B.RandomMatrix(0, 60, 12334564);

  CudaMult(A, B, C, 16);
  Matrix R = A*B;
  R.PrintMatrix();
  C.PrintMatrix();

  return 0;
}
