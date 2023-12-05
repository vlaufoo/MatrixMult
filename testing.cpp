#include"Functions.hpp"

#define TYPE int

int main(int argc, char** argv){
  using namespace std;
  Matrix<TYPE> A(10, 4);
  Matrix<TYPE> B(4, 15);

  Matrix<TYPE> C(A.Rows(), B.Columns());
  Matrix<TYPE> T = C;

  A.RandomMatrix(0, 60, 94234852);
  B.RandomMatrix(0, 60, 12334564);

  CudaMult<TYPE>(A, B, C, 0);
  CudaMult<TYPE>(A, B, T, 1);

  Matrix<TYPE> R = A*B;

  R.PrintMatrix();
  C.PrintMatrix();
  T.PrintMatrix();

  cout<<"\nNon tiled operation: ";
  if(R == C){
    cout<<"SUCCESS\n";
  }else{
    cout<<"FAILED (results don't match)\n";
  }

  cout<<"\nTiled operation: ";
  if(R == T){
    cout<<"SUCCESS\n";
  }else{
    cout<<"FAILED (results don't match)\n";
  }

  return 0;
}
