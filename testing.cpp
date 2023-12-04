#include"Functions.hpp"

int main(int argc, char** argv){
  using namespace std;
  Matrix A(10, 4);
  Matrix B(4, 15);

  Matrix C(A.Rows(), B.Columns());
  Matrix T = C;

  A.RandomMatrix(0, 60, 94234852);
  B.RandomMatrix(0, 60, 12334564);

  CudaMult(A, B, C, 0);
  CudaMult(A, B, T, 1);

  Matrix R = A*B;

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
