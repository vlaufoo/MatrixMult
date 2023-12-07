#include"Functions.hpp"

#define TYPE int

int main(int argc, char** argv){
  using namespace std;
  Matrix<TYPE> A(240, 240);
  Matrix<TYPE> B(240, 480);

  Matrix<TYPE> C(A.Rows(), B.Columns());
  Matrix<TYPE> T = C;

  A.RandomMatrix(0, 60, 94234852);
  B.RandomMatrix(0, 60, 12334564);

  int Rdiv, Cdiv;

  int tSize = BestSquareTiling<TYPE>(A, B, 2, 6, Rdiv, Cdiv);

  Matrix<TYPE> PC = C.ForceAddTilingPadding(tSize, Rdiv, Cdiv);

  cout<<"\n\nRows of tiles: "<<PC.Rows()/tSize<<" while Rdiv is "<<Rdiv<<"\n\n";
  cout<<"\n\nCols of tiles: "<<PC.Columns()/tSize<<" while Cdiv is "<<Cdiv<<"\n\n";

  
  return 0;
}
