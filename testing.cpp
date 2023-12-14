#include "Functions.hpp"

#define TYPE int

int main(int argc, char** argv){
  using namespace std;
  Matrix<TYPE> A(80, 50);
  Matrix<TYPE> B(50, 500);

  Matrix<TYPE> C(A.Rows(), B.Columns());
  Matrix<TYPE> T = C;

  A.RandomMatrix(1, 40, 94234852);
  B.RandomMatrix(1, 100, 12334564);

  C=A*B;

  T.ZeroMatrix();

  double cuda_tile_t = CudaMult(A, B, T, 1);


  T.PrintMatrix();
  int always_less = 1;
  int count = 0;
  for(int i=0; i<T.Rows(); i++){
    for(int j=0; j<T.Columns(); j++){
      cout<<T.GetElement(i, j)<<"\t"<<C.GetElement(i, j)
        //<<"\t"<<A.GetElement(i, j)
        <<endl;
      if(T.GetElement(i, j) > C.GetElement(i, j)){
        cout<<i<<", "<<j<<":\t"
          <<T.GetElement(i, j)<<"\t!=\t"<<C.GetElement(i, j)<<endl;
        always_less = 0;
        count++;
        cout<<"^^^ Here!----------------------------\n";
      }
    }
  }
  cout<<"Result: \n"<< (T==C)<<endl<<always_less<<endl;
  cout<<"Error count = "<<count<<endl;

  return 0;
}
