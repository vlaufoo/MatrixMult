#include"Classes.h"
#include<ostream>

int main(){

  Matrix A(23, 10);
  A.TestMatrix();

  Matrix B = A;

  Matrix C = A.AddTilingPadding(4);



  return 0;
}
