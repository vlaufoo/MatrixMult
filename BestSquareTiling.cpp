#include "Functions.hpp"

template <typename T = int>
int BestSquareTiling(Matrix<T>& A, Matrix<T>& B, int form_factor_result, int threads, int& big_div, int &small_div)
{
  using namespace std;
  //PREP FOR TILING OF THE MATRICES

  //find the best square tiling for this matrix (allowing for padding)
  int div_1;
  if(form_factor_result >= 1){
    div_1 = round(sqrt(threads*form_factor_result));
  }else{
    div_1 = round(sqrt(threads/form_factor_result));
  }

  if((unsigned)div_1 <= threads){
    while(threads % div_1 != 0){
      div_1 = div_1 + 1;
    }
  }else if(div_1 == 0){
    div_1 = 1;
  }

  cout<<"Closest divider is "<<div_1<<"\n";
  int div_2 = (threads/div_1 > (unsigned)div_1) ? div_1 : threads/div_1;
  div_1 = threads/div_2;
  if(div_2 == 0){
    div_2 = 1;
    div_1 = threads;
  }

  small_div = div_2;
  big_div = div_1;

  int tSize;
  if(form_factor_result >= 1){
    tSize = A.Rows()/div_2;
    while(A.Rows() > div_2*tSize || B.Columns() > div_1*tSize){
      tSize++;
    }
  }else{
    tSize = B.Columns()/div_2;
    while(A.Rows() > div_1*tSize || B.Columns() > div_2*tSize){
      tSize++;
    }
  }

  return tSize;

}
