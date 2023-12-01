/*
#include <iostream>
#include <cstdio>
#include <ctime>
#include <thread>
#include <vector>
*/
#include "Functions.hpp"

double OpTile(Matrix<TYPE> &A, Matrix<TYPE> &B, Matrix<TYPE> &C, int form_factor_result, int threads, int& tRows, int& tCols)
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
  int div_2 = ((unsigned)threads/div_1 > (unsigned)div_1) ? div_1 : threads/div_1;
  div_1 = threads/div_2;
  if(div_2 == 0){
    div_2 = 1;
    div_1 = threads;
  }

  //PREP FOR THE DISUNIFORM TILING OF THE OPTIMIZED METHOD
  int Rdiv = div_2;
  int Cdiv = div_1;
  if(form_factor_result <= 1){
    Rdiv = div_1;
    Cdiv = div_2;
  }

  int tR = A.Rows()/Rdiv + A.Rows()%Rdiv;
  int tC = B.Columns()/Cdiv + B.Columns()%Cdiv;

  //START OF THE OPTIMIZED PARALLEL OPERATION (NO PADDING NEEDED)
  std::vector<std::thread> tiles;
  clock_t tic_2 = clock();
  //parallel execution
  for(int i=0; i<Rdiv; i++){
    for(int j=0; j<Cdiv; j++){
      tiles.emplace_back(&Matrix<TYPE>::GetResultTile, ref(C), ref(A), ref(B), i, j, tR, tC);
    }
  }

  for(auto& thread :tiles){
    thread.join();
  }

  clock_t toc_2 = clock();

  double execution_time = (double)(toc_2-tic_2)/CLOCKS_PER_SEC;

#ifdef PRINT_NUMBERS
  cout<<"Matrix T after parallel operation: \n\n";
  C.PrintMatrix();
  cout<<"\n";
  cout<<"Simple execution in "<<execution_time<<" seconds.\n\n";
#endif

  tRows = tR;
  tCols = tC;

  return execution_time;

}
