/*
#include <ctime>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>
*/
#include "Functions.hpp"

double UnopTile(Matrix<TYPE> &A, Matrix<TYPE> &B, Matrix<TYPE> &C, int form_factor_result, int threads, int& tileSize, int& ThNumber)
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

  cout<<"Sto qui";

  //PADDING OF THE MATRICES FOR THE GENERAL PARALLEL OPERATION
  int tSize;
  if(form_factor_result >= 1){
    tSize = C.Rows()/div_2;
    while(C.Rows() > div_2*tSize || C.Columns() > div_1*tSize){
      tSize++;
      //cout<<tSize<<endl;
    }
    //calcola la dimensione tile e aggiustala se è tale che il tiling venga poi fatto
    //in modo improprio. Se le righe di C contengono tSize div_2 volte, ma con un resto 
    //allora otterremmo 3 righe di tile invece che due, quindi ridimensioniamo la tile
    //per averne 2
    A = A.ForceAddTilingPaddingRows(tSize, div_2);
    B = B.ForceAddTilingPaddingColumns(tSize, div_1);
    C = C.ForceAddTilingPadding(tSize, div_2, div_1);

    cout<<"The tile size is "<<tSize<<", from division of Rows by "<<div_2<<endl;
  }else{
    tSize = C.Columns()/div_2;

    while(C.Columns() > div_2*tSize || C.Rows() > div_1*tSize){
      tSize++;
      //cout<<tSize<<endl;
    }

    A = A.ForceAddTilingPaddingRows(tSize, div_1);
    B = B.ForceAddTilingPaddingColumns(tSize, div_2);
    C = C.ForceAddTilingPadding(tSize, div_1, div_2);
    //stesso procedimento per questo caso
    cout<<"The tile size is "<<tSize<<", from division of Columns by "<<div_2<<endl;
  }

  //cout<<"tSize = "<<tSize<<"\n\n";


  //cout<<p*step<<" righe, "<<p*column_factor<<" colonne.\n\n";


#ifdef VERBOSE
  cout<<"tSize = "<<tSize<<"\n\n";
#endif

  //calculating iteration number and prepping threads
  std::vector<std::thread> kernels;
  int ThN = 0;
  int iterations = A.Columns()/tSize;
  if(A.Columns() % tSize != 0)
    iterations++;

#ifdef VERBOSE
  printf("Matrice destinazione (A): %p\n", &A);
#endif


#if defined(PRINT_NUMBERS)
  cout<<"Matrices A and B: \n\n";

  //NOW THE MATRICES HAVE PADDING
  A.PrintMatrix();
  B.PrintMatrix();
#endif

  //GENERALIZED PARALLEL OPERATION
  
  clock_t tic = clock();

  for(int i=0; i<A.Rows()/tSize; i++){
    for(int j=0; j<B.Columns()/tSize; j++){
      kernels.emplace_back(SingleTileThread, ThN, std::ref(C), std::ref(A), std::ref(B), iterations, i, j, tSize);
      ThN++;
    }
  }

  for(auto& thread :kernels){
    thread.join();
  }

  clock_t toc = clock();

  double execution_time = (double)(toc-tic)/CLOCKS_PER_SEC;

#ifdef PRINT_NUMBERS
  cout<<"Matrix C after parallel operation: \n\n";
  C.PrintMatrix();
  cout<<"\n";
  cout<<"Parallel execution in "<<execution_time<<" seconds.\n\n";
#endif

  tileSize = tSize;
  ThNumber = ThN;

  return execution_time;
}
