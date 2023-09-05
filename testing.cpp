#include<iostream>
#include<cstdio>
#include"Classes.h"
#include<thread>
#include<vector>
#include<mutex>
#include<ctime>
#include<cmath>

void SingleTileThread(Matrix& Destination, Matrix& A, Matrix& B, int iterations, int i, int j, int tSize){

  for(int k=0; k<iterations; k++){
    Destination.MultiplyTilesOnce(A, B, k, i, j, tSize);
  }
}



int main(int argc, char **argv){

  using namespace std;


//  int** result = new int*[p*R_STEP];
//  for(int i = 0; i<p*R_STEP; i++){
//    result[i] = new int[p*column_factor];
//    for(int j=0; j<p*column_factor; j++){
//      result[i][j] = 0;
//      cout<<result[i][j]<<"\t";
//    }
//    cout<<"\n";
//  }
//  cout<<"\n";



  Matrix A(12, 7);
  Matrix B(7, 6);

  Matrix X(12, 6);
  Matrix Y(12, 6);

  A.RandomMatrix(0, 20, 290083);
  B.RandomMatrix(3, 15, 879273);

  A.PrintMatrix();
  B.PrintMatrix();

  int tSize = 3;

  Matrix PA = A.AddTilingPaddingRows(tSize);
  Matrix PB = B.AddTilingPaddingColumns(tSize);
  Matrix PX = X.AddTilingPadding(tSize);
  Matrix PY = Y.AddTilingPadding(tSize);

  Matrix PT = PY;

  PA.PrintMatrix();
  PB.PrintMatrix();

  cout<<"\n\nTile size is: "<<tSize<<endl;

  std::vector<std::thread> threads;

  int ThN = 0;
  int i, j, k;
  int iterations = PA.Columns()/tSize + 1;

  PX = PA*PB;
  PX.PrintMatrix();
/*
  //	PX.PrintMatrix();	
  //serial execution
  for(i=0; i<PA.Rows()/tSize; i++){
    for(j=0; j<PB.Columns()/tSize; j++){
      for(int k=0; k<iterations; k++){
        PY.MultiplyTilesOnce(PA, PB, k, i, j, tSize);
      }
    }
  }
*/

  //pariallel execution
    for(i=0; i<A.Rows()/tSize; i++){
      for(j=0; j<B.Columns()/tSize; j++){
        threads.emplace_back(SingleTileThread, std::ref(Y), std::ref(A), std::ref(B), iterations, i, j, tSize);
//        threads.emplace_back(SingleTileThread, std::ref(PY), std::ref(PA), std::ref(PB), iterations, i, j, tSize);
       // threads.emplace_back(i, SingleTileThread, &PA);
        ThN++;
      }
    }
  for(auto& thread :threads){
    thread.join();
  }

  cout<<"Out of the parallel section\n\n";
  
/*
  PY.MultiplyTilesOnce(PA, PB, 0, 0, 0, 4);
  PY.MultiplyTilesOnce(PA, PB, 1, 0, 0, 4);
  PY.MultiplyTilesOnce(PA, PB, 2, 0, 0, 4);
*/

//  PY.GetResultTile(PA, PB, iterations, 0, 0, tSize);
//  PY.GetResultTile(PA, PB, iterations, 0, 1, tSize);
//  PY.GetResultTile(PA, PB, iterations, 1, 0, tSize);
//  PY.GetResultTile(PA, PB, iterations, 1, 1, tSize);
//  PY.GetResultTile(PA, PB, iterations, 2, 0, tSize);
//  PY.GetResultTile(PA, PB, iterations, 2, 1, tSize);

  Y.PrintMatrix();

//  cout<<"Number of Threads: "<<ThN<<endl;
//  
//  for(int i = 0; i<p*column_factor; i++){
//    for(int j=0; j<p*column_factor; j++){
//      cout<<result[i][j]<<"\t";
//    }
//    cout<<"\n";
//
//    delete[] result[i];
//  }
//  delete[] result;

  return 0;
}
