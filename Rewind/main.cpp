#include<iostream>
#include<cstdio>
#include"Ten.hpp"
#include<thread>
#include<vector>
#include<mutex>
#include<cmath>


void SingleTileThread(int threadId, Tensor &Destination, Tensor A, Tensor B, int iterations, int i, int j, int tSize, Tensor results){
  //   Destination.MultiplyTilesOnce(A, B, k, i, j, tSize);
  for(int k=0; k<iterations; k++){
    Destination.MultiplyTilesOnce(A, 0, B, 0, k, i, j, tSize);
  }
}




int main(int argc, char **argv){

  using namespace std;

  Tensor Piccolo(2, 4, 2);
  Tensor Grande(3, 7, 4);


  Piccolo.Random(0, 15, 7623548);
  Grande.Random(0, 13, 123455);

  Piccolo.Print();
  Tensor Coso = Piccolo || Grande;
  Coso.Print();


  Tensor A(10, 6);
  Tensor B(6, 6);

  Tensor X(10, 6);

  A.Random(0, 15, 7623548);
  B.Random(0, 2, 7623548);

  A.Print();
  B.Print();

  cout<<"Layer di A: "<<A.Layers()<<endl;
  A.Print();
  B.Print();

  cout<<"Element 3, 3: "<<A.GetElement(0, 3, 3)<<endl;
  const int tSize = 3;

  std::vector<std::thread> threads;
  int ThN = 0;

  Tensor PA = A.AddTilingPaddingRows(tSize);
  cout<<"hello there"<<endl;
  Tensor PB = B.AddTilingPaddingColumns(tSize);

  PA.Print();

  std::cout<<"Iterazioni: "<<PA.Columns()/tSize + 1<<std::endl;

  //   int*** results;

  //   results = new int** [PA.Columns()/tSize + 1];

  Tensor results(PA.Columns()/tSize + 1, PA.Rows(), PB.Columns());

  Tensor PX = X.AddTilingPadding(tSize);

  int i, j, k;
  int iterations = PA.Columns()/tSize + 1;

  for(i=0; i<PA.Rows()/tSize; i++){
    for(j=0; j<PB.Columns()/tSize; j++){
      for(k=0; k<iterations; k++){
        PX.MultiplyTilesOnce(PA, 0, PB, 0, k, i, j, tSize);
      }
    }
  }

  for(i=0; i<PA.Rows()/tSize; i++){
    for(j=0; j<PB.Columns()/tSize; j++){
      threads.emplace_back([ThN, iterations, i, j, &PA, &PB, &PX, &tSize, &results]() {
      //  SingleTileThread(ThN, PX, PA, PB, iterations, i, j, tSize, results);
      });
      ThN++;
    }
  }


  /*
   for(i=0; i<PA.Rows()/tSize; i++){
      cout<<"loop 1"<<endl;
      for(j=0; j<PB.Columns()/tSize; j++){
         for(k=0; k<iterations; k++){
            PX.MultiplyTiles(PA, PB, k, i, j, tSize, results);
         }
      }
   }
*/


  for(auto& thread :threads){
    thread.join();
  }
  results.Print();
  Tensor prova(2, 1);
  prova.CollapseTensor();
  prova.Print();

  cout<<"\n\n-------Somma--------\n\n";
  Tensor Template = A*B;
  Template.Print();

  cout<<"Number of Threads: "<<ThN<<endl;

  return 0;
}

