#include<iostream>
#include<cstdio>
#include"Ten.hpp"
#include<thread>
#include<vector>
#include<mutex>
#include<cmath>


void SingleTileThread(int threadId, Tensor& Destination, Tensor& A, Tensor& B, int iterations, int i, int j, int tSize){
  //   Destination.MultiplyTilesOnce(A, B, k, i, j, tSize);
  for(int k=0; k<iterations; k++){
    Destination.MultiplyTilesOnce(A, 0, B, 0, k, i, j, tSize);
  }
}




int main(int argc, char **argv){

  using namespace std;
/* //test per l'operatore || con i tensori (funziona)

  Tensor Piccolo(2, 4, 2);
  Tensor Grande(3, 7, 4);


  Piccolo.Random(0, 15, 7623548);
  Grande.Random(0, 13, 123455);

  Piccolo.Print();
  Grande.Print();
  Tensor Coso = Piccolo || Grande;
  Coso.Print();
*/

  Tensor A(10, 6);
  Tensor B(6, 6);


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


  //   int*** results;

  //   results = new int** [PA.Columns()/tSize + 1];


  int i, j, k;
  int iterations = PA.Columns()/tSize;
  //int iterations = PA.Columns()/tSize + (PA.Columns()%tSize == 0) ? 0 : 1;
  if(PA.Columns()%tSize != 0)
    iterations++;

  std::cout<<"Iterazioni: "<<iterations<<std::endl;

  Tensor results(iterations, PA.Rows(), PB.Columns());
  cout<<"here is fine\n";

  for(i=0; i<PA.Rows()/tSize; i++){
    for(j=0; j<PB.Columns()/tSize; j++){
      for(k=0; k<iterations; k++){
      //  results.MultiplyTilesOnce(PA, 0, PB, 0, k, i, j, tSize);
      //  results.Print(k);
      }
    }
  }

  for(i=0; i<PA.Rows()/tSize; i++){
    for(j=0; j<PB.Columns()/tSize; j++){
      threads.emplace_back(SingleTileThread(ThN, results, PA, PB, iterations, i, j, tSize));
      ThN++;
      results.Print();
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

  cout<<"\n\n\n\n\n\n\n\n\n";
  results.Print();
  results.CollapseTensor();
  
  cout<<"padding rows = "<<results.PaddingRows()<<"\n"
      <<"padding columns = "<<results.PaddingColumns()<<"\n"
      <<"padding layers = "<<results.PaddingLayers()<<"\n";

  results = results.RemovePadding();

  cout<<"\n\n-------Somma--------\n\n";
  Tensor Template = A*B;
  Template.Print();
  results.Print(0);

  cout<<"Number of Threads: "<<ThN<<endl;

  return 0;
}

