#include<iostream>
#include<cstdio>
#include"Tensor.hpp"
#include<thread>
#include<vector>
#include<mutex>
#include<math>


void SingleTileThread(int threadId, Matrix &Destination, Matrix A, Matrix B, int iterations, int i, int j, int tSize, Tensor results){
//   Destination.MultiplyTilesOnce(A, B, k, i, j, tSize);
   for(int k=0; k<iterations; k++){
      Destination.MultiplyTiles(A, B, k, i, j, tSize, results);
   }
}


   

int main(int argc, char **argv){

   using namespace std;

   Tensor Piccolo(2, 4, 2);
   Tensor Grande(3, 7, 4);

   Piccolo.TestTensor();
   Grande.TestTensor();

   Tensor Coso = Piccolo || Grande;
   Coso.PrintTensor();


   Matrix A(10, 5);
   Matrix B(5, 6);

   Matrix X(10, 6);

   A.TestMatrix();
   B.TestMatrix();

   A.PrintMatrix();
   B.PrintMatrix();

   cout<<"Layer di A: "<<A.Layers()<<endl;
   A.PrintTensor();
   A.PrintMatrix();
   cout<<A.GetElement(3, 3)<<endl;
   const int tSize = 3;
   
   std::vector<std::thread> threads;
   int ThN = 0;

   Matrix PA = A.AddTilingPaddingRows(tSize);
   cout<<"hello there"<<endl;
   Matrix PB = B.AddTilingPaddingColumns(tSize);
   
   

   std::cout<<"Iterazioni: "<<PA.Columns()/tSize + 1<<std::endl;

//   int*** results;

//   results = new int** [PA.Columns()/tSize + 1];
   
   Tensor results(PA.Columns()/tSize + 1, PA.Rows(), PB.Columns());

   results.PrintTensor();

   Matrix PX = X.AddTilingPadding(tSize);

   int i, j, k;
   int iterations = PA.Columns()/tSize + 1;

   for(i=0; i<PA.Rows()/tSize; i++){
      for(j=0; j<PB.Columns()/tSize; j++){
         threads.emplace_back([ThN, iterations, i, j, &PA, &PB, &PX, &tSize, &results]() {
            SingleTileThread(ThN, PX, PA, PB, iterations, i, j, tSize, results);
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
   results.PrintTensor();
   Matrix prova(2, 1);
   prova.CollapseTensor(results);
   prova.PrintTensor();

   cout<<"\n\n-------Somma--------\n\n";
   Matrix Template = A*B;
   Template.PrintMatrix();

   cout<<"Number of Threads: "<<ThN<<endl;

   return 0;
}
   
