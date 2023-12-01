#include"Functions.hpp"

void SingleTileThread(int threadId, Matrix<TYPE>& Destination, Matrix<TYPE>& A, Matrix<TYPE>& B, int iterations, int i, int j, int tSize){

  for(int k=0; k<iterations; k++){
    Destination.MultiplyTilesOnce(A, B, k, i, j, tSize);
  }
}
