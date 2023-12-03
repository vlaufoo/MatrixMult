#include "Functions.hpp"

template <typename T = int>
double UnopTile(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C, int tSize, int& ThNumber)
{
  using namespace std;
  Matrix<TYPE> PA = A.ForceAddTilingPaddingRows(tSize);
  Matrix<TYPE> PB = B.ForceAddTilingPaddingColumns(tSize);
  Matrix<TYPE> PC = C.ForceAddTilingPadding(tSize);

#ifdef VERBOSE
  cout<<"tSize = "<<tSize<<"\n\n";
#endif

  //calculating iteration number and prepping threads
  std::vector<std::thread> kernels;
  int ThN = 0;
  int iterations = PA.Columns()/tSize;
  if(PA.Columns() % tSize != 0)
    iterations++;

#ifdef VERBOSE
  printf("Matrice destinazione (PA): %p\n", &PA);
#endif


#if defined(PRINT_NUMBERS)
  cout<<"Matrices PA and PB: \n\n";

  //NOW THE MATRICES HAVE PADDING
  PA.PrintMatrix();
  PB.PrintMatrix();
#endif

  //GENERALIZED PARALLEL OPERATION
  
  clock_t tic = clock();

  for(int i=0; i<PA.Rows()/tSize; i++){
    for(int j=0; j<PB.Columns()/tSize; j++){
      kernels.emplace_back(SingleTileThread, ThN, std::ref(PC), std::ref(PA), std::ref(PB), iterations, i, j, tSize);
      ThN++;
    }
  }

  for(auto& thread :kernels){
    thread.join();
  }

  clock_t toc = clock();

  double execution_time = (double)(toc-tic)/CLOCKS_PER_SEC;

#ifdef PRINT_NUMBERS
  cout<<"Matrix PC after parallel operation: \n\n";
  PC.PrintMatrix();
  cout<<"\n";
  cout<<"Parallel execution in "<<execution_time<<" seconds.\n\n";
#endif

  ThNumber = ThN;

  return execution_time;
}
