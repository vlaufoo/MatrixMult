#include "Functions.hpp"

double OpTile(Matrix<TYPE> &A, Matrix<TYPE> &B, Matrix<TYPE> &C, int& div_1, int& div_2)
{
  using namespace std;
  //PREP FOR THE DISUNIFORM TILING OF THE OPTIMIZED METHOD
  /*
  int Rdiv = div_2;
  int Cdiv = div_1;
  if(form_factor_result <= 1){
    Rdiv = div_1;
    Cdiv = div_2;
  }
  */
  //int tR = A.Rows()/Rdiv;
  //int tC = B.Columns()/Cdiv;
  
  int Rdiv = (A.Rows()>B.Columns()) ? div_1 : div_2;
  int Cdiv = (A.Rows()>B.Columns()) ? div_2 : div_2;

  int tR = A.Rows()/Rdiv;
  int tC = B.Columns()/Cdiv;

  if(A.Rows()%Rdiv != 0)
    tR++;
  if(B.Columns()%Cdiv != 0)
    tC++;

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


  return execution_time;

}
