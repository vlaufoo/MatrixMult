#include<iostream>
#include<cstdio>
#include"Functions.hpp"
#include<thread>
#include<vector>
#include<mutex>
#include<ctime>
#include<cmath>
#include<string>

//#define TYPE int


int main(int argc, char **argv){

  using namespace std;

  if(argc != 8){
    cout<<"Number of arguments is incorrect. Set all of the necessary arguments:\n"<<
      "Threads, Seed, Number of steps, Step height, FormFactor of operands, FormFactor of result, Start dimensions (in steps)"
      <<endl;
    exit(-1);
  }

  const unsigned int threads =        atoi(argv[1]);
  const unsigned int seed =           atoi(argv[2]);
  const unsigned int steps_amt =      atoi(argv[3]);
  const unsigned int step =           atoi(argv[4]);
  const float form_factor_operands =  atof(argv[5]);
  const float form_factor_result  =   atof(argv[6]);
  const int start_step    =           atoi(argv[7]);
  cout<<"Arguments received: "<<threads<<" "<<seed<<" "<<steps_amt<<" "<<step<<" "<<endl;

  int p;
  int max = steps_amt;
  int best_dimensions = 0;
  double speedup[max + 1];
  double best_result = 0;
  int column_factor = step*form_factor_result;
  int i, j;

  cout<<"Rows\tColumns\tthreads\tTile\tOperandsFF\tResultFF\tSerial\tParallel\t\tSpeedup\n";

  string filename = "./" + string(argv[2]) + "_" + string(argv[3]) +  "_" + string(argv[4]) + "_" + string(argv[7]) + ".txt";
  FILE *fp;


  //THE MAIN LOOP:


  for(p = max; p>0; p--){

    //OPEN LOG FILE:
    fp = fopen(filename.c_str(), "a");
    if(fp == NULL){
      cout<<"Errore file.\n";
      return 1;
    }
    if(p == max)
      fprintf(fp, "\n");

    //INITIALIZE MATRIX:
    Matrix<TYPE> A((start_step+p)*step, (start_step+p)*step*form_factor_operands);
    Matrix<TYPE> B((start_step+p)*step*form_factor_operands, (start_step+p)*column_factor);
    A.RandomMatrix(50, 400, seed);
    B.RandomMatrix(50, 400, seed);
    Matrix<TYPE> X((start_step+p)*step, (start_step+p)*column_factor);
    Matrix<TYPE> Y((start_step+p)*step, (start_step+p)*column_factor);
    //testing if the loop function is slowing down the operation
    Matrix<TYPE> T = X;

#if defined(PRINT_NUMBERS)
    cout<<"Matrices A and B: \n\n";
    A.PrintMatrix();
    B.PrintMatrix();

    cout<<X.Rows()<<" righe, "<<X.Columns()<<" colonne.\n\n";
#endif


    //NORMAL OPERATION
    clock_t tic_1 = clock();
    X = A*B;
    clock_t toc_1 = clock();


#if defined(PRINT_NUMBERS)
    cout<<"Matrix X after serial operation: \n\n";
    X.PrintMatrix();
//    cout<<"Serial execution in "<<(double)(toc_1 - tic_1)/CLOCKS_PER_SEC<<" seconds.\n\n";
#endif

    int big_divider, small_divider, ThN;

    int tSize = BestSquareTiling(A, B, form_factor_result, threads, big_divider, small_divider);

    double serial_time = (double)(toc_1 - tic_1)/CLOCKS_PER_SEC;
    double optimized_time = OpTile(A, B, T, big_divider, small_divider);
    double unoptimized_time = UnopTile(A, B, Y, tSize, ThN);

    speedup[p] = (serial_time/min(optimized_time, unoptimized_time));

#ifdef CUDA
    speedup[p] = (serial_time/min(optimized_time, unoptimized_time, cuda_time));
#endif

    if(p == max){

    }else{
      fprintf(fp, "%d\t%d\t%d\t%d\t%d\t%d\t%2f\t%2f\t%5f\t%5f\t%5f\t%5f\t", 
              X.Rows(),
              X.Columns(),
              ThN, //ThN, previously confirmed the length of the thread vector, now useless
              tSize,
              form_factor_operands,
              form_factor_result,
              serial_time,
              unoptimized_time,
              optimized_time,
              speedup[p]
             );

      std::cout 
        << X.Rows() << "\t"
        << X.Columns() << "\t"
        << ThN << "\t"
        << tSize << "\t"
        << form_factor_operands << "\t"
        << form_factor_result << "\t"
        << std::fixed 
        << serial_time
        << unoptimized_time
        << optimized_time
        << speedup[p] << "\t";

      //segnare miglior risultato per curiosità
      if(best_result < speedup[p]){
        best_result = speedup[p];
        best_dimensions = p;
        //A.PrintMatrix();
        //B.PrintMatrix();
        cout<<"Best result yet!\n";
        fprintf(fp, "Best result yet!\n");
      }else{
        cout<<endl;
        fprintf(fp, "\n");
      }
    }

    fclose(fp);
  }


  cout<<"\n\n\nBest: "<<(best_result)<<" obtained with "<<best_dimensions*step<<" rows, "<<best_dimensions*column_factor<<" columns."<<endl;


  return 0;
}
