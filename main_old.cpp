#include"Functions.hpp"

#ifndef TYPE
#define TYPE int
#endif

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
  //int best_dimensions = 0;
  double speedup[max + 1];
#ifdef CUDA
  double cuda_speedup[max + 1];
  int cuda_tiled_failures = 0;
  int cuda_normal_failures = 0;
#endif
  //double best_result = 0;
  int column_factor = step*form_factor_result;

  cout<<"Rows\tColumns\tthreads\tTile\ttRows\tCols\tOperandsFF\tResultFF\tSerial\tParallel\t\tSpeedup\n";

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
    double serial_time = A.MeasureMultTime(B, X);


#if defined(PRINT_NUMBERS)
    cout<<"Matrix X after serial operation: \n\n";
    X.PrintMatrix();
//    cout<<"Serial execution in "<<serial_time/CLOCKS_PER_SEC<<" seconds.\n\n";
#endif

    int ThN, Rdiv, Cdiv;

    int tSize = BestSquareTiling<TYPE>(A, B, form_factor_result, threads, Rdiv, Cdiv);

    double optimized_time = OpTile<TYPE>(A, B, T, Rdiv, Cdiv);
#ifdef CHECK_RESULT
    if(!(X == T)){
      cout<<"\n\n\033[1;37;41mOptimized tiled op. has failed!\033[0m\n\n";
      return 1;
    }
#endif

    double unoptimized_time = UnopTile<TYPE>(A, B, T, tSize, ThN, Rdiv, Cdiv);
#ifdef CHECK_RESULT
    if(!(X == T)){
      cout<<"\n\n\033[1;37;41mUnoptimized tiled op. has failed!\033[0m\n\n";
      return 2;
    }
#endif

    speedup[p] = (serial_time/min(optimized_time, unoptimized_time));

//NOW THE SECTION THAT USES CUDA
#ifdef CUDA

    double cuda_tiled_time = CudaMult<TYPE>(A, B, T, Rdiv, Cdiv, 1);
#ifdef CHECK_RESULT
    if(!(X == T)){
      cout<<"\n\n\033[1;37;41mTiled cuda op. has failed!\033[0m\n\n";
      //return 4;
      cuda_tiled_failures++;
    }
#endif

/*
    double cuda_normal_time = CudaMult<TYPE>(A, B, T, Rdiv, Cdiv, 0);
#ifdef CHECK_RESULT
    if(!(X == T)){
      cout<<"\n\n\033[1;37;41mNormal cuda op. has failed!\033[0m\n\n";
      //return 3;
      cuda_normal_failures++;
    }
#endif
*/

    cuda_speedup[p] = (serial_time/cuda_tiled_time);
#endif

    if(p == max){

    }else{

#ifndef CUDA
      fprintf(fp, "%d\t%d\t%d\t%d\t%2f\t%2f\t%5f\t%5f\t%5f\t%5f\t\n", 
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



      printf("%d\t%d\t%d\t%d\t%2f\t%2f\t%5f\t%5f\t%5f\t%5f\t\n", 
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
      //segnare miglior risultato per curiosità
#endif

#ifdef CUDA
      fprintf(fp, "%d\t%d\t%d\t%d\t%2f\t%2f\t%5f\t%5f\t%5f\t%5f\t%d\t%5f\t%5f\t\n", 
              X.Rows(),
              X.Columns(),
              ThN, //ThN, previously confirmed the length of the thread vector, now useless
              tSize,
              form_factor_operands,
              form_factor_result,
              serial_time,
              unoptimized_time,
              optimized_time,
              speedup[p],
              BLOCK_SIZE,
//              cuda_normal_time,
              cuda_tiled_time,
              cuda_speedup[p]
             );



      printf("%d\t%d\t%d\t%d\t%2f\t%2f\t%5f\t%5f\t%5f\t%5f\t%d\t%5f\t%5f\t\n", 
              X.Rows(),
              X.Columns(),
              ThN, //ThN, previously confirmed the length of the thread vector, now useless
              tSize,
              form_factor_operands,
              form_factor_result,
              serial_time,
              unoptimized_time,
              optimized_time,
              speedup[p],
              BLOCK_SIZE,
//              cuda_normal_time,
              cuda_tiled_time,
              cuda_speedup[p]
             );
#endif
    }

    fclose(fp);
  }

#ifdef CUDA
  cout<<"Tiled Cuda Failures: "<<cuda_tiled_failures<<endl;
  cout<<"Normal Cuda Failures: "<<cuda_normal_failures<<endl;
#endif

  return 0;
}
