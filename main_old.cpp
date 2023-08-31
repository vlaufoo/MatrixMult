#include<iostream>
#include<cstdio>
#include"Classes.h"
#include<thread>
#include<vector>
#include<mutex>
#include<ctime>
#include<cmath>
#include<string>

/*
void SingleTileThread(int threadId, Matrix& Destination, Matrix& A, Matrix& B, int iterations, int i, int j, int tSize){

  for(int k=0; k<iterations; k++){
    Destination.MultiplyTilesOnce(A, B, k, i, j, tSize);
  }
}
*/
void MultiplyLoop(Matrix& PY, const Matrix& PA, const Matrix& PB, int iterations, int i, int j, int tSize) {
  for (int k = 0; k < iterations; k++) {
    //MultiplyTilesOnce(PY, PA, PB, i, j, tSize);
  }
}

int main(int argc, char **argv){

  using namespace std;


  if(argc != 5){
    cout<<"Number of arguments is incorrect. Set all of the necessary arguments:\n"<<
      "Threads, Seed, Number of steps, Step height"<<endl;
    exit(-1);
  }

  const unsigned int threads =    atoi(argv[1]);
  const unsigned int seed =       atoi(argv[2]);
  const unsigned int steps_amt =  atoi(argv[3]);
  const unsigned int step =       atoi(argv[4]);

  cout<<"Arguments received: "<<threads<<" "<<seed<<" "<<steps_amt<<" "<<step<<" "<<endl;


  int p;
  int max = steps_amt + threads;
  int best_dimensions = 0;
  double speedup[max];
  double best_result = 0;
  int column_factor = 4*step/threads;

  cout<<"Rows\tColumns\tthreads\tTile\tSerial\t\tParallel\t\tSpeedup\n";

  string filename = "./" + string(argv[1]) + "T_" + string(argv[2]) + ".txt";
  FILE *fp = fopen(filename.c_str(), "w");
  if(fp == NULL){
    cout<<"Errore file.\n";
    return 1;
  }


  for(p = max-1; p>0; p--){

    int tSize = 2*p*step/threads;

    cout<<"tSize = "<<tSize<<endl<<endl;



    Matrix A(p*step, p*step);
    Matrix B(p*step, p*column_factor);

    Matrix X(p*step, p*column_factor);
    Matrix Y(p*step, p*column_factor);

    A.RandomMatrix(100, 350, seed);
    B.RandomMatrix(40, 500, seed);

#if defined(PRINT_NUMBERS)
    cout<<"Matrices A and B: \n\n";
    A.PrintMatrix();
    B.PrintMatrix();



    cout<<p*step<<" righe, "<<p*column_factor<<" colonne.\n\n";
#endif

    std::vector<std::thread> threads;
    int ThN = 0;
    int i, j;

    //testing if it works in serial execution

    clock_t tic_1 = clock();

    X = A*B;

    clock_t toc_1 = clock();

    Matrix PA = A.AddTilingPaddingRows(tSize);
    Matrix PB = B.AddTilingPaddingColumns(tSize);
    Matrix PX = X.AddTilingPadding(tSize);
    Matrix PY = Y.AddTilingPadding(tSize);


    int iterations = PA.Columns()/tSize + 1/*(PA.Columns()%tSize == 0) ? 0 : 1*/;

    cout<<"sono ancora qua";
    int ***result;
    result = new int** [iterations];
    for(int k=0; k<iterations; k++){
      cout<<k;
      result[k] = new int* [PA.Rows()];
      for(int i=0; i<PA.Rows(); i++){
      cout<<i;
        result[k][j] = new int [PB.Columns()];
        for(int j=0; j<PB.Columns(); j++){
      cout<<j;
          result[k][i][j] = 0;
        }
      }
    }

    cout<<endl;

#if defined(PRINT_NUMBERS)
    cout<<"Matrices PA and PB: \n\n";
    PA.PrintMatrix();
    PB.PrintMatrix();
#endif


#if defined(PRINT_NUMBERS)
    cout<<"Matrix X after serial operation: \n\n";
    X.PrintMatrix();
    cout<<"Serial execution in "<<(double)(toc_1 - tic_1)/CLOCKS_PER_SEC<<" seconds.\n\n";
#endif

#if defined(PRINT_NUMBERS)
    cout<<"\n\n\n--------------Fine esecuzione sequenziale----------------\n\n\n";
    cout<<"\n\n\n--------------Inizio esecuzione parallela----------------\n\n\n";
#endif

/*
    for(i=0; i<PA.Rows()/tSize; i++){
      for(j=0; j<PB.Columns()/tSize; j++){
        for(int k=0; k<iterations; k++){
          PY.MultiplyTilesOnce(PA, PB, k, i, j, tSize);
        }
      }
    }
*/

    cout<<"After serial execution:\n";
    PY.PrintMatrix();

    clock_t tic = clock();

    //parallel execution
    for(i=0; i<PA.Rows()/tSize; i++){
      for(j=0; j<PB.Columns()/tSize; j++){
 //       threads.emplace_back(&Matrix::WriteResultTile, std::ref(PY), std::ref(PA), std::ref(PB), iterations, i, j, tSize, result[0]);
        ThN++;
      }
    }

    for(auto& thread :threads){
      thread.join();
    }

    clock_t toc = clock();

#if defined(PRINT_NUMBERS) 
    cout<<"Matrix X after parallel operation: \n\n";
    PY.PrintMatrix();
    cout<<endl;
    cout<<"Parallel execution in "<<(double)(toc-tic)/CLOCKS_PER_SEC<<" seconds.\n\n";
#endif


    speedup[p-1] = ((double)(toc_1-tic_1)/(double)(toc-tic));

    if(p == max - 1){

    }else{
      fprintf(fp, "%d\t%d\t%d\t%d\t%5f\t%5f\t%5f\t", PX.Rows(), PX.Columns(), ThN, tSize, (double)(toc_1-tic_1)/CLOCKS_PER_SEC, (double)(toc-tic)/CLOCKS_PER_SEC, speedup[p-1]);

      std::cout << p * step << "\t"
        << p * column_factor << "\t"
        << ThN << "\t"
        << tSize << "\t"
        << std::fixed << (double)(toc_1-tic_1)/CLOCKS_PER_SEC << "\t"
        << (double)(toc-tic)/CLOCKS_PER_SEC << "\t"
        << speedup[p-1] << "\t";
      if(best_result < speedup[p-1]){
        best_result = speedup[p-1];
        best_dimensions = p-1;
        //PA.PrintMatrix();
        //PB.PrintMatrix();
        cout<<"Best result yet!\n";
        fprintf(fp, "Best result yet!\n");
      }else{
        cout<<endl;
        fprintf(fp, "\n");
      }
    }

    //deallocazione contenitore risultati
    for(int k=0; k<iterations; k++){
      for(i=0; i<PA.Rows(); i++){
        for(j=0; j<PB.Columns(); j++){
          cout<<result[k][i][j]<<"\t\t";
        }
        cout<<"\n\n";
        delete result[k][i];
        cout<<"deleted result (row "<<i<<")\n";
      }
      cout<<"\n\n";
      delete[] result[k];
    }
      delete result;
  }


  cout<<"\n\n\nBest: "<<(best_result)<<" obtained with "<<best_dimensions*step<<" rows, "<<best_dimensions*column_factor<<" columns."<<endl;

  fclose(fp);

  return 0;
}
