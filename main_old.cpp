#include<iostream>
#include<cstdio>
#include"Classes.h"
#include<thread>
#include<vector>
#include<mutex>
#include<ctime>
#include<cmath>
#include<string>

#ifndef FORM_FACTOR_OPERANDS
  #define FORM_FACTOR_OPERANDS 1
#endif

#ifndef FORM_FACTOR_RESULT
  #define FORM_FACTOR_RESULT 0.3 
#endif

void SingleTileThread(int threadId, Matrix& Destination, Matrix& A, Matrix& B, int iterations, int i, int j, int tSize){

  for(int k=0; k<iterations; k++){
    Destination.MultiplyTilesOnce(A, B, k, i, j, tSize);
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
  int max = steps_amt;
  int best_dimensions = 0;
  double speedup[max];
  double best_result = 0;
  int column_factor = step*FORM_FACTOR_RESULT;

  cout<<"Rows\tColumns\tthreads\tTile\tSerial\t\tParallel\t\tSpeedup\n";

  string filename = "./" + string(argv[1]) + "T_" + string(argv[2]) + ".txt";
  FILE *fp = fopen(filename.c_str(), "w");
  if(fp == NULL){
    cout<<"Errore file.\n";
    return 1;
  }


  for(p = max; p>0; p--){

    Matrix A(p*step, p*step*FORM_FACTOR_OPERANDS);
    Matrix B(p*step*FORM_FACTOR_OPERANDS, p*column_factor);


    A.RandomMatrix(100, 350, seed);
    B.RandomMatrix(40, 500, seed);


    Matrix X(p*step, p*column_factor);
    Matrix Y(p*step, p*column_factor);

    //standard execution

    clock_t tic_1 = clock();
    X = A*B;
    clock_t toc_1 = clock();

    
    //find the best square tiling for this matrix (allowing for padding)
    int div_1;
    if(FORM_FACTOR_RESULT >= 1){
      div_1 = round(sqrt(threads*FORM_FACTOR_RESULT));
    }else{
      div_1 = round(sqrt(threads/FORM_FACTOR_RESULT));
    }

    cout<<"div_1 = "<<div_1<<endl;
    while(threads % div_1 != 0){
      div_1 = div_1 + 1;
    }

    cout<<"Closest divider is "<<div_1<<"\n";
    int div_2 = threads/div_1;
  
    int tSize;

    if(FORM_FACTOR_RESULT >= 1){
      tSize = X.Columns()/div_1;
      if(X.Columns()%div_1 != 0)
        tSize++;
      //calcola la dimensione tile e aggiustala se è tale che il tiling venga poi fatto
      //in modo improprio. Se le righe di X contengono tSize div_2 volte, ma con un resto 
      //allora otterremmo 3 righe di tile invece che due, quindi ridimensioniamo la tile
      //per averne 2
      A = A.ForceAddTilingPaddingRows(tSize, div_2);
      B = B.ForceAddTilingPaddingColumns(tSize, div_1);
      Y = Y.ForceAddTilingPadding(tSize, div_2, div_1);
 
      cout<<"The tile size is "<<tSize<<", from division of columns by "<<div_1<<endl;
    }else{
      tSize = X.Rows()/div_1 + ceil((X.Columns()%div_1)/div_1);
      if(X.Rows()%div_1 != 0)
        tSize++;

      A = A.ForceAddTilingPaddingRows(tSize, div_1);
      B = B.ForceAddTilingPaddingColumns(tSize, div_2);
      Y = Y.ForceAddTilingPadding(tSize, div_1, div_2);
      //stesso procedimento per questo caso
      cout<<"The tile size is "<<tSize<<", from division of Rows by "<<div_1<<endl;
    }
    cout<<"tSize = "<<tSize<<"\n\n";

#if defined(PRINT_NUMBERS)

    cout<<p*step<<" righe, "<<p*column_factor<<" colonne.\n\n";
#endif



#ifdef VERBOSE
    cout<<"tSize = "<<tSize<<"\n\n";
#endif


#if defined(PRINT_NUMBERS)
    cout<<"Matrices A and B: \n\n";
    A.PrintMatrix();
    B.PrintMatrix();

    cout<<p*step<<" righe, "<<p*column_factor<<" colonne.\n\n";
#endif

    std::vector<std::thread> threads;
    int ThN = 0;
    int i, j;

/*
    if(FORM_FACTOR_RESULT >= 1){
      if(A.Rows()%tSize == 0){
        Matrix A = A;  
      }else{
        Matrix A = A.AddTilingPaddingRows(tSize);
      }

      if(B.Columns()%tSize == 0){
        Matrix B = B; 
      }else{
        Matrix B = B.AddTilingPaddingColumns(tSize);
      }
    }
*/



    int iterations = A.Columns()/tSize;
    if(A.Columns() % tSize != 0)
      iterations++;

#ifdef VERBOSE
    printf("Matrice destinazione (A): %p\n", &A);
#endif


#if defined(PRINT_NUMBERS)
    cout<<"Matrices A and B: \n\n";
    A.PrintMatrix();
    B.PrintMatrix();
#endif


#if defined(PRINT_NUMBERS)
    cout<<"Matrix X after serial operation: \n\n";
    X.PrintMatrix();
    cout<<"Serial execution in "<<(double)(toc_1 - tic_1)/CLOCKS_PER_SEC<<" seconds.\n\n";
#endif

#if defined(PRINT_NUMBERS)
    cout<<"\n--------------Fine esecuzione sequenziale----------------\n\n";
    cout<<"\n--------------Inizio esecuzione parallela----------------\n\n";
#endif

//test dell'esecuzione sequenziale con la MultiplyTiles, ora non necessario

//    for(i=0; i<A.Rows()/tSize; i++){
//      for(j=0; j<B.Columns()/tSize; j++){
//        for(int k=0; k<iterations; k++){
//          Y.MultiplyTilesOnce(A, B, k, i, j, tSize);
//        }
//      }
//    }


//    cout<<"After serial execution:\n";
//    Y.PrintMatrix();


    clock_t tic = clock();

    //parallel execution
    for(i=0; i<A.Rows()/tSize; i++){
      for(j=0; j<B.Columns()/tSize; j++){
        threads.emplace_back(SingleTileThread, ThN, std::ref(Y), std::ref(A), std::ref(B), iterations, i, j, tSize);
        ThN++;
      }
    }

    for(auto& thread :threads){
      thread.join();
    }

    clock_t toc = clock();
#ifdef PRINT_NUMBERS
    cout<<"Matrix Y after parallel operation: \n\n";
    Y.PrintMatrix();
    cout<<"\n";
    cout<<"Parallel execution in "<<(double)(toc-tic)/CLOCKS_PER_SEC<<" seconds.\n\n";
#endif



    speedup[p-1] = ((double)(toc_1-tic_1)/(double)(toc-tic));

    if(p == max - 1){

    }else{
      fprintf(fp, "%d\t%d\t%d\t%d\t%5f\t%5f\t%5f\t", X.Rows(), X.Columns(), ThN, tSize, (double)(toc_1-tic_1)/CLOCKS_PER_SEC, (double)(toc-tic)/CLOCKS_PER_SEC, speedup[p-1]);

      std::cout << p * step << "\t"
        << p * column_factor << "\t"
        << ThN << "\t"
        << tSize << "\t"
        << std::fixed << (double)(toc_1-tic_1)/CLOCKS_PER_SEC << "\t"
        << (double)(toc-tic)/CLOCKS_PER_SEC << "\t"
        << speedup[p-1] << "\t";

      //segnare miglior risultato per curiosità
      if(best_result < speedup[p-1]){
        best_result = speedup[p-1];
        best_dimensions = p-1; //capisci se va corretto con p+1
        //A.PrintMatrix();
        //B.PrintMatrix();
        cout<<"Best result yet!\n";
        fprintf(fp, "Best result yet!\n");
      }else{
        cout<<endl;
        fprintf(fp, "\n");
      }
    }

  }


  cout<<"\n\n\nBest: "<<(best_result)<<" obtained with "<<best_dimensions*step<<" rows, "<<best_dimensions*column_factor<<" columns."<<endl;

  fclose(fp);

  return 0;
}
