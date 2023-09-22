#include<iostream>
#include<cstdio>
#include"Classes.h"
#include<thread>
#include<vector>
#include<mutex>
#include<ctime>
#include<cmath>
#include<string>

void SingleTileThread(int threadId, Matrix& Destination, Matrix& A, Matrix& B, int iterations, int i, int j, int tSize){

  for(int k=0; k<iterations; k++){
    Destination.MultiplyTilesOnce(A, B, k, i, j, tSize);
  }
}


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
    Matrix A((start_step+p)*step, (start_step+p)*step*form_factor_operands);
    Matrix B((start_step+p)*step*form_factor_operands, (start_step+p)*column_factor);
    A.RandomMatrix(50, 400, seed);
    B.RandomMatrix(50, 400, seed);
    Matrix X((start_step+p)*step, (start_step+p)*column_factor);
    Matrix Y((start_step+p)*step, (start_step+p)*column_factor);
    //testing if the loop function is slowing down the operation
    Matrix T = X;

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


    //PREP FOR TILING OF THE MATRICES
    //
    //find the best square tiling for this matrix (allowing for padding)
    int div_1;
    if(form_factor_result >= 1){
      div_1 = round(sqrt(threads*form_factor_result));
    }else{
      div_1 = round(sqrt(threads/form_factor_result));
    }

    if((unsigned)div_1 <= threads){
      while(threads % div_1 != 0){
        div_1 = div_1 + 1;
      }
    }else if(div_1 == 0){
      div_1 = 1;
    }

    cout<<"Closest divider is "<<div_1<<"\n";
    int div_2 = (threads/div_1 > (unsigned)div_1) ? div_1 : threads/div_1;
    div_1 = threads/div_2;
    if(div_2 == 0){
      div_2 = 1;
      div_1 = threads;
    }

    cout<<"div_1 = "<<div_1<<endl;

    //PREP FOR THE DISUNIFORM TILING OF THE OPTIMIZED METHOD
    int Rdiv = div_2;
    int Cdiv = div_1;
    if(form_factor_result <= 1){
      Rdiv = div_1;
      Cdiv = div_2;
    }
  
    int tR = A.Rows()/Rdiv + A.Rows()%Rdiv;
    int tC = B.Columns()/Cdiv + B.Columns()%Cdiv;


    //START OF THE OPTIMIZED PARALLEL OPERATION (NO PADDING NEEDED)
    std::vector<std::thread> tiles;
    clock_t tic_2 = clock();
    //parallel execution
    for(i=0; i<Rdiv; i++){
      for(j=0; j<Cdiv; j++){
        tiles.emplace_back(&Matrix::GetResultTile, ref(T), ref(A), ref(B), i, j, tR, tC);
      }
    }

    for(auto& thread :tiles){
      thread.join();
    }

    clock_t toc_2 = clock();

#ifdef PRINT_NUMBERS
    cout<<"Matrix T after parallel operation: \n\n";
    T.PrintMatrix();
    cout<<"\n";
    cout<<"Simple execution in "<<(double)(toc_2-tic_2)/CLOCKS_PER_SEC<<" seconds.\n\n";
#endif


    //PADDING OF THE MATRICES FOR THE GENERAL PARALLEL OPERATION
    int tSize;
    if(form_factor_result >= 1){
      tSize = X.Rows()/div_2;
      while(X.Rows() > div_2*tSize || X.Columns() > div_1*tSize){
        tSize++;
        //cout<<tSize<<endl;
      }
      //calcola la dimensione tile e aggiustala se è tale che il tiling venga poi fatto
      //in modo improprio. Se le righe di X contengono tSize div_2 volte, ma con un resto 
      //allora otterremmo 3 righe di tile invece che due, quindi ridimensioniamo la tile
      //per averne 2
      A = A.ForceAddTilingPaddingRows(tSize, div_2);
      B = B.ForceAddTilingPaddingColumns(tSize, div_1);
      Y = Y.ForceAddTilingPadding(tSize, div_2, div_1);
 
      cout<<"The tile size is "<<tSize<<", from division of Rows by "<<div_2<<endl;
    }else{
      tSize = X.Columns()/div_2;

      while(X.Columns() > div_2*tSize || X.Rows() > div_1*tSize){
        tSize++;
        //cout<<tSize<<endl;
      }

      A = A.ForceAddTilingPaddingRows(tSize, div_1);
      B = B.ForceAddTilingPaddingColumns(tSize, div_2);
      Y = Y.ForceAddTilingPadding(tSize, div_1, div_2);
      //stesso procedimento per questo caso
      cout<<"The tile size is "<<tSize<<", from division of Columns by "<<div_2<<endl;
    }

    //cout<<"tSize = "<<tSize<<"\n\n";


    //cout<<p*step<<" righe, "<<p*column_factor<<" colonne.\n\n";


#ifdef VERBOSE
    cout<<"tSize = "<<tSize<<"\n\n";
#endif

    //calculating iteration number and prepping threads
    std::vector<std::thread> threads;
    int ThN = 0;
    int iterations = A.Columns()/tSize;
    if(A.Columns() % tSize != 0)
      iterations++;

#ifdef VERBOSE
    printf("Matrice destinazione (A): %p\n", &A);
#endif


#if defined(PRINT_NUMBERS)
    cout<<"Matrices A and B: \n\n";
    //NOW THE MATRICES HAVE PADDING
    A.PrintMatrix();
    B.PrintMatrix();
#endif


#if defined(PRINT_NUMBERS)
    cout<<"\n--------------Fine esecuzione sequenziale----------------\n\n";
    cout<<"\n--------------Inizio esecuzione parallela----------------\n\n";
#endif


    //GENERALIZED PARALLEL OPERATION
    clock_t tic = clock();

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



    speedup[p] = ((double)(toc_1-tic_1)/min((double)(toc-tic), (double)(toc_2-tic_2)));

    if(p == max){

    }else{
      fprintf(fp, "%d\t%d\t%d\t%d\t%d\t%d\t%2f\t%2f\t%5f\t%5f\t%5f\t%5f\t", 
              X.Rows(),
              X.Columns(),
              ThN,
              tSize,
              tR,
              tC,
              form_factor_operands,
              form_factor_result,
              (double)(toc_1-tic_1)/CLOCKS_PER_SEC,
              (double)(toc-tic)/CLOCKS_PER_SEC,
              (double)(toc_2-tic_2)/CLOCKS_PER_SEC,
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
        << (double)(toc_1-tic_1)/CLOCKS_PER_SEC << "\t"
        << (double)(toc-tic)/CLOCKS_PER_SEC << "\t"
        << (double)(toc_2-tic_2)/CLOCKS_PER_SEC << "\t"
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
