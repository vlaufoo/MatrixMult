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

  if(argc != 7){
    cout<<"Number of arguments is incorrect. Set all of the necessary arguments:\n"<<
      "Threads, Seed, Number of steps, Step height, FormFactor of operands, FormFactor of result"
      <<endl;
    exit(-1);
  }

  const unsigned int threads =        atoi(argv[1]);
  const unsigned int seed =           atoi(argv[2]);
  const unsigned int steps_amt =      atoi(argv[3]);
  const unsigned int step =           atoi(argv[4]);
  const float form_factor_operands =  atof(argv[5]);
  const float form_factor_result  =   atof(argv[6]);

  cout<<"Arguments received: "<<threads<<" "<<seed<<" "<<steps_amt<<" "<<step<<" "<<endl;

  int p;
  int max = steps_amt;
  int best_dimensions = 0;
  double speedup[max + 1];
  double best_result = 0;
  int column_factor = step*form_factor_result;

  cout<<"Rows\tColumns\tthreads\tTile\tOperandsFF\tResultFF\tSerial\t\tParallel\t\tSpeedup\n";

  string filename = "./" + string(argv[2]) + "_" + string(argv[3]) +  "_" + string(argv[4]) + ".txt";
  FILE *fp;


  for(p = max; p>0; p--){

    
    fp = fopen(filename.c_str(), "a");
    if(fp == NULL){
      cout<<"Errore file.\n";
      return 1;
    }

    if(p == max)
      fprintf(fp, "\n");

    Matrix A(p*step, p*step*form_factor_operands);
    Matrix B(p*step*form_factor_operands, p*column_factor);


    A.RandomMatrix(50, 400, seed);
    B.RandomMatrix(50, 400, seed);


    Matrix X(p*step, p*column_factor);
    Matrix Y(p*step, p*column_factor);

    //standard execution

    clock_t tic_1 = clock();
    X = A*B;
    clock_t toc_1 = clock();

    
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
    if(form_factor_result >= 1){
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



    speedup[p] = ((double)(toc_1-tic_1)/(double)(toc-tic));

    if(p == max){

    }else{
      fprintf(fp, "%d\t%d\t%d\t%d\t%f\t%f\t%5f\t%5f\t%5f\t", 
              X.Rows(), 
              X.Columns(), 
              ThN, 
              tSize, 
              form_factor_operands, 
              form_factor_result, 
              (double)(toc_1-tic_1)/CLOCKS_PER_SEC, 
              (double)(toc-tic)/CLOCKS_PER_SEC, speedup[p]
             );

      std::cout << p * step << "\t"
        << p * column_factor << "\t"
        << ThN << "\t"
        << tSize << "\t"
        << form_factor_operands << "\t"
        << form_factor_result << "\t"
        << std::fixed << (double)(toc_1-tic_1)/CLOCKS_PER_SEC << "\t"
        << (double)(toc-tic)/CLOCKS_PER_SEC << "\t"
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
