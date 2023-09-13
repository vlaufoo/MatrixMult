#include<iostream>
#include<ostream>
#include<cstdio>
#include<thread>
#include<vector>
#include<mutex>
#include<cmath>

  
class Matrix
{

  private:
    int rows;
    int columns;
    int padded_rows;
    int padded_columns;
    int **matrixpt;

  public:


    Matrix(int r, int c){
      rows = r;
      columns = c;
      padded_rows = 0;
      padded_columns = 0;

      matrixpt = new int* [rows];
      for(int i=0; i<rows; i++)
        matrixpt[i] = new int [columns];

      for(int i=0; i<rows; i++){
        for(int j=0; j<columns; j++){
          matrixpt[i][j] = 0;
        }
      }
  std::cout<<"My constructor created a "<<rows<<"x"<<columns<<" matrix.\n";
    }
    
    //copy constructor
    Matrix(const Matrix& other) {
      rows = other.rows;
      columns = other.columns;
      padded_rows = other.padded_rows;
      padded_columns = other.padded_columns;

      matrixpt = new int*[rows];
      for (int i = 0; i < rows; i++) {
        matrixpt[i] = new int[columns];
        for (int j = 0; j < columns; j++) {
          matrixpt[i][j] = other.matrixpt[i][j];
        }
      }
      std::cout<<"Copy constructor created a "<<rows<<"x"<<columns<<" matrix.\n";
    }

    ~Matrix(){
      for(int i=0; i<rows; i++){
        delete[] matrixpt[i];
      }
      delete[] matrixpt;
    }

    //FUNCTIONS
    int Rows(){
      return rows;
    }

    int Columns(){
      return columns;
    }

    int GetElement(int n, int m){
      if(n >= rows || m >= columns){
        std::exit(1);
      }
      return matrixpt[n][m];
    }

    void RandomMatrix(int minVal, int maxVal, const unsigned int seed){
      srand(seed);
      for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
          matrixpt[i][j] = rand() % (maxVal - minVal + 1) + minVal;
        }
      }
    }

    void PrintMatrix(){
      for(int i=0; i<rows; i++){
        for(int j=0; j<columns; j++){
          std::cout<<matrixpt[i][j]<<"\t";
        }
        std::cout<<"\n\n";
      }
      std::cout<<"\n";
    }

    void MultiplyTilesOnce(Matrix A, Matrix B, int IdxAcol, int IdxArow, int IdxBcol, int tSize){
      std::cout<<"\nTile row "<<IdxArow<<" column "<<IdxBcol<<". Iteration "<<IdxAcol<<"\n\n";

      if(A.columns != B.rows){
        std::cout<<"The matrices are incompatible!\n";
        exit(1);
      }

      std::cout<<"Good.\n";

      //all indices are referred to A matrix and transposed for the B matrix
      int tileCstart = IdxBcol*tSize;
      int tileCend = (IdxBcol+1)*tSize;
      int tileRstart = IdxArow*tSize;
      int tileRend = (IdxArow+1)*tSize;
      int ThisTileEnd = (IdxAcol+1)*tSize;
      int ThisTileStart = IdxAcol*tSize;

      //adjustment for when A columns and B rows are not multiple of tSize
      if(ThisTileEnd > A.columns){
        ThisTileEnd = A.columns;
        std::cout<<"Abnormal tile encountered...................."<<std::endl;
      }

      //IdxAcol is equal to the iteration number so in the tile multiplication
      //the index of the destination tile columns is defined with IdxBcol, instead 
      //the inner most loop uses IdxAcol.

      //setting the padding rows and columns depending on the operands
      //padded_rows = A.padded_rows;
      //padded_columns = B.padded_columns;
      //
      std::cout<<"Good..\n";

      if(IdxAcol == 0){
        //if it's the first iteration set destination matrix to 0)
        for(int i=0; i<tSize; i++){
          for(int j=0; j<tSize; j++){
            matrixpt[tileRstart+i][tileCstart+j] = 0;
            std::cout<<matrixpt[tileRstart+i][tileCstart+j]<<", "<<A.matrixpt[tileRstart+i][tileCstart+j]<<"\t";
          }
          std::cout<<"\n";
        }
        std::cout<<"First iter. check is true.\n";
      }
      

      //normal matrix multiplication for one tile
      for(int i=tileRstart; i<tileRend; i++){
        for(int j=tileCstart; j<tileCend; j++){
          for(int k=ThisTileStart; k<ThisTileEnd; k++){
            std::cout<<i<<", "<<j<<", "<<k<<"\n";
            std::cout<<matrixpt[i][j]<<std::endl;
            //std::cout<<matrixpt[i][j]<<"\n";
            //std::cout<<A.matrixpt[i][k]<<"\n";
            //std::cout<<B.matrixpt[8][3]<<"\n";
            matrixpt[i][j] += A.matrixpt[i][k]*B.matrixpt[k][j];
//            matrixpt[i][j] += A.GetElement(i, k)*B.GetElement(k, j);
          }
          std::cout<<"Element "<<i<<" "<<j<<" pass "<<IdxAcol<<" done\n";
        }
      }
    }


    //operators
    Matrix& operator=(const Matrix& other) {
      // Check for self-assignment
      if (this != &other)
      {
        // Deallocate the current memory and allocate the new required space
        for (int i = 0; i < rows; i++) {
          delete[] matrixpt[i];
        }
        delete[] matrixpt;

        // Allocate new memory
        rows = other.rows;
        columns = other.columns;
        padded_rows = other.padded_rows;
        padded_columns = other.padded_columns;

        matrixpt = new int*[rows];
        for (int i = 0; i < rows; i++) {
          matrixpt[i] = new int[columns];
          for (int j = 0; j < columns; j++) {
            matrixpt[i][j] = other.matrixpt[i][j];
          }
        }
      }
      return *this;
    }

    Matrix operator*(Matrix second){

      Matrix result(rows, second.columns);
      if(columns != second.rows){
        std::cout<<"Matrices have incompatible dimensions.\n";
        std::exit(403);
      }

      for(int i=0; i<rows; i++){
        for(int j=0; j<second.columns; j++){
          for(int k=0; k<columns; k++){
            result.matrixpt[i][j]+=matrixpt[i][k]*second.matrixpt[k][j];
          }
        }
      }
      return result;
    }

};

/*
I have omitted many other methods, which seem to work from the little testing i have done of them.
*/


void SingleTileThread(Matrix Destination, Matrix A, Matrix B, int iterations, int i, int j, int tSize){

  for(int k=0; k<iterations; k++){
    Destination.MultiplyTilesOnce(A, B, k, i, j, tSize);
  }
}

int main(int argc, char **argv){

  using namespace std;


  cout<<__cplusplus<<"\n";

  Matrix A(12, 9);
  Matrix B(9, 6);

  Matrix X(12, 6);
  Matrix Y = X;
  Matrix T = X;

  A.RandomMatrix(0, 20, 290083);
  B.RandomMatrix(3, 15, 879273);

  A.PrintMatrix();
  B.PrintMatrix();

  cout<<A.GetElement(0, 0)<<" is A(0, 0)\n";
  cout<<B.GetElement(0, 0)<<" is B(0, 0)\n";

  std::vector<std::thread> threads;

  int tSize = 3;
  int ThN = 0;
  int i, j;
  int iterations = A.Columns()/tSize + (bool)(A.Columns() % tSize) ? 1 : 0;

  X = A*B;
  X.PrintMatrix();
  
  //serial execution
  for(i=0; i<A.Rows()/tSize; i++){
    for(j=0; j<B.Columns()/tSize; j++){
      for(int k=0; k<iterations; k++){
        T.MultiplyTilesOnce(ref(A), ref(B), k, i, j, tSize);
      }
    }
  }

  cout<<"Out of serial section\n\n";

  //parallel execution
  for(i=0; i<A.Rows()/tSize; i++){
    for(j=0; j<B.Columns()/tSize; j++){
      threads.emplace_back(SingleTileThread, ref(Y), ref(A), ref(B), iterations, i, j, tSize);
//      this_thread::sleep_for(chrono::milliseconds(10));
      ThN++;
    }
  }
  
  for(auto& thread :threads){
    thread.join();
  }

  cout<<"Out of the parallel section\n\n";

  Y.PrintMatrix();

  return 0;
}
