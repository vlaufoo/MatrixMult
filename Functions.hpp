#include<iostream>
#include<sstream>
#include<random>
#include<thread>
#include<ctime>
#include<cstdio>
#include<vector>
#include<type_traits>
#include<cmath>
#include<string>


#ifdef CUDA
#include<assert.h>
#include<cuda_runtime.h>
#include<cuda_profiler_api.h>
#include<helper_functions.h>
#include<helper_cuda.h>

template <typename T = int>
struct mat {
  int width; 
  int height;
  int padd_width;
  int padd_height;
  T* elements;
};

#endif


template <typename T = int>
class Matrix {
private:
  int rows;
  int columns;
  int padded_rows;
  int padded_columns;
  T **matrixpt;
  int i, j, k;

public:


  //CONSTRUCTORS
  Matrix(){
    rows=100;
    columns=100;
    padded_rows = 0;
    padded_columns = 0;

    matrixpt = new T* [rows];
    for(i=0; i<rows; i++)
      matrixpt[i] = new T [columns];
    for(i=0; i<rows; i++){
      for(j=0; j<columns; j++){
        matrixpt[i][j] = 0;
      }
    }
#ifdef VERBOSE
    std::cout<<"Creo matrice "<<rows<<"x"<<columns<<"\n";
#endif
  }

  Matrix(int r, int c){
    rows = r;
    columns = c;
    padded_rows = 0;
    padded_columns = 0;

    matrixpt = new T* [rows];
    for(i=0; i<rows; i++)
      matrixpt[i] = new T [columns];

    for(i=0; i<rows; i++){
      for(j=0; j<columns; j++){
        matrixpt[i][j] = 0;
      }
    }
#ifdef VERBOSE
    std::cout<<"Creo matrice "<<rows<<"x"<<columns<<"\n";
#endif
  }

  // Copy constructor
  Matrix(const Matrix& other){

    rows = other.rows;
    columns = other.columns;
    padded_rows = other.padded_rows;
    padded_columns = other.padded_columns;
#ifdef VERBOSE      
    std::cout<<"Constructing a copy of a "<<rows<<"x"<<columns<<" matrix. From ("<<(void*)&other<<") to ("<<(void*)this<<")\n";
#endif
    matrixpt = new T*[rows];
    for (int i = 0; i < rows; i++) {
      matrixpt[i] = new T[columns];
      for (int j = 0; j < columns; j++) {
        matrixpt[i][j] = other.matrixpt[i][j];
      }
    }
  }

  // Copy assignment operator
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

      matrixpt = new T*[rows];
      for (int i = 0; i < rows; i++) {
        matrixpt[i] = new T[columns];
        for (int j = 0; j < columns; j++) {
          matrixpt[i][j] = other.matrixpt[i][j];
        }
      }
    }
    return *this;
  }

  //FUNCTIONS
  int Rows(){
    return rows;
  }

  int Columns(){
    return columns;
  }

  int PaddingRows(){
    return padded_rows;
  }

  int PaddingColumns(){
    return padded_columns;
  }

  void SetElement(int n, int m, T val){
    if(n >= rows || m >= columns){
#ifdef VERBOSE
      std::cout<<"Invalid indices.";
#endif
    }
    matrixpt[n][m] = val;
  }

  void ZeroElement(int n, int m){
    if(n >= rows || m >= columns){
#ifdef VERBOSE
      std::cout<<"Invalid indices.";
#endif
    }
    matrixpt[n][m] = 0;
  }

  T GetElement(int n, int m){
    if(n >= rows || m >= columns){
#ifdef VERBOSE
      std::cout<<"Invalid indices.";
#endif
    }
    return matrixpt[n][m];
  }

  void Identity(){
    for(i=0; i<rows; i++){
      for(j=0; j<columns; j++){
        if(i==j){
          matrixpt[i][j] = 1;
        }else{
          matrixpt[i][j] = 0;
        }
      }
    }
  }

  void PrintMatrix(){
    for(i=0; i<rows; i++){
      for(j=0; j<columns; j++){
        std::cout<<matrixpt[i][j]<<"\t";
        //		 std::cout<<"Printed element "<<i<<", "<<j<<"\n";
      }
      std::cout<<"\n\n";
    }
    std::cout<<"\n";
  }

  void ZeroMatrix(){
    for(i=0; i<rows; i++){
      for(j=0; j<columns; j++){
        matrixpt[i][j] = 0;
      }
    }
  }

  void TestMatrix(){
    i=0;
    k=0;
    while(i<rows*columns){
      matrixpt[i/columns][i%columns] = (T)k++;
      if(k > 128)
        k=0;
      i++;
    }
  }

  void RandomMatrix(const T min, const T max, const unsigned int seed){
    srand(seed);

    for(int i=0; i<rows; i++){
      for(int j=0; j<columns; j++){
        matrixpt[i][j] = (T)min + (T)rand()/(T)(RAND_MAX/(max-min));
      }
    }
  }

  void InitMatrix(T* values, int length){
    if(length != rows*columns){
      std::cout<<"!\n!\n!\nData length and matrix dimentions are incompatilble!\n!\n!\n";
      std::cout<<"Check that the bounds correspond.\n";
    }

    for(i=0; i<length; i++){
      k = i/columns;
      j = i%columns;
      matrixpt[k][j] = *(values + i);
    }

  }

  void BlurtMatrix(T* values){
    for(i=0; i<rows; i++){
      for(j=0; j<columns; j++){
        values[i*columns + j] = matrixpt[i][j];
      }
    }
  }

  Matrix ForceAddTilingPaddingRows(int tSize){

    int padding_rows = (rows%tSize == 0) ? tSize : (tSize-rows%tSize);

    Matrix temp(rows+padding_rows, columns);
    Matrix result = *this || temp;
#ifdef VERBOSE
    printf("matrice chiamante: %p, matrice ritornata: %p\n", this, &result);
#endif
    result.padded_rows = padding_rows+padded_rows;
    return result;
  }

  Matrix ForceAddTilingPaddingColumns(int tSize){

    int padding_columns = (columns%tSize == 0) ? tSize : (tSize-columns%tSize);

    Matrix temp(rows, columns+padding_columns);
    Matrix result = *this || temp;
    result.padded_columns = padding_columns+padded_columns;
    return result;
  }

  Matrix ForceAddTilingPadding(int tSize){

    int padding_rows = (rows%tSize == 0) ? tSize : (tSize-rows%tSize);
    int padding_columns = (columns%tSize == 0) ? tSize : (tSize-columns%tSize);

    Matrix temp(rows+padding_rows, columns+padding_columns);
    Matrix result = *this || temp;
    result.padded_rows = padding_rows + padded_rows;
    result.padded_columns = padding_columns + padded_columns;
    return result;
  }

  Matrix AddTilingPaddingRows(int tSize){
    if(rows%tSize == 0){
      std::cout<<"Copia la matrice chiamante\n";
      return *this;
    }else{
      int padding_rows = tSize - rows%tSize;

      Matrix temp(rows+padding_rows, columns);
      Matrix result = *this || temp;
      result.padded_rows = padding_rows+padded_rows;
      return result;
    }
  }

  Matrix AddTilingPaddingColumns(int tSize){
    if(columns%tSize == 0){
      return *this;
    }else{
      int padding_columns = tSize - columns%tSize;

      Matrix temp(rows, columns+padding_columns);
      Matrix result = *this || temp;
      result.padded_columns = padding_columns+padded_columns;
      return result;
    }
  }


  Matrix AddPaddingRows(int padding_rows){
    Matrix temp(rows+padding_rows, columns);
    Matrix result = *this || temp;
    result.padded_rows = padding_rows+padded_rows;
    return result;
  }

  Matrix AddPaddingColumns(int padding_columns){

    Matrix temp(rows, columns+padding_columns);
    Matrix result = *this || temp;
    result.padded_columns = padding_columns+padded_columns;
    return result;
  }

  Matrix AddPadding(int p_r, int p_c){
    Matrix temp(rows + p_r, columns + p_c);
    Matrix result = *this || temp;
    result.padded_rows = padded_rows + p_r;
    result.padded_columns = padded_columns + p_c;
    return result;
  }

  Matrix AddTilingPadding(int tSize){
    if(rows%tSize == 0 && columns%tSize == 0){
      return *this;
    }else{
      int padding_rows = 0;
      int padding_columns = 0;

      if(rows%tSize != 0)
        padding_rows = tSize-rows%tSize;
      if(columns%tSize != 0)
        padding_columns = tSize-columns%tSize;

      Matrix temp(rows+padding_rows, columns+padding_columns);
      Matrix result = *this || temp;
      result.padded_rows = padding_rows + padded_rows;
      result.padded_columns = padding_columns + padded_columns;
      return result;
    }
  }

  Matrix RemovePaddingRows(){
    Matrix result(rows-padded_rows, columns);
    for(i=0; i<result.rows; i++){
      for(j=0; j<result.columns; j++){
        result.matrixpt[i][j] = matrixpt[i][j];
      }
    }

    return result;
  }

  Matrix RemovePaddingColumns(){
    Matrix result(rows, columns-padded_columns);

    for(i=0; i<result.rows; i++){
      for(j=0; j<result.columns; j++){
        result.matrixpt[i][j] = matrixpt[i][j];
      }
    }

    return result;
  }

  Matrix RemovePadding(){
    Matrix result(rows-padded_rows, columns-padded_columns);

    for(i=0; i<result.rows; i++){
      for(j=0; j<result.columns; j++){
        result.matrixpt[i][j] = matrixpt[i][j];
      }
    }

    return result;
  }


  //tiling of the operands
  void GetResultTile(Matrix &A, Matrix &B, int i, int j, int tR, int tC){

    int tileStartR = i*tR;
    int tileEndR = (i+1)*tR;
    int tileStartC = j*tC;
    int tileEndC = (j+1)*tC;

    if(tileEndR > A.rows)
      tileEndR = A.rows;

    if(tileEndC > B.columns)
      tileEndC = B.columns;

    for(int r=tileStartR; r < tileEndR; r++){
      for(int c=tileStartC; c < tileEndC; c++){
        for(int k=0; k<A.columns; k++){
          matrixpt[r][c] += A.matrixpt[r][k] * B.matrixpt[k][c];
        }
      }
    }
  }


  //tiling of the result matrix

  void MultiplyTilesOnce(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize){
    std::stringstream msg;
#ifdef VERBOSE
    msg << "\nTile row "<<IdxArow<<" column "<<IdxBcol<<". Iteration "<<IdxAcol<<"\n\n";
    std::cout<< msg.str();
    msg.str("");
#endif

    if(A.columns != B.rows){
      std::cout<<"The matrices are incompatible!\n";
      exit(1);
    }


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
#ifdef VERBOSE
      std::cout<<"Abnormal tile encountered....................\n";
#endif
    }

    //IdxAcol is equal to the iteration number so in the tile multiplication
    //the index of the destination tile columns is defined with IdxBcol, instead 
    //the inner most loop uses IdxAcol.

    //setting the padding rows and columns depending on the operands


    if(IdxAcol == 0){
      //if it's the first iteration set destination matrix to 0)
      if(IdxArow == 0 && IdxBcol == 0){
        padded_rows = A.padded_rows;
        padded_columns = B.padded_columns;
#ifdef PRINT_NUMBERS
        std::cout<<"Beginning tiled multiplication: padding rows/columns copied from operands to result.\n";
        msg << "Padd rows = "<<padded_rows<<" Padd cols = "<<padded_columns<<std::endl;
        std::cout<<msg.str();
        msg.str("");
#endif
      }
#ifdef VERBOSE
      std::cout<<"First iter. check is true.\n";
#endif
    }

    //normal matrix multiplication for one tile
    for(int i=tileRstart; i<tileRend; i++){
      for(int j=tileCstart; j<tileCend; j++){
        for(int k=ThisTileStart; k<ThisTileEnd; k++){
          //std::cout<<matrixpt[i][j]<<"\n";
          //std::cout<<A.matrixpt[i][k]<<"\n";
          //std::cout<<B.matrixpt[8][3]<<"\n";
          matrixpt[i][j] += A.matrixpt[i][k]*B.matrixpt[k][j];
#ifdef VERBOSE
          msg << i<<", "<<j<<", "<<k<<"\n"<<"sum is now: "<<matrixpt[i][j]<<"\n";
          std::cout<< msg.str();
          msg.str("");
#endif
        }
#ifdef VERBOSE
        msg << "Element "<<i<<" "<<j<<" pass "<<IdxAcol<<" done\n";
        std::cout<<msg.str();
        msg.str("");
#endif
      }
    }
  }

#ifdef CUDA

  void TransferMatrix(T* data){
    for (int i = 0; i<rows; ++i){
      for(int j = 0; j<columns; ++j){
        data[i*columns + j] = matrixpt[i][j];
      }
    }
  }

  //GPU implementation  (CUDA) (commented since it's unadvisable to do it in a class, especially
  //                            if the object I'll be using will be in host memory)
  //  __global__ void MultiplyTilesOnce_CUDA(Matrix& A, Matrix& B, int tSize){
  //
  //    namespace cg = cooperative_groups;
  //    cg::thread_block cta = cg::this_thread_block();
  //
  //    __shared__ T shareA[tSize][tSize];
  //    __shared__ T shareB[tSize][tSize];
  //
  //    int bx = blockIdx.x;
  //    int by = blockIdx.y;
  //    int tx = threadIdx.x;
  //    int ty = threadIdx.y;
  //
  //    int row = by*tSize + ty;
  //    int column = bx*tSize + tx;
  //    T temp = 0;
  //
  //    for (int i = 0; i<columns/tSize; ++i){
  //      shareA[tx][ty] = A.matrixpt[row][i*tSize + ty];
  //      shareB[tx][ty] = B.matrixpt[i*tSize + tx][column];
  //      __syncthreads();
  //
  //      for (int k = 0; k < tSize; ++k){
  //        temp += shareA[ty][k] * shareB[k][tx];
  //        __syncthreads();
  //      }
  //    }
  //    //Questa siga non la capisco, temp è uno scalare o una matrice?
  //    C.matrixpt[row][column] = temp;
  //  }
#endif


  //OPERATORS
  Matrix operator+(Matrix second){
    if(rows != second.rows || columns != second.columns){
      std::cout<<"\e[93mMatrix dimensions don't match in the sum operation.\e[39m\n";
      std::exit(403);
    }
    Matrix result(rows, columns);

    for(i=0; i<rows; i++){
      for(j=0; j<columns; j++){
        result.matrixpt[i][j] = matrixpt[i][j]+second.matrixpt[i][j];
      }
    }

#ifdef VERBOSE
    std::cout<<"Sum done. \n";
#endif
    return result;
  }

  Matrix operator||(Matrix second){   //an extended sum where if the value of index 
    //i, j i spresent in one matrix but not the other
    //that value is taken as the result, so the result matrix
    //always has the most rows and most columns among the
    //operands
#ifdef VERBOSE
    if(rows != second.rows || columns != second.columns){
      std::cout<<"\e[93mMatrix dimensions don't match in the sum operation.\e[39m\n";
    }
#endif
    Matrix result(std::max(rows, second.rows), std::max(columns, second.columns));

    for(i=0; i<result.rows; i++){
      if(i<rows && i<second.rows){
        for(j=0; j<result.columns; j++){
          if(j<columns && j<second.columns){
            result.matrixpt[i][j] = matrixpt[i][j]+second.matrixpt[i][j];
          }else if(j<columns){
            result.matrixpt[i][j] = matrixpt[i][j];
          }else{
            result.matrixpt[i][j] = second.matrixpt[i][j];
          }
#ifdef VERBOSE
          std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
#endif
        }
      }else if(i<rows){
        for(j=0; j<result.columns; j++){
          if(j<columns){
            result.matrixpt[i][j] = matrixpt[i][j];
          }else{
            result.matrixpt[i][j] = 0;
          }
#ifdef VERBOSE
          std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
#endif
        }
      }else{
        for(j=0; j<result.columns; j++){
          if(j<second.columns){
            result.matrixpt[i][j] = second.matrixpt[i][j];
          }else{
            result.matrixpt[i][j] = 0;
          }
#ifdef VERBOSE
          std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
#endif
        }
      }
    }

#ifdef VERBOSE
    std::cout<<"Sum done. \n";
#endif
    return result;
  }

  Matrix operator*(Matrix second){

    Matrix result(rows, second.columns);
    if(columns != second.rows){
      std::cout<<"Matrices have incompatible dimensions.\n";
      std::exit(403);
    }

    for(i=0; i<rows; i++){
      for(j=0; j<second.columns; j++){
        for(k=0; k<columns; k++){
          result.matrixpt[i][j]+=matrixpt[i][k]*second.matrixpt[k][j];
          //		    std::cout<<i<<", "<<k<<"\t";
        }
        //		 std::cout<<"Risultato indice ("<<i<<", "<<j<<"):\n";
        //		 std::cout<<result.matrixpt[i][j]<<"\n";
      }
    }
    return result;
  }

  bool operator==(Matrix second){

    for(int i=0; i<std::min(rows, second.rows); i++){
      for(int j=0; j<std::min(columns, second.columns); j++){
        if(matrixpt[i][j] != second.matrixpt[i][j])
          return false;
      }
    }

    return true;
  }



  //DESTRUCTOR
  ~Matrix(){
#ifdef VERBOSE
    std::stringstream msg;
    msg << "Distruggo matrice "<<rows<<"x"<<columns<<"("<<(void*)this<<")...\n";
    std::cout << msg.str();
    msg.str("");
#endif
    for(i=0; i<rows; i++){
      delete[] matrixpt[i];
#ifdef VERBOSE
      msg << "Cancellata riga "<<i<<".\n";
      std::cout << msg.str();
      msg.str("");
#endif
    }
    delete[] matrixpt;
#ifdef VERBOSE
    msg << "Cancellata tutta la matrice "<<rows<<"x"<<columns<<".\n";
    std::cout << msg.str();
    msg.str("");
#endif
  }


};

#ifdef CUDA
#include "CudaFunctions.cu"
#endif

//OTHER FUNCTIONS
template <typename T = int>
void SingleTileThread(int threadId, Matrix<T>& Destination, Matrix<T>& A, Matrix<T>& B, int iterations, int i, int j, int tSize){

  for(int k=0; k<iterations; k++){
    Destination.MultiplyTilesOnce(A, B, k, i, j, tSize);
  }
}
template <typename T = int>
int BestSquareTiling(Matrix<T>& A, Matrix<T>& B, int form_factor_result, int threads, int& big_div, int &small_div)
{
  using namespace std;
  //PREP FOR TILING OF THE MATRICES

  //find the best square tiling for this matrix (allowing for padding)
  int div_1;
  if(form_factor_result >= 1){
    div_1 = round(sqrt(threads*form_factor_result));
  }else{
    div_1 = round(sqrt(threads/form_factor_result));
  }

  if(div_1 <= threads){
    while(threads % div_1 != 0){
      div_1 = div_1 + 1;
    }
  }else if(div_1 == 0){
    div_1 = 1;
  }

  cout<<"Closest divider is "<<div_1<<"\n";
  int div_2 = (threads/div_1 > div_1) ? div_1 : threads/div_1;
  div_1 = threads/div_2;
  if(div_2 == 0){
    div_2 = 1;
    div_1 = threads;
  }

  small_div = div_2;
  big_div = div_1;

  int tSize;
  if(form_factor_result >= 1){
    tSize = A.Rows()/div_2;
    while(A.Rows() > div_2*tSize || B.Columns() > div_1*tSize){
      tSize++;
    }
  }else{
    tSize = B.Columns()/div_2;
    while(A.Rows() > div_1*tSize || B.Columns() > div_2*tSize){
      tSize++;
    }
  }

  return tSize;

}
template <typename T = int>
double UnopTile(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C, int tSize, int& ThNumber)
{
  using namespace std;
  Matrix<T> PA = A.ForceAddTilingPaddingRows(tSize);
  Matrix<T> PB = B.ForceAddTilingPaddingColumns(tSize);
  Matrix<T> PC = C.ForceAddTilingPadding(tSize);

#ifdef VERBOSE
  cout<<"tSize = "<<tSize<<"\n\n";
#endif

  //calculating iteration number and prepping threads
  vector<thread> kernels;
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
      kernels.emplace_back(SingleTileThread<T>, ThN, ref(PC), ref(PA), ref(PB), iterations, i, j, tSize);
      ThN++;
    }
  }

  for(auto& thread : kernels){
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


template <typename T = int>
double OpTile(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C, int& div_1, int& div_2)
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
      tiles.emplace_back(&Matrix<T>::GetResultTile, ref(C), ref(A), ref(B), i, j, tR, tC);
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

