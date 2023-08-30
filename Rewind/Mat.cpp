#include"Mat.hpp"


Matrix::Matrix(){
  rows=100;
  columns=100;
  depth=1;
  padded_layers = 0;
  padded_rows = 0;
  padded_columns = 0;
  data = new int* [rows];
  for(i=0; i<rows; i++)
    data[i] = new int [columns];
  for(i=0; i<rows; i++){
    for(j=0; j<columns; j++){
      data[i][j] = 0;
    }
  }

  std::cout<<"Creo matrice "<<rows<<"x"<<columns<<"\n";
}

Matrix::Matrix(int r, int c){
  rows = r;
  columns = c;
  depth = 1;
  padded_layers = 0;
  padded_rows = 0;
  padded_columns = 0;
  data = new int* [rows];
  for(i=0; i<rows; i++)
    data[i] = new int [columns];

  for(i=0; i<rows; i++){
    for(j=0; j<columns; j++){
      data[i][j] = 0;
    }
  }

  std::cout<<"Creo matrice "<<rows<<"x"<<columns<<"\n";
}

// Copy constructor
Matrix::Matrix(Matrix& other) {
  rows = other.rows;
  columns = other.columns;
  padded_rows = other.padded_rows;
  padded_columns = other.padded_columns;
  std::cout<<"Matrix ("<<rows<<"x"<<columns<<")compy constructor has been called.\n\n";

  data = new int*[rows];
  for (int i = 0; i < rows; i++) {
    data[i] = new int[columns];
    for (int j = 0; j < columns; j++) {
      data[i][j] = other.data[i][j];
    }
  }
}

//FUNCTIONS
void Matrix::Identity(){
  for(i=0; i<rows; i++){
    for(j=0; j<columns; j++){
      if(i==j){
        data[i][j] = 1;
      }else{
        data[i][j] = 0;
      }
    }
  }
}

int Matrix::GetElement(int n, int m){
  if(n >= rows || m >= columns){
    std::cout<<"Invalid indices.\n";
  }
  return data[n][m];
}

int Matrix::SetElement(int n, int m, int val){
  if(n >= rows || m >= columns){
    std::cout<<"Invalid indices.\n";
  }
  data[n][m] = val;
  return val;
}

int Matrix::ZeroElement(int n, int m){
  if(n >= rows || m >= columns){
    std::cout<<"Invalid indices.\n";
  }
  data[n][m] = 0;
  return 0;
}

Tensor Matrix::ChangeLayerFromMatrix(Tensor& A, int layer_idx){
  if(A.depth <= layer_idx || rows > A.rows || columns > A.columns){
    std::cout<<"Error in ChangeLayerMatrix: The matrix is to big or the layer index is too big.";
  }

  Tensor result(depth, rows, columns);
  for(int l=0; l<depth; l++){
    for(int i=0; i<rows; i++){
      for(int j=0; j<columns; j++){
        result.data[l][i][j] = A.data[l][i][j];
      }
    }

  }


  result.padded_rows = padded_rows;
  result.padded_columns = padded_columns;
  result.padded_layers = padded_layers;

  for(int i=0; i<rows; i++){
    for(int j=0; j<columns; j++){
      A.data[layer_idx][i][j] = data[i][j];
    }
  }

  return result;
}

Tensor Matrix::AppendMatrix(Tensor& A){

  if(rows != A.rows || columns != A.columns){
    std::cout<<"Error in AppendLayerMatrix: matrix dimensions are too big";
  }

  Tensor result(depth+1, rows, columns);
  for(int l=0; l<depth; l++){
    for(int i=0; i<rows; i++){
      for(int j=0; j<columns; j++){
        result.data[l][i][j] = A.data[l][i][j];
      }
    }
  }

  result.padded_rows = padded_rows;
  result.padded_columns = padded_columns;
  result.padded_layers = padded_layers;

  for(int i=0; i<rows; i++){
    for(int j=0; j<columns; j++){
      result.data[depth][i][j] = data[i][j];
    }
  }

  return result;
}
//to be added:
//--a function that appends a Matrix object as a layer of the Tensor
//    (can only e called by a tensor object) [ done! ]
//--a function that collapses a tensor into a matrix by adding all the layers
//    (can only be called by a Matrix object since the return value is of type Matrix)
//--a new tiling function that writes to a tensor instead of a pointer.




void Matrix::CollapseTensor(const Tensor& T){
  if(T.rows != rows || T.columns != columns){
    for (int i = 0; i < rows; i++) {
      delete[] data[i];
    }
    delete[] data;

    rows = T.rows;
    columns = T.columns;
    padded_rows = T.padded_rows;
    padded_columns = T.padded_columns;

    // Deallocate the current memory and allocate the new required space

    data = new int*[rows];
    for(int i = 0; i < rows; i++){
      data[i] = new int[columns];
      for(int j = 0; j < columns; j++){
        data[i][j] = 0;
        for(int k=0; k<T.depth; k++){
          data[i][j] += T.data[k][i][j];
        }
      }
    }      
  }
}

//TILING FUNCTIONS

//tiling of the result matrix
void Matrix::CalculateTile(Matrix A, Matrix B, int tSize, int tRow, int tCol){
  //this does the product of one tSizextSize tile, at position (tRow, tCol)
  //int the A matrix, and puts the result in the calling matrix, which has 
  //the destinetion tile inizialied at zero before the calculation;
  //the case in which the tiling is not perfect (not all tiles are square) is 
  //accounted for with the STD::min functions below

  int start_row = tRow * tSize;
  int start_column = tCol * tSize;
  int end_row = std::min(start_row + tSize, A.rows);
  int end_column = std::min(start_column + tSize, B.columns);

  for(i=start_row; i<end_row; i++){
    for(j=start_column; j<end_column; j++){
      data[i][j] = 0;
    }
  }

  for(i=start_row; i<end_row; i++){
    for(j=start_column; j<end_column; j++){
      for(k=0; k<A.columns; k++){
        data[i][j] += A.data[i][k]*B.data[k][j];
      }
      // std::cout<<"Element ("<<i<<","<<j<<") = "<<sum<<"\n";
    }
  }
}

Matrix Matrix::AddTilingPaddingRows(int tSize){
  int padding_rows = tSize - rows%tSize -tSize*(!(bool(rows%tSize)));

  Matrix temp(rows+padding_rows, columns);
  Matrix result = *this || temp;

  std::cout<<"Hello here"<<std::endl;
  result.padded_rows = padding_rows+padded_rows;
  return result;
}

Matrix Matrix::AddTilingPaddingColumns(int tSize){
  int padding_columns = tSize - columns%tSize - tSize*(!(bool(columns%tSize)));

  Matrix temp(rows, columns+padding_columns);
  Matrix result = *this || temp;
  result.padded_columns = padding_columns+padded_columns;
  return result;
}


Matrix Matrix::AddPaddingRows(int padding_rows){

  Matrix temp(rows+padding_rows, columns);
  Matrix result = *this || temp;
  result.padded_rows = padding_rows+padded_rows;
  return result;
}

Matrix Matrix::AddPaddingColumns(int padding_columns){

  Matrix temp(rows, columns+padding_columns);
  Matrix result = *this || temp;
  result.padded_columns = padding_columns+padded_columns;
  return result;
}

Matrix Matrix::AddTilingPadding(int tSize){
  int padding_rows = tSize-rows%tSize - tSize*(!bool(rows%tSize));
  int padding_columns = tSize-columns%tSize - tSize*(!bool(columns%tSize));

  Matrix temp(rows+padding_rows, columns+padding_columns);
  Matrix result = *this || temp;
  result.padded_rows = padding_rows + padded_rows;
  result.padded_columns = padding_columns + padded_columns;
  return result;
}

Matrix Matrix::RemovePaddingRows(){
  Matrix result(rows-padded_rows, columns);
  for(i=0; i<result.rows; i++){
    for(j=0; j<result.columns; j++){
      result.data[i][j] = data[i][j];
    }
  }

  return result;
}

Matrix Matrix::RemovePaddingColumns(){
  Matrix result(rows, columns-padded_columns);

  for(i=0; i<result.rows; i++){
    for(j=0; j<result.columns; j++){
      result.data[i][j] = data[i][j];
    }
  }

  return result;
}

Matrix Matrix::RemovePadding(){
  Matrix result(rows-padded_rows, columns-padded_columns);

  for(i=0; i<result.rows; i++){
    for(j=0; j<result.columns; j++){
      result.data[i][j] = data[i][j];
    }
  }

  return result;
}


//tiling of the operands
void Matrix::MultiplyTilesOnce(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize){

  //debugging
  std::cout<<"\nTile row "<<IdxArow<<" column "<<IdxBcol<<". Iteration "<<IdxAcol<<"\n\n";

  if(A.columns != B.rows){
    std::cout<<"The matrices are incompatible!\n";
    exit(404);
  }
  //all indices are referred to A matrix and transposed for the B matrix
  int tileCstart = IdxBcol*tSize;
  int tileCend = (IdxBcol+1)*tSize;
  int tileRstart = IdxArow*tSize;
  int tileRend = (IdxArow+1)*tSize;
  int ThisTileEnd = (IdxAcol+1)*tSize;
  int ThisTileStart = IdxAcol*tSize;

  //adjustment for when A clumns and B rows are not multiple of tSize
  if(ThisTileEnd>A.columns){
    ThisTileEnd = A.columns;
    std::cout<<"Abnormal tile encountered...................."<<std::endl;
  }

  //IdxAcol is equal to the iteration number so in the tile multiplication
  //the index of the destination tile is defined with IdxBcol, instead 
  //the inner most loop uses IdxAcol.

  //setting the padding rows and columns depending on the operands
  padded_rows = A.padded_rows;
  padded_columns = B.padded_columns;

  if(IdxAcol == 0){
    //if it's the first iteration set destination matrix to 0)
    for(i=0; i<tSize; i++){
      for(j=0; j<tSize; j++){
        data[tileRstart+i][tileCstart+j] = 0;
      }
    }
  }

  //normal matrix multiplication for one tile
  for(i=tileRstart; i<tileRend; i++){
    for(j=tileCstart; j<tileCend; j++){
      for(k=ThisTileStart; k<ThisTileEnd; k++){
        data[i][j] += A.data[i][k]*B.data[k][j];
      }
      std::cout<<"Element cumulative value("<<i<<","<<j<<") = "<<data[i][j]<<"\n";
    }
  } 
  printf("\e[93mUscita dalla funzione MultiplyTilesOnce...\e[39mDistruzione delle variabili locali\n");
}

//tiling of the operands with the tile stored in a given pointer to pointer topointer (iteration > row > col)
void Matrix::MultiplyTiles(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize, Tensor& results){

  //debugging
  std::cout<<"\nTile row "<<IdxArow<<" column "<<IdxBcol<<". Iteration "<<IdxAcol<<"\n\n";

  if(A.columns != B.rows){
    std::cout<<"The matrices are incompatible!\n";
    exit(404);
  }
  //all indices are referred to A matrix and transposed for the B matrix
  int tileCstart = IdxBcol*tSize;
  int tileCend = (IdxBcol+1)*tSize;
  int tileRstart = IdxArow*tSize;
  int tileRend = (IdxArow+1)*tSize;
  int ThisTileEnd = (IdxAcol+1)*tSize;
  int ThisTileStart = IdxAcol*tSize;

  //adjustment for when A clumns and B rows are not multiple of tSize
  if(ThisTileEnd>A.columns){
    ThisTileEnd = A.columns;
    std::cout<<"Abnormal tile encountered...................."<<std::endl;
  }

  //IdxAcol is equal to the iteration number so in the tile multiplication
  //the index of the destination tile is defined with IdxBcol, instead 
  //the inner most loop uses IdxAcol.

  //setting the padding rows and columns depending on the operands
  results.padded_rows = A.padded_rows;
  results.padded_columns = B.padded_columns;

  //normal matrix multiplication for one tile
  for(i=tileRstart; i<tileRend; i++){
    for(j=tileCstart; j<tileCend; j++){
      for(k=ThisTileStart; k<ThisTileEnd; k++){
        results.data[IdxAcol][i][j] += A.data[i][k]*B.data[k][j];
      }
    }
  } 
  printf("\e[93mUscita dalla funzione MultiplyTiles...\e[39mDistruzione delle variabili locali\n");
}




//OPERATORS
Matrix Matrix::operator+(Matrix& second){
  if(rows != second.rows || columns != second.columns){
    std::cout<<"\e[93mMatrix dimensions don't match in the sum operation.\e[39m\n";
    std::exit(403);
  }
  Matrix result(rows, columns);

  for(i=0; i<rows; i++){
    for(j=0; j<columns; j++){
      result.data[i][j] = data[i][j]+second.data[i][j];
    }
  }

  //	   std::cout<<"Sum done. \n";
  return result;
}

Matrix Matrix::operator||(Matrix& second){   //an extended sum where if the value of index 
  //i, j i spresent in one matrix but not the other
  //that value is taken as the result, so the result matrix
  //always has the most rows and most columns among the
  //operands
  if(rows != second.rows || columns != second.columns){
    std::cout<<"\e[93mMatrix dimensions don't match in the sum operation.\e[39m\n";
  }
  Matrix result(std::max(rows, second.rows), std::max(columns, second.columns));

  for(i=0; i<result.rows; i++){
    if(i<rows && i<second.rows){
      for(j=0; j<result.columns; j++){
        if(j<columns && j<second.columns){
          result.data[i][j] = data[i][j]+second.data[i][j];
        }else if(j<columns){
          result.data[i][j] = data[i][j];
        }else{
          result.data[i][j] = second.data[i][j];
        }
        std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
      }
    }else if(i<rows){
      for(j=0; j<result.columns; j++){
        if(j<columns){
          result.data[i][j] = data[i][j];
        }else{
          result.data[i][j] = 0;
        }
        std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
      }
    }else{
      for(j=0; j<result.columns; j++){
        if(j<second.columns){
          result.data[i][j] = second.data[i][j];
        }else{
          result.data[i][j] = 0;
        }
        std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
      }
    }
  }

  std::cout<<"Sum done. \n";
  return result;
}


Matrix Matrix::operator*(Matrix& second){

  Matrix result(rows, second.columns);
  if(columns != second.rows){
    std::cout<<"Matrices have incompatible dimensions.\n";
    std::exit(403);
  }

  for(i=0; i<rows; i++){
    for(j=0; j<second.columns; j++){
      for(k=0; k<columns; k++){
        result.data[i][j]+=data[i][k]*second.data[k][j];
        //		    std::cout<<i<<", "<<k<<"\t";
      }
      //		 std::cout<<"Risultato indice ("<<i<<", "<<j<<"):\n";
      //		 std::cout<<result.data[i][j]<<"\n";
    }
  }
  return result;
}

// Copy assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
  // Check for self-assignment
  if (this != &other)
  {
    // Deallocate the current memory and allocate the new required space
    for (int i = 0; i < rows; i++) {
      delete[] data[i];
    }
    delete[] data;

    // Allocate new memory
    rows = other.rows;
    columns = other.columns;
    padded_rows = other.padded_rows;
    padded_columns = other.padded_columns;

    data = new int*[rows];
    for (int i = 0; i < rows; i++) {
      data[i] = new int[columns];
      for (int j = 0; j < columns; j++) {
        data[i][j] = other.data[i][j];
      }
    }
  }
  return *this;
}


//DESTRUCTOR
Matrix::~Matrix(){
  std::cout<<"Distruggo matrice "<<rows<<"x"<<columns<<"...\n";
  for(i=0; i<rows; i++){
    delete[] data[i];
    std::cout<<"Cancellata riga "<<i<<".\n";
  }
  delete[] data;
  std::cout<<"Cancellata tutta la matrice "<<rows<<"x"<<columns<<".\n";
}

