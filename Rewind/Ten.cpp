#include"Ten.hpp"
//constructors
Tensor::Tensor(){
  rows = 10;
  columns = 10;
  depth = 10;
  padded_rows = 0;
  padded_columns = 0;
  padded_layers = 0;
  std::cout<<"Generating Tensor "<<depth<<"x"<<rows<<"x"<<columns<<"\n\n";

  data = new int** [depth];
  for(int k=0; k<depth; k++){
    data[k] = new int *[rows];
    for(int i=0; i<rows; i++){
      data[k][i] = new int [columns];
      for(int j=0; j<columns; j++){
        data[k][i][j] = 0;
      }
    }
  }
}

Tensor::Tensor(int r, int c){ 
  rows = r;
  columns = c;
  depth = 1;
  padded_rows = 0;
  padded_columns = 0;
  padded_layers = 0;
  std::cout<<"Generating Tensor "<<depth<<"x"<<rows<<"x"<<columns<<"\n\n";

  data = new int** [depth];
  for(int k=0; k<depth; k++){
    data[k] = new int *[rows];
    for(int i=0; i<rows; i++){
      data[k][i] = new int [columns];
      for(int j=0; j<columns; j++){
        data[k][i][j] = 0;
      }
    }
  }
}

Tensor::Tensor(int dep, int r, int c){ 
  rows = r;
  columns = c;
  depth = dep;
  padded_rows = 0;
  padded_columns = 0;
  padded_layers = 0;
  std::cout<<"Generating Tensor "<<depth<<"x"<<rows<<"x"<<columns<<"\n\n";

  data = new int** [depth];
  for(int k=0; k<depth; k++){
    data[k] = new int *[rows];
    for(int i=0; i<rows; i++){
      data[k][i] = new int [columns];
      for(int j=0; j<columns; j++){
        data[k][i][j] = 0;
      }
    }
  }
}

//Copy constructor
Tensor::Tensor(const Tensor& other){
  rows = other.rows;
  columns = other.columns;
  depth = other.depth;
  padded_rows = other.padded_rows;
  padded_columns = other.padded_columns;
  padded_layers = other.padded_layers;

  data = new int** [depth];
  for(int k=0; k<depth; k++){
    data[k] = new int* [rows];
    for(int i=0; i<rows; i++){
      data[k][i] = new int [columns];
      for(int j=0; j<columns; j++){
        data[k][i][j] = other.data[k][i][j];
      }
    }
  }
}

//FUNCTIONS
int Tensor::Rows(){
  return rows;
}

int Tensor::Columns(){
  return columns;
}

int Tensor::Layers(){
  return depth;
}

int Tensor::PaddingRows(){
  return padded_rows;
}

int Tensor::PaddingColumns(){
  return padded_columns;
}

int Tensor::PaddingLayers(){
  return padded_layers;
}

int Tensor::SetElement(int d, int n, int m, int val){
  if(n >= rows || m >= columns || d >= depth){
    std::cout<<"Invalid indices.\n";
  }
  data[d][n][m] = val;
  return val;
}

int Tensor::ZeroElement(int d, int n, int m){
  if(n >= rows || m >= columns || d >= depth){
    std::cout<<"Invalid indices.\n";
  }
  data[d][n][m] = 0;
  return 0;
}

int Tensor::GetElement(int d, int n, int m){
  if(n >= rows || m >= columns || d >= depth){
    std::cout<<"Invalid indices.\n";
  }
  return data[d][n][m];
}

void Tensor::Print(){
  for(int k=0; k<depth; k++){
    std::cout<<"\e[0;92m--------Layer "<<k<<"--------\e[0;39m\n\n";
    for(int i=0; i<rows; i++){
      for(int j=0; j<columns; j++){
        std::cout<<data[k][i][j]<<"\t";
      }
      std::cout<<"\n\n";
    }
    std::cout<<"\n\n";
  }
}

void Tensor::Print(int layer){
    std::cout<<"\e[0;92m--------Layer "<<layer<<"--------\e[0;39m\n\n";
    for(int i=0; i<rows; i++){
      for(int j=0; j<columns; j++){
        std::cout<<data[layer][i][j]<<"\t";
      }
      std::cout<<"\n\n";
    }
    std::cout<<"\n\n";
}


void Tensor::TestValues(){ //fix it! (only first layer working)
  int i=0;
  int j=0;
  int k=0;

  for(k=0; k<depth; k++){   
    while(i<rows*columns){
      data[k][i/columns][i%columns] = j++;
      if(j > 512)
        j=0;
      i++;
    }
  }
}

void Tensor::Random(int minVal, int maxVal, const unsigned int seed){
  srand(seed);
  for(int k=0; k<depth; k++){
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < columns; j++){
        data[k][i][j] = rand() % (maxVal - minVal + 1) + minVal;
      }
    }
  }
}

void Tensor::Load(int *values, int length){
  if(length >= rows*columns*depth){
    std::cout<<"!\n!\n!\nData length is excessive!\n!\n!\n";
    std::cout<<"Check that the bounds correspond.\n";
  }

  for(int m=0; m<length; m++){
    int k = m/columns*rows;
    int i = (m%(columns*rows))/columns;
    int j = (m%(columns*rows))%columns;
    data[k][i][j] = *(values + m);
  }
}


void Tensor::CopyLayer(Tensor& A, int src_layer_idx, int dst_layer_idx){
  if(A.depth <= src_layer_idx || depth <= dst_layer_idx || rows > A.rows || columns > A.columns){
    std::cout<<"Error in ChangeLayerTensor: The matrix is to big or the layer index is too big.";
  }

  for(int i=0; i<rows; i++){
    for(int j=0; j<columns; j++){
      data[dst_layer_idx][i][j] = A.data[src_layer_idx][i][j];
    }
  }
  
  padded_rows = (padded_rows<A.padded_rows) ? padded_rows : A.padded_rows;
  padded_columns = (padded_columns<A.padded_columns) ? padded_columns : A.padded_columns;
  padded_layers = (padded_layers<A.padded_layers) ? padded_layers : A.padded_layers;

}

Tensor Tensor::AppendTensor(Tensor& A){

  if(rows != A.rows || columns != A.columns){
    std::cout<<"Error in AppendLayerTensor: matrix dimensions are too big";
  }

  Tensor result(depth+A.depth, rows, columns);
  for(int l=0; l<depth; l++){
    for(int i=0; i<rows; i++){
      for(int j=0; j<columns; j++){
        result.data[l][i][j] = data[l][i][j];
      }
    }
  }
  for(int l=depth; l<A.depth+depth; l++){
    for(int i=0; i<rows; i++){
      for(int j=0; j<columns; j++){
        result.data[l][i][j] = A.data[l][i][j];
      }
    }
  }

  result.padded_rows = padded_rows;
  result.padded_columns = padded_columns;
  result.padded_layers = padded_layers;

  return result;
}
//to be added:
//--a function that appends a Tensor object as a layer of the Tensor
//    (can only e called by a tensor object) [ done! ]
//--a function that collapses a tensor into a matrix by adding all the layers
//    (can only be called by a Tensor object since the return value is of type Tensor)
//--a new tiling function that writes to a tensor instead of a pointer.


void Tensor::CollapseTensor(){
  for(int l=1; l<depth; l++){
    for(int i=0; i<rows; i++){
      for(int j=0; j<columns; j++){
       data[0][i][j] += data[l][i][j];
      }
    }
  }
}

//TILING FUNCTIONS

//tiling of the result matrix
/*
void Tensor::CalculateTile(Tensor A, Tensor B, int tSize, int tRow, int tCol){
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
*/

Tensor Tensor::AddTilingPaddingRows(int tSize){
  int padding_rows = tSize - rows%tSize -tSize*(!(bool(rows%tSize)));

  Tensor temp(depth, rows+padding_rows, columns);
  Tensor result = *this || temp;

  std::cout<<"Hello here"<<std::endl;
  result.padded_rows = padding_rows+padded_rows;
  return result;
}

Tensor Tensor::AddTilingPaddingColumns(int tSize){
  int padding_columns = tSize - columns%tSize - tSize*(!(bool(columns%tSize)));

  Tensor temp(depth, rows, columns+padding_columns);
  Tensor result = *this || temp;
  result.padded_columns = padding_columns+padded_columns;
  return result;
}

Tensor Tensor::AddPaddingRows(int padding_rows){

  Tensor temp(depth, rows+padding_rows, columns);
  Tensor result = *this || temp;
  result.padded_rows = padding_rows+padded_rows;
  return result;
}

Tensor Tensor::AddPaddingColumns(int padding_columns){

  Tensor temp(depth, rows, columns+padding_columns);
  Tensor result = *this || temp;
  result.padded_columns = padding_columns+padded_columns;
  return result;
}

Tensor Tensor::AddTilingPadding(int tSize){
  int padding_rows = tSize-rows%tSize - tSize*(!bool(rows%tSize));
  int padding_columns = tSize-columns%tSize - tSize*(!bool(columns%tSize));

  Tensor temp(depth, rows+padding_rows, columns+padding_columns);
  Tensor result = *this || temp;
  result.padded_rows = padding_rows + padded_rows;
  result.padded_columns = padding_columns + padded_columns;
  return result;
}

Tensor Tensor::RemovePaddingRows(){
  Tensor result(depth, rows-padded_rows, columns);
  for(k=0; k<result.depth; k++){
    for(i=0; i<result.rows; i++){
      for(j=0; j<result.columns; j++){
        result.data[k][i][j] = data [k][i][j];
      }
    }
  }
  return result;
}

Tensor Tensor::RemovePaddingColumns(){
  Tensor result(depth, rows, columns-padded_columns);
  for(k=0; k<result.depth; k++){
    for(i=0; i<result.rows; i++){
      for(j=0; j<result.columns; j++){
        result.data[k][i][j] = data [k][i][j];
      }
    }
  }
  return result;
}

Tensor Tensor::RemovePadding(){
  Tensor result(depth-padded_layers, rows-padded_rows, columns-padded_columns);
  for(k=0; k<result.depth; k++){
    for(i=0; i<result.rows; i++){
      for(j=0; j<result.columns; j++){
        result.data[k][i][j] = data [k][i][j];
      }
    }
  }
  return result;
}


//tiling of the operands
void Tensor::MultiplyTilesOnce(Tensor& A, int a_layer, Tensor& B, int b_layer,
                               int iteration, int IdxArow, int IdxBcol, int tSize){

  //debugging
  std::cout<<"\nTile row "<<IdxArow<<" column "<<IdxBcol<<". Iteration "<<iteration<<"\n\n";

  if(A.columns != B.rows){
    std::cout<<"The matrices are incompatible!\n";
    exit(404);
  }
  //all indices are referred to A matrix and transposed for the B matrix
  int tileCstart = IdxBcol*tSize;
  int tileCend = (IdxBcol+1)*tSize;
  int tileRstart = IdxArow*tSize;
  int tileRend = (IdxArow+1)*tSize;
  int ThisTileStart = iteration*tSize;
  int ThisTileEnd = (iteration+1)*tSize;

  //adjustment for when A clumns and B rows are not multiple of tSize
  if(ThisTileEnd>A.columns){
    ThisTileEnd = A.columns;
    std::cout<<"Abnormal tile encountered..................\n"
              <<"Last tile iteration is not square.\n"<<std::endl;
  }

  //iteration is equal to the iteration number so in the tile multiplication
  //the index of the destination tile is defined with IdxBcol, instead 
  //the inner most loop uses iteration.

  //setting the padding rows and columns depending on the operands
  padded_rows = A.padded_rows;
  padded_columns = B.padded_columns;

  if(iteration == 0){
    //if it's the first iteration set destination matrix to 0)
    for(i=0; i<tSize; i++){
      for(j=0; j<tSize; j++){
        data[iteration][tileRstart+i][tileCstart+j] = 0;
      }
    }
  }

  //normal matrix multiplication for one tile
  for(i=tileRstart; i<tileRend; i++){
    for(j=tileCstart; j<tileCend; j++){
      for(k=ThisTileStart; k<ThisTileEnd; k++){
        data[iteration][i][j] += A.data[a_layer][i][k]*B.data[b_layer][k][j];
      }
      std::cout<<"Element cumulative value("<<i<<","<<j<<") = "<<data[iteration][i][j]<<"\n";
    }
  } 
  printf("\e[93mUscita dalla funzione MultiplyTilesOnce...\e[39mDistruzione delle variabili locali\n");
}


/*
//tiling of the operands with the tile stored in a given pointer to pointer topointer (iteration > row > col)
void Tensor::MultiplyTiles(Tensor& A, Tensor& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize, Tensor& results){

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
*/

//OPERATORS
Tensor Tensor::operator+(const Tensor& second){
  if(rows != second.rows || columns != second.columns){
    std::cout<<"\e[93mTensor dimensions don't match in the sum operation.\e[39m\n";
    std::exit(403);
  }
  Tensor result(depth, rows, columns);

  for(int k=0; k<depth; k++){
    for(int i=0; i<rows; i++){
      for(int j=0; j<columns; j++){
        result.data[k][i][j] = data[k][i][j]+second.data[k][i][j];
      }
    }
  }

  //	   std::cout<<"Sum done. \n";
  return result;
}

Tensor Tensor::operator||(const Tensor& second){
  std::cout<<"Entering || operator\n";
  Tensor result(std::max(depth, second.depth), std::max(rows, second.rows), std::max(columns, second.columns));
  std::cout<<"Tensor "<<result.depth<<"x"<<result.rows<<"x"<<result.columns<<" created.\n\n";
  for(int k=0; k<result.depth; k++){
    if(k<depth && k<second.depth){
      for(int i=0; i<result.rows; i++){
        if(i<rows && i<second.rows){
          for(int j=0; j<result.columns; j++){
            if(j<columns && j<second.columns){
              result.data[k][i][j] = data[k][i][j]+second.data[k][i][j];
            }else if(j<columns){
              result.data[k][i][j] = data[k][i][j];
            }else{
              result.data[k][i][j] = second.data[k][i][j];
            }
            std::cout<<"Doing sum of element "<<k<<", "<<i<<", "<<j<<"\n";
          }
        }else if(i<rows){
          for(int j=0; j<result.columns; j++){
            if(j<columns){
              result.data[k][i][j] = data[k][i][j];
            }else{
              result.data[k][i][j] = 0;
            }
            std::cout<<"Doing sum of element "<<k<<", "<<i<<", "<<j<<"\n";
          }
        }else{
          for(int j=0; j<result.columns; j++){
            if(j<second.columns){
              result.data[k][i][j] = second.data[k][i][j];
            }else{
              result.data[k][i][j] = 0;
            }
            std::cout<<"Doing sum of element "<<k<<", "<<i<<", "<<j<<"\n";
          }
        }
      } 
    }else if(k<depth){        //now it's much simpler since I know that only "this.data" has valid values
      //i just need to be careful to set to zero the other locations (even though
      //they should be zzero already since the inizialization of the result tensor)
      for(int i=0; i<result.rows; i++){
        if(i<rows){
          for(int j=0; j<result.columns; j++){
            if(j<columns){
              result.data[k][i][j] = data[k][i][j];
            }else{
              result.data[k][i][j] = 0;
            }
            //		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
            std::cout<<"Doing sum of element "<<k<<", "<<i<<", "<<j<<"\n";
          }
        }else{
          for(int j=0; j<result.columns; j++){
            result.data[k][i][j] = 0;
          }
        }
      }
    }else{            
      for(int i=0; i<result.rows; i++){
        if(i<second.rows){
          for(int j=0; j<result.columns; j++){
            if(j<second.columns){
              result.data[k][i][j] = second.data[k][i][j];
            }else{
              result.data[k][i][j] = 0;
            }
            //		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
            std::cout<<"Doing sum of element "<<k<<", "<<i<<", "<<j<<"\n";
          }
        }else{
          for(int j=0; j<result.columns; j++){
            result.data[k][i][j] = 0;
          }
        }
      }
    }
  }
  return result;
}

//Copy assignment operator
Tensor& Tensor::operator=(const Tensor& other){
  std::cout<<"entering copy assignment\n\n";
  if(this != &other){
    for(int k=0; k<depth; k++){
      for(int i=0; i<rows; i++){
        delete[] data[k][i];
      }
      delete[] data[k];
    }
    delete[] data;

    //allocate new memory
    depth = other.depth;
    rows = other.rows;
    columns = other.columns;
    padded_layers = other.padded_layers;
    padded_rows = other.padded_rows;
    padded_columns = other.padded_columns;

    data = new int** [depth];
    for(int k=0; k<depth; k++){
      data[k] = new int* [rows];
      for(int i=0; i<rows; i++){
        data[k][i] = new int [columns];
        for(int j=0; j<columns; j++){
          data[k][i][j] = other.data[k][i][j];
        }
      }
    }
  }
  return *this;
}

Tensor Tensor::operator*(Tensor& second){

  Tensor result(depth, rows, second.columns);
  if(depth != second.depth){
    std::cout<<"Tensors have are of different depth. Use another method.\n\n";
    std::exit(403);
  }
  if(columns != second.rows){
    std::cout<<"Matrices have incompatible dimensions.\n";
    std::exit(403);
  }

  for(int l=0; l<depth; l++){
    for(i=0; i<rows; i++){
      for(j=0; j<second.columns; j++){
        for(k=0; k<columns; k++){
          result.data[l][i][j]+=data[l][i][k]*second.data[l][k][j];
          //		    std::cout<<i<<", "<<k<<"\t";
        }
        //		 std::cout<<"Risultato indice ("<<i<<", "<<j<<"):\n";
        //		 std::cout<<result.data[i][j]<<"\n";
      }
    }
  }
  return result;
}
//DESTRUCTOR

Tensor::~Tensor(){
  std::cout<<"Deleting Tensor "<<depth<<"x"<<rows<<"x"<<columns<<std::endl;
  for(int k=0; k<depth; k++){  
    for(int i=0; i<rows; i++){
      delete[] data[k][i];
      std::cout<<"Deleted row "<<i<<" of layer "<<k<<std::endl;
    }
    delete[] data[k];
    std::cout<<"Deleted layer "<<k<<std::endl;
  }
  delete[] data;
  std::cout<<"Deleted Tensor "<<depth<<"x"<<rows<<"x"<<columns<<std::endl;
}

