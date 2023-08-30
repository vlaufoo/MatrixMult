#include"Mat.hpp"


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

void Tensor::TestValues(){
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

