#include<iostream>


class Tensor
{
      protected:
   int rows;
   int columns;
   int depth;
   int padded_rows;
   int padded_columns;
   int padded_layers;
   int ***tensorpt;
   friend class Matrix;  //i had built the class thinking Matrix could then access
                         //protected members of it, but it can only access the members of its
                         //own objects (that are inherited), 
                         //unless it is declared a friend of Tensor.
                         
      public:
   //constructors
   Tensor(){
      std::cout<<"Generating Tensor\n\n";
      rows = 10;
      columns = 10;
      depth = 10;
      padded_rows = 0;
      padded_columns = 0;
      padded_layers = 0;

      tensorpt = new int** [depth];
      for(int k=0; k<depth; k++){
         tensorpt[k] = new int *[rows];
         for(int i=0; i<rows; i++){
            tensorpt[k][i] = new int [columns];
            for(int j=0; j<columns; j++){
               tensorpt[k][i][j] = 0;
            }
         }
      }
   }

   Tensor(int dep, int r, int c){ 
      std::cout<<"Generating Tensor\n\n";
      rows = r;
      columns = c;
      depth = dep;
      padded_rows = 0;
      padded_columns = 0;
      padded_layers = 0;

      tensorpt = new int** [depth];
      for(int k=0; k<depth; k++){
         tensorpt[k] = new int *[rows];
         for(int i=0; i<rows; i++){
            tensorpt[k][i] = new int [columns];
            for(int j=0; j<columns; j++){
               tensorpt[k][i][j] = 0;
            }
         }
      }
   }

   //Copy constructor
   Tensor(const Tensor& other){
      rows = other.rows;
      columns = other.columns;
      depth = other.depth;
      padded_rows = other.padded_rows;
      padded_columns = other.padded_columns;
      padded_layers = other.padded_layers;

      tensorpt = new int** [depth];
      for(int k=0; k<depth; k++){
         tensorpt[k] = new int* [rows];
         for(int i=0; i<rows; i++){
            tensorpt[k][i] = new int [columns];
            for(int j=0; j<columns; j++){
               tensorpt[k][i][j] = other.tensorpt[k][i][j];
            }
         }
      }
   }

   //FUNCTIONS
   int Rows(){
	   return rows;
	}

	int Columns(){
	   return columns;
	}

   int Layers(){
      return depth;
   }

	int PaddingRows(){
	   return padded_rows;
	}

	int PaddingColumns(){
	   return padded_columns;
	}

	int PaddingLayers(){
	   return padded_layers;
	}

	void SetElement(int d, int n, int m, int val){
	   if(n >= rows || m >= columns || d >= depth){
	      std::cout<<"Invalid indices.\n";
	   }
	   tensorpt[d][n][m] = val;
	}

	void ZeroElement(int d, int n, int m){
	   if(n >= rows || m >= columns || d >= depth){
	      std::cout<<"Invalid indices.\n";
	   }
	   tensorpt[d][n][m] = 0;
	}

	int GetElement(int d, int n, int m){
	   if(n >= rows || m >= columns || d >= depth){
	      std::cout<<"Invalid indices.\n";
	   }
	   return tensorpt[d][n][m];
	}

	void PrintTensor(){
      for(int k=0; k<depth; k++){
         std::cout<<"\e[0;92m--------Layer "<<k<<"--------\e[0;39m\n\n";
         for(int i=0; i<rows; i++){
            for(int j=0; j<columns; j++){
               std::cout<<tensorpt[k][i][j]<<"\t";
            }
            std::cout<<"\n\n";
         }
         std::cout<<"\n\n";
      }
	}

	void TestTensor(){
	   int i=0;
      int j=0;
	   int k=0;

      for(k=0; k<depth; k++){   
   	   while(i<rows*columns){
   	      tensorpt[k][i/columns][i%columns] = j++;
   	      if(j > 512)
   	        j=0;
   	      i++;
   	   }
      }

	}

	void InitTensor(int *values, int length){
	   if(length >= rows*columns*depth){
	      std::cout<<"!\n!\n!\nData length is excessive!\n!\n!\n";
	      std::cout<<"Check that the bounds correspond.\n";
	   }

	   for(int m=0; m<length; m++){
	      int k = m/columns*rows;
	      int i = (m%(columns*rows))/columns;
         int j = (m%(columns*rows))%columns;
	      tensorpt[k][i][j] = *(values + m);
	   }
   }


   //OPERATORS
   Tensor operator+(const Tensor& second){
	   if(rows != second.rows || columns != second.columns){
	      std::cout<<"\e[93mTensor dimensions don't match in the sum operation.\e[39m\n";
	      std::exit(403);
	   }
  	   Tensor result(depth, rows, columns);

	   for(int k=0; k<depth; k++){
	      for(int i=0; i<rows; i++){
            for(int j=0; j<columns; j++){
   	         result.tensorpt[k][i][j] = tensorpt[k][i][j]+second.tensorpt[k][i][j];
            }
	      }
	   }
   
//	   std::cout<<"Sum done. \n";
	   return result;
   }

   Tensor operator||(const Tensor& second){
      std::cout<<"Entering || operator\n";
      Tensor result(std::max(depth, second.depth), std::max(rows, second.rows), std::max(columns, second.columns));
      std::cout<<"Tensor "<<result.depth<<"x"<<result.rows<<"x"<<result.columns<<" created.\n\n";
      for(int k=0; k<result.depth; k++){
         if(k<depth && k<second.depth){
            for(int i=0; i<result.rows; i++){
	            if(i<rows && i<second.rows){
	               for(int j=0; j<result.columns; j++){
		               if(j<columns && j<second.columns){
		                  result.tensorpt[k][i][j] = tensorpt[k][i][j]+second.tensorpt[k][i][j];
		               }else if(j<columns){
		                  result.tensorpt[k][i][j] = tensorpt[k][i][j];
		               }else{
		                  result.tensorpt[k][i][j] = second.tensorpt[k][i][j];
		               }
		               std::cout<<"Doing sum of element "<<k<<", "<<i<<", "<<j<<"\n";
	               }
	            }else if(i<rows){
	               for(int j=0; j<result.columns; j++){
		               if(j<columns){
		                  result.tensorpt[k][i][j] = tensorpt[k][i][j];
		               }else{
		                  result.tensorpt[k][i][j] = 0;
		               }
		               std::cout<<"Doing sum of element "<<k<<", "<<i<<", "<<j<<"\n";
		            }
	            }else{
	               for(int j=0; j<result.columns; j++){
		               if(j<second.columns){
		                  result.tensorpt[k][i][j] = second.tensorpt[k][i][j];
		               }else{
		                  result.tensorpt[k][i][j] = 0;
		               }
		               std::cout<<"Doing sum of element "<<k<<", "<<i<<", "<<j<<"\n";
		            }
   	         }
	         } 
         }else if(k<depth){        //now it's much simpler since I know that only "this.tensorpt" has valid values
                                   //i just need to be careful to set to zero the other locations (even though
                                   //they should be zzero already since the inizialization of the result tensor)
            for(int i=0; i<result.rows; i++){
	            if(i<rows){
	               for(int j=0; j<result.columns; j++){
		               if(j<columns){
		                  result.tensorpt[k][i][j] = tensorpt[k][i][j];
		               }else{
		                  result.tensorpt[k][i][j] = 0;
		               }
//		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
		               std::cout<<"Doing sum of element "<<k<<", "<<i<<", "<<j<<"\n";
		            }
	            }else{
                  for(int j=0; j<result.columns; j++){
                     result.tensorpt[k][i][j] = 0;
                  }
               }
	         }
         }else{            
            for(int i=0; i<result.rows; i++){
	            if(i<second.rows){
	               for(int j=0; j<result.columns; j++){
		               if(j<second.columns){
		                  result.tensorpt[k][i][j] = second.tensorpt[k][i][j];
		               }else{
		                  result.tensorpt[k][i][j] = 0;
		               }
//		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
		               std::cout<<"Doing sum of element "<<k<<", "<<i<<", "<<j<<"\n";
		            }
	            }else{
                  for(int j=0; j<result.columns; j++){
                     result.tensorpt[k][i][j] = 0;
                  }
               }
	         }
         }
      }
      return result;
   }

   //Copy assignment operator
   Tensor& operator=(const Tensor& other){
      std::cout<<"entering copy assignment\n\n";
      if(this != &other){
         for(int k=0; k<depth; k++){
            for(int i=0; i<rows; i++){
               delete[] tensorpt[k][i];
            }
            delete[] tensorpt[k];
         }
         delete[] tensorpt;

         //allocate new memory
         depth = other.depth;
         rows = other.rows;
         columns = other.columns;
         padded_layers = other.padded_layers;
         padded_rows = other.padded_rows;
         padded_columns = other.padded_columns;

         tensorpt = new int** [depth];
         for(int k=0; k<depth; k++){
            tensorpt[k] = new int* [rows];
            for(int i=0; i<rows; i++){
               tensorpt[k][i] = new int [columns];
               for(int j=0; j<columns; j++){
                  tensorpt[k][i][j] = other.tensorpt[k][i][j];
               }
            }
         }
      }
      return *this;
   }


   //DESTRUCTOR

   ~Tensor(){
      std::cout<<"Deleting Tensor "<<depth<<"x"<<rows<<"x"<<columns<<std::endl;
      for(int k=0; k<depth; k++){  
         for(int i=0; i<rows; i++){
            delete[] tensorpt[k][i];
            std::cout<<"Deleted row "<<i<<" of layer "<<k<<std::endl;
         }
         delete[] tensorpt[k];
         std::cout<<"Deleted layer "<<k<<std::endl;
      }
      delete[] tensorpt;
      std::cout<<"Deleted Tensor "<<depth<<"x"<<rows<<"x"<<columns<<std::endl;
   }

};

class Matrix : public Tensor
{
      private:
  // int rows;
//	int columns;
//	int padded_rows;
//	int padded_columns;
//	int **tensorpt[0];
	int i, j, k;

      public:
//CONSTRUCTORS
	Matrix(){
	   rows=100;
	   columns=100;
      depth=1;
      padded_layers = 0;
	   padded_rows = 0;
	   padded_columns = 0;
      tensorpt = new int**;
	   tensorpt[0] = new int* [rows];
	   for(i=0; i<rows; i++)
	      tensorpt[0][i] = new int [columns];
	   for(i=0; i<rows; i++){
	      for(j=0; j<columns; j++){
	         tensorpt[0][i][j] = 0;
	      }
	   }

	   std::cout<<"Creo matrice "<<rows<<"x"<<columns<<"\n";
	}

	Matrix(int r, int c){
	   rows = r;
	   columns = c;
      depth = 1;
      padded_layers = 0;
	   padded_rows = 0;
	   padded_columns = 0;
	   tensorpt[0] = new int* [rows];
	   for(i=0; i<rows; i++)
	      tensorpt[0][i] = new int [columns];

	   for(i=0; i<rows; i++){
	      for(j=0; j<columns; j++){
	         tensorpt[0][i][j] = 0;
	      }
	   }

	   std::cout<<"Creo matrice "<<rows<<"x"<<columns<<"\n";
	}
/*
	// Copy constructor
    Matrix(const Matrix& other) {
        rows = other.rows;
        columns = other.columns;
	padded_rows = other.padded_rows;
	padded_columns = other.padded_columns;

        tensorpt[0] = new int*[rows];
        for (int i = 0; i < rows; i++) {
            tensorpt[0][i] = new int[columns];
            for (int j = 0; j < columns; j++) {
                tensorpt[0][i][j] = other.tensorpt[0][i][j];
            }
        }
    }
*/

//FUNCTIONS
	void Identity(){
	   for(i=0; i<rows; i++){
	      for(j=0; j<columns; j++){
	         if(i==j){
		    tensorpt[0][i][j] = 1;
		 }else{
		    tensorpt[0][i][j] = 0;
		 }
	      }
	   }
	}

	int GetElement(int n, int m){
	   if(n >= rows || m >= columns){
	      std::cout<<"Invalid indices.\n";
	   }
	   return tensorpt[0][n][m];
	}

	void PrintMatrix(){
	   for(i=0; i<rows; i++){
	      for(j=0; j<columns; j++){
                 std::cout<<tensorpt[0][i][j]<<"\t";
//		 std::cout<<"Printed element "<<i<<", "<<j<<"\n";
	      }
	      std::cout<<"\n\n";
	   }
	   std::cout<<"\n";
	}

	void TestMatrix(){
	   i=0;
	   k=0;
	   while(i<rows*columns){
	      tensorpt[0][i/columns][i%columns] = k++;
	      if(k > 128)
	         k=0;
	      i++;
	   }
	}

	void InitMatrix(int *values, int length){
	   if(length != rows*columns){
	      std::cout<<"!\n!\n!\nData length and matrix dimentions are incompatilble!\n!\n!\n";
	   std::cout<<"Check that the bounds correspond.\n";
	   }

	   for(i=0; i<length; i++){
	      k = i/columns;
	      j = i%columns;
	      tensorpt[0][k][j] = *(values + i);
	   }
	   
	}

   void ChangeLayerMatrix(Tensor& A, int layer_idx){
      if(A.depth <= layer_idx || rows > A.rows || columns > A.columns){
         std::cout<<"Error in ChangeLayerMatrix: The matrix is to big or the layer index is too big.";
      }

      for(int i=0; i<rows; i++){
         for(int j=0; j<columns; j++){
            A.tensorpt[layer_idx][i][j] = tensorpt[0][i][j];
         }
      }
   }

   //to be added:
   //--a function that appends a Matrix object as a layer of the Tensor
   //    (can only e called by a tensor object) [ done! ]
   //--a function that collapses a tensor into a matrix by adding all the layers
   //    (can only be called by a Matrix object since the return value is of type Matrix)
   //--a new tiling function that writes to a tensor instead of a pointer.

   Tensor AppendLayerMatrix(Tensor& A){

      if(rows != A.rows || columns != A.columns){
         std::cout<<"Error in AppendLayerMatrix: matrix dimensions are too big";
      }
      Tensor result(depth+1, rows, columns);
      for(int k=0; k<A.depth; k++){
         for(int i=0; i<A.rows; i++){
            for(int j=0; j<A.columns; j++){
               result.tensorpt[k][i][j] = A.tensorpt[k][i][j];
            }  
         }
      }

      result.padded_rows = A.padded_rows;
      result.padded_columns = A.padded_columns;
      result.padded_layers = A.padded_layers;

      for(int i=0; i<rows; i++){
         for(int j=0; j<columns; j++){
            result.tensorpt[depth][i][j] = tensorpt[0][i][j];
         }  
      }
      return result;
   }


   void CollapseTensor(const Tensor& data){
      if(data.rows != rows || data.columns != columns){
         for (int i = 0; i < rows; i++) {
             delete[] tensorpt[0][i];
         }
         delete[] tensorpt[0];

         rows = data.rows;
         columns = data.columns;
         padded_rows = data.padded_rows;
         padded_columns = data.padded_columns;

			// Deallocate the current memory and allocate the new required space

         tensorpt[0] = new int*[rows];
         for(int i = 0; i < rows; i++){
            tensorpt[0][i] = new int[columns];
            for(int j = 0; j < columns; j++){
               tensorpt[0][i][j] = 0;
               for(int k=0; k<data.depth; k++){
                  tensorpt[0][i][j] += data.tensorpt[k][i][j];
               }
            }
         }      
      }
   }

//TILING FUNCTIONS

	//tiling of the result matrix
	void CalculateTile(Matrix A, Matrix B, int tSize, int tRow, int tCol){
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
	         tensorpt[0][i][j] = 0;
	      }
	   }

	   for(i=start_row; i<end_row; i++){
	      for(j=start_column; j<end_column; j++){
		 for(k=0; k<A.columns; k++){
		    tensorpt[0][i][j] += A.tensorpt[0][i][k]*B.tensorpt[0][k][j];
	         }
		// std::cout<<"Element ("<<i<<","<<j<<") = "<<sum<<"\n";
	      }
	   }
	}

	Matrix AddTilingPaddingRows(int tSize){
	   int padding_rows = tSize - rows%tSize -tSize*(!(bool(rows%tSize)));
	   
	   Matrix temp(rows+padding_rows, columns);
	   Matrix result = *this || temp;

      std::cout<<"Hello here"<<std::endl;
	   result.padded_rows = padding_rows+padded_rows;
	   return result;
	}

	Matrix AddTilingPaddingColumns(int tSize){
	   int padding_columns = tSize - columns%tSize - tSize*(!(bool(columns%tSize)));

	   Matrix temp(rows, columns+padding_columns);
	   Matrix result = *this || temp;
	   result.padded_columns = padding_columns+padded_columns;
	   return result;
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

	Matrix AddTilingPadding(int tSize){
	   int padding_rows = tSize-rows%tSize - tSize*(!bool(rows%tSize));
	   int padding_columns = tSize-columns%tSize - tSize*(!bool(columns%tSize));

     	   Matrix temp(rows+padding_rows, columns+padding_columns);
	   Matrix result = *this || temp;
	   result.padded_rows = padding_rows + padded_rows;
	   result.padded_columns = padding_columns + padded_columns;
	   return result;
	}

	Matrix RemovePaddingRows(){
	   Matrix result(rows-padded_rows, columns);
	   for(i=0; i<result.rows; i++){
	      for(j=0; j<result.columns; j++){
	         result.tensorpt[0][i][j] = tensorpt[0][i][j];
	      }
	   }

	   return result;
	}

	Matrix RemovePaddingColumns(){
	   Matrix result(rows, columns-padded_columns);
	  
	   for(i=0; i<result.rows; i++){
	      for(j=0; j<result.columns; j++){
	         result.tensorpt[0][i][j] = tensorpt[0][i][j];
	      }
	   }

	   return result;
	}

	Matrix RemovePadding(){
	   Matrix result(rows-padded_rows, columns-padded_columns);

	   for(i=0; i<result.rows; i++){
	      for(j=0; j<result.columns; j++){
	         result.tensorpt[0][i][j] = tensorpt[0][i][j];
	      }
	   }

	   return result;
	}


	//tiling of the operands
	void MultiplyTilesOnce(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize){

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
   		      tensorpt[0][tileRstart+i][tileCstart+j] = 0;
		      }
	      }
	   }

	   //normal matrix multiplication for one tile
	   for(i=tileRstart; i<tileRend; i++){
	      for(j=tileCstart; j<tileCend; j++){
   		   for(k=ThisTileStart; k<ThisTileEnd; k++){
		         tensorpt[0][i][j] += A.tensorpt[0][i][k]*B.tensorpt[0][k][j];
	         }
		      std::cout<<"Element cumulative value("<<i<<","<<j<<") = "<<tensorpt[0][i][j]<<"\n";
	      }
	   } 
	   printf("\e[93mUscita dalla funzione MultiplyTilesOnce...\e[39mDistruzione delle variabili locali\n");
	}

	//tiling of the operands with the tile stored in a given pointer to pointer topointer (iteration > row > col)
	void MultiplyTiles(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize, Tensor& results){

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
		         results.tensorpt[IdxAcol][i][j] += A.tensorpt[0][i][k]*B.tensorpt[0][k][j];
	         }
	      }
	   } 
	   printf("\e[93mUscita dalla funzione MultiplyTiles...\e[39mDistruzione delle variabili locali\n");
	}




//OPERATORS
	Matrix operator+(Matrix second){
	   if(rows != second.rows || columns != second.columns){
	      std::cout<<"\e[93mMatrix dimensions don't match in the sum operation.\e[39m\n";
	      std::exit(403);
	   }
  	   Matrix result(rows, columns);

	   for(i=0; i<rows; i++){
	      for(j=0; j<columns; j++){
	         result.tensorpt[0][i][j] = tensorpt[0][i][j]+second.tensorpt[0][i][j];
	      }
	   }

//	   std::cout<<"Sum done. \n";
	   return result;
	}

	Matrix operator||(Matrix& second){   //an extended sum where if the value of index 
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
		       result.tensorpt[0][i][j] = tensorpt[0][i][j]+second.tensorpt[0][i][j];
		    }else if(j<columns){
		       result.tensorpt[0][i][j] = tensorpt[0][i][j];
		    }else{
		       result.tensorpt[0][i][j] = second.tensorpt[0][i][j];
		    }
		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
	         }
	      }else if(i<rows){
	         for(j=0; j<result.columns; j++){
		    if(j<columns){
		       result.tensorpt[0][i][j] = tensorpt[0][i][j];
		    }else{
		       result.tensorpt[0][i][j] = 0;
		    }
		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
		 }
	      }else{
	         for(j=0; j<result.columns; j++){
		    if(j<second.columns){
		       result.tensorpt[0][i][j] = second.tensorpt[0][i][j];
		    }else{
		       result.tensorpt[0][i][j] = 0;
		    }
		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
		 }
   	      }
	   }

	   std::cout<<"Sum done. \n";
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
		    result.tensorpt[0][i][j]+=tensorpt[0][i][k]*second.tensorpt[0][k][j];
//		    std::cout<<i<<", "<<k<<"\t";
		 }
//		 std::cout<<"Risultato indice ("<<i<<", "<<j<<"):\n";
//		 std::cout<<result.tensorpt[0][i][j]<<"\n";
 	      }
	   }
	   return result;
	}

/*
	// Copy assignment operator
    Matrix& operator=(const Matrix& other) {
		// Check for self-assignment
      if (this != &other)
		{
			// Deallocate the current memory and allocate the new required space
            for (int i = 0; i < rows; i++) {
                delete[] tensorpt[0][i];
            }
            delete[] tensorpt[0];

            // Allocate new memory
            rows = other.rows;
            columns = other.columns;
      	   padded_rows = other.padded_rows;
	         padded_columns = other.padded_columns;

            tensorpt[0] = new int*[rows];
            for (int i = 0; i < rows; i++) {
                tensorpt[0][i] = new int[columns];
                for (int j = 0; j < columns; j++) {
                    tensorpt[0][i][j] = other.tensorpt[0][i][j];
                }
            }
        }
        return *this;
    }
*/


//DESTRUCTOR
/*
   ~Matrix(){
	   std::cout<<"Distruggo matrice "<<rows<<"x"<<columns<<"...\n";
	   for(i=0; i<rows; i++){
	      delete[] tensorpt[0][i];
	      std::cout<<"Cancellata riga "<<i<<".\n";
	   }
	   delete[] tensorpt[0];
	   std::cout<<"Cancellata tutta la matrice "<<rows<<"x"<<columns<<".\n";
      delete[] tensorpt;
	}
*/
};


//todo:
//--function to collapse a tensor into a matrix by addistion of layers
//--modified tiling function to deposit the results on layers of a tensor, to be then collapsed with the previous 
//function
//--


