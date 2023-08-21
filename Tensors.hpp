class Tensor = {
      protected:
   int rows;
   int cols;
   int depth;
   int padded_rows;
   int padded_columns;
   int ***tensorpt;
      
      public:
   //constructors
   Tensor(){
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

   //copy constructor
   Tensor(const Tensor& other){
      rows = other.rows;
      columns = other.columns;
      depth = other.depth;
      padded_rows = other.padded_rows;
      padded_columns = other.padded_columns;
      padded_layers = other.padded_layers;

      tensorpt = new int** [depth];
      for(int k=0; k<depth; k++){
         tensorpt[k] = new int *[rows];
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
	      std::cout<<"Invalid indices.";
	   }
	   tensorpt[d][n][m] = val;
	}

	void ZeroElement(int d, int n, int m){
	   if(n >= rows || m >= columns || d >= depth){
	      std::cout<<"Invalid indices.";
	   }
	   tensorpt[d][n][m] = 0;
	}

	int GetElement(int d, int n, int m){
	   if(n >= rows || m >= columns || d >= depth){
	      std::cout<<"Invalid indices.";
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
         std::cout<<"\n\n"
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
	      k = i/columns*rows;
	      i = (i%(columns*rows))/columns;
         j = (i%(columns*rows))%columns;
	      tensorpt[k][i][j] = *(values + i);
	   }
	   
	}

   void ChangeLayerMatrix(Matrix A, int layer_idx){
      if(depth <= layer_idx || A.rows > rows || A.columns > columns){
         std::cout<<"Error in ChangeLayerMatrix: The matrix is to big or the layer index is too big."
      }

      for(int i=0; i<A.rows; i++){
         for(int j=0; j<columns; j++){
            tensorpt[layer_idx][i][j] = A.matrixpt[i][j];
         }
      }

   }

   //to be added:
   //--a function that appends a Matrix object as a layer of the Tensor
   //    (can only e called by a tensor object) [ done! ]
   //--a function that collapses a tensor into a matrix by adding all the layers
   //    (can only be called by a Matrix object since the return value is of type Matrix)


   Tensor AppendLayerMatrix(Matrix A){

      if(rows != A.rows || columns != A.columns){
         std::cout<<"Error in AppendLayerMatrix: matrix dimensions are too big"
      }
      Tensor result(depth+1, rows, columns);
      for(int k=0; k<depth; k++){
         for(int i=0; i<rows; i++){
            for(int j=0; j<columns; j++){
               result.tensorpt[k][i][j] = tensorpt[k][i][j];
            }  
         }
      }

      result.padded_rows = padded_rows;
      result.padded_columns = padded_columns;
      result.padded_layers = paded_layers;

      for(int i=0; i<rows; i++){
         for(int j=0; j<columns; j++){
            result.tensorpt[depth][i][j] = A.matrixpt[i][j];
         }  
      }
   }

   //OPERATORS
   Tensor operator+(const Tensor& second){
	   if(rows != second.rows || columns != second.columns){
	      std::cout<<"\e[93mTensor dimensions don't match in the sum operation.\e[39m\n";
	      std::exit(403);
	   }
  	   Tensor result(depth, rows, columns);

	   for(k=0; k<rows; k++){
	      for(i=0; i<columns; i++){
	         result.tensorpt[k][i][j] = tensorpt[k][i][j]+second.tensorpt[k][i][j];
	      }
	   }
   
//	   std::cout<<"Sum done. \n";
	   return result;
   }

   Tensor operator||(const Tensor& second){
      Tensor result(std::max(depth, second.depth), std::max(rows, second.rows), std::max(columns, second.columns))
      for(int k=0; k<result.depth; k++){

         if(k<depth && k<second.depth){
            for(i=0; i<result.rows; i++){
	            if(i<rows && i<second.rows){
	               for(j=0; j<result.columns; j++){
		               if(j<columns && j<second.columns){
		                  result.tensorpt[k][i][j] = tensorpt[k][i][j]+second.tensorpt[k][i][j];
		               }else if(j<columns){
		                  result.tensorpt[k][i][j] = tensorpt[k][i][j];
		               }else{
		                  result.tensorpt[k][i][j] = second.tensorpt[k][i][j];
		               }
//		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
	               }
	            }else if(i<rows){
	               for(j=0; j<result.columns; j++){
		               if(j<columns){
		                  result.tensorpt[k][i][j] = tensorpt[k][i][j];
		               }else{
		                  result.tensorpt[k][i][j] = 0;
		               }
//		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
		            }
	            }else{
	               for(j=0; j<result.columns; j++){
		               if(j<second.columns){
		                  result.tensorpt[k][i][j] = second.tensorpt[k][i][j];
		               }else{
		                  result.tensorpt[k][i][j] = 0;
		               }
//		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
		            }
   	         }
	         } 
         }else if(k<depth){        //now it's much simpler since I know that only "this.tensorpt" has valid values
                                   //i just need to be careful to set to zero the other locations (even though
                                   //they should be zzero already since the inizialization of the result tensor)
            for(i=0; i<result.rows; i++){
	            if(i<rows){
	               for(j=0; j<result.columns; j++){
		               if(j<columns){
		                  result.tensorpt[k][i][j] = tensorpt[k][i][j];
		               }else{
		                  result.tensorpt[k][i][j] = 0;
		               }
//		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
		            }
	            }else{
                  for(j=0; j<result.columns; j++){
                     result.tensorpt[k][i][j] = 0;
                  }
               }
	         }
         }else{            
            for(i=0; i<result.rows; i++){
	            if(i<second.rows){
	               for(j=0; j<result.columns; j++){
		               if(j<second.columns){
		                  result.tensorpt[k][i][j] = tensorpt[k][i][j];
		               }else{
		                  result.tensorpt[k][i][j] = 0;
		               }
//		    std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
		            }
	            }else{
                  for(j=0; j<result.columns; j++){
                     result.tensorpt[k][i][j] = 0;
                  }
               }
	         }
         }
      }
      return result;
   }

   //copy operator
   Tensor& operator=(const Tensor& other){
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
                  tensorpt[k][i][j] = other.matrixpt[k][i][j];
               }
            }
         }
      }
      return *this;
   }


   //DESTRUCTOR

   ~Tensor(){
      for(int k=0; k<depth; k++){  
         for(int i=0; i<rows; i++){
            delete[] tensorpt[k][i];
         }
         delete[] tensorpt[k];
      }
      delete[] tensorpt;
   }

};
