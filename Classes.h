#include<iostream>
#include<sstream>

class Matrix
  {
  private:
    int rows;
    int columns;
    int padded_rows;
    int padded_columns;
    int **matrixpt;
    int i, j, k;

  public:
    //CONSTRUCTORS
    Matrix(){
      rows=100;
      columns=100;
      padded_rows = 0;
      padded_columns = 0;

      matrixpt = new int* [rows];
      for(i=0; i<rows; i++)
        matrixpt[i] = new int [columns];
      for(i=0; i<rows; i++){
        for(j=0; j<columns; j++){
          matrixpt[i][j] = 0;
        }
      }
#ifdef VERBOY
      std::cout<<"Creo matrice "<<rows<<"x"<<columns<<"\n";
#endif
    }

    Matrix(int r, int c){
      rows = r;
      columns = c;
      padded_rows = 0;
      padded_columns = 0;

      matrixpt = new int* [rows];
      for(i=0; i<rows; i++)
        matrixpt[i] = new int [columns];

      for(i=0; i<rows; i++){
        for(j=0; j<columns; j++){
          matrixpt[i][j] = 0;
        }
      }
#ifdef VERBOY
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
      matrixpt = new int*[rows];
      for (int i = 0; i < rows; i++) {
        matrixpt[i] = new int[columns];
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

    void SetElement(int n, int m, int val){
      if(n >= rows || m >= columns){
#ifdef VERBOY
        std::cout<<"Invalid indices.";
#endif
      }
      matrixpt[n][m] = val;
    }

    void ZeroElement(int n, int m){
      if(n >= rows || m >= columns){
#ifdef VERBOY
        std::cout<<"Invalid indices.";
#endif
      }
      matrixpt[n][m] = 0;
    }

    int GetElement(int n, int m){
      if(n >= rows || m >= columns){
#ifdef VERBOY
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
        matrixpt[i/columns][i%columns] = k++;
        if(k > 128)
          k=0;
        i++;
      }
    }

    void RandomMatrix(int minVal, int maxVal, const unsigned int seed){
      srand(seed);
      for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
          matrixpt[i][j] = rand() % (maxVal - minVal + 1) + minVal;
        }
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
        matrixpt[k][j] = *(values + i);
      }

    }

    //TILING FUNCTIONS
/*
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
          matrixpt[i][j] = 0;
        }
      }

      for(i=start_row; i<end_row; i++){
        for(j=start_column; j<end_column; j++){
          for(k=0; k<A.columns; k++){
            matrixpt[i][j] += A.matrixpt[i][k]*B.matrixpt[k][j];
          }
#ifdef VERBOY
          std::cout<<"Element ("<<i<<","<<j<<") = "<<matrixpt[i][j]<<"\n";
#endif
        }
      }
    }
*/

    Matrix ForceAddTilingPaddingRows(int tSize, int tile_rows){
      int padding_rows = tSize*tile_rows - rows;

      Matrix temp(rows+padding_rows, columns);
      Matrix result = *this || temp;
      printf("matrice chiamante: %p, matrice ritornata: %p\n", this, &result);
      result.padded_rows = padding_rows+padded_rows;
      return result;
    }

    Matrix ForceAddTilingPaddingColumns(int tSize, int tile_columns){
      int padding_columns = tSize*tile_columns - columns;

      Matrix temp(rows, columns+padding_columns);
      Matrix result = *this || temp;
      result.padded_columns = padding_columns+padded_columns;
      return result;
    }

    Matrix ForceAddTilingPadding(int tSize, int tile_rows, int tile_columns){

      int padding_rows = tSize*tile_rows-rows;
      int padding_columns = tSize*tile_columns-columns;

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
    void GetResultTile(const Matrix& A, const Matrix& B, int iterations, int IdxArow, int IdxBcol, int tSize){

      std::stringstream msg;
      
      for(int IdxAcol=0; IdxAcol<iterations; IdxAcol++){
        //debugging
//#ifdef VERBOY
        std::cout<<"\nTile row "<<IdxArow<<" column "<<IdxBcol<<". Iteration "<<IdxAcol<<"\n\n";
//#endif
        printf("Sono all'inizio\n");
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
//#ifdef VERBOY
          std::cout<<"Abnormal tile encountered...................."<<std::endl;
//#endif
        }

        //IdxAcol is equal to the iteration number so in the tile multiplication
        //the index of the destination tile is defined with IdxBcol, instead 
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

        for(int i=0; i<tSize; i++){
          for(int j=0; j<tSize; j++){
            matrixpt[tileRstart+i][tileCstart+j] = 0;
          }
        }
#ifdef VERBOY
        std::cout<<"First iter. check is true.\n";
#endif
      }

        //normal matrix multiplication for one tile
        for(i=tileRstart; i<tileRend; i++){
          for(j=tileCstart; j<tileCend; j++){
            for(k=ThisTileStart; k<ThisTileEnd; k++){
              matrixpt[i][j] += A.matrixpt[i][k]*B.matrixpt[k][j];
            }
          }
        }
      }
     // printf("\e[33m");
     // PrintMatrix();
     // printf("\e[0m");
#ifdef VERBOY
      std::cout<<"\e[93mUscita dalla funzione MultiplyTilesOnce...\e[39mDistruzione delle variabili locali\n";
#endif
    }









    //tiling of the operands

    void MultiplyTilesOnce(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize){
      std::stringstream msg;
#ifdef VERBOY
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
#ifdef VERBOY
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

        for(int i=0; i<tSize; i++){
          for(int j=0; j<tSize; j++){
            matrixpt[tileRstart+i][tileCstart+j] = 0;
          }
        }
#ifdef VERBOY
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
//            matrixpt[i][j] += A.GetElement(i, k)*B.GetElement(k, j);
#ifdef VERBOY
            msg << i<<", "<<j<<", "<<k<<"\n"<<"sum is now: "<<matrixpt[i][j]<<"\n";
            std::cout<< msg.str();
            msg.str("");
#endif
          }
#ifdef VERBOY
          msg << "Element "<<i<<" "<<j<<" pass "<<IdxAcol<<" done\n";
          std::cout<<msg.str();
          msg.str("");
#endif
        }
      }
    }














    //tiling of the operands with the tile stored in a given pointer to pointer topointer (iteration > row > col)
    void MultiplyTiles(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize, int** results){

      //debugging
#ifdef VERBOY
      std::cout<<"\nTile row "<<IdxArow<<" column "<<IdxBcol<<". Iteration "<<IdxAcol<<"\n\n";
#endif

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
#ifdef VERBOY
        std::cout<<"Abnormal tile encountered...................."<<std::endl;
#endif
      }

      //IdxAcol is equal to the iteration number so in the tile multiplication
      //the index of the destination tile is defined with IdxBcol, instead 
      //the inner most loop uses IdxAcol.

      //setting the padding rows and columns depending on the operands
      //padded_rows = A.padded_rows;
      //padded_columns = B.padded_columns;

      //the initialization to zero is left to the moment of allocation of the "results" pointer (before this 
      //function is called)

      //normal matrix multiplication for one tile
      for(i=tileRstart; i<tileRend; i++){
        for(j=tileCstart; j<tileCend; j++){
          for(k=ThisTileStart; k<ThisTileEnd; k++){
            results[i][j] += A.matrixpt[i][k]*B.matrixpt[k][j];
          }
        }
      } 
      //	   printf("\e[93mUscita dalla funzione MultiplyTiles...\e[39mDistruzione delle variabili locali\n");
    }







    //tiling of the operands with the tile stored in a given pointer to pointer topointer (iteration > row > col)
    void WriteResultTile(Matrix& A, Matrix& B, int iterations, int IdxArow, int IdxBcol, int tSize, int** results){

      for(int IdxAcol=0; IdxAcol<iterations; IdxAcol++){
        //debugging
#ifdef VERBOY
        std::cout<<"\nTile row "<<IdxArow<<" column "<<IdxBcol<<". Iteration "<<IdxAcol<<"\n\n";
#endif

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
#ifdef VERBOY
          std::cout<<"Abnormal tile encountered...................."<<std::endl;
#endif
        }

        //IdxAcol is equal to the iteration number so in the tile multiplication
        //the index of the destination tile is defined with IdxBcol, instead 
        //the inner most loop uses IdxAcol.

        //setting the padding rows and columns depending on the operands
        //padded_rows = A.padded_rows;
        //padded_columns = B.padded_columns;

        //the initialization to zero is left to the moment of allocation of the "results" pointer (before this 
        //function is called)

        //normal matrix multiplication for one tile
        for(i=tileRstart; i<tileRend; i++){
          for(j=tileCstart; j<tileCend; j++){
            for(k=ThisTileStart; k<ThisTileEnd; k++){
              results[i][j] += A.matrixpt[i][k]*B.matrixpt[k][j];
            }
          }
        } 
        //	   printf("\e[93mUscita dalla funzione MultiplyTiles...\e[39mDistruzione delle variabili locali\n");
      }
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
          result.matrixpt[i][j] = matrixpt[i][j]+second.matrixpt[i][j];
        }
      }

#ifdef VERBOY
      std::cout<<"Sum done. \n";
#endif
      return result;
    }

    Matrix operator||(Matrix second){   //an extended sum where if the value of index 
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
              result.matrixpt[i][j] = matrixpt[i][j]+second.matrixpt[i][j];
            }else if(j<columns){
              result.matrixpt[i][j] = matrixpt[i][j];
            }else{
              result.matrixpt[i][j] = second.matrixpt[i][j];
            }
#ifdef VERBOY
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
#ifdef VERBOY
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
#ifdef VERBOY
            std::cout<<"Doing sum of element "<<i<<", "<<j<<"\n";
#endif
          }
        }
      }

#ifdef VERBOY
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






