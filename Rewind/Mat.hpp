#ifndef MATRIX_H
#define MATRIX_H

#include"Ten.hpp"

class Matrix : public Tensor
{
  private:
    int **data;
    int i, j, k;

  public:
    
    Matrix();
    Matrix(int r, int c);
    Matrix(Matrix& other);
    ~Matrix();
    
    void Identity();
    int GetElement(int n, int m);
    int ZeroElement(int n, int m);
    int SetElement(int n, int m, int v);

    void CollapseTensor(const Tensor& target);
    Tensor ChangeLayerFromMatrix(Tensor& target, int layer_idx);
    Tensor AppendMatrix(Tensor& target);

    Matrix AddTilingPaddingRows(int tSize);
    Matrix AddTilingPaddingColumns(int tSize);
    Matrix AddTilingPadding(int tSize);
    Matrix AddPaddingRows(int padding_rows);
    Matrix AddPaddingColumns(int padding_columns);

    Matrix RemovePadding();
    Matrix RemovePaddingRows();
    Matrix RemovePaddingColumns();

    void CalculateTile(Matrix A, Matrix B, int tSize, int tRow, int tCol);
    void MultiplyTilesOnce(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize);
    void MultiplyTiles(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize, Tensor& results);


    Matrix operator+(Matrix& second);
    Matrix operator*(Matrix& second);
    Matrix operator||(Matrix& second);
    Matrix& operator=(const Matrix& other);
};

#endif
