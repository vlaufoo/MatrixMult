#ifndef TENSOR_H
#define TENSOR_H

#include<iostream>
#include<cmath>
#include<sstream>

class Tensor
{

  protected:

    int rows;
    int columns;
    int depth;
    int padded_rows;
    int padded_columns;
    int padded_layers;
    int ***data;
    int i, j, k;

  public:

    Tensor();
    Tensor(int dep, int r, int c);
    Tensor(int r, int c); //generates a matrix
    Tensor(const Tensor& other);
    ~Tensor();

    int Rows();
    int Columns();
    int Layers();
    int PaddingRows();
    int PaddingColumns();
    int PaddingLayers();
    int SetElement(int d, int n, int m, int val);
    int ZeroElement(int d, int n, int m);
    int GetElement(int d, int n, int m);
    void Print();
    void Print(int layer);
    void TestValues();
    void Random(int minVal, int maxVal, const unsigned int seed);
    void Load(int* vlaues, int length);

    void CollapseTensor();
    void CopyLayer(Tensor& A, int src_layer_idx, int dst_layer_idx);
    Tensor AppendTensor(Tensor& target);

    Tensor AddTilingPaddingRows(int tSize);
    Tensor AddTilingPaddingColumns(int tSize);
    Tensor AddTilingPadding(int tSize);
    Tensor AddPaddingRows(int padding_rows);
    Tensor AddPaddingColumns(int padding_columns);

    Tensor RemovePadding();
    Tensor RemovePaddingRows();
    Tensor RemovePaddingColumns();

    //void CalculateTile(Matrix A, Matrix B, int tSize, int tRow, int tCol);
    void MultiplyTilesOnce(Tensor& A, int a_layer, Tensor& B, int b_layer,
                           int iteration, int IdxArow, int IdxBcol, int tSize);

    //void MultiplyTiles(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize, Tensor& results);

    Tensor operator+(const Tensor& second);
    Tensor operator||(const Tensor& second);
    Tensor operator*(Tensor& second);
    Tensor& operator=(const Tensor& other);
};

#endif
