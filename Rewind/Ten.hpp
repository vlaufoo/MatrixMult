#ifndef TENSOR_H
#define TENSOR_H

#include<iostream>
#include<cmath>

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

  public:

    Tensor();
    Tensor(int dep, int r, int c);
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
    void TestValues();
    void Load(int* vlaues, int length);


    Tensor operator+(const Tensor& second);
    Tensor operator||(const Tensor& second);
    Tensor& operator=(const Tensor& other);
};

#endif
