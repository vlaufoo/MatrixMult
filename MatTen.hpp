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
    int ***tensorpt;
  public:

    Tensor();
    Tensor(ind dep, int r, int c);
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

    void ChangeLayerFromMatrix(Matrix& target, int layer_idx);
    void AppendMatrix(Matrix& target);

    Tensor operator+(const Tensor& second);
    Tensor operator||(const Tensor& second);
    Tensor& operator=(const Tensor& other);
};

class Matrix : public Tensor
{
  private:
    int i, j, k;

  public:
    
    Matrix();
    Matrix(int r, int c);
    Matrix(const Matrix& other);
    ~Matrix();
    
    void Identity();
    int GetElement(int n, int m);
    int ZeroElement(int n, int m);
    int WriteElement(int n, int m, int v);

    void CollapseTensor(Tensor& target);




};
