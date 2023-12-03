
template <typename T = int>
__global__ void CudaMultKernel(struct mat<T> A, struct mat<T> B, struct mat<T> C)
{
  T Cvalue = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.x;
  int column = blockIdx.x * blockDim.x + threadIdx.y;

  for(int i=0; i<A.width; i++){
    Cvalue += A.elements[row * A.width + i] * B.elements[i * B.width + column];
  }

  C.elements[row * C.width + column] = Cvalue;
}
//__global__ void TiledCudaMultKernel();

template <typename T = int>
double CudaMult(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, const int bSize) 
//bSize deve essere potenza di due, o meglio è preferibile che lo sia
{
  using namespace std;

  Matrix<T> CA = A.ForceAddTilingPaddingRows(bSize);
  Matrix<T> CB = B.ForceAddTilingPaddingColumns(bSize);
  Matrix<T> CC = C.ForceAddTilingPadding(bSize);
  struct mat<T> h_A, h_B, h_C;

  //h for host
  h_A.width = CA.Columns();
  h_A.height = CA.Rows();
  h_A.padd_height = CA.PaddingRows();
  h_A.padd_width = CA.PaddingColumns();
  h_B.width = CB.Columns();
  h_B.height = CB.Rows();
  h_B.padd_height = CB.PaddingRows();
  h_B.padd_width = CB.PaddingColumns();
  h_C.width = CC.Columns();
  h_C.height = CC.Rows();
  h_C.padd_height = CC.PaddingRows();
  h_C.padd_width = CC.PaddingColumns();

  h_A.elements = new T[CA.Rows()*CA.Columns()];
  h_B.elements = new T[CB.Rows()*CB.Columns()];
  h_C.elements = new T[CC.Rows()*CC.Columns()];

  CA.BlurtMatrix(h_A.elements);
  CB.BlurtMatrix(h_B.elements);
  CC.BlurtMatrix(h_C.elements);

  for(int i=0; i<CA.Rows()*CA.Columns(); i++){
    cout<<CA.GetElement(i/CA.Columns(), i%CA.Columns())
        << "-vs-"
      <<h_A.elements[i]
      <<"\t";
  }

  clock_t tic = clock();


  size_t Asize = h_A.height * h_A.width * sizeof(TYPE);
  size_t Bsize = h_B.height * h_B.width * sizeof(TYPE);
  size_t Csize = h_C.height * h_C.width * sizeof(TYPE);

  //now to allocate GPU memory
  struct mat<T> d_A;
  d_A.width = h_A.width;
  d_A.height = h_A.height;
  d_A.padd_height = h_A.padd_height;
  d_A.padd_width = h_A.padd_width;
  cudaMalloc(&d_A.elements, Asize);

  struct mat<T> d_B;
  d_B.width = h_B.width;
  d_B.height = h_A.height;
  d_B.padd_height = h_B.padd_height;
  d_B.padd_width = h_B.padd_width;
  cudaMalloc(&d_B.elements, Bsize);

  struct mat<T> d_C;
  d_C.width = h_C.width;
  d_C.height = h_A.height;
  d_C.padd_height = h_C.padd_height;
  d_C.padd_width = h_C.padd_width;
  cudaMalloc(&d_C.elements, Csize);

  //now to populate the memory
  cudaMemcpy(d_A.elements, h_A.elements, Asize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B.elements, h_B.elements, Bsize, cudaMemcpyHostToDevice);
  //C will be populated when it's calculated

  dim3 dimBlock(bSize, bSize);
  dim3 dimGrid(B.Columns() / dimBlock.x, A.Rows() / dimBlock.y);
  CudaMultKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);

  cudaMemcpy(h_C.elements, d_C.elements, Csize, cudaMemcpyDeviceToHost);

  clock_t toc = clock();

  CC.InitMatrix(h_C.elements, h_C.height * h_C.width);

  double execution_time = (double)(toc-tic)/CLOCKS_PER_SEC;

  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);

  delete h_A.elements;
  delete h_B.elements;
  delete h_C.elements;

  return execution_time;
}//double TiledCudaMult();
