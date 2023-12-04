#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

template <typename T = int>
__global__ void CudaMultKernel(struct mat<T> A, struct mat<T> B, struct mat<T> C)
{
  T Cvalue = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.x;
  int column = blockIdx.x * blockDim.x + threadIdx.y;

  //__syncthreads();

  for(int i=0; i<A.width; i++){
    Cvalue += A.elements[row * A.width + i] * B.elements[i * B.width + column];
  }

  C.elements[row * C.width + column] = Cvalue;
}

template <typename T = int>
__global__ void TiledCudaMultKernel(struct mat<T> A, struct mat<T> B, struct mat<T> C)
{
  T Cvalue = 0;
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  for(int thisTileStart=0; thisTileStart < A.width; thisTileStart+=BLOCK_SIZE){
    
    __shared__ T Atile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Btile[BLOCK_SIZE][BLOCK_SIZE];

    Btile[ty][tx] = B.elements[(thisTileStart + ty) * B.width + bx * BLOCK_SIZE + tx];
    Atile[ty][tx] = A.elements[(by * BLOCK_SIZE + ty) * A.width + (thisTileStart * BLOCK_SIZE) + tx];

    __syncthreads();

    for(int k=0; k < BLOCK_SIZE; k++){
      Cvalue += Atile[ty][k] * Btile[k][tx];
    }

    __syncthreads();
  }

  C.elements[(by * BLOCK_SIZE + ty) * C.width + bx * BLOCK_SIZE + tx] = Cvalue;
}


template <typename T = int>
double CudaMult(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, bool tiled_mode)
//BLOCK_SIZE deve essere potenza di due, o meglio è preferibile che lo sia
{
  using namespace std;

  Matrix<T> CA = A.ForceAddTilingPaddingRows(BLOCK_SIZE);
  Matrix<T> CB = B.ForceAddTilingPaddingColumns(BLOCK_SIZE);
  Matrix<T> CC = C.ForceAddTilingPadding(BLOCK_SIZE);
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

  size_t Asize = h_A.height * h_A.width * sizeof(T);
  size_t Bsize = h_B.height * h_B.width * sizeof(T);
  size_t Csize = h_C.height * h_C.width * sizeof(T);


  checkCudaErrors(cudaMallocHost(&h_A.elements, Asize));
  checkCudaErrors(cudaMallocHost(&h_B.elements, Bsize));
  checkCudaErrors(cudaMallocHost(&h_C.elements, Csize));

  CA.BlurtMatrix(h_A.elements);
  CB.BlurtMatrix(h_B.elements);

  clock_t tic = clock();



  //now to allocate GPU memory
  struct mat<T> d_A;
  d_A.width = h_A.width;
  d_A.height = h_A.height;
  d_A.padd_height = h_A.padd_height;
  d_A.padd_width = h_A.padd_width;
  checkCudaErrors(cudaMalloc(&d_A.elements, Asize));

  struct mat<T> d_B;
  d_B.width = h_B.width;
  d_B.height = h_A.height;
  d_B.padd_height = h_B.padd_height;
  d_B.padd_width = h_B.padd_width;
  checkCudaErrors(cudaMalloc(&d_B.elements, Bsize));

  struct mat<T> d_C;
  d_C.width = h_C.width;
  d_C.height = h_A.height;
  d_C.padd_height = h_C.padd_height;
  d_C.padd_width = h_C.padd_width;
  checkCudaErrors(cudaMalloc(&d_C.elements, Csize));

  //now to populate the memory
  checkCudaErrors(cudaMemcpy(d_A.elements, h_A.elements, Asize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B.elements, h_B.elements, Bsize, cudaMemcpyHostToDevice));
  //C will be populated when it's calculated

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(h_B.width / dimBlock.x, h_A.height / dimBlock.y);

#ifdef VERBOSE
  cout<<"The thread block dimensions are: "<<dimBlock.y<<"x"<<dimBlock.x<<"\n";
  cout<<"The block grid dimensions are: "<<dimGrid.y<<"x"<<dimGrid.x<<"\n";
#endif

  if(tiled_mode){
    CudaMultKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);
  }else{
    TiledCudaMultKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);
  }

  checkCudaErrors(cudaMemcpy(h_C.elements, d_C.elements, Csize, cudaMemcpyDeviceToHost));

  clock_t toc = clock();

  CC.InitMatrix(h_C.elements, h_C.height * h_C.width);
  C = CC.RemovePadding();

  double execution_time = (double)(toc-tic)/CLOCKS_PER_SEC;

  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);

  checkCudaErrors(cudaFreeHost(h_A.elements));
  checkCudaErrors(cudaFreeHost(h_B.elements));
  checkCudaErrors(cudaFreeHost(h_C.elements));

#ifdef PRINT_NUMBERS
  if(tiled_mode){
    cout<<"result of tiled cuda multiplication:\n\n";
  }else{
    cout<<"result of normal cuda multiplication:\n\n";
  }
  C.PrintMatrix();
#endif

  return execution_time;
}

