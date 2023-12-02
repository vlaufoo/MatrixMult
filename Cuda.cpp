//Questa funzione fa uso del kernel definito nel file Functions.hpp, che descrive la funzione svolta dai thread blocks di cuda. Qui invece svolgo funzioni di allocazione e preparazione

//i kernel da definire in fondo per la moltiplicazione prima element wise e poi tile wise
__global__ void CudaMultKernel();
__global__ void TiledCudaMultKernel();

double CudaMult(Matrix& A, Matrix& B, Matrix& C, )
{
  TYPE* matrix_A;
  TYPE* matrix_B;

  
}
