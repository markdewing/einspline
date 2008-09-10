#define BLOCK_SIZE 64

#include <stdio.h>

__global__ static void
update_inverse (float *AinvT, float *u, int N, int rowstride, int k)
{
  // Store the product Ainv * u in shared memory
  __shared__ float Ainv_u[BLOCK_SIZE], Ainv_u_k[BLOCK_SIZE];
  Ainv_u[threadIdx.x] = 0.0;
  __syncthreads();

  for (int row=0; row < N; row++)
    Ainv_u[threadIdx.x] += AinvT[row*rowstride+threadIdx.x]*u[row];
  
  // Compute lambda = [A^(-1)]_k dot u
  float lambda = 0.0;
  for (int i=0; i<N; i += BLOCK_SIZE) {
    Ainv_u_k[threadIdx.x] = AinvT[i+threadIdx.x] * u[i+threadIdx.x];
    __syncthreads();
    for (int j=BLOCK_SIZE>>2; j!=0; j >>=1) {
      if (threadIdx.x < j)
	Ainv_u_k[threadIdx.x] = Ainv_u_k[2*threadIdx.x] + Ainv_u_k[2*threadIdx.x+1];
      lambda += Ainv_u_k[0];
    }
    float prefact = 1.0/(1.0+lambda);
  }
    
}


main()
{
  int N = 64;
  float *A, *AinvT_h, *AinvT_d, *u_h, *u_d;

  A      = (float*)malloc (N*N*sizeof(float));
  AinvT_h = (float*)malloc (N*N*sizeof(float));
  cudaMalloc((void**)&AinvT_d, N*N*sizeof(float));
  
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++) 
      A[i*N+j] = AinvT_h[i*N+j] = drand48();
  
  cudaMemcpy (AinvT_d, AinvT_h, N*N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(1);

  update_inverse<<<dimGrid,dimBlock>>>(AinvT_h, u_d, N, N, 0);

}
