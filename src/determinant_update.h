#define DET_BLOCK_SIZE 64

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>


// The first kernel just computes Ainv * u and also stores the kth
// row of Ainv in global memory
__global__ static void
update_inverse_cuda1 (float *Ainv_g[], float *u_g[], float *Ainv_u_g[],
		      float *Ainv_colk_g[], int N, int rowstride, int k)
{
  __shared__ float *Ainv, *u, *Ainv_u, *Ainv_colk;
  if (threadIdx.x==0) {
    Ainv     = Ainv_g[blockIdx.y];
    u         = u_g[blockIdx.y];
    Ainv_u    = Ainv_u_g[blockIdx.y];
    Ainv_colk = Ainv_colk_g[blockIdx.y];
  }

  __syncthreads();

  // Store the product Ainv * u in shared memory
  __shared__ float Ainv_u_shared[DET_BLOCK_SIZE], 
    Ainv_colk_shared[DET_BLOCK_SIZE], u_shared[DET_BLOCK_SIZE];
  Ainv_u_shared[threadIdx.x] = 0.0;
  int col = blockIdx.x*DET_BLOCK_SIZE + threadIdx.x;
  int numblocks = N / DET_BLOCK_SIZE;

  if (blockIdx.x*DET_BLOCK_SIZE <= k && k < (blockIdx.x+1)*DET_BLOCK_SIZE) {
    for (int block=0; block<numblocks; block++) {
      u_shared[threadIdx.x] = u[block*DET_BLOCK_SIZE+threadIdx.x];
      __syncthreads();
      for (int i=0; i<DET_BLOCK_SIZE; i++) {
	int row = block*DET_BLOCK_SIZE + i;
	
	float a = Ainv[row*rowstride+col];
	if (col == k)
	  Ainv_colk_shared[i] = a;
	Ainv_u_shared[threadIdx.x] += a*u_shared[i];
      }
      __syncthreads();
      Ainv_colk[block*DET_BLOCK_SIZE+threadIdx.x] = Ainv_colk_shared[threadIdx.x];
    }
  }
  else {
    for (int block=0; block<numblocks; block++) {
      u_shared[threadIdx.x] = u[block*DET_BLOCK_SIZE+threadIdx.x];
      __syncthreads();
      for (int i=0; i<DET_BLOCK_SIZE; i++) {
	int row = block*DET_BLOCK_SIZE + i;
	Ainv_u_shared[threadIdx.x] += Ainv[row*rowstride+col]*u_shared[i];
      }
    }
  }

  __syncthreads();
  
  // Write the data back to global memory
  Ainv_u[col]    = Ainv_u_shared[threadIdx.x];
}

__global__ static void
update_inverse_cuda2 (float *Ainv_g[], float *u_g[], float *Ainv_u_g[],
		      float *Ainv_colk_g[], int N, int rowstride, int k)
{
  __shared__ float *Ainv, *Ainv_u, *Ainv_colk;
  if (threadIdx.x==0) {
    Ainv     = Ainv_g[blockIdx.y];
    Ainv_u    = Ainv_u_g[blockIdx.y];
    Ainv_colk = Ainv_colk_g[blockIdx.y];
  }
  __syncthreads();

  __shared__ float Ainv_u_shared[DET_BLOCK_SIZE];
  __shared__ float  Ainv_colk_shared[DET_BLOCK_SIZE];
  int col = blockIdx.x*DET_BLOCK_SIZE + threadIdx.x;
  // Read the data back from global memory
  Ainv_u_shared[threadIdx.x] = Ainv_u[col];
  Ainv_colk_shared[threadIdx.x] = Ainv_colk[col];
  __shared__ float prefact;
  if (threadIdx.x == 0)
    prefact = -1.0f/(1.0f+Ainv_u[k]);
  __syncthreads();
		   
  int numblocks = N / DET_BLOCK_SIZE;
  for (int block=0; block<numblocks; block++) {
    Ainv_colk_shared[threadIdx.x] = prefact*Ainv_colk[block*DET_BLOCK_SIZE+threadIdx.x];
    __syncthreads();
    for (int i=0; i<DET_BLOCK_SIZE; i++) {
      int row = block*DET_BLOCK_SIZE + i;
      Ainv[row*rowstride+col] += Ainv_u_shared[threadIdx.x]*Ainv_colk_shared[i];
    }
  }
}
