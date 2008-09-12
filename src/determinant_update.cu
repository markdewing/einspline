#define BLOCK_SIZE 64

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>


// The first kernel just computes Ainv * u and also stores the kth
// row of Ainv in global memory
__global__ static void
update_inverse_cuda1 (float *AinvT, float *u, float *Ainv_u,
		      float *Ainv_rowk, int N, int rowstride, int k)
{
  // Store the product Ainv * u in shared memory
  __shared__ float Ainv_u_shared[BLOCK_SIZE], Ainv_rowk_shared[BLOCK_SIZE];
  __shared__ float u_shared[BLOCK_SIZE];
  Ainv_u_shared[threadIdx.x] = 0.0;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  int numblocks = N / BLOCK_SIZE;

  if (blockIdx.x*BLOCK_SIZE <= k && k < (blockIdx.x+1)*BLOCK_SIZE) {
    for (int block=0; block<numblocks; block++) {
      u_shared[threadIdx.x] = u[block*BLOCK_SIZE+threadIdx.x];
      __syncthreads();
      for (int i=0; i<BLOCK_SIZE; i++) {
	int row = block*BLOCK_SIZE + i;
	
	float a = AinvT[row*rowstride+col];
	if (col == k)
	  Ainv_rowk_shared[i] = a;
	Ainv_u_shared[threadIdx.x] += a*u_shared[i];
      }
      __syncthreads();
      Ainv_rowk[block*BLOCK_SIZE+threadIdx.x] = Ainv_rowk_shared[threadIdx.x];
    }
  }
  else {
    for (int block=0; block<numblocks; block++) {
      u_shared[threadIdx.x] = u[block*BLOCK_SIZE+threadIdx.x];
      __syncthreads();
      for (int i=0; i<BLOCK_SIZE; i++) {
	int row = block*BLOCK_SIZE + i;
	Ainv_u_shared[threadIdx.x] += AinvT[row*rowstride+col]*u_shared[i];
      }
    }
  }

  __syncthreads();

  // Write the data back to global memory
  Ainv_u[col]    = Ainv_u_shared[threadIdx.x];
  //Ainv_rowk[col] = Ainv_rowk_shared[threadIdx.x];
}


__global__ static void
update_inverse_cuda2 (float *AinvT, float *u, float *Ainv_u,
		      float *Ainv_rowk, int N, int rowstride, int k)
{
  __shared__ float Ainv_u_shared[BLOCK_SIZE], Ainv_rowk_shared[BLOCK_SIZE];
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  // Read the data back from global memory
  Ainv_u_shared[threadIdx.x] = Ainv_u[col];
  Ainv_rowk_shared[threadIdx.x] = Ainv_rowk[col];
  __shared__ float prefact;
  if (threadIdx.x == 0)
    prefact = 1.0f/(1.0f+Ainv_u[k]);
  __syncthreads();
		   
  int numblocks = N / BLOCK_SIZE;
  for (int block=0; block<numblocks; block++) {
    Ainv_rowk_shared[threadIdx.x] = prefact*Ainv_rowk[block*BLOCK_SIZE+threadIdx.x];
    __syncthreads();
    for (int i=0; i<BLOCK_SIZE; i++) {
      int row = block*BLOCK_SIZE + i;
      AinvT[row*rowstride+col] -= Ainv_u_shared[threadIdx.x]*Ainv_rowk_shared[i];
    }
  }
}



// __global__ static void
// update_inverse_cuda (float *AinvT, float *u, int N, int rowstride, int k)
// {
//   // Store the product Ainv * u in shared memory
//   __shared__ float Ainv_u[BLOCK_SIZE], Ainv_u_k[BLOCK_SIZE];
//   Ainv_u[threadIdx.x] = 0.0;
//   __syncthreads();

//   for (int row=0; row < N; row++)
//     Ainv_u[threadIdx.x] += AinvT[row*rowstride+threadIdx.x]*u[row];
  
//   // Compute lambda = [A^(-1)]_k dot u
//   float lambda = 0.0;
//   for (int i=0; i<N; i += BLOCK_SIZE) {
//     Ainv_u_k[threadIdx.x] = AinvT[i+threadIdx.x] * u[i+threadIdx.x];
//     __syncthreads();
//     for (int j=BLOCK_SIZE>>1; j!=0; j >>=1) {
//       if (threadIdx.x < j)
// 	Ainv_u_k[threadIdx.x] = Ainv_u_k[2*threadIdx.x] + Ainv_u_k[2*threadIdx.x+1];
//       lambda += Ainv_u_k[0];
//     }
//     float prefact = 1.0/(1.0+lambda);
//   }

//   // Now, subtract off outer product
// }





void
update_inverse (float *AinvT, float *u, int N, int k)
{
  float Ainv_u[N], Ainv_rowk[N];
  
  for (int i=0; i<N; i++) {
    Ainv_u[i] = 0.0f;
    Ainv_rowk[i] = AinvT[N*i+k];
    for (int j=0; j<N; j++)
      Ainv_u[i] += AinvT[j*N+i] * u[j];
  }

  float prefact = 1.0/(1.0+Ainv_u[k]);

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      AinvT[j*N+i] -= prefact * Ainv_u[i]*Ainv_rowk[j];
}



// Replaces A with its inverse by gauss-jordan elimination with full pivoting
// Adapted from Numerical Recipes in C
void GJInverse (double *A, int n)
{
  const int maxSize = 2000;

  if (n == 2) { // Special case for 2x2
    double a=A[0]; double b=A[1];
    double c=A[2]; double d=A[3];
    double detInv = 1.0/(a*d-b*c);
    A[0] = d*detInv;
    A[1] = -b*detInv;
    A[2] = -c*detInv;
    A[3] =  a*detInv;
    return;
  }

  int colIndex[maxSize], rowIndex[maxSize], ipiv[maxSize];
  double big, pivInv;
  int icol, irow;
  
  for (int j=0; j<n; j++)
    ipiv[j] = -1;

  for (int i=0; i<n; i++) {
    big = 0.0;
    for (int j=0; j<n; j++) 
      if (ipiv[j] != 0)
	for (int k=0; k<n; k++) {
	  if (ipiv[k] == -1) {
	    if (fabs(A[n*j+k]) >= big) {
	      big = fabs(A[n*j+k]);
	      irow = j; 
	      icol = k;
	    }
	  }
	  else if (ipiv[k] > 0) {
	    fprintf (stderr, "GJInverse: Singular matrix!\n");
	    exit(1);
	  }
	}
    ++(ipiv[icol]); 
    
    if (irow != icol) 
      for (int l=0; l<n; l++) {
	double tmp = A[n*irow+l];
	A[n*irow+l] = A[n*icol+l];
	A[n*icol+l] = tmp;
	// swap (A[n*irow+l], A[n*icol+l]);
      }
			     
    
    rowIndex[i] = irow;
    colIndex[i] = icol;
    if (A[n*icol+icol] == 0.0) { 
      fprintf (stderr, "GJInverse: Singular matrix!\n");
      exit(1);
    }
    pivInv = 1.0/A[n*icol+icol];
    A[n*icol+icol] = 1.0;
    for (int l=0; l<n; l++)
      A[n*icol+l] *= pivInv;
    for (int ll=0; ll<n; ll++)
      if (ll != icol) {
	double dum = A[n*ll+icol];
	A[n*ll+icol] = 0.0;
	for (int l=0; l<n; l++)
	  A[n*ll+l] -= A[n*icol+l]*dum;
      }
  }
  // Now unscramble the permutations
  for (int l=n-1; l>=0; l--) {
    if (rowIndex[l] != colIndex[l])
      for (int k=0; k<n ; k++) {
	double tmp = A[n*k+rowIndex[l]];
	A[n*k+rowIndex[l]] = A[n*k+colIndex[l]];
	A[n*k+colIndex[l]] = tmp;
	// swap (A(k,rowIndex[l]),A(k, colIndex[l]));
      }
  }
}



main()
{
  int N = 128;
  double *A, *Ainv, *ident;
  float *AinvT_h, *u_h;
  float *AinvT_d, *Ainv_u_d, *Ainv_rowk_d, *u_d;

  A       = (double*)malloc (N*N*sizeof(double));
  Ainv    = (double*)malloc (N*N*sizeof(double));
  ident   = (double*)malloc (N*N*sizeof(double));
  AinvT_h = (float*)malloc (N*N*sizeof(float));
  u_h     = (float*)malloc (N*sizeof(float));
  cudaMalloc((void**)&AinvT_d, N*N*sizeof(float));
  cudaMalloc((void**)&u_d, N*sizeof(float));
  cudaMalloc((void**)&Ainv_u_d, N*sizeof(float));
  cudaMalloc((void**)&Ainv_rowk_d, N*sizeof(float));
  
  for (int i=0; i<N; i++) {
    u_h[i] = drand48();
    for (int j=0; j<N; j++) 
      A[i*N+j] = Ainv[i*N+j] = drand48();
  }
  GJInverse(Ainv, N);

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++) {
      double ident = 0.0;
      for (int k=0; k<N; k++)
	ident += Ainv[i*N+k]*A[k*N+j];
      if ((i==j && fabs(ident - 1.0) > 1.0e-8) ||
	  (i!=j && fabs(ident) > 1.0e-8))
	fprintf (stderr, "Error in matrix inverse.\n");
    }

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++) 
      AinvT_h[j*N+i] = (float)Ainv[i*N+j];

  cudaMemcpy (AinvT_d, AinvT_h, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy (u_d, u_h, N*sizeof(float), cudaMemcpyHostToDevice);

  
  int col = 1;

  update_inverse (AinvT_h, u_h, N, col);

  for (int i=0; i<N; i++)
    A[i*N+col] += u_h[i];

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++) {
      double ident = 0.0;
      for (int k=0; k<N; k++)
	ident += AinvT_h[k*N+i]*A[k*N+j];
      if ((i==j && fabs(ident - 1.0) > 1.0e-5) ||
	  (i!=j && fabs(ident) > 1.0e-5))
	fprintf (stderr, "Error in matrix inverse, (%d, %d) = %1.8f\n", i, j, ident);
    }

  clock_t host_start = clock();
  for (int i=0; i<100000; i++) 
    update_inverse (AinvT_h, u_h, N, col);
  clock_t host_end = clock();
  double host_time = (double)(host_end - host_start)/(double)(CLOCKS_PER_SEC);
  double host_rate = 1.0e5/host_time;
  fprintf (stderr, "Host rate = %1.8f updates per seconds.\n", host_rate);


  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(N/BLOCK_SIZE);

  update_inverse_cuda1<<<dimGrid,dimBlock>>>
    (AinvT_d, u_d, Ainv_u_d, Ainv_rowk_d, N, N, col);
  update_inverse_cuda2<<<dimGrid,dimBlock>>>
    (AinvT_d, u_d, Ainv_u_d, Ainv_rowk_d, N, N, col);

  cudaMemcpy (AinvT_h, AinvT_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  fprintf (stderr, "Device test:  ");
  bool passed = true;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++) {
      double ident = 0.0;
      for (int k=0; k<N; k++)
	ident += AinvT_h[k*N+i]*A[k*N+j];
      if ((i==j && fabs(ident - 1.0) > 1.0e-5) ||
	  (i!=j && fabs(ident) > 1.0e-5)) {
	fprintf (stderr, "Error in matrix inverse, (%d, %d) = %1.8f\n", i, j, ident);
	passed = false;
      }
    }
  if (passed)
    fprintf (stderr, "Passed.\n");
  else
    fprintf (stderr, "Failed.\n");
    

  dim3 dimGrid2(N/BLOCK_SIZE, 1000);

  clock_t start = clock();
  for (int i=0; i<1000; i++) {
    update_inverse_cuda1<<<dimGrid2,dimBlock>>>
      (AinvT_d, u_d, Ainv_u_d, Ainv_rowk_d, N, N, col);
    update_inverse_cuda2<<<dimGrid2,dimBlock>>>
      (AinvT_d, u_d, Ainv_u_d, Ainv_rowk_d, N, N, col);
  }
  clock_t end = clock();

  double time = (double)(end-start)/(double)CLOCKS_PER_SEC;
  double rate = 1.0e6/time;

  fprintf (stderr, "Device rate = %1.8f updates per seconds.\n", rate);


}
