#define BLOCK_SIZE 64

#include <stdio.h>

__global__ void 
eval_multi_UBspline_3d_cuda_c (float *coefs, float *abc, float *vals,
			       int ix, int iy, int iz,
			       int xs, int ys, int zs, int N)
{
  int block = blockIdx.x;
  int thr   = threadIdx.x;
  int offset = block*BLOCK_SIZE+thr;

  float val= 0.0;
  //int index=0;
  val = 0.0;
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      for (int k=0; k<4; k++) {
	float *base_addr = coefs + (ix+i)*xs + (iy+j)*ys + (iz+k)*zs;
	val += abc[(16*i+4*j+k)*BLOCK_SIZE + thr] * base_addr[offset];
	//index++;
      }
  vals[offset] = val;
}


void
test_cuda()
{
  float *coefs  , *abc  , *abc2, *vals;
  float *coefs_d, *abc_d, *vals_d;
  int xs, ys, zs, N;
  int Nx, Ny, Nz;

  N = 512;
  Nx = Ny = Nz = 32;
  xs = Nx*Ny*Nz;
  ys = Ny*Nz;
  zs = Nz;
  
  int size = Nx*Ny*Nz*N*sizeof(float);
  posix_memalign((void**)&coefs, 16, size);
  cudaMalloc((void**)&coefs_d, size);
  for (int ix=0; ix<Nx; ix++)
    for (int iy=0; iy<Ny; iy++)
      for (int iz=0; iz<Nz; iz++)
	for (int n=0; n<N; n++)
	  coefs[ix*xs + iy*ys + iz*zs + n] = drand48();
  cudaMemcpy(coefs_d, coefs, size, cudaMemcpyHostToDevice);

  posix_memalign ((void**)&abc, 16, 64*sizeof(float));
  posix_memalign ((void**)&abc2, 16, 64*BLOCK_SIZE*sizeof(float));
  cudaMalloc((void**)&abc_d, 64*BLOCK_SIZE*sizeof(float));
  for (int i=0; i<64; i++) {
    abc[i] = drand48();
    for (int j=0; j<BLOCK_SIZE; j++)
      abc2[i*BLOCK_SIZE+j] = abc[i];
  }
  cudaMemcpy(abc_d, abc2, 64*BLOCK_SIZE*sizeof(float), 
	     cudaMemcpyHostToDevice);

  posix_memalign((void**)&vals, 16, N*sizeof(float));
  cudaMalloc((void**)&vals_d, N*sizeof(float));

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(N/BLOCK_SIZE);

  int ix=1; 
  int iy=2;
  int iz=3;
  
  clock_t start, end;
  start = clock();
  for (int i=0; i<100000; i++) {
    eval_multi_UBspline_3d_cuda_c<<<dimGrid,dimBlock>>> 
      (coefs_d, abc_d, vals_d, ix, iy, iz, xs, ys, zs, N);
  }
  end = clock();
  double time = (double)(end-start)/(double)(CLOCKS_PER_SEC*100000*N);
  fprintf (stderr, "Evals per second = %1.8e\n", 1.0/time);

  cudaMemcpy (vals, vals_d, N*sizeof(float), cudaMemcpyDeviceToHost);

  float vals2[512];
  
  for (int n=0; n<N; n++) {
    vals2[n] = 0.0;
    int index=0;
    for(int i=0; i<4; i++)
      for (int j=0; j<4; j++)
	for (int k=0; k<4; k++)  {
	  vals2[n] += abc[index] * coefs[(ix+i)*xs+(iy+j)*ys+(iz+k)*zs+n];
	  index++;
	}
  }


  for (int i=0; i<N; i++)	
    fprintf (stderr, "%1.9f %1.9f\n", vals[i], vals2[i]); 


  cudaFree(abc_d);
  cudaFree(coefs_d);
  cudaFree(vals_d);
}


main()
{
  test_cuda();
}
