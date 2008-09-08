#define BLOCK_SIZE 64

#include <stdio.h>
#include "multi_bspline.h"

__constant__ float A[48];

typedef struct
{
  float *coefs;
  uint3 stride;
  float3 gridInv;
  int num_splines;
} multi_UBspline_3d_s_cuda;


multi_UBspline_3d_s_cuda*
create_CUDA_multi_UBspline_3d_s (multi_UBspline_3d_s* spline)
{
  multi_UBspline_3d_s_cuda *cuda_spline =
    (multi_UBspline_3d_s_cuda*) malloc (sizeof (multi_UBspline_3d_s_cuda*));
  
  cuda_spline->num_splines = spline->num_splines;

  int Nx = spline->x_grid.num+3;
  int Ny = spline->y_grid.num+3;
  int Nz = spline->z_grid.num+3;

  int N = spline->num_splines;
  if ((N%BLOCK_SIZE) != 0)
    N += 64 - (N%BLOCK_SIZE);
  cuda_spline->stride.x = Ny*Nz*N;
  cuda_spline->stride.y = Nz*N;
  cuda_spline->stride.z = N;

  size_t size = Nx*Ny*Nz+N*sizeof(float);

  cudaMalloc((void**)&(cuda_spline->coefs), size);
  
  float *spline_buff = (float*)malloc(size);

  for (int ix=0; ix<Nx; ix++)
    for (int iy=0; iy<Ny; iy++)
      for (int iz=0; iz<Nz; iz++) 
	for (int isp=0; isp<spline->num_splines; isp++) {
	  spline_buff[ix*cuda_spline->stride.x +
		      iy*cuda_spline->stride.y +
		      iz*cuda_spline->stride.z + isp] =
	    spline->coefs[ix*spline->x_stride +
			  iy*spline->y_stride +
			  iz*spline->z_stride + isp];
	}
  cudaMemcpy(cuda_spline->coefs, spline_buff, size, cudaMemcpyHostToDevice);

  free(spline_buff);

  return cuda_spline;
}



__global__ static void
eval_multi_multi_UBspline_3d_s_cuda (float *pos, float3 drInv, 
				     float *coefs, float *vals[], uint3 strides)
{
  int block = blockIdx.x;
  int thr   = threadIdx.x;
  int ir    = blockIdx.y;
  int off   = block*BLOCK_SIZE+thr;

  __shared__ float *myval;
  __shared__ float abc[64];

  __shared__ float3 r;
  if (thr == 0) {
    r.x = pos[4*ir+0];
    r.y = pos[4*ir+1];
    r.z = pos[4*ir+2];
    myval = vals[ir];
  }
  __syncthreads();
  
  int3 index;
  float3 t;
  float s, sf;
  float4 tp[3];

  s = r.x * drInv.x;
  sf = floor(s);
  index.x = (int)sf;
  t.x = s - sf;

  s = r.y * drInv.y;
  sf = floor(s);
  index.y = (int)sf;
  t.y = s - sf;

  s = r.z * drInv.z;
  sf = floor(s);
  index.z = (int)sf;
  t.z = s - sf;
  
  tp[0] = make_float4(1.0, t.x, t.x*t.x, t.x*t.x*t.x);
  tp[1] = make_float4(1.0, t.y, t.y*t.y, t.y*t.y*t.y);
  tp[2] = make_float4(1.0, t.z, t.z*t.z, t.z*t.z*t.z);

  __shared__ float a[4], b[4], c[4];
  if (thr < 4) {
    a[thr] = A[4*thr+0]*tp[0].x + A[4*thr+1]*tp[0].y + A[4*thr+2]*tp[0].z + A[4*thr+3]*tp[0].z;
    b[thr] = A[4*thr+0]*tp[1].x + A[4*thr+1]*tp[1].y + A[4*thr+2]*tp[1].z + A[4*thr+3]*tp[1].z;
    c[thr] = A[4*thr+0]*tp[2].x + A[4*thr+1]*tp[2].y + A[4*thr+2]*tp[2].z + A[4*thr+3]*tp[2].z;
  }
  __syncthreads();

  int i = (thr>>4)&3;
  int j = (thr>>2)&3;
  int k = (thr & 3);
  
  abc[thr] = a[i]*b[j]*c[k];
  __syncthreads();


  float val = 0.0;
  for (int i=0; i<4; i++) {
    for (int j=0; j<4; j++) {
      float *base = coefs + (index.x+i)*strides.x + (index.y+j)*strides.y + index.z*strides.z;
      for (int k=0; k<4; k++) 
  	val += abc[16*i+4*j+k] * base[off+k*strides.z];
    }
  }
  myval[off] = val;
}



__global__ static void
eval_multi_multi_UBspline_3d_s_vgh_cuda (float *pos, float3 drInv,  float *coefs, 
					 float *vals[], float *grads[], float *hess[],
					 uint3 strides)
{
  int block = blockIdx.x;
  int thr   = threadIdx.x;
  int ir    = blockIdx.y;
  int off   = block*BLOCK_SIZE+thr;

  __shared__ float *myval, *mygrad, *myhess;
  __shared__ float3 r;
  if (thr == 0) {
    r.x = pos[4*ir+0];
    r.y = pos[4*ir+1];
    r.z = pos[4*ir+2];
    myval  = vals[ir];
    mygrad = grads[ir];
    myhess = hess[ir];
  }
  __syncthreads();
  
  int3 index;
  float3 t;
  float s, sf;
  float4 tp[3];

  s = r.x * drInv.x;
  sf = floor(s);
  index.x = (int)sf;
  t.x = s - sf;

  s = r.y * drInv.y;
  sf = floor(s);
  index.y = (int)sf;
  t.y = s - sf;

  s = r.z * drInv.z;
  sf = floor(s);
  index.z = (int)sf;
  t.z = s - sf;
  
  tp[0] = make_float4(1.0, t.x, t.x*t.x, t.x*t.x*t.x);
  tp[1] = make_float4(1.0, t.y, t.y*t.y, t.y*t.y*t.y);
  tp[2] = make_float4(1.0, t.z, t.z*t.z, t.z*t.z*t.z);

  // First 4 of a are value, second 4 are derivative, last four are
  // second derivative.
  __shared__ float a[12], b[12], c[12];
  if (thr < 12) {
    a[thr] = A[4*thr+0]*tp[0].x + A[4*thr+1]*tp[0].y + A[4*thr+2]*tp[0].z + A[4*thr+3]*tp[0].z;
    b[thr] = A[4*thr+0]*tp[1].x + A[4*thr+1]*tp[1].y + A[4*thr+2]*tp[1].z + A[4*thr+3]*tp[1].z;
    c[thr] = A[4*thr+0]*tp[2].x + A[4*thr+1]*tp[2].y + A[4*thr+2]*tp[2].z + A[4*thr+3]*tp[2].z;
  }
  __syncthreads();

  __shared__ float abc[640];
  int i = (thr>>4)&3;
  int j = (thr>>2)&3;
  int k = (thr & 3);

  abc[10*(16*i+4*j+k)+0] = a[i+0]*b[j+0]*c[k+0]; // val
  abc[10*(16*i+4*j+k)+1] = a[i+4]*b[j+0]*c[k+0]; // d/dx
  abc[10*(16*i+4*j+k)+2] = a[i+0]*b[j+4]*c[k+0]; // d/dy
  abc[10*(16*i+4*j+k)+3] = a[i+0]*b[j+0]*c[k+4]; // d/dz
  abc[10*(16*i+4*j+k)+4] = a[i+8]*b[j+0]*c[k+0]; // d2/dx2
  abc[10*(16*i+4*j+k)+5] = a[i+4]*b[j+4]*c[k+0]; // d2/dxdy
  abc[10*(16*i+4*j+k)+6] = a[i+4]*b[j+0]*c[k+4]; // d2/dxdz
  abc[10*(16*i+4*j+k)+7] = a[i+0]*b[j+8]*c[k+0]; // d2/dy2
  abc[10*(16*i+4*j+k)+8] = a[i+0]*b[j+4]*c[k+4]; // d2/dydz
  abc[10*(16*i+4*j+k)+9] = a[i+0]*b[j+0]*c[k+8]; // d2/dz2

  __syncthreads();

  float v = 0.0;
  float g0=0.0,  g1=0.0, g2=0.0, 
    h00=0.0, h01=0.0, h02=0.0, h11=0.0, h12=0.0, h22=0.0;
  int n = 0;
  for (int i=0; i<4; i++) {
    for (int j=0; j<4; j++) {
      float *base = coefs + (index.x+i)*strides.x + (index.y+j)*strides.y + index.z*strides.z;
      for (int k=0; k<4; k++) {
	float c = base[off+k*strides.z];
	v   += abc[n+0] * c;
	g0  += abc[n+1] * c;
	g1  += abc[n+2] * c;
	g2  += abc[n+3] * c;
	h00 += abc[n+4] * c;
	h01 += abc[n+5] * c;
	h02 += abc[n+6] * c;
	h11 += abc[n+7] * c;
	h12 += abc[n+8] * c;
	h22 += abc[n+9] * c;
	n += 10;
      }
    }
  }
  g0 *= drInv.x; 
  g1 *= drInv.y; 
  g2 *= drInv.z; 

  h00 *= drInv.x * drInv.x;  
  h01 *= drInv.x * drInv.y;  
  h02 *= drInv.x * drInv.z;  
  h11 *= drInv.y * drInv.y;  
  h12 *= drInv.y * drInv.z;  
  h22 *= drInv.z * drInv.z;  

  
  __shared__ float buff[6*BLOCK_SIZE];
  // Note, we can reuse abc, by replacing buff with abc.
  myval[off] = v;
  buff[3*thr+0] = g0; 
  buff[3*thr+1] = g1; 
  buff[3*thr+2] = g2; 
  __syncthreads();
  for (int i=0; i<3; i++) 
    mygrad[(3*block+i)*BLOCK_SIZE+thr] = buff[i*BLOCK_SIZE+thr]; 
  __syncthreads();

  // Write first half of Hessians
  buff[6*thr+0]  = h00;
  buff[6*thr+1]  = h01;
  buff[6*thr+2]  = h02;
  buff[6*thr+3]  = h11;
  buff[6*thr+4]  = h12;
  buff[6*thr+5]  = h22;
  __syncthreads();
  for (int i=0; i<6; i++) 
    myhess[(6*block+i)*BLOCK_SIZE+thr] = buff[i*BLOCK_SIZE+thr];
}

				    


static void *
test_multi_cuda(void *thread)
{
  cudaSetDevice((int)(size_t)thread);
  fprintf (stderr, "In thread %p\n", thread);

  int numWalkers = 200;
  float *coefs  ,  __device__ *vals[numWalkers], *grads[numWalkers], *hess[numWalkers];
  float *coefs_d, __device__ **vals_d, **grads_d, **hess_d;
  float A_h[48] = { -1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
		     3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
		    -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
		     1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0,
		         0.0,     -0.5,      1.0,    -0.5,
  		         0.0,      1.5,     -2.0,     0.0,
		         0.0,     -1.5,      1.0,     0.5,
		         0.0,      0.5,      0.0,     0.0,
		         0.0,      0.0,     -1.0,     1.0,
		         0.0,      0.0,      3.0,    -2.0,
		         0.0,      0.0,     -3.0,     1.0,
		         0.0,      0.0,      1.0,     0.0 };

  // Copy A to host
  cudaMemcpy(A, A_h, 48*sizeof(float), cudaMemcpyHostToDevice); 

  float *r_d, *r_h;
  int xs, ys, zs, N;
  int Nx, Ny, Nz;

  N = 128;
  Nx = Ny = Nz = 16;
  xs = Ny*Nz*N;
  ys = Nz*N;
  zs = N;

  float3 drInv;
  drInv.x = 1.0/float(Nx);
  drInv.y = 1.0/float(Ny);
  drInv.z = 1.0/float(Nz);

  // Setup Bspline coefficients
  int size = Nx*Ny*Nz*N*sizeof(float);
  posix_memalign((void**)&coefs, 16, size);
  for (int ix=0; ix<Nx; ix++)
    for (int iy=0; iy<Ny; iy++)
      for (int iz=0; iz<Nz; iz++)
	for (int n=0; n<N; n++)
	  coefs[ix*xs + iy*ys + iz*zs + n] = drand48();


  fprintf (stderr, "Filled in coefs.\n");
  fprintf (stderr, "size = %d\n", size);
  
  // Setup CUDA coefficients
  fprintf (stderr, "Before first CUDA mallocs.\n");
  cudaMalloc((void**)&coefs_d, 2*size);
  fprintf (stderr, "Before Memcpy.\n");
  cudaMemcpy(coefs_d, coefs, size, cudaMemcpyHostToDevice);
  fprintf (stderr, "After Memcpy.\n");  

  // Setup device value storage
  int numVals = N*numWalkers*10;
  float *valBlock_d, *valBlock_h;
  cudaMalloc((void**)&(valBlock_d),     numVals*sizeof(float));
  cudaMallocHost((void**)&(valBlock_h), numVals*sizeof(float));
  cudaMalloc((void**)&(vals_d),  numWalkers*sizeof(float*));
  cudaMalloc((void**)&(grads_d), numWalkers*sizeof(float*));
  cudaMalloc((void**)&(hess_d),  numWalkers*sizeof(float*));
  fprintf (stderr, "valBlock_d = %p\n", valBlock_d);
  for (int i=0; i<numWalkers; i++) {
    vals[i]  = valBlock_d + i*N;
    grads[i] = valBlock_d + N*numWalkers + 3*i*N;
    hess[i]  = valBlock_d + 4*N*numWalkers + 6*i*N;
  }
  cudaMemcpy(vals_d,  vals,  numWalkers*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(grads_d, grads, numWalkers*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(hess_d,  hess,  numWalkers*sizeof(float*), cudaMemcpyHostToDevice);
  
  fprintf (stderr, "Finished cuda allocations.\n");


  // Setup walker positions
  cudaMalloc((void**)&(r_d),     4*numWalkers*sizeof(float));
  cudaMallocHost((void**)&(r_h), 4*numWalkers*sizeof(float));

  for (int ir=0; ir<numWalkers; ir++) {
    r_h[4*ir+0] = 0.5*drand48();
    r_h[4*ir+1] = 0.5*drand48();
    r_h[4*ir+2] = 0.5*drand48();
  }

  uint3 strides;
  strides.x = xs;
  strides.y = ys;
  strides.z = zs;

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(N/BLOCK_SIZE,numWalkers);
  
  clock_t start, end;

  start = clock();
  for (int i=0; i<10000; i++) {
    if ((i%1000) == 0) 
      fprintf (stderr, "i = %d\n", i);
    cudaMemcpy(r_d, r_h, 4*numWalkers*sizeof(float), cudaMemcpyHostToDevice);
    eval_multi_multi_UBspline_3d_s_cuda<<<dimGrid,dimBlock>>> 
       (r_d, drInv, coefs_d, vals_d, strides);
  }
  end = clock();
  double time = (double)(end-start)/(double)((double)CLOCKS_PER_SEC*(double)10000*N*numWalkers);
  fprintf (stderr, "VGH evals per second = %1.8e\n", 1.0/time);

  start = clock();
  for (int i=0; i<10000; i++) {
    if ((i%1000) == 0) 
      fprintf (stderr, "i = %d\n", i);
    cudaMemcpy(r_d, r_h, 4*numWalkers*sizeof(float), cudaMemcpyHostToDevice);
    eval_multi_multi_UBspline_3d_s_vgh_cuda<<<dimGrid,dimBlock>>> 
       (r_d, drInv, coefs_d, vals_d, grads_d, hess_d, strides);
  }
  end = clock();
  time = (double)(end-start)/(double)((double)CLOCKS_PER_SEC*(double)10000*N*numWalkers);
  fprintf (stderr, "Evals per second = %1.8e\n", 1.0/time);
  
  cudaFree (valBlock_d);
  cudaFree (vals_d);
  cudaFree (coefs_d);
  cudaFree (r_d);

  return NULL;

}




main()
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  fprintf (stderr, "Detected %d CUDA devices.\n", deviceCount);

  // test_cuda();

  for (int device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    fprintf (stderr, "Device %d:\n", device);
    fprintf (stderr, "  Global memory:     %10d\n",
	     deviceProp.totalGlobalMem);
    fprintf (stderr, "  MultiProcessors:   %10d\n",
	     deviceProp.multiProcessorCount);
    fprintf (stderr, "  Registers:         %10d\n", 
	     deviceProp.regsPerBlock);
    fprintf (stderr, "  Constant memory:   %10d\n", 
	     deviceProp.totalConstMem);
    fprintf (stderr, "  Shared memory:     %10d\n", 
	     deviceProp.sharedMemPerBlock);
  }

  test_multi_cuda((void*)0);
}
