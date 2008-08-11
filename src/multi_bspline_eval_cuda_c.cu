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
  __shared__ float abcs[64];
  abcs[thr] = abc[thr];
  
  __syncthreads();

  float val= 0.0;
  //int index=0;
  val = 0.0;
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      for (int k=0; k<4; k++) {
	float *base_addr = coefs + (ix+i)*xs + (iy+j)*ys + (iz+k)*zs;
	//val += abc[(16*i+4*j+k)*BLOCK_SIZE + thr] * base_addr[offset];
	val += abcs[16*i+4*j+k] * base_addr[offset];	
	//index++;
      }
  vals[offset] = val;
}


__constant__ float A[16];

__global__ void
eval_multi_multi_UBspline_3d_cuda_c (float3 *pos, float3 drInv, 
				     float *coefs_real, float *coefs_imag,
				     float **vals_real, float **vals_imag, int3 strides)
{
  int block = blockIdx.x;
  int thr   = threadIdx.x;
  int ir    = blockIdx.y;
  int offset = block*BLOCK_SIZE+thr;
  
  float3 r = pos[ir];
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
  
  tp[0].x = 1.0;
  tp[0].y = t.x;
  tp[0].z = t.x * t.x;
  tp[0].w = t.x*t.x*t.x;

  tp[1].x = 1.0;
  tp[1].y = t.y;
  tp[1].z = t.y * t.y;
  tp[1].w = t.y*t.y*t.y;

  tp[2].x = 1.0;
  tp[2].y = t.z;
  tp[2].z = t.z*t.z;
  tp[2].w = t.z*t.z*t.z;

  float a[4], b[4], c[4];
  a[0] = A[ 0]*tp[0].x + A[ 1]*tp[0].y + A[ 2]*tp[0].z + A[ 3]*tp[0].w;
  a[1] = A[ 4]*tp[0].x + A[ 5]*tp[0].y + A[ 6]*tp[0].z + A[ 7]*tp[0].w;
  a[2] = A[ 8]*tp[0].x + A[ 9]*tp[0].y + A[10]*tp[0].z + A[11]*tp[0].w;
  a[3] = A[12]*tp[0].x + A[13]*tp[0].y + A[14]*tp[0].z + A[15]*tp[0].w;

  b[0] = A[ 0]*tp[1].x + A[ 1]*tp[1].y + A[ 2]*tp[1].z + A[ 3]*tp[1].w;
  b[1] = A[ 4]*tp[1].x + A[ 5]*tp[1].y + A[ 6]*tp[1].z + A[ 7]*tp[1].w;
  b[2] = A[ 8]*tp[1].x + A[ 9]*tp[1].y + A[10]*tp[1].z + A[11]*tp[1].w;
  b[3] = A[12]*tp[1].x + A[13]*tp[1].y + A[14]*tp[1].z + A[15]*tp[1].w;

  c[0] = A[ 0]*tp[2].x + A[ 1]*tp[2].y + A[ 2]*tp[2].z + A[ 3]*tp[2].w;
  c[1] = A[ 4]*tp[2].x + A[ 5]*tp[2].y + A[ 6]*tp[2].z + A[ 7]*tp[2].w;
  c[2] = A[ 8]*tp[2].x + A[ 9]*tp[2].y + A[10]*tp[2].z + A[11]*tp[2].w;
  c[3] = A[12]*tp[2].x + A[13]*tp[2].y + A[14]*tp[2].z + A[15]*tp[2].w;

  __shared__ float abc[64];

  int i = (thr>>4)&3;
  int j = (thr>>2)&3;
  int k = (thr & 3);
  
  abc[thr] = a[i]*b[j]*c[k];
  __syncthreads();

  float val_real = 0.0;
  float val_imag = 0.0;
  //int index=0;
  val_real = val_imag = 0.0;
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      for (int k=0; k<4; k++) {
	float *base_real = coefs_real + (index.x+i)*strides.x + (index.y+j)*strides.y + (index.z+k)*strides.z;
	float *base_imag = coefs_imag + (index.x+i)*strides.x + (index.y+j)*strides.y + (index.z+k)*strides.z;
	//val += abc[(16*i+4*j+k)*BLOCK_SIZE + thr] * base_addr[offset];
	val_real += abc[16*i+4*j+k] * base_real[offset];
	val_imag += abc[16*i+4*j+k] * base_imag[offset];
	//index++;
      }
  // vals_real[ir][offset] = val_real;
  // vals_imag[ir][offset] = val_imag;
}
				    


// __global__ void 
// eval_multi_UBspline_3d_cuda_c2 (float3 r,
// 				float *coefs, float *vals,
// 				int xs, int ys, int zs, int N)
// {
//   int block = blockIdx.x;
//   int thr   = threadIdx.x;

//   __shared__ float abcs[64];
//   abcs[thr] = abc[thr];

//   float dxInv = 0.0625f;
//   float v, dv;

//   v = floor(dxInv*r.x);
//   dv = dxInv*r.x - v;
//   int ix = (int) v;

//   v = floor(dxInv*r.x);
//   dv = dxInv*r.x - v;
//   int iy = (int) v;

//   v = floor(dxInv*r.y);
//   dv = dxInv*r.y - v;
//   int iz = (int) v;

//   int offset = block*BLOCK_SIZE+thr;
//   __shared__ float abcs[64];
//   abcs[thr] = abc[thr];
  

//   float val= 0.0;
//   //int index=0;
//   val = 0.0;
//   for (int i=0; i<4; i++)
//     for (int j=0; j<4; j++)
//       for (int k=0; k<4; k++) {
// 	float *base_addr = coefs + (ix+i)*xs + (iy+j)*ys + (iz+k)*zs;
// 	//val += abc[(16*i+4*j+k)*BLOCK_SIZE + thr] * base_addr[offset];
// 	val += abcs[16*i+4*j+k] * base_addr[offset];	
// 	//index++;
//       }
//   vals[offset] = val;
// }


void
test_cuda()
{
  float *coefs  , *abc  , *abc2, *vals;
  float *coefs_d, *abc_d, *vals_d;
  int xs, ys, zs, N;
  int Nx, Ny, Nz;

  N = 4096;
  Nx = Ny = Nz = 16;
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
  //  cudaMemcpy(abc_d, abc2, 64*BLOCK_SIZE*sizeof(float), 
  //     cudaMemcpyHostToDevice);
  cudaMemcpy(abc_d, abc, 64*sizeof(float), 
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

  float vals2[N];
  
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


  for (int i=0; i<N/256; i++)	
    fprintf (stderr, "%1.9f %1.9f\n", vals[i], vals2[i]); 


  cudaFree(abc_d);
  cudaFree(coefs_d);
  cudaFree(vals_d);
}


void
test_multi_cuda()
{
  int numWalkers = 100;
  float *coefs  , *vals[numWalkers], *vals_real[numWalkers], *vals_imag[numWalkers];
  float *coefs_d, *vals_real_d[numWalkers], *vals_imag_d[numWalkers];
  float3 r[numWalkers], *r_d;
  int xs, ys, zs, N;
  int Nx, Ny, Nz;

  N = 256;
  Nx = Ny = Nz = 16;
  xs = Nx*Ny*Nz;
  ys = Ny*Nz;
  zs = Nz;

  float3 drInv;
  drInv.x = 1.0/float(Nx);
  drInv.y = 1.0/float(Ny);
  drInv.z = 1.0/float(Nz);

  // Setup Bspline coefficients
  int size = Nx*Ny*Nz*N*sizeof(float);
  posix_memalign((void**)&coefs, 16, size);
  cudaMalloc((void**)&coefs_d, size);
  for (int ix=0; ix<Nx; ix++)
    for (int iy=0; iy<Ny; iy++)
      for (int iz=0; iz<Nz; iz++)
	for (int n=0; n<N; n++)
	  coefs[ix*xs + iy*ys + iz*zs + n] = drand48();
  cudaMemcpy(coefs_d, coefs, size, cudaMemcpyHostToDevice);

  // Setup values
  posix_memalign((void**)&vals, 16, N*sizeof(float));

  // Setup walker positions
  cudaMalloc((void**)&r_d, numWalkers*sizeof(float3));

  for (int ir=0; ir<numWalkers; ir++) {
    r[ir].x = 0.5*drand48();
    r[ir].y = 0.5*drand48();
    r[ir].z = 0.5*drand48();
  }
  cudaMemcpy(r_d, r, numWalkers*sizeof(float3), cudaMemcpyHostToDevice);

  // Setup device value storage
  int numVals = 2*N*numWalkers;
  float *valBlock_d;
  cudaMalloc((void**)&valBlock_d, numVals*sizeof(float));
  cudaMalloc((void**)&vals_real_d, numWalkers*sizeof(float*));
  cudaMalloc((void**)&vals_imag_d, numWalkers*sizeof(float*));
  for (int i=0; i<numWalkers; i++) {
    vals_real[i] = valBlock_d + 2*i*N;
    vals_imag[i] = valBlock_d + (2*i+1)*N;
  }
  cudaMemcpy(vals_real_d, vals_real, numWalkers*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(vals_imag_d, vals_imag, numWalkers*sizeof(float*), cudaMemcpyHostToDevice);



  int3 strides;
  strides.x = xs;
  strides.y = ys;
  strides.z = zs;


  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(N/BLOCK_SIZE,numWalkers);
  
  clock_t start, end;
  start = clock();
  for (int i=0; i<100000; i++) {
    eval_multi_multi_UBspline_3d_cuda_c<<<dimGrid,dimBlock>>> 
      (r_d, drInv, coefs, coefs, vals_real_d, vals_imag_d, strides);
  }
  end = clock();
  double time = (double)(end-start)/(double)((double)CLOCKS_PER_SEC*(double)100000*N*numWalkers);
  fprintf (stderr, "Evals per second = %1.8e\n", 1.0/time);

  // cudaMemcpy (vals, vals_d, N*sizeof(float), cudaMemcpyDeviceToHost);

  // float vals2[N];
  
  // for (int n=0; n<N; n++) {
  //   vals2[n] = 0.0;
  //   int index=0;
  //   for(int i=0; i<4; i++)
  //     for (int j=0; j<4; j++)
  // 	for (int k=0; k<4; k++)  {
  // 	  vals2[n] += abc[index] * coefs[(ix+i)*xs+(iy+j)*ys+(iz+k)*zs+n];
  // 	  index++;
  // 	}
  // }


  // for (int i=0; i<N/256; i++)	
  //   fprintf (stderr, "%1.9f %1.9f\n", vals[i], vals2[i]); 


  // cudaFree(abc_d);
  // cudaFree(coefs_d);
  // cudaFree(vals_d);
}




main()
{
  test_multi_cuda();
}
