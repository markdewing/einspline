#include "multi_bspline.h"
#include "multi_bspline_create_cuda.h"
#include "multi_bspline_structs_cuda.h"
#include "multi_bspline_eval_cuda.h"


void
test_float()
{
  int numWalkers = 1000;
  float *vals[numWalkers], *grads[numWalkers], *hess[numWalkers];
  float *coefs, __device__ **vals_d, **grads_d, **hess_d;
  float *r_d, *r_h;
  int xs, ys, zs, N;
  int Nx, Ny, Nz;

  N = 128;
  Nx = Ny = Nz = 32;
  xs = Ny*Nz*N;
  ys = Nz*N;
  zs = N;

  // Setup Bspline coefficients
  int size = Nx*Ny*Nz*N*sizeof(float);
  posix_memalign((void**)&coefs, 16, size);
  for (int ix=0; ix<Nx; ix++)
    for (int iy=0; iy<Ny; iy++)
      for (int iz=0; iz<Nz; iz++)
	for (int n=0; n<N; n++)
	  coefs[ix*xs + iy*ys + iz*zs + n] = drand48();

  Ugrid x_grid, y_grid, z_grid;
  x_grid.start = 0.0; x_grid.end = 1.0; x_grid.num = Nx;
  y_grid.start = 0.0; y_grid.end = 1.0; y_grid.num = Ny;
  z_grid.start = 0.0; z_grid.end = 1.0; z_grid.num = Nz;
  BCtype_s xBC, yBC, zBC;
  xBC.lCode = xBC.rCode = PERIODIC;
  yBC.lCode = yBC.rCode = PERIODIC;
  zBC.lCode = zBC.rCode = PERIODIC;
  

  multi_UBspline_3d_s *spline = 
    create_multi_UBspline_3d_s (x_grid, y_grid, z_grid, xBC, yBC, zBC, N);
  for (int i=0; i<N; i++) 
    set_multi_UBspline_3d_s (spline, i, coefs);

  multi_UBspline_3d_s_cuda *cudaspline = 
    create_multi_UBspline_3d_s_cuda (spline);

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
  cudaMalloc((void**)&(r_d),     3*numWalkers*sizeof(float));
  cudaMallocHost((void**)&(r_h), 3*numWalkers*sizeof(float));

  for (int ir=0; ir<numWalkers; ir++) {
    r_h[3*ir+0] = 0.5*drand48();
    r_h[3*ir+1] = 0.5*drand48();
    r_h[3*ir+2] = 0.5*drand48();
  }

  dim3 dimBlock(SPLINE_BLOCK_SIZE);
  dim3 dimGrid(N/SPLINE_BLOCK_SIZE,numWalkers);
  
  float vals_host[N], vals_cuda[N];

  // Check value
  for (int w=0; w<numWalkers; w++) {
    eval_multi_UBspline_3d_s (spline, r_h[3*w+0], r_h[3*w+1], r_h[3*w+2], vals_host);
    cudaMemcpy(r_d, r_h, 3*numWalkers*sizeof(float), cudaMemcpyHostToDevice);
    eval_multi_multi_UBspline_3d_s_cuda (cudaspline, r_d, vals_d, numWalkers);
    cudaMemcpy(vals_cuda, valBlock_d+(N*w), N*sizeof(float), cudaMemcpyDeviceToHost);
    //for (int i=0; i<N; i++)
      fprintf (stderr, "%3i  %15.8e %15.8e\n", w, vals_host[0], vals_cuda[0]);
  }


  clock_t start, end;
  start = clock();
  for (int i=0; i<10000; i++) {
    if ((i%1000) == 0) 
      fprintf (stderr, "i = %d\n", i);
    cudaMemcpy(r_d, r_h, 3*numWalkers*sizeof(float), cudaMemcpyHostToDevice);
    eval_multi_multi_UBspline_3d_s_cuda (cudaspline, r_d, vals_d, numWalkers);
  }
  end = clock();
  double time = (double)(end-start)/(double)((double)CLOCKS_PER_SEC*(double)10000*N*numWalkers);
  fprintf (stderr, "Evals per second = %1.8e\n", 1.0/time);

  start = clock();
  for (int i=0; i<10000; i++) {
    if ((i%1000) == 0) 
      fprintf (stderr, "i = %d\n", i);
    cudaMemcpy(r_d, r_h, 3*numWalkers*sizeof(float), cudaMemcpyHostToDevice);
    eval_multi_multi_UBspline_3d_s_vgh_cuda (cudaspline, r_d, vals_d, grads_d, hess_d, numWalkers);
  }
  end = clock();
  time = (double)(end-start)/(double)((double)CLOCKS_PER_SEC*(double)10000*N*numWalkers);
  fprintf (stderr, "VGH Evals per second = %1.8e\n", 1.0/time);
  
  cudaFree (spline->coefs);
  cudaFree (valBlock_d);
  cudaFree (vals_d);
  cudaFree (grads_d);
  cudaFree (hess_d);
  cudaFree (r_d);
}



void
test_double()
{
  int numWalkers = 1000;
  double *vals[numWalkers], *grads[numWalkers], *hess[numWalkers];
  double *coefs, __device__ **vals_d, **grads_d, **hess_d;
  double *r_d, *r_h;
  int xs, ys, zs, N;
  int Nx, Ny, Nz;

  N = 128;
  Nx = Ny = Nz = 32;
  xs = Ny*Nz*N;
  ys = Nz*N;
  zs = N;

  // Setup Bspline coefficients
  int size = Nx*Ny*Nz*N*sizeof(double);
  posix_memalign((void**)&coefs, 16, size);
  for (int ix=0; ix<Nx; ix++)
    for (int iy=0; iy<Ny; iy++)
      for (int iz=0; iz<Nz; iz++)
	for (int n=0; n<N; n++)
	  coefs[ix*xs + iy*ys + iz*zs + n] = drand48();

  Ugrid x_grid, y_grid, z_grid;
  x_grid.start = 0.0; x_grid.end = 1.0; x_grid.num = Nx;
  y_grid.start = 0.0; y_grid.end = 1.0; y_grid.num = Ny;
  z_grid.start = 0.0; z_grid.end = 1.0; z_grid.num = Nz;
  BCtype_d xBC, yBC, zBC;
  xBC.lCode = xBC.rCode = PERIODIC;
  yBC.lCode = yBC.rCode = PERIODIC;
  zBC.lCode = zBC.rCode = PERIODIC;
  

  multi_UBspline_3d_d *spline = 
    create_multi_UBspline_3d_d (x_grid, y_grid, z_grid, xBC, yBC, zBC, N);
  for (int i=0; i<N; i++) 
    set_multi_UBspline_3d_d (spline, i, coefs);

  multi_UBspline_3d_d_cuda *cudaspline = 
    create_multi_UBspline_3d_d_cuda (spline);

  // Setup device value storage
  int numVals = N*numWalkers*10;
  double *valBlock_d, *valBlock_h;
  cudaMalloc((void**)&(valBlock_d),     numVals*sizeof(double));
  cudaMallocHost((void**)&(valBlock_h), numVals*sizeof(double));
  cudaMalloc((void**)&(vals_d),  numWalkers*sizeof(double*));
  cudaMalloc((void**)&(grads_d), numWalkers*sizeof(double*));
  cudaMalloc((void**)&(hess_d),  numWalkers*sizeof(double*));
  fprintf (stderr, "valBlock_d = %p\n", valBlock_d);
  for (int i=0; i<numWalkers; i++) {
    vals[i]  = valBlock_d + i*N;
    grads[i] = valBlock_d + N*numWalkers + 3*i*N;
    hess[i]  = valBlock_d + 4*N*numWalkers + 6*i*N;
  }
  cudaMemcpy(vals_d,  vals,  numWalkers*sizeof(double*), cudaMemcpyHostToDevice);
  cudaMemcpy(grads_d, grads, numWalkers*sizeof(double*), cudaMemcpyHostToDevice);
  cudaMemcpy(hess_d,  hess,  numWalkers*sizeof(double*), cudaMemcpyHostToDevice);
  fprintf (stderr, "Finished cuda allocations.\n");

  // Setup walker positions
  cudaMalloc((void**)&(r_d),     3*numWalkers*sizeof(double));
  cudaMallocHost((void**)&(r_h), 3*numWalkers*sizeof(double));

  for (int ir=0; ir<numWalkers; ir++) {
    r_h[3*ir+0] = 0.5*drand48();
    r_h[3*ir+1] = 0.5*drand48();
    r_h[3*ir+2] = 0.5*drand48();
  }

  dim3 dimBlock(SPLINE_BLOCK_SIZE);
  dim3 dimGrid(N/SPLINE_BLOCK_SIZE,numWalkers);
  
  double vals_host[N], vals_cuda[N];

  // Check value
  for (int w=0; w<numWalkers; w++) {
    eval_multi_UBspline_3d_d (spline, r_h[3*w+0], r_h[3*w+1], r_h[3*w+2], vals_host);
    cudaMemcpy(r_d, r_h, 3*numWalkers*sizeof(double), cudaMemcpyHostToDevice);
    eval_multi_multi_UBspline_3d_d_cuda (cudaspline, r_d, vals_d, numWalkers);
    cudaMemcpy(vals_cuda, valBlock_d+(N*w), N*sizeof(double), cudaMemcpyDeviceToHost);
    //for (int i=0; i<N; i++)
      fprintf (stderr, "%3i  %15.8e %15.8e\n", w, vals_host[0], vals_cuda[0]);
  }


  clock_t start, end;
  start = clock();
  for (int i=0; i<10000; i++) {
    if ((i%1000) == 0) 
      fprintf (stderr, "i = %d\n", i);
    cudaMemcpy(r_d, r_h, 3*numWalkers*sizeof(double), cudaMemcpyHostToDevice);
    eval_multi_multi_UBspline_3d_d_cuda (cudaspline, r_d, vals_d, numWalkers);
  }
  end = clock();
  double time = (double)(end-start)/(double)((double)CLOCKS_PER_SEC*(double)10000*N*numWalkers);
  fprintf (stderr, "Evals per second = %1.8e\n", 1.0/time);

  start = clock();
  for (int i=0; i<10000; i++) {
    if ((i%1000) == 0) 
      fprintf (stderr, "i = %d\n", i);
    cudaMemcpy(r_d, r_h, 3*numWalkers*sizeof(double), cudaMemcpyHostToDevice);
    eval_multi_multi_UBspline_3d_d_vgh_cuda (cudaspline, r_d, vals_d, grads_d, hess_d, numWalkers);
  }
  end = clock();
  time = (double)(end-start)/(double)((double)CLOCKS_PER_SEC*(double)10000*N*numWalkers);
  fprintf (stderr, "VGH Evals per second = %1.8e\n", 1.0/time);
  
  cudaFree (spline->coefs);
  cudaFree (valBlock_d);
  cudaFree (vals_d);
  cudaFree (grads_d);
  cudaFree (hess_d);
  cudaFree (r_d);
}


main()
{
  fprintf(stderr, "Testing single-precision routines:\n");
  test_float();
  fprintf(stderr, "Testing double-precision routines:\n");
  test_double();
}
