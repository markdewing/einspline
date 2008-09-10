#ifndef MULTI_BSPLINE_STRUCTS_CUDA_H
#define MULTI_BSPLINE_STRUCTS_CUDA_H

#define BLOCK_SIZE 64

typedef struct
{
  float *coefs;
  uint3 stride;
  float3 gridInv;
  int num_splines;
} multi_UBspline_3d_s_cuda;

typedef struct
{
  float *coefs_real, *coefs_imag;
  uint3 stride;
  float3 gridInv;
  int num_splines;
} multi_UBspline_3d_c_cuda;

#endif
