#ifndef NUBSPLINE_STRUCTS_H
#define NUBSPLINE_STRUCTS_H

#include "bspline_base.h"
#include "nubasis.h"

#ifdef __cplusplus
typedef complex<float>  complex_float;
typedef complex<double> complex_double;
#else
#include <complex.h>
typedef complex float  complex_float;
typedef complex double complex_double;
#endif

///////////////////////////
// Single precision real //
///////////////////////////
typedef struct
{
  spline_code code;
  float* restrict coefs;
  NUgrid *restrict  x_grid;
  NUBasis *restrict x_basis;
  BCtype_s xBC;
} NUBspline_1d_s;

typedef struct
{
  spline_code code;
  float* restrict coefs;
  int x_stride;
  NUgrid  *restrict x_grid,  *restrict y_grid;
  NUBasis *restrict x_basis, *restrict y_basis;
  BCtype_s xBC, yBC;
} NUBspline_2d_s;

typedef struct
{
  spline_code code;
  float* restrict coefs;
  int x_stride, y_stride;
  NUgrid  *restrict x_grid,  *restrict y_grid,  *restrict z_grid;
  NUBasis *restrict x_basis, *restrict y_basis, *restrict z_basis;
  BCtype_s xBC, yBC, zBC;
} NUBspline_3d_s;

///////////////////////////
// Double precision real //
///////////////////////////
typedef struct
{
  spline_code code;
  double* restrict coefs;
  NUgrid* restrict x_grid;
  NUBasis* restrict x_basis;
  BCtype_d xBC;
} NUBspline_1d_d;

typedef struct
{
  spline_code code;
  double* restrict coefs;
  int x_stride;
  NUgrid * restrict x_grid, * restrict y_grid;
  NUBasis * restrict x_basis, * restrict y_basis;
  BCtype_d xBC, yBC;
} NUBspline_2d_d;

typedef struct
{
  spline_code code;
  double* restrict coefs;
  int x_stride, y_stride;
  NUgrid  *restrict x_grid,  *restrict y_grid,  *restrict z_grid;
  NUBasis *restrict x_basis, *restrict y_basis, *restrict z_basis;
  BCtype_d xBC, yBC, zBC;
} NUBspline_3d_d;

//////////////////////////////
// Single precision complex //
//////////////////////////////
typedef struct
{
  spline_code code;
  complex_float* restrict coefs;
  NUgrid* restrict x_grid;
  NUBasis* restrict x_basis;
  BCtype_c xBC;
} NUBspline_1d_c;

typedef struct
{
  spline_code code;
  complex_float* restrict coefs;
  int x_stride;
  NUgrid* restrict x_grid, *restrict y_grid;
  NUBasis* restrict x_basis, *restrict y_basis;
  BCtype_c xBC, yBC;
} NUBspline_2d_c;

typedef struct
{
  spline_code code;
  complex_float* restrict coefs;
  int x_stride, y_stride;
  NUgrid  *restrict x_grid,  *restrict y_grid,  *restrict z_grid;
  NUBasis *restrict x_basis, *restrict y_basis, *restrict z_basis;
  BCtype_c xBC, yBC, zBC;
} NUBspline_3d_c;

//////////////////////////////
// Double precision complex //
//////////////////////////////
typedef struct
{
  spline_code code;
  complex_double* restrict coefs;
  NUgrid  *restrict x_grid;
  NUBasis *restrict x_basis;
  BCtype_z xBC;
} NUBspline_1d_z;

typedef struct
{
  spline_code code;
  complex_double* restrict coefs;
  int x_stride;
  NUgrid  *restrict x_grid,  *restrict y_grid;
  NUBasis *restrict x_basis, *restrict y_basis;
  BCtype_z xBC, yBC;
} NUBspline_2d_z;

typedef struct
{
  spline_code code;
  complex_double* restrict coefs;
  int x_stride, y_stride;
  NUgrid  *restrict x_grid,  *restrict y_grid,  *restrict z_grid;
  NUBasis *restrict x_basis, *restrict y_basis, *restrict z_basis;
  BCtype_z xBC, yBC, zBC;
} NUBspline_3d_z;

#endif
