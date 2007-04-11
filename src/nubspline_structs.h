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
  int coefs_size;
  NUgrid *x_grid;
  NUBasis *x_basis;
  BCtype_s xBC;
} NUBspline_1d_s;

///////////////////////////
// Double precision real //
///////////////////////////
typedef struct
{
  spline_code code;
  double* restrict coefs;
  int coefs_size;
  NUgrid *x_grid;
  NUBasis *x_basis;
  BCtype_d xBC;
} NUBspline_1d_d;

//////////////////////////////
// Single precision complex //
//////////////////////////////
typedef struct
{
  spline_code code;
  complex_float* restrict coefs;
  int coefs_size;
  NUgrid *x_grid;
  NUBasis *x_basis;
  BCtype_c xBC;
} NUBspline_1d_c;

//////////////////////////////
// Double precision complex //
//////////////////////////////
typedef struct
{
  spline_code code;
  complex_double* restrict coefs;
  int coefs_size;
  NUgrid *x_grid;
  NUBasis *x_basis;
  BCtype_z xBC;
} NUBspline_1d_z;

#endif
