#ifndef NUBSPLINE_BASH_H
#define NUBSPLINE_BASH_H

#include <stdbool.h>

typedef enum { LINEAR, GENERAL, CENTER } grid_type;

// Nonuniform grid base structure
typedef struct
{
  // public data
  grid_type code;
  double start, end;
  double* restrict points;
  int num_points;
  int (*reverse_map)(void *grid, double x);
} NUgrid;

typedef struct
{
  // public data
  grid_type code;
  double start, end;
  double* restrict points;
  int num_points;
  int (*reverse_map)(void *grid, double x);

  // private data
  double a, aInv, b, bInv, center, even_half;
  int half_points, odd_one;
  bool odd;
} center_grid;

typedef struct
{
  NUgrid* restrict grid;
  // xVals is just the grid points, augmented by two extra points on
  // either side.  These are necessary to generate enough basis
  // functions. 
  double* restrict xVals;
  // dxInv[3*i+j] = 1.0/(grid(i+j-1)-grid(i-2))
  double* restrict dxInv;
  bool periodic;
} NUBasis;

#endif
