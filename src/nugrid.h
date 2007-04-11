#ifndef NUGRID_H
#define NUGRID_H

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

NUgrid*
create_general_grid (double *points, int num_points);

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

NUgrid*
create_center_grid (double start, double end, double ratio, 
		    int num_points);
#endif
