#include "nubspline_create.h"
#include <math.h>

int
center_grid_reverse_map (void* gridptr, double x)
{
  center_grid *grid = (center_grid *)gridptr;

  x -= grid->center;
  double index = 
    copysign (log1p(fabs(x)*grid->aInv)*grid->bInv, x);
  return (int)floor(grid->half_points + index - grid->even_half);
}


int
general_grid_reverse_map (void* gridptr, double x)
{
  NUgrid* grid = (NUgrid*) gridptr;
  int N = grid->num_points;
  double *points = grid->points;
  if (x <= points[0])
    return (0);
  else if (x >= points[N-1])
    return (N-1);
  else {
    int hi = N-1;
    int lo = 0;
    bool done = false;
    while (!done) {
      int i = (hi+lo)>>1;
      if (points[i] > x)
	hi = i;
      else
	lo = i;
      done = (hi-lo)<2;
    }
    return (lo);
  }
}

