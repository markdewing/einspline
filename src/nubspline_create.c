#include "nubspline_create.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>

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

NUgrid*
create_center_grid (double start, double end, double ratio, 
		    int num_points)
{
  center_grid *grid = malloc (sizeof (center_grid));
  if (grid != NULL) {
    assert (ratio > 1.0);
    grid->start       = start;
    grid->end         = end;
    grid->center      = 0.5*(start + end);
    grid->num_points  = num_points;
    grid->half_points = num_points/2;
    grid->odd = ((num_points % 2) == 1);
    grid->b = log(ratio) / (double)(grid->half_points-1);
    grid->bInv = 1.0/grid->b;
    grid->points = malloc (num_points * sizeof(double));
    if (grid->odd) {
      grid->even_half = 0.0;  
      grid->odd_one   = 1;
      grid->a = 0.5*(end-start)/expm1(grid->b*grid->half_points);
      grid->aInv = 1.0/grid->a;
      for (int i=-grid->half_points; i<=grid->half_points; i++) {
	double sign;
	if (i<0) 
	  sign = -1.0;
	else
	  sign =  1.0;
	grid->points[i+grid->half_points] = 
	  sign * grid->a*expm1(grid->b*abs(i))+grid->center;
      }
    }
    else {
      grid->even_half = 0.5;  
      grid->odd_one   = 0;
      grid->a = 
	0.5*(end-start)/expm1(grid->b*(-0.5+grid->half_points));
      grid->aInv = 1.0/grid->a;
      for (int i=-grid->half_points; i<grid->half_points; i++) {
	double sign;
	if (i<0) sign = -1.0; 
	else     sign =  1.0;
	grid->points[i+grid->half_points] = 
	  sign * grid->a*expm1(grid->b*fabs(0.5+i)) + grid->center;
      }
    }
    grid->reverse_map = center_grid_reverse_map;
    grid->code = CENTER;
  }
  return (NUgrid*) grid;
}


NUgrid*
create_general_grid (double *points, int num_points)
{
  NUgrid* grid = malloc (sizeof(NUgrid));
  if (grid != NULL) {
    grid->reverse_map = general_grid_reverse_map;
    grid->code = GENERAL;
    grid->points = malloc (num_points*sizeof(double));
    grid->start = points[0];
    grid->end   = points[num_points-1];
    grid->num_points = num_points;
    for (int i=0; i<num_points; i++) 
      grid->points[i] = points[i];
  }
  return grid;
}


NUBasis*
create_NUBasis (NUgrid *grid, bool periodic)
{
  NUBasis* restrict basis = malloc (sizeof(NUBasis));
  basis->grid = grid;
  basis->periodic = periodic;
  int N = grid->num_points;
  basis->xVals = malloc ((N+2)*sizeof(double));
  basis->dxInv = malloc (3*(N+4)*sizeof(double));
  for (int i=0; i<N; i++)
    basis->xVals[i+2] = grid->points[i];
  double*  restrict g = grid->points;
  // Extend grid points on either end to provide enough points to
  // construct a full basis set
  if (!periodic) {
    basis->xVals[0]   = g[ 0 ] - 2.0*(g[1]-g[0]);
    basis->xVals[1]   = g[ 0 ] - 1.0*(g[1]-g[0]);
    basis->xVals[N+2] = g[N-1] + 1.0*(g[N-1]-g[N-2]);
    basis->xVals[N+3] = g[N-1] + 2.0*(g[N-1]-g[N-2]);
  }
  else {
    basis->xVals[1]   = g[ 0 ] - (g[N-1] - g[N-2]);
    basis->xVals[0]   = g[ 0 ] - (g[N-1] - g[N-3]);
    basis->xVals[N+2] = g[N-1] + (g[ 1 ] - g[ 0 ]);
    basis->xVals[N+3] = g[N-1] + (g[ 2 ] - g[ 0 ]);
  }
  for (int i=0; i<N+2; i++) 
    for (int j=0; j<3; j++) 
      basis->dxInv[3*i+j] = 1.0/(xVals(i+j+1)-xVals(i));
  return basis;
}
