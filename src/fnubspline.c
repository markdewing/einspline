#include "fnubspline.h"

#include "config.h"

#ifdef __cplusplus
#define CFUNC extern "C" /* Avoid name mangling in C++ */
#else
#define CFUNC
#endif

CFUNC void 
F77_FUNC_(fcreate_general_grid,FCREATE_GENERAL_GRID)
  (double *points, int *num_points, NUgrid **grid)
{
  *grid = create_general_grid (points, *num_points);
}

CFUNC void 
F77_FUNC_(fcreate_center_grid,FCREATE_CENTER_GRID)
  (double *start, double *end, double *ratio,
   int *num_points, NUgrid **grid)
{
  *grid = create_center_grid (*start, *end, *ratio, *num_points);
}

CFUNC void
F77_FUNC_(fdestroy_grid,FDESTROY_GRID)
  (NUgrid **grid)
{
  destroy_grid (*grid);
}
