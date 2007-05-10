#ifndef F_NUBSPLINE_H
#define F_NUBSPLINE_H

#include "config.h"
#include "nugrid.h"
#include "nubspline_structs.h"

#ifdef __cplusplus
#define CFUNC extern "C" /* Avoid name mangling in C++ */
#else
#define CFUNC
#endif

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
////                    Grid Creation routines                    ////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

CFUNC void 
F77_FUNC_(fcreate_general_grid,FCREATE_GENERAL_GRID)
  (double *points, int *num_points, NUgrid **grid);

CFUNC void 
F77_FUNC_(fcreate_center_grid,FCREATE_CENTER_GRID)
  (double *start, double *end, double *ratio,
   int *num_points, NUgrid **grid);

CFUNC void
F77_FUNC_(fdestroy_grid,FDESTROY_GRID)
  (NUgrid **grid);

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
////            Nonuniform spline creation routines               ////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


////////
// 1D //
////////
CFUNC void
F77_FUNC_(fcreate_nubspline_1d_s,FCREATE_NUBSPLINE_1D_S)
  (NUgrid **x_grid, 
   int* x0_code, float *x0_val, int *x1_code, float *x1_val,
   float *data, NUBspline_1d_s **spline);

CFUNC void
F77_FUNC_(fcreate_nubspline_1d_d,FCREATE_NUBSPLINE_1D_D)
  (NUgrid **x_grid, 
   int *x0_code, double *x0_val, int *x1_code, double *x1_val,
   double *data, NUBspline_1d_d **spline);

CFUNC void
F77_FUNC_(fcreate_nubspline_1d_c,FCREATE_NUBSPLINE_1D_C)
  (NUgrid **x_grid, 
   int *x0_code, complex_float *x0_val, 
   int *x1_code, complex_float *x1_val,
   complex_float *data, NUBspline_1d_c **spline);

CFUNC void
F77_FUNC_(fcreate_nubspline_1d_z,FCREATE_NUBSPLINE_1D_Z)
  (NUgrid **x_grid, 
   int *x0_code, complex_double *x0_val, 
   int *x1_code, complex_double *x1_val,
   complex_double *data, NUBspline_1d_z **spline);

////////
// 2D //
////////
CFUNC void
F77_FUNC_(fcreate_nubspline_2d_s,FCREATE_NUBSPLINE_2D_S)
  (NUgrid **x_grid, NUgrid **y_grid, 
   int* x0_code, float *x0_val, int *x1_code, float *x1_val,
   int* y0_code, float *y0_val, int *y1_code, float *y1_val,
   float *data, NUBspline_2d_s **spline);

CFUNC void
F77_FUNC_(fcreate_nubspline_2d_d,FCREATE_NUBSPLINE_2D_D)
  (NUgrid **x_grid, NUgrid **y_grid,
   int *x0_code, double *x0_val, int *x1_code, double *x1_val,
   int *y0_code, double *y0_val, int *y1_code, double *y1_val,
   double *data, NUBspline_2d_d **spline);

CFUNC void
F77_FUNC_(fcreate_nubspline_2d_c,FCREATE_NUBSPLINE_2D_C)
  (NUgrid **x_grid, NUgrid **y_grid,
   int *x0_code, complex_float *x0_val, 
   int *x1_code, complex_float *x1_val,
   int *y0_code, complex_float *y0_val, 
   int *y1_code, complex_float *y1_val,
   complex_float *data, NUBspline_2d_c **spline);

CFUNC void
F77_FUNC_(fcreate_nubspline_2d_z,FCREATE_NUBSPLINE_2D_Z)
  (NUgrid **x_grid, NUgrid **y_grid,
   int *x0_code, complex_double *x0_val, 
   int *x1_code, complex_double *x1_val,
   int *y0_code, complex_double *y0_val, 
   int *y1_code, complex_double *y1_val,
   complex_double *data, NUBspline_2d_z **spline);

////////
// 3D //
////////
CFUNC void
F77_FUNC_(fcreate_nubspline_3d_s,FCREATE_NUBSPLINE_3D_S)
  (NUgrid **x_grid, NUgrid **y_grid, NUgrid **z_grid,
   int* x0_code, float *x0_val, int *x1_code, float *x1_val,
   int* y0_code, float *y0_val, int *y1_code, float *y1_val,
   int* z0_code, float *z0_val, int *z1_code, float *z1_val,
   float *data, NUBspline_3d_s **spline);

CFUNC void
F77_FUNC_(fcreate_nubspline_3d_d,FCREATE_NUBSPLINE_3D_D)
  (NUgrid **x_grid, NUgrid **y_grid, NUgrid **z_grid,
   int *x0_code, double *x0_val, int *x1_code, double *x1_val,
   int *y0_code, double *y0_val, int *y1_code, double *y1_val,
   int* z0_code, float *z0_val, int *z1_code, float *z1_val,
   double *data, NUBspline_3d_d **spline);

CFUNC void
F77_FUNC_(fcreate_nubspline_3d_c,FCREATE_NUBSPLINE_3D_C)
  (NUgrid **x_grid, NUgrid **y_grid, NUgrid **z_grid,
   int *x0_code, complex_float *x0_val, 
   int *x1_code, complex_float *x1_val,
   int *y0_code, complex_float *y0_val, 
   int *y1_code, complex_float *y1_val,
   int *z0_code, complex_float *z0_val, 
   int *z1_code, complex_float *z1_val,
   complex_float *data, NUBspline_3d_c **spline);

CFUNC void
F77_FUNC_(fcreate_nubspline_3d_z,FCREATE_NUBSPLINE_3D_Z)
  (NUgrid **x_grid, NUgrid **y_grid, NUgrid **z_grid,
   int *x0_code, complex_double *x0_val, 
   int *x1_code, complex_double *x1_val,
   int *y0_code, complex_double *y0_val, 
   int *y1_code, complex_double *y1_val,
   int *z0_code, complex_float *z0_val, 
   int *z1_code, complex_float *z1_val,
   complex_double *data, NUBspline_3d_z **spline);

#endif
