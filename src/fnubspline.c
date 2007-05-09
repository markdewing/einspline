#include "fnubspline.h"

#include "config.h"
#include "nubspline_create.h"

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

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
////            Nonuniform spline creation routines               ////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
CFUNC void
F77_FUNC_(fcreate_nubspline_1d_s,FCREATE_NUBSPLINE_1D_S)
  (NUgrid **x_grid, 
   int* x0_code, float *x0_val, int *x1_code, float *x1_val,
   float *data, NUBspline_2d_s **spline)
{
  BCtype_s xBC;
  xBC.lCode = (bc_code) *x0_code;
  xBC.rCode = (bc_code) *x1_code;
  xBC.lVal  = *x0_val;
  xBC.rVal  = *x1_val;

  *spline = create_NUBspline_1d_s (*x_grid, xBC, data);
}


CFUNC void
F77_FUNC_(fcreate_nubspline_1d_d,FCREATE_NUBSPLINE_1D_D)
  (NUgrid **x_grid, 
   int *x0_code, double *x0_val, int *x1_code, double *x1_val,
   double *data, NUBspline_2d_d **spline)
{
  BCtype_d xBC;
  xBC.lCode = (bc_code) *x0_code;
  xBC.rCode = (bc_code) *x1_code;
  xBC.lVal  = *x0_val;
  xBC.rVal  = *x1_val;

  *spline = create_NUBspline_1d_d (*x_grid, xBC, data);
}

CFUNC void
F77_FUNC_(fcreate_nubspline_1d_c,FCREATE_NUBSPLINE_1D_C)
  (NUgrid **x_grid, 
   int *x0_code, complex_float *x0_val, 
   int *x1_code, complex_float *x1_val,
   complex_float *data, NUBspline_2d_c **spline)
{
  BCtype_c xBC;
  xBC.lCode = (bc_code) *x0_code;
  xBC.rCode = (bc_code) *x1_code;
  xBC.lVal_r  = crealf(*x0_val);
  xBC.lVal_i  = cimagf(*x0_val);
  xBC.rVal_r  = crealf(*x1_val);
  xBC.rVal_i  = cimagf(*x1_val);

  *spline = create_NUBspline_1d_c (*x_grid, xBC, data);
}

CFUNC void
F77_FUNC_(fcreate_nubspline_1d_z,FCREATE_NUBSPLINE_1D_Z)
  (NUgrid **x_grid, 
   int *x0_code, complex_double *x0_val, 
   int *x1_code, complex_double *x1_val,
   complex_double *data, NUBspline_2d_z **spline)
{
  BCtype_z xBC;
  xBC.lCode = (bc_code) *x0_code;
  xBC.rCode = (bc_code) *x1_code;
  xBC.lVal_r  = creal(*x0_val);
  xBC.lVal_i  = cimag(*x0_val);
  xBC.rVal_r  = creal(*x1_val);
  xBC.rVal_i  = cimag(*x1_val);

  *spline = create_NUBspline_1d_z (*x_grid, xBC, data);
}
