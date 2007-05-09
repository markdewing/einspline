#include "bspline_create.h"
#include "bspline.h"
#include "fbspline.h"
#include "config.h"

#ifdef __cplusplus
#define CFUNC "C" /* Avoid name mangling in C++ */
#else
#define CFUNC
#endif




CFUNC void
F77_FUNC_(fcreate_ubspline_1d_s,FCREATE_UBSPLINE_1D_S)
  (double *x0,   double    *x1, int   *num_x, 
   int *x0_code, float *x0_val, int *x1_code, float *x1_val,
   float *data,  UBspline_1d_s **spline)
{
  Ugrid xgrid;
  BCtype_s xBC;
  xgrid.start = *x0;
  xgrid.end   = *x1;
  xgrid.num   = *num_x;
 
  xBC.lCode = (bc_code) *x0_code;
  xBC.rCode = (bc_code) *x1_code;
  xBC.lVal  = *x0_val;
  xBC.rVal  = *x1_val;

  *spline = create_UBspline_1d_s (xgrid, xBC, data);
}


CFUNC void
F77_FUNC_(feval_ubspline_1d_s,FEVAL_UBSPLINE_1D_S)
  (UBspline_1d_s **spline, double *x, float *val)
{
  eval_UBspline_1d_s (*spline, *x, val);
}
