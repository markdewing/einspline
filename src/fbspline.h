#ifndef F_BSPLINE_H
#define F_BSPLINE_H

#include "config.h"
#include "bspline_create.h"

#ifdef __cplusplus
#define CFUNC extern "C" /* Avoid name mangling in C++ */
#else
#define CFUNC
#endif

///////////////////////
// Creation routines //
///////////////////////

////////
// 1D //
////////
CFUNC void 
F77_FUNC_(fcreate_ubspline_1d_s,FCREATE_UBSPLINE_1D_S)
  (double   *x0, double    *x1, int   *num_x, 
   int *x0_code, float *x0_val, int *x1_code, float *x1_val,
   float *data, UBspline_1d_s **spline);
CFUNC void 
F77_FUNC_(fcreate_ubspline_1d_d,FCREATE_UBSPLINE_1D_D)
  (double   *x0, double     *x1, int   *num_x, 
   int *x0_code, double *x0_val, int *x1_code, double *x1_val,
   double *data, UBspline_1d_d **spline);
CFUNC void 
F77_FUNC_(fcreate_ubspline_1d_c,FCREATE_UBSPLINE_1D_C)
  (double   *x0, double    *x1, int   *num_x, 
   int *x0_code, complex_float *x0_val, int *x1_code, complex_float *x1_val,
   complex_float *data, UBspline_1d_c **spline);
CFUNC void 
F77_FUNC_(fcreate_ubspline_1d_z,FCREATE_UBSPLINE_1D_Z)
  (double   *x0, double     *x1, int   *num_x, 
   int *x0_code, complex_double *x0_val, int *x1_code, complex_double *x1_val,
   complex_double *data, UBspline_1d_z **spline);

////////
// 2D //
////////
CFUNC void 
F77_FUNC_(fcreate_ubspline_2d_s,FCREATE_UBSPLINE_2D_S)
  (double   *x0, double    *x1, int   *num_x, 
   double   *y0, double    *y1, int   *num_y, 
   int *x0_code, float *x0_val, int *x1_code, float *x1_val,
   int *y0_code, float *y0_val, int *y1_code, float *y1_val,
   float *data, UBspline_2d_s **spline);
CFUNC void 
F77_FUNC_(fcreate_ubspline_2d_d,FCREATE_UBSPLINE_2D_D)
  (double   *x0, double     *x1, int   *num_x, 
   double   *y0, double     *y1, int   *num_y, 
   int *x0_code, double *x0_val, int *x1_code, double *x1_val,
   int *y0_code, double *y0_val, int *y1_code, double *y1_val,
   double *data, UBspline_2d_d **spline);
CFUNC void 
F77_FUNC_(fcreate_ubspline_2d_c,FCREATE_UBSPLINE_2D_C)
  (double   *x0, double    *x1, int   *num_x, 
   double   *y0, double    *y1, int   *num_y, 
   int *x0_code, complex_float *x0_val, int *x1_code, complex_float *x1_val,
   int *y0_code, complex_float *y0_val, int *y1_code, complex_float *y1_val,
   complex_float *data, UBspline_2d_c **spline);
CFUNC void 
F77_FUNC_(fcreate_ubspline_2d_z,FCREATE_UBSPLINE_2D_Z)
  (double *x0, double *x1, int *num_x, 
   double *y0, double *y1, int *num_y, 
   int *x0_code, complex_double *x0_val, int *x1_code, complex_double *x1_val,
   int *y0_code, complex_double *y0_val, int *y1_code, complex_double *y1_val,
   complex_double *data, UBspline_2d_z **spline);


////////
// 3D //
////////
CFUNC void 
F77_FUNC_(fcreate_ubspline_3d_s,FCREATE_UBSPLINE_3D_S)
  (double   *x0, double    *x1, int   *num_x, 
   double   *y0, double    *y1, int   *num_y, 
   double   *z0, double    *z1, int   *num_z, 
   int *x0_code, float *x0_val, int *x1_code, float *x1_val,
   int *y0_code, float *y0_val, int *y1_code, float *y1_val,
   int *z0_code, float *z0_val, int *z1_code, float *z1_val,
   float *data, UBspline_3d_s **spline);
CFUNC void 
F77_FUNC_(fcreate_ubspline_3d_d,FCREATE_UBSPLINE_3D_D)
  (double   *x0, double     *x1, int   *num_x, 
   double   *y0, double     *y1, int   *num_y, 
   double   *z0, double     *z1, int   *num_z, 
   int *x0_code, double *x0_val, int *x1_code, double *x1_val,
   int *y0_code, double *y0_val, int *y1_code, double *y1_val,
   int *z0_code, double *z0_val, int *z1_code, double *z1_val,
   double *data, UBspline_3d_d **spline);
CFUNC void 
F77_FUNC_(fcreate_ubspline_3d_c,FCREATE_UBSPLINE_3D_C)
  (double *x0, double *x1, int *num_x, 
   double *y0, double *y1, int *num_y, 
   double *z0, double *z1, int *num_z, 
   int *x0_code, complex_float *x0_val, int *x1_code, complex_float *x1_val,
   int *y0_code, complex_float *y0_val, int *y1_code, complex_float *y1_val,
   int *z0_code, complex_float *z0_val, int *z1_code, complex_float *z1_val,
   complex_float *data, UBspline_3d_c **spline);
CFUNC void 
F77_FUNC_(fcreate_ubspline_3d_z,FCREATE_UBSPLINE_3D_Z)
  (double *x0, double *x1, int *num_x, 
   double *y0, double *y1, int *num_y, 
   double *z0, double *z1, int *num_z, 
   int *x0_code,  complex_double *x0_val, int *x1_code, complex_double *x1_val,
   int *y0_code,  complex_double *y0_val, int *y1_code, complex_double *y1_val,
   int *z0_code,  complex_double *z0_val, int *z1_code, complex_double *z1_val,
   complex_double *data, UBspline_3d_z **spline);

/////////////////////////
// Evaluation routines //
/////////////////////////
CFUNC void
F77_FUNC_(feval_ubspline_1d_s,FEVAL_UBSPLINE_1D_S)
  (UBspline_1d_s **spline, double *x, float *val);


#undef CFUNC
#endif
