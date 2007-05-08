#ifndef F_BSPLINE_H
#define F_BSPLINE_H

#include "config.h"
#include "bspline_create.h"

#ifdef __cplusplus
#define CFUNC extern "C" /* Avoid name mangling in C++ */
#else
#define CFUNC
#endif

// Creation routines
CFUNC void 
F77_FUNC(fcreate_ubspline_1d_s,FCREATE_UBSPLINE_1D_S)
  (double   *x0, double    *x1, int   *num_x, 
   int *x0_code, float *x0_val, int *x1_code, float *x1_val,
   float *data, UBspline_1d_s **spline);


#undef CFUNC
#endif
