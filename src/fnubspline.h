#ifndef F_NUBSPLINE_H
#define F_NUBSPLINE_H

#include "config.h"
#include "nugrid.h"
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
  (double *points, int *num_points, NUgrid **grid);

CFUNC void 
F77_FUNC_(fcreate_center_grid,FCREATE_CENTER_GRID)
  (double *start, double *end, double *ratio,
   int *num_points, NUgrid **grid);

CFUNC void
F77_FUNC_(fdestroy_grid,FDESTROY_GRID)
  (NUgrid **grid);


#endif
