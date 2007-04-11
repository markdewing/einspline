#ifndef NUBASIS_H
#define NUBASIS_H

#include "nugrid.h"
#include <stdbool.h>

typedef struct
{
  NUgrid* restrict grid;
  // xVals is just the grid points, augmented by two extra points on
  // either side.  These are necessary to generate enough basis
  // functions. 
  double* restrict xVals;
  // dxInv[3*i+j] = 1.0/(grid(i+j-1)-grid(i-2))
  double* restrict dxInv;
  bool periodic;
} NUBasis;

/////////////////
// Constructor //
/////////////////
NUBasis*
create_NUBasis (NUgrid *grid, bool periodic);

////////////////
// Destructor //
////////////////
void
destroy_NUBasis (NUBasis *basis);


////////////////////////////////////////////////
// Single-precision basis function evaluation //
////////////////////////////////////////////////
int
get_NUBasis_funcs_s (NUBasis* restrict basis, double x,
		     float bfuncs[4]);
void
get_NUBasis_funcs_si (NUBasis* restrict basis, int i,
		      float bfuncs[4]);

int
get_NUBasis_dfuncs_s (NUBasis* restrict basis, double x,
		      float bfuncs[4], float dbfuncs[4]);
void
get_NUBasis_dfuncs_si (NUBasis* restrict basis, int i,
		       float bfuncs[4], float dbfuncs[4]);

int
get_NUBasis_d2funcs_s (NUBasis* restrict basis, double x,
		       float bfuncs[4], float dbfuncs[4], float d2bfuncs[4]);
void
get_NUBasis_d2funcs_si (NUBasis* restrict basis, int i,
			float bfuncs[4], float dbfuncs[4], float d2bfuncs[4]);

////////////////////////////////////////////////
// Double-precision basis function evaluation //
////////////////////////////////////////////////
int
get_NUBasis_funcs_d (NUBasis* restrict basis, double x,
		     double bfuncs[4]);
void
get_NUBasis_funcs_di (NUBasis* restrict basis, int i,
		      double bfuncs[4]);
int
get_NUBasis_dfuncs_d (NUBasis* restrict basis, double x,
		      double bfuncs[4], double dbfuncs[4]);
void
get_NUBasis_dfuncs_di (NUBasis* restrict basis, int i,
		       double bfuncs[4], double dbfuncs[4]);
int
get_NUBasis_d2funcs_d (NUBasis* restrict basis, double x,
		       double bfuncs[4], double dbfuncs[4], 
		       double d2bfuncs[4]);
void
get_NUBasis_d2funcs_di (NUBasis* restrict basis, int i,
			double bfuncs[4], double dbfuncs[4], 
			double d2bfuncs[4]);

#endif
