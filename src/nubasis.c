#include "nubasis.h"

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
      basis->dxInv[3*i+j] = 
	1.0/(basis->xVals[i+j+1]-basis->xVals[i]);
  return basis;
}


int
get_NUBasis_funcs_s (NUBasis* restrict basis, double x,
		     float bfuncs[4])
{
  double b1[2], b2[3];
  int i = (*basis->grid->reverse_map)(basis->grid, x);
  int i2 = i+2;
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals;

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2];
  return i;
}


void
get_NUBasis_funcs_si (NUBasis* restrict basis, int i,
		     float bfuncs[4])
{
  int i2 = i+2;
  double b1[2], b2[3];
  double x = basis->grid->points[i];
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals; 

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2];
}

int
get_NUBasis_dfuncs_s (NUBasis* restrict basis, double x,
		      float bfuncs[4], float dbfuncs[4])
{
  double b1[2], b2[3];
  int i = (*basis->grid->reverse_map)(basis->grid, x);
  int i2 = i+2;
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals;

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2]; 

  dbfuncs[0] = -3.0 * (dxInv[3*(i  )+2] * b2[0]);
  dbfuncs[1] =  3.0 * (dxInv[3*(i  )+2] * b2[0] - dxInv[3*(i+1)+2] * b2[1]);
  dbfuncs[2] =  3.0 * (dxInv[3*(i+1)+2] * b2[1] - dxInv[3*(i+2)+2] * b2[2]);
  dbfuncs[3] =  3.0 * (dxInv[3*(i+2)+2] * b2[2]);

  return i;
}


void
get_NUBasis_dfuncs_si (NUBasis* restrict basis, int i,
		       float bfuncs[4], float dbfuncs[4])
{
  double b1[2], b2[3];
  double x = basis->grid->points[i];
  int i2 = i+2;
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals;

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2]; 

  dbfuncs[0] = -3.0 * (dxInv[3*(i  )+2] * b2[0]);
  dbfuncs[1] =  3.0 * (dxInv[3*(i  )+2] * b2[0] - dxInv[3*(i+1)+2] * b2[1]);
  dbfuncs[2] =  3.0 * (dxInv[3*(i+1)+2] * b2[1] - dxInv[3*(i+2)+2] * b2[2]);
  dbfuncs[3] =  3.0 * (dxInv[3*(i+2)+2] * b2[2]);
}


int
get_NUBasis_d2funcs_s (NUBasis* restrict basis, double x,
		       float bfuncs[4], float dbfuncs[4], float d2bfuncs[4])
{
  double b1[2], b2[3];
  int i = (*basis->grid->reverse_map)(basis->grid, x);
  int i2 = i+2;
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals;

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2]; 

  dbfuncs[0] = -3.0 * (dxInv[3*(i  )+2] * b2[0]);
  dbfuncs[1] =  3.0 * (dxInv[3*(i  )+2] * b2[0] - dxInv[3*(i+1)+2] * b2[1]);
  dbfuncs[2] =  3.0 * (dxInv[3*(i+1)+2] * b2[1] - dxInv[3*(i+2)+2] * b2[2]);
  dbfuncs[3] =  3.0 * (dxInv[3*(i+2)+2] * b2[2]);

  d2bfuncs[0] = 6.0 * (+dxInv[3*(i+0)+2]* dxInv[3*(i+1)+1]*b1[0]);
  d2bfuncs[1] = 6.0 * (-dxInv[3*(i+1)+1]*(dxInv[3*(i+0)+2]+dxInv[3*(i+1)+2])*b1[0] +
		        dxInv[3*(i+1)+2]* dxInv[3*(i+2)+1]*b1[1]);
  d2bfuncs[2] = 6.0 * (+dxInv[3*(i+1)+2]* dxInv[3*(i+1)+1]*b1[0] -
		        dxInv[3*(i+2)+1]*(dxInv[3*(i+1)+2] + dxInv[3*(i+2)+2])*b1[1]);
  d2bfuncs[3] = 6.0 * (+dxInv[3*(i+2)+2]* dxInv[3*(i+2)+1]*b1[1]);

  return i;
}


void
get_NUBasis_d2funcs_si (NUBasis* restrict basis, int i,
			float bfuncs[4], float dbfuncs[4], float d2bfuncs[4])
{
  double b1[2], b2[3];
  double x = basis->grid->points[i];
  int i2 = i+2;
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals;

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2]; 

  dbfuncs[0] = -3.0 * (dxInv[3*(i  )+2] * b2[0]);
  dbfuncs[1] =  3.0 * (dxInv[3*(i  )+2] * b2[0] - dxInv[3*(i+1)+2] * b2[1]);
  dbfuncs[2] =  3.0 * (dxInv[3*(i+1)+2] * b2[1] - dxInv[3*(i+2)+2] * b2[2]);
  dbfuncs[3] =  3.0 * (dxInv[3*(i+2)+2] * b2[2]);

  d2bfuncs[0] = 6.0 * (+dxInv[3*(i+0)+2]* dxInv[3*(i+1)+1]*b1[0]);
  d2bfuncs[1] = 6.0 * (-dxInv[3*(i+1)+1]*(dxInv[3*(i+0)+2]+dxInv[3*(i+1)+2])*b1[0] +
		        dxInv[3*(i+1)+2]* dxInv[3*(i+2)+1]*b1[1]);
  d2bfuncs[2] = 6.0 * (+dxInv[3*(i+1)+2]* dxInv[3*(i+1)+1]*b1[0] -
		        dxInv[3*(i+2)+1]*(dxInv[3*(i+1)+2] + dxInv[3*(i+2)+2])*b1[1]);
  d2bfuncs[3] = 6.0 * (+dxInv[3*(i+2)+2]* dxInv[3*(i+2)+1]*b1[1]);
}


//////////////////////////////
// Double-precision version //
//////////////////////////////
int
get_NUBasis_funcs_d (NUBasis* restrict basis, double x,
		     double bfuncs[4])
{
  double b1[2], b2[3];
  int i = (*basis->grid->reverse_map)(basis->grid, x);
  int i2 = i+2;
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals;

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2];
  return i;
}


void
get_NUBasis_funcs_di (NUBasis* restrict basis, int i,
		      double bfuncs[4])
{
  int i2 = i+2;
  double b1[2], b2[3];
  double x = basis->grid->points[i];
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals; 

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2];
}

int
get_NUBasis_dfuncs_d (NUBasis* restrict basis, double x,
		      double bfuncs[4], double dbfuncs[4])
{
  double b1[2], b2[3];
  int i = (*basis->grid->reverse_map)(basis->grid, x);
  int i2 = i+2;
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals;

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2]; 

  dbfuncs[0] = -3.0 * (dxInv[3*(i  )+2] * b2[0]);
  dbfuncs[1] =  3.0 * (dxInv[3*(i  )+2] * b2[0] - dxInv[3*(i+1)+2] * b2[1]);
  dbfuncs[2] =  3.0 * (dxInv[3*(i+1)+2] * b2[1] - dxInv[3*(i+2)+2] * b2[2]);
  dbfuncs[3] =  3.0 * (dxInv[3*(i+2)+2] * b2[2]);

  return i;
}


void
get_NUBasis_dfuncs_di (NUBasis* restrict basis, int i,
		       double bfuncs[4], double dbfuncs[4])
{
  double b1[2], b2[3];
  double x = basis->grid->points[i];
  int i2 = i+2;
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals;

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2]; 

  dbfuncs[0] = -3.0 * (dxInv[3*(i  )+2] * b2[0]);
  dbfuncs[1] =  3.0 * (dxInv[3*(i  )+2] * b2[0] - dxInv[3*(i+1)+2] * b2[1]);
  dbfuncs[2] =  3.0 * (dxInv[3*(i+1)+2] * b2[1] - dxInv[3*(i+2)+2] * b2[2]);
  dbfuncs[3] =  3.0 * (dxInv[3*(i+2)+2] * b2[2]);
}


int
get_NUBasis_d2funcs_d (NUBasis* restrict basis, double x,
		       double bfuncs[4], double dbfuncs[4], double d2bfuncs[4])
{
  double b1[2], b2[3];
  int i = (*basis->grid->reverse_map)(basis->grid, x);
  int i2 = i+2;
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals;

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2]; 

  dbfuncs[0] = -3.0 * (dxInv[3*(i  )+2] * b2[0]);
  dbfuncs[1] =  3.0 * (dxInv[3*(i  )+2] * b2[0] - dxInv[3*(i+1)+2] * b2[1]);
  dbfuncs[2] =  3.0 * (dxInv[3*(i+1)+2] * b2[1] - dxInv[3*(i+2)+2] * b2[2]);
  dbfuncs[3] =  3.0 * (dxInv[3*(i+2)+2] * b2[2]);

  d2bfuncs[0] = 6.0 * (+dxInv[3*(i+0)+2]* dxInv[3*(i+1)+1]*b1[0]);
  d2bfuncs[1] = 6.0 * (-dxInv[3*(i+1)+1]*(dxInv[3*(i+0)+2]+dxInv[3*(i+1)+2])*b1[0] +
		        dxInv[3*(i+1)+2]* dxInv[3*(i+2)+1]*b1[1]);
  d2bfuncs[2] = 6.0 * (+dxInv[3*(i+1)+2]* dxInv[3*(i+1)+1]*b1[0] -
		        dxInv[3*(i+2)+1]*(dxInv[3*(i+1)+2] + dxInv[3*(i+2)+2])*b1[1]);
  d2bfuncs[3] = 6.0 * (+dxInv[3*(i+2)+2]* dxInv[3*(i+2)+1]*b1[1]);

  return i;
}


void
get_NUBasis_d2funcs_di (NUBasis* restrict basis, int i,
			double bfuncs[4], double dbfuncs[4], 
			double d2bfuncs[4])
{
  double b1[2], b2[3];
  double x = basis->grid->points[i];
  int i2 = i+2;
  double* restrict dxInv = basis->dxInv;
  double* restrict xVals = basis->xVals;

  b1[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+2)+0];
  b1[1]     = (x-xVals[i2])    * dxInv[3*(i+2)+0];
  b2[0]     = (xVals[i2+1]-x)  * dxInv[3*(i+1)+1] * b1[0];
  b2[1]     = ((x-xVals[i2-1]) * dxInv[3*(i+1)+1] * b1[0]+
	       (xVals[i2+2]-x) * dxInv[3*(i+2)+1] * b1[1]);
  b2[2]     = (x-xVals[i2])    * dxInv[3*(i+2)+1] * b1[1];
  bfuncs[0] = (xVals[i2+1]-x)  * dxInv[3*(i  )+2] * b2[0];
  bfuncs[1] = ((x-xVals[i2-2]) * dxInv[3*(i  )+2] * b2[0] +
	       (xVals[i2+2]-x) * dxInv[3*(i+1)+2] * b2[1]);
  bfuncs[2] = ((x-xVals[i2-1]) * dxInv[3*(i+1)+2] * b2[1] +
	       (xVals[i2+3]-x) * dxInv[3*(i+2)+2] * b2[2]);
  bfuncs[3] = (x-xVals[i2])    * dxInv[3*(i+2)+2] * b2[2]; 

  dbfuncs[0] = -3.0 * (dxInv[3*(i  )+2] * b2[0]);
  dbfuncs[1] =  3.0 * (dxInv[3*(i  )+2] * b2[0] - dxInv[3*(i+1)+2] * b2[1]);
  dbfuncs[2] =  3.0 * (dxInv[3*(i+1)+2] * b2[1] - dxInv[3*(i+2)+2] * b2[2]);
  dbfuncs[3] =  3.0 * (dxInv[3*(i+2)+2] * b2[2]);

  d2bfuncs[0] = 6.0 * (+dxInv[3*(i+0)+2]* dxInv[3*(i+1)+1]*b1[0]);
  d2bfuncs[1] = 6.0 * (-dxInv[3*(i+1)+1]*(dxInv[3*(i+0)+2]+dxInv[3*(i+1)+2])*b1[0] +
		        dxInv[3*(i+1)+2]* dxInv[3*(i+2)+1]*b1[1]);
  d2bfuncs[2] = 6.0 * (+dxInv[3*(i+1)+2]* dxInv[3*(i+1)+1]*b1[0] -
		        dxInv[3*(i+2)+1]*(dxInv[3*(i+1)+2] + dxInv[3*(i+2)+2])*b1[1]);
  d2bfuncs[3] = 6.0 * (+dxInv[3*(i+2)+2]* dxInv[3*(i+2)+1]*b1[1]);
}
