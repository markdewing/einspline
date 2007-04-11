#include "nubspline_create.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>

void
solve_NUB_deriv_interp_1d_s (NUBasis* restrict basis, 
			     float* restrict data, int datastride,
			     float* restrict    p, int pstride,
			     float abcdInitial[4], float abcdFinal[4])
{
  int M = basis->grid->num_points;
  int N = M+2;
  // Banded matrix storage.  The first three elements in the
  // tinyvector store the tridiagonal coefficients.  The last element
  // stores the RHS data.
  float bands[4*N];

  // Fill up bands
  for (int i=0; i<4; i++) {
    bands[i]         = abcdInitial[i];
    bands[4*(N-1)+i] = abcdFinal[i];
  }
  for (int i=0; i<M; i++) {
    get_NUBasis_funcs_si (basis, i, &(bands[4*(i+1)]));
    bands[4*(i+1)+3] = data[datastride*i];
  }
    
  // Now solve:
  // First and last rows are different
  bands[4*0+1] /= bands[4*0+0];
  bands[4*0+2] /= bands[4*0+0];
  bands[4*0+3] /= bands[4*0+0];
  bands[4*0+0] = 1.0;
  bands[4*1+1] -= bands[4*1+0]*bands[4*0+1];
  bands[4*1+2] -= bands[4*1+0]*bands[4*0+2];
  bands[4*1+3] -= bands[4*1+0]*bands[4*0+3];
  bands[4*0+0] = 0.0;
  bands[4*1+2] /= bands[4*1+1];
  bands[4*1+3] /= bands[4*1+1];
  bands[4*1+1] = 1.0;
  
  // Now do rows 2 through M+1
  for (int row=2; row < N-1; row++) {
    bands[4*(row)+1] -= bands[4*(row)+0]*bands[4*(row-1)+2];
    bands[4*(row)+3] -= bands[4*(row)+0]*bands[4*(row-1)+3];
    bands[4*(row)+2] /= bands[4*(row)+1];
    bands[4*(row)+3] /= bands[4*(row)+1];
    bands[4*(row)+0] = 0.0;
    bands[4*(row)+1] = 1.0;
  }

  // Do last row
  bands[4*(M+1)+1] -= bands[4*(M+1)+0]*bands[4*(M-1)+2];
  bands[4*(M+1)+3] -= bands[4*(M+1)+0]*bands[4*(M-1)+3];
  bands[4*(M+1)+2] -= bands[4*(M+1)+1]*bands[4*(M)+2];
  bands[4*(M+1)+3] -= bands[4*(M+1)+1]*bands[4*(M)+3];
  bands[4*(M+1)+3] /= bands[4*(M+1)+2];
  bands[4*(M+1)+2] = 1.0;

  p[pstride*(M+1)] = bands[4*(M+1)+3];
  // Now back substitute up
  for (int row=M; row>0; row--)
    p[pstride*(row)] = bands[4*(row)+3] - bands[4*(row)+2]*p[pstride*(row+1)];
  
  // Finish with first row
  p[0] = bands[3*(0)+3] - bands[4*(0)+1]*p[pstride*1] - bands[4*(0)+2]*p[pstride*2];
}


void
solve_NUB_periodic_interp_1d_s (NUBasis* restrict basis,
				float* restrict data, int datastride,
				float* restrict p, int pstride)
{
  int M = basis->grid->num_points;
  int N = M+3;

  // Banded matrix storage.  The first three elements in the
  // tinyvector store the tridiagonal coefficients.  The last element
  // stores the RHS data.
  float bands[4*M], lastCol[M];

  // Fill up bands
  for (int i=0; i<M; i++) {
    get_NUBasis_funcs_si (basis, i, &(bands[4*i])); 
    bands[4*(i)+3] = data[i*datastride];
  }
    
  // Now solve:
  // First and last rows are different
  bands[4*(0)+2] /= bands[4*(0)+1];
  bands[4*(0)+0] /= bands[4*(0)+1];
  bands[4*(0)+3] /= bands[4*(0)+1];
  bands[4*(0)+1]  = 1.0;
  bands[4*(M-1)+1] -= bands[4*(M-1)+2]*bands[4*(0)+0];
  bands[4*(M-1)+3] -= bands[4*(M-1)+2]*bands[4*(0)+3];
  bands[4*(M-1)+2]  = -bands[4*(M-1)+2]*bands[4*(0)+2];
  lastCol[0] = bands[4*(0)+0];
  
  for (int row=1; row < (M-1); row++) {
    bands[4*(row)+1] -= bands[4*(row)+0] * bands[4*(row-1)+2];
    bands[4*(row)+3] -= bands[4*(row)+0] * bands[4*(row-1)+3];
    lastCol[row]   = -bands[4*(row)+0] * lastCol[row-1];
    bands[4*(row)+0] = 0.0;
    bands[4*(row)+2] /= bands[4*(row)+1];
    bands[4*(row)+3] /= bands[4*(row)+1];
    lastCol[row]  /= bands[4*(row)+1];
    bands[4*(row)+1]  = 1.0;
    if (row < (M-2)) {
      bands[4*(M-1)+3] -= bands[4*(M-1)+2]*bands[4*(row)+3];
      bands[4*(M-1)+1] -= bands[4*(M-1)+2]*lastCol[row];
      bands[4*(M-1)+2] = -bands[4*(M-1)+2]*bands[4*(row)+2];
    }
  }
  
  // Now do last row
  // The [2] element and [0] element are now on top of each other 
  bands[4*(M-1)+0] += bands[4*(M-1)+2];
  bands[4*(M-1)+1] -= bands[4*(M-1)+0] * (bands[4*(M-2)+2]+lastCol[M-2]);
  bands[4*(M-1)+3] -= bands[4*(M-1)+0] *  bands[4*(M-2)+3];
  bands[4*(M-1)+3] /= bands[4*(M-1)+1];
  p[pstride*M] = bands[4*(M-1)+3];
  for (int row=M-2; row>=0; row--) 
    p[pstride*(row+1)] = bands[4*(row)+3] - 
      bands[4*(row)+2]*p[pstride*(row+2)] - lastCol[row]*p[pstride*M];
  
  p[pstride*  0  ] = p[pstride*M];
  p[pstride*(M+1)] = p[pstride*1];
  p[pstride*(M+2)] = p[pstride*2];
}



void
find_NUBcoefs_1d_s (NUBasis* restrict basis, BCtype_s bc,
		    float *data,  int dstride,
		    float *coefs, int cstride)
{
  int M = basis->grid->num_points;
  if (bc.lCode == PERIODIC) 
    solve_NUB_periodic_interp_1d_s (basis, data, dstride, coefs, cstride);
  else {
    float bands[(M+2)*4];
    // Setup boundary conditions
    float bfuncs[4], dbfuncs[4], abcd_left[4], abcd_right[4];
    // Left boundary
    if (bc.lCode == FLAT || bc.lCode == NATURAL)
      bc.lVal = 0.0;
    if (bc.lCode == FLAT || bc.lCode == DERIV1) {
      get_NUBasis_dfuncs_si (basis, 0, bfuncs, abcd_left);
      abcd_left[3] = bc.lVal;
    }
    if (bc.lCode == NATURAL || bc.lCode == DERIV2) {
      get_NUBasis_d2funcs_si (basis, 0, bfuncs, dbfuncs, abcd_left);
      abcd_left[3] = bc.lVal;
    }
    
    // Right boundary
    if (bc.rCode == FLAT || bc.rCode == NATURAL)
      bc.rVal = 0.0;
    if (bc.rCode == FLAT || bc.rCode == DERIV1) {
      get_NUBasis_dfuncs_si (basis, M-1, bfuncs, abcd_right);
      abcd_right[3] = bc.rVal;
    }
    if (bc.rCode == NATURAL || bc.rCode == DERIV2) {
      get_NUBasis_d2funcs_si (basis, M-1, bfuncs, dbfuncs, abcd_right);
      abcd_right[3] = bc.rVal;
    }
    // Now, solve for coefficients
    solve_NUB_deriv_interp_1d_s (basis, data, dstride, coefs, cstride,
				 abcd_left, abcd_right);
  }
}




NUBspline_1d_s *
create_NUBspline_1d_s (NUgrid* restrict x_grid, BCtype_s xBC, float *data)
{
  // First, create the spline structure
  NUBspline_1d_s* spline = (NUBspline_1d_s*) malloc (sizeof(NUBspline_1d_s));
  if (spline == NULL)
    return spline;

  // Next, create the basis
  spline->x_basis = create_NUBasis (x_grid, xBC.lCode==PERIODIC);
  int M = x_grid->num_points;
  int N;

  // Allocate coefficients and solve
  if (xBC.lCode == PERIODIC) {
    assert (xBC.rCode == PERIODIC);
    N = M+3;
  }
  else 
    N = M+2;
  
  spline->coefs = malloc(N*sizeof(float));
  find_NUBcoefs_1d_s (spline->x_basis, xBC, data, 1, spline->coefs, 1);
    
  return spline;
}
