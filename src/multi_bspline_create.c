/////////////////////////////////////////////////////////////////////////////
//  einspline:  a library for creating and evaluating B-splines            //
//  Copyright (C) 2007 Kenneth P. Esler, Jr.                               //
//                                                                         //
//  This program is free software; you can redistribute it and/or modify   //
//  it under the terms of the GNU General Public License as published by   //
//  the Free Software Foundation; either version 2 of the License, or      //
//  (at your option) any later version.                                    //
//                                                                         //
//  This program is distributed in the hope that it will be useful,        //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of         //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
//  GNU General Public License for more details.                           //
//                                                                         //
//  You should have received a copy of the GNU General Public License      //
//  along with this program; if not, write to the Free Software            //
//  Foundation, Inc., 51 Franklin Street, Fifth Floor,                     //
//  Boston, MA  02110-1301  USA                                            //
/////////////////////////////////////////////////////////////////////////////

#include "multi_bspline_create.h"
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif
#ifndef __USE_XOPEN2K
  #define __USE_XOPEN2K
#endif
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

int posix_memalign(void **memptr, size_t alignment, size_t size);

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////       Helper functions for spline creation         ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
void init_sse_data();

void
find_coefs_1d_d (Ugrid grid, BCtype_d bc, 
		 double *data,  int dstride,
		 double *coefs, int cstride);

void 
solve_deriv_interp_1d_s (float bands[], float coefs[],
			 int M, int cstride);

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
void 
solve_periodic_interp_1d_s (float bands[], float coefs[],
			    int M, int cstride);

void
find_coefs_1d_s (Ugrid grid, BCtype_s bc, 
		 float *data,  int dstride,
		 float *coefs, int cstride);

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////     Single-Precision, Real Creation Routines       ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
multi_UBspline_1d_s*
create_multi_UBspline_1d_s (Ugrid x_grid, BCtype_s xBC, int num_splines)
{
  // Create new spline
  multi_UBspline_1d_s* restrict spline = malloc (sizeof(multi_UBspline_1d_s));
  spline->spcode = MULTI_U1D;
  spline->tcode  = SINGLE_REAL;
  spline->xBC = xBC; spline->x_grid = x_grid;
  spline->num_splines = num_splines;

  // Setup internal variables
  int M = x_grid.num;
  int N;

  if (xBC.lCode == PERIODIC) {
    x_grid.delta     = (x_grid.end-x_grid.start)/(double)(x_grid.num);
    N = M+3;
  }
  else {
    x_grid.delta     = (x_grid.end-x_grid.start)/(double)(x_grid.num-1);
    N = M+2;
  }

  spline->spline_stride = N;
  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;
#ifndef HAVE_SSE2
  spline->coefs = malloc (sizeof(float)*N*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, (sizeof(float)*N*num_splines));
#endif

  init_sse_data();    
  return spline;
}

void
set_multi_UBspline_1d_s (multi_UBspline_1d_s *spline, int num,
			 float *data)
{
  float *coefs = spline->coefs + num*spline->spline_stride;
  find_coefs_1d_s (spline->x_grid, spline->xBC, data, 1, 
		   coefs, 1);
}


multi_UBspline_2d_s*
create_multi_UBspline_2d_s (Ugrid x_grid, Ugrid y_grid,
			    BCtype_s xBC, BCtype_s yBC, int num_splines)
{
  // Create new spline
  multi_UBspline_2d_s* restrict spline = malloc (sizeof(multi_UBspline_2d_s));
  spline->spcode = MULTI_U2D;
  spline->tcode  = SINGLE_REAL;
  spline->xBC = xBC; 
  spline->yBC = yBC; 
  spline->num_splines = num_splines;
  // Setup internal variables
  int Mx = x_grid.num;
  int My = y_grid.num;
  int Nx, Ny;

  if (xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                           Nx = Mx+2;
  x_grid.delta = (x_grid.end - x_grid.start)/(double)(Nx-3);
  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC)     Ny = My+3;
  else                           Ny = My+2;
  y_grid.delta = (y_grid.end - y_grid.start)/(double)(Ny-3);
  y_grid.delta_inv = 1.0/y_grid.delta;
  spline->y_grid   = y_grid;
  spline->x_stride = Ny;
  spline->spline_stride = Nx*Ny;
#ifndef HAVE_SSE2
  spline->coefs = malloc ((size_t)sizeof(float)*Nx*Ny*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, 
		  sizeof(float)*Nx*Ny*num_splines);
#endif

  init_sse_data();
  return spline;
}

void
set_multi_UBspline_2d_s (multi_UBspline_2d_s* spline, int num, float *data)
{
  int Mx = spline->x_grid.num;
  int My = spline->y_grid.num;
  int Nx, Ny;

  if (spline->xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                                   Nx = Mx+2;
  if (spline->yBC.lCode == PERIODIC)     Ny = My+3;
  else                                   Ny = My+2;

  intptr_t offset = num*spline->spline_stride;
  float *coefs = spline->coefs + offset;
  // First, solve in the X-direction 
  for (int iy=0; iy<My; iy++) {
    intptr_t doffset = iy;
    intptr_t coffset = iy;
    find_coefs_1d_s (spline->x_grid, spline->xBC, data+doffset, My,
		     coefs+coffset, Ny);
  }
  
  // Now, solve in the Y-direction
  for (int ix=0; ix<Nx; ix++) {
    intptr_t doffset = ix*Ny;
    intptr_t coffset = ix*Ny;
    find_coefs_1d_s (spline->y_grid, spline->yBC, coefs+doffset, 1, 
		     coefs+coffset, 1);
  }
}


multi_UBspline_3d_s*
create_multi_UBspline_3d_s (Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
			    BCtype_s xBC, BCtype_s yBC, BCtype_s zBC,
			    int num_splines)
{
  // Create new spline
  multi_UBspline_3d_s* restrict spline = malloc (sizeof(multi_UBspline_3d_s));
  spline->spcode = MULTI_U3D;
  spline->tcode  = SINGLE_REAL;
  spline->xBC = xBC; 
  spline->yBC = yBC; 
  spline->zBC = zBC; 
  spline->num_splines = num_splines;
  // Setup internal variables
  int Mx = x_grid.num;  int My = y_grid.num; int Mz = z_grid.num;
  int Nx, Ny, Nz;

  if (xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                           Nx = Mx+2;
  x_grid.delta = (x_grid.end - x_grid.start)/(double)(Nx-3);
  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC)     Ny = My+3;
  else                           Ny = My+2;
  y_grid.delta = (y_grid.end - y_grid.start)/(double)(Ny-3);
  y_grid.delta_inv = 1.0/y_grid.delta;
  spline->y_grid   = y_grid;

  if (zBC.lCode == PERIODIC)     Nz = Mz+3;
  else                           Nz = Mz+2;
  z_grid.delta = (z_grid.end - z_grid.start)/(double)(Nz-3);
  z_grid.delta_inv = 1.0/z_grid.delta;
  spline->z_grid   = z_grid;

  spline->spline_stride = Nx*Ny*Nz;
  spline->x_stride      = Ny*Nz;
  spline->y_stride      = Nz;

#ifndef HAVE_SSE2
  spline->coefs      = malloc (sizeof(float)*Nx*Ny*Nz);
#else
  posix_memalign ((void**)&spline->coefs, 16, 
		  ((size_t)sizeof(float)*Nx*Ny*Nz));
#endif

  init_sse_data();
  return spline;
}

void
set_multi_UBspline_3d_s (multi_UBspline_3d_s* spline, int num, float *data)
{
  intptr_t offset = spline->spline_stride * num;
  int Mx = spline->x_grid.num;
  int My = spline->y_grid.num;
  int Mz = spline->z_grid.num;
  int Nx, Ny, Nz;

  if (spline->xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                                   Nx = Mx+2;
  if (spline->yBC.lCode == PERIODIC)     Ny = My+3;
  else                                   Ny = My+2;
  if (spline->zBC.lCode == PERIODIC)     Nz = Mz+3;
  else                                   Nz = Mz+2;

  float *coefs = spline->coefs + offset;

  // First, solve in the X-direction 
  for (int iy=0; iy<My; iy++) 
    for (int iz=0; iz<Mz; iz++) {
      int doffset = iy*Mz+iz;
      int coffset = iy*Nz+iz;
      find_coefs_1d_s (spline->x_grid, spline->xBC, data+doffset, My*Mz,
		       coefs+coffset, Ny*Nz);
    }
  
  // Now, solve in the Y-direction
  for (int ix=0; ix<Nx; ix++) 
    for (int iz=0; iz<Nz; iz++) {
      int doffset = ix*Ny*Nz + iz;
      int coffset = ix*Ny*Nz + iz;
      find_coefs_1d_s (spline->y_grid, spline->yBC, coefs+doffset, Nz, 
		       coefs+coffset, Nz);
    }

  // Now, solve in the Z-direction
  for (int ix=0; ix<Nx; ix++) 
    for (int iy=0; iy<Ny; iy++) {
      int doffset = (ix*Ny+iy)*Nz + offset;
      int coffset = (ix*Ny+iy)*Nz + offset;
      find_coefs_1d_s (spline->z_grid, spline->zBC, coefs+doffset, 1, 
		       coefs+coffset, 1);
    }
}


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////    Single-Precision, Complex Creation Routines     ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
multi_UBspline_1d_c*
create_multi_UBspline_1d_c (Ugrid x_grid, BCtype_c xBC, int num_splines)
{
  // Create new spline
  multi_UBspline_1d_c* restrict spline = malloc (sizeof(multi_UBspline_1d_c));
  spline->spcode = MULTI_U1D;
  spline->tcode  = SINGLE_COMPLEX;
  spline->xBC = xBC; 
  spline->num_splines = num_splines;
  // Setup internal variables
  int M = x_grid.num;
  int N;

  if (xBC.lCode == PERIODIC) {
    x_grid.delta     = (x_grid.end-x_grid.start)/(double)(x_grid.num);
    N = M+3;
  }
  else {
    x_grid.delta     = (x_grid.end-x_grid.start)/(double)(x_grid.num-1);
    N = M+2;
  }

  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;
  spline->spline_stride = N;
#ifndef HAVE_SSE2
  spline->coefs = malloc (sizeof(float)*N*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, 2*sizeof(float)*N*num_splines);
#endif

  init_sse_data();    
  return spline;
}

void
set_UBspline_1d_c (multi_UBspline_1d_c* spline, int num, complex_float *data)
{
  intptr_t offset = spline->spline_stride * num;
  complex_float *coefs = spline->coefs + offset;

  BCtype_s xBC_r, xBC_i;
  xBC_r.lCode = spline->xBC.lCode;  xBC_r.rCode = spline->xBC.rCode;
  xBC_r.lVal  = spline->xBC.lVal_r; xBC_r.rVal  = spline->xBC.rVal_r;
  xBC_i.lCode = spline->xBC.lCode;  xBC_i.rCode = spline->xBC.rCode;
  xBC_i.lVal  = spline->xBC.lVal_i; xBC_i.rVal  = spline->xBC.rVal_i;

  // Real part
  find_coefs_1d_s (spline->x_grid, xBC_r, 
		   (float*)data, 2, (float*)coefs, 2);
  // Imaginarty part
  find_coefs_1d_s (spline->x_grid, xBC_i, 
		   ((float*)data)+1, 2, ((float*)coefs+1), 2);
}



multi_UBspline_2d_c*
create_multi_UBspline_2d_c (Ugrid x_grid, Ugrid y_grid,
			    BCtype_c xBC, BCtype_c yBC, int num_splines)
{
  // Create new spline
  multi_UBspline_2d_c* restrict spline = malloc (sizeof(multi_UBspline_2d_c));
  spline->spcode = MULTI_U2D;
  spline->tcode  = SINGLE_COMPLEX;
  spline->xBC = xBC; 
  spline->yBC = yBC;
  spline->num_splines = num_splines;

  // Setup internal variables
  int Mx = x_grid.num;
  int My = y_grid.num;
  int Nx, Ny;

  if (xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                           Nx = Mx+2;
  x_grid.delta = (x_grid.end - x_grid.start)/(double)(Nx-3);
  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC)     Ny = My+3;
  else                           Ny = My+2;
  y_grid.delta = (y_grid.end - y_grid.start)/(double)(Ny-3);
  y_grid.delta_inv = 1.0/y_grid.delta;
  spline->y_grid   = y_grid;
  spline->x_stride = Ny;
  spline->spline_stride = Nx * Ny;

#ifndef HAVE_SSE2
  spline->coefs = malloc (2*sizeof(float)*Nx*Ny*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, 
		  2*sizeof(float)*Nx*Ny*num_splines);
#endif

  init_sse_data();
  return spline;
}


void
set_multi_UBspline_2d_c (multi_UBspline_2d_c* spline, int num, complex_float *data)
{
  // Setup internal variables
  int Mx = spline->x_grid.num;
  int My = spline->y_grid.num;
  int Nx, Ny;

  complex_float* coefs = spline->coefs + spline->spline_stride * num;

  if (spline->xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                                   Nx = Mx+2;
  if (spline->yBC.lCode == PERIODIC)     Ny = My+3;
  else                                   Ny = My+2;

  BCtype_s xBC_r, xBC_i, yBC_r, yBC_i;
  xBC_r.lCode = spline->xBC.lCode;  xBC_r.rCode = spline->xBC.rCode;
  xBC_r.lVal  = spline->xBC.lVal_r; xBC_r.rVal  = spline->xBC.rVal_r;
  xBC_i.lCode = spline->xBC.lCode;  xBC_i.rCode = spline->xBC.rCode;
  xBC_i.lVal  = spline->xBC.lVal_i; xBC_i.rVal  = spline->xBC.rVal_i;
  yBC_r.lCode = spline->yBC.lCode;  yBC_r.rCode = spline->yBC.rCode;
  yBC_r.lVal  = spline->yBC.lVal_r; yBC_r.rVal  = spline->yBC.rVal_r;
  yBC_i.lCode = spline->yBC.lCode;  yBC_i.rCode = spline->yBC.rCode;
  yBC_i.lVal  = spline->yBC.lVal_i; yBC_i.rVal  = spline->yBC.rVal_i;
 
  // First, solve in the X-direction 
  for (int iy=0; iy<My; iy++) {
    int doffset = 2*iy;
    int coffset = 2*iy;
    // Real part
    find_coefs_1d_s (spline->x_grid, xBC_r, ((float*)data)+doffset, 2*My,
		     (float*)coefs+coffset, 2*Ny);
    // Imag part
    find_coefs_1d_s (spline->x_grid, xBC_i, ((float*)data)+doffset+1, 2*My,
		     ((float*)coefs)+coffset+1, 2*Ny);
  }
  
  // Now, solve in the Y-direction
  for (int ix=0; ix<Nx; ix++) {
    int doffset = 2*ix*Ny;
    int coffset = 2*ix*Ny;
    // Real part
    find_coefs_1d_s (spline->y_grid, yBC_r, ((float*)coefs)+doffset, 2, 
		     ((float*)coefs)+coffset, 2);
    // Imag part
    find_coefs_1d_s (spline->y_grid, yBC_i, ((float*)coefs)+doffset+1, 2, 
		     ((float*)coefs)+coffset+1, 2);
  }  
}

multi_UBspline_3d_c*
create_multi_UBspline_3d_c (Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
		      BCtype_c xBC, BCtype_c yBC, BCtype_c zBC,
		      int num_splines)
{
  // Create new spline
  multi_UBspline_3d_c* restrict spline = malloc (sizeof(multi_UBspline_3d_c));
  spline->spcode = MULTI_U3D;
  spline->tcode  = SINGLE_COMPLEX;
  spline->xBC = xBC; 
  spline->yBC = yBC; 
  spline->zBC = zBC; 
  spline->num_splines = num_splines;

  // Setup internal variables
  int Mx = x_grid.num;  int My = y_grid.num; int Mz = z_grid.num;
  int Nx, Ny, Nz;

  if (xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                           Nx = Mx+2;
  x_grid.delta = (x_grid.end - x_grid.start)/(double)(Nx-3);
  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC)     Ny = My+3;
  else                           Ny = My+2;
  y_grid.delta = (y_grid.end - y_grid.start)/(double)(Ny-3);
  y_grid.delta_inv = 1.0/y_grid.delta;
  spline->y_grid   = y_grid;

  if (zBC.lCode == PERIODIC)     Nz = Mz+3;
  else                           Nz = Mz+2;
  z_grid.delta = (z_grid.end - z_grid.start)/(double)(Nz-3);
  z_grid.delta_inv = 1.0/z_grid.delta;
  spline->z_grid   = z_grid;

  spline->x_stride = Ny*Nz;
  spline->y_stride = Nz;
  spline->spline_stride = Nx*Ny*Nz;

#ifndef HAVE_SSE2
  spline->coefs      = malloc ((size_t)2*sizeof(float)*Nx*Ny*Nz*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, (size_t)2*sizeof(float)*Nx*Ny*Nz*num_splines);
#endif

  init_sse_data();
  return spline;
}

void
set_multi_UBspline_3d_c (multi_UBspline_3d_c* spline, int num, complex_float *data)
{
  // Setup internal variables
  int Mx = spline->x_grid.num;
  int My = spline->y_grid.num;
  int Mz = spline->z_grid.num;
  int Nx, Ny, Nz;

  if (spline->xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                           Nx = Mx+2;
  if (spline->yBC.lCode == PERIODIC)     Ny = My+3;
  else                           Ny = My+2;
  if (spline->zBC.lCode == PERIODIC)     Nz = Mz+3;
  else                           Nz = Mz+2;

  BCtype_s xBC_r, xBC_i, yBC_r, yBC_i, zBC_r, zBC_i;
  xBC_r.lCode = spline->xBC.lCode;  xBC_r.rCode = spline->xBC.rCode;
  xBC_r.lVal  = spline->xBC.lVal_r; xBC_r.rVal  = spline->xBC.rVal_r;
  xBC_i.lCode = spline->xBC.lCode;  xBC_i.rCode = spline->xBC.rCode;
  xBC_i.lVal  = spline->xBC.lVal_i; xBC_i.rVal  = spline->xBC.rVal_i;
  yBC_r.lCode = spline->yBC.lCode;  yBC_r.rCode = spline->yBC.rCode;
  yBC_r.lVal  = spline->yBC.lVal_r; yBC_r.rVal  = spline->yBC.rVal_r;
  yBC_i.lCode = spline->yBC.lCode;  yBC_i.rCode = spline->yBC.rCode;
  yBC_i.lVal  = spline->yBC.lVal_i; yBC_i.rVal  = spline->yBC.rVal_i;
  zBC_r.lCode = spline->zBC.lCode;  zBC_r.rCode = spline->zBC.rCode;
  zBC_r.lVal  = spline->zBC.lVal_r; zBC_r.rVal  = spline->zBC.rVal_r;
  zBC_i.lCode = spline->zBC.lCode;  zBC_i.rCode = spline->zBC.rCode;
  zBC_i.lVal  = spline->zBC.lVal_i; zBC_i.rVal  = spline->zBC.rVal_i;

  complex_float *coefs = spline->coefs + spline->spline_stride * num;
  // First, solve in the X-direction 
  for (int iy=0; iy<My; iy++) 
    for (int iz=0; iz<Mz; iz++) {
      int doffset = 2*(iy*Mz+iz);
      int coffset = 2*(iy*Nz+iz);
      // Real part
      find_coefs_1d_s (spline->x_grid, xBC_r, ((float*)data)+doffset, 2*My*Mz,
		       ((float*)coefs)+coffset, 2*Ny*Nz);
      // Imag part
      find_coefs_1d_s (spline->x_grid, xBC_i, ((float*)data)+doffset+1, 2*My*Mz,
		       ((float*)coefs)+coffset+1, 2*Ny*Nz);
    }
  
  // Now, solve in the Y-direction
  for (int ix=0; ix<Nx; ix++) 
    for (int iz=0; iz<Nz; iz++) {
      int doffset = 2*(ix*Ny*Nz + iz);
      int coffset = 2*(ix*Ny*Nz + iz);
      // Real part
      find_coefs_1d_s (spline->y_grid, yBC_r, ((float*)coefs)+doffset, 2*Nz, 
		       ((float*)coefs)+coffset, 2*Nz);
      // Imag part
      find_coefs_1d_s (spline->y_grid, yBC_i, ((float*)coefs)+doffset+1, 2*Nz, 
		       ((float*)coefs)+coffset+1, 2*Nz);
    }

  // Now, solve in the Z-direction
  for (int ix=0; ix<Nx; ix++) 
    for (int iy=0; iy<Ny; iy++) {
      int doffset = 2*((ix*Ny+iy)*Nz);
      int coffset = 2*((ix*Ny+iy)*Nz);
      // Real part
      find_coefs_1d_s (spline->z_grid, zBC_r, ((float*)coefs)+doffset, 2, 
		       ((float*)coefs)+coffset, 2);
      // Imag part
      find_coefs_1d_s (spline->z_grid, zBC_i, ((float*)coefs)+doffset+1, 2, 
		       ((float*)coefs)+coffset+1, 2);
    }
}


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////     Double-Precision, Real Creation Routines       ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
void 
solve_deriv_interp_1d_d (double bands[], double coefs[],
			 int M, int cstride);

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
void 
solve_periodic_interp_1d_d (double bands[], double coefs[],
			    int M, int cstride);

void
find_coefs_1d_d (Ugrid grid, BCtype_d bc, 
		 double *data,  int dstride,
		 double *coefs, int cstride);

multi_UBspline_1d_d*
create_multi_UBspline_1d_d (Ugrid x_grid, BCtype_d xBC, int num_splines)
{
  // Create new spline
  multi_UBspline_1d_d* restrict spline = malloc (sizeof(multi_UBspline_1d_d));
  spline->spcode = MULTI_U1D;
  spline->tcode  = DOUBLE_REAL;
  spline->xBC = xBC; 
  spline->num_splines = num_splines;

  // Setup internal variables
  int M = x_grid.num;
  int N;

  if (xBC.lCode == PERIODIC) {
    x_grid.delta     = (x_grid.end-x_grid.start)/(double)(x_grid.num);
    N = M+3;
  }
  else {
    x_grid.delta     = (x_grid.end-x_grid.start)/(double)(x_grid.num-1);
    N = M+2;
  }

  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;
  spline->spline_stride = N;

#ifndef HAVE_SSE2
  spline->coefs = malloc (sizeof(double)*N*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, sizeof(double)*N*num_splines);
#endif
    
  init_sse_data();
  return spline;
}

void
set_multi_UBspline_1d_d (multi_UBspline_1d_d* spline, int num, double *data)
{
  double *coefs = spline->coefs + spline->spline_stride * num;
  find_coefs_1d_d (spline->x_grid, spline->xBC, data, 1, coefs, 1);
}


multi_UBspline_2d_d*
create_multi_UBspline_2d_d (Ugrid x_grid, Ugrid y_grid,
			    BCtype_d xBC, BCtype_d yBC, int num_splines)
{
  // Create new spline
  multi_UBspline_2d_d* restrict spline = malloc (sizeof(multi_UBspline_2d_d));
  spline->spcode = MULTI_U2D;
  spline->tcode  = DOUBLE_REAL;
  spline->xBC = xBC; 
  spline->yBC = yBC; 
  spline->num_splines = num_splines;
 
  // Setup internal variables
  int Mx = x_grid.num;
  int My = y_grid.num;
  int Nx, Ny;

  if (xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                           Nx = Mx+2;
  x_grid.delta = (x_grid.end - x_grid.start)/(double)(Nx-3);
  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC)     Ny = My+3;
  else                           Ny = My+2;
  y_grid.delta = (y_grid.end - y_grid.start)/(double)(Ny-3);
  y_grid.delta_inv = 1.0/y_grid.delta;
  spline->y_grid   = y_grid;
  spline->x_stride = Ny;
  spline->spline_stride = Nx*Ny;

#ifndef HAVE_SSE2
  spline->coefs = malloc (sizeof(double)*Nx*Ny*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, (sizeof(double)*Nx*Ny*num_splines));
#endif

  init_sse_data();
  return spline;
}

void
set_multi_UBspline_2d_d (multi_UBspline_2d_d* spline, int num, double *data)
{
  int Mx = spline->x_grid.num;
  int My = spline->y_grid.num;
  int Nx, Ny;
  double *coefs = spline->coefs + num*spline->spline_stride;

  if (spline->xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                                   Nx = Mx+2;

  if (spline->yBC.lCode == PERIODIC)     Ny = My+3;
  else                                   Ny = My+2;

  // First, solve in the X-direction 
  for (int iy=0; iy<My; iy++) {
    int doffset = iy;
    int coffset = iy;
    find_coefs_1d_d (spline->x_grid, spline->xBC, data+doffset, My,
		     coefs+coffset, Ny);
  }
  
  // Now, solve in the Y-direction
  for (int ix=0; ix<Nx; ix++) {
    int doffset = ix*Ny;
    int coffset = ix*Ny;
    find_coefs_1d_d (spline->y_grid, spline->yBC, coefs+doffset, 1, 
		     coefs+coffset, 1);
  }
}


multi_UBspline_3d_d*
create_multi_UBspline_3d_d (Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
		      BCtype_d xBC, BCtype_d yBC, BCtype_d zBC,
		      int num_splines)
{
  // Create new spline
  multi_UBspline_3d_d* restrict spline = malloc (sizeof(multi_UBspline_3d_d));
  spline->spcode = MULTI_U3D;
  spline->tcode  = DOUBLE_REAL;
  spline->xBC = xBC; 
  spline->yBC = yBC; 
  spline->zBC = zBC; 
  spline->num_splines = num_splines;

  // Setup internal variables
  int Mx = x_grid.num;  int My = y_grid.num; int Mz = z_grid.num;
  int Nx, Ny, Nz;

  if (xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                           Nx = Mx+2;
  x_grid.delta = (x_grid.end - x_grid.start)/(double)(Nx-3);
  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC)     Ny = My+3;
  else                           Ny = My+2;
  y_grid.delta = (y_grid.end - y_grid.start)/(double)(Ny-3);
  y_grid.delta_inv = 1.0/y_grid.delta;
  spline->y_grid   = y_grid;

  if (zBC.lCode == PERIODIC)     Nz = Mz+3;
  else                           Nz = Mz+2;
  z_grid.delta = (z_grid.end - z_grid.start)/(double)(Nz-3);
  z_grid.delta_inv = 1.0/z_grid.delta;
  spline->z_grid   = z_grid;

  spline->x_stride = Ny*Nz;
  spline->y_stride = Nz;
  spline->spline_stride = Nx*Ny*Nz;

#ifndef HAVE_SSE2
  spline->coefs      = malloc ((size_t)sizeof(double)*Nx*Ny*Nz*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, ((size_t)sizeof(double)*Nx*Ny*Nz*num_splines));
#endif

  init_sse_data();
  return spline;
}

void
set_multi_UBspline_3d_d (multi_UBspline_3d_d* spline, int num, double *data)
{
  int Mx = spline->x_grid.num;  
  int My = spline->y_grid.num; 
  int Mz = spline->z_grid.num;
  int Nx, Ny, Nz;

  if (spline->xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                                   Nx = Mx+2;
  if (spline->yBC.lCode == PERIODIC)     Ny = My+3;
  else                                   Ny = My+2;
  if (spline->zBC.lCode == PERIODIC)     Nz = Mz+3;
  else                                   Nz = Mz+2;

  double *coefs = spline->coefs + spline->spline_stride * num;

  // First, solve in the X-direction 
  for (int iy=0; iy<My; iy++) 
    for (int iz=0; iz<Mz; iz++) {
      int doffset = iy*Mz+iz;
      int coffset = iy*Nz+iz;
      find_coefs_1d_d (spline->x_grid, spline->xBC, data+doffset, My*Mz,
		       coefs+coffset, Ny*Nz);
    }
  
  // Now, solve in the Y-direction
  for (int ix=0; ix<Nx; ix++) 
    for (int iz=0; iz<Nz; iz++) {
      int doffset = ix*Ny*Nz + iz;
      int coffset = ix*Ny*Nz + iz;
      find_coefs_1d_d (spline->y_grid, spline->yBC, coefs+doffset, Nz, 
		       coefs+coffset, Nz);
    }

  // Now, solve in the Z-direction
  for (int ix=0; ix<Nx; ix++) 
    for (int iy=0; iy<Ny; iy++) {
      int doffset = (ix*Ny+iy)*Nz;
      int coffset = (ix*Ny+iy)*Nz;
      find_coefs_1d_d (spline->z_grid, spline->zBC, coefs+doffset, 1, 
		       coefs+coffset, 1);
    }
}


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////    Double-Precision, Complex Creation Routines     ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs


multi_UBspline_1d_z*
create_multi_UBspline_1d_z (Ugrid x_grid, BCtype_z xBC, int num_splines)
{
  // Create new spline
  multi_UBspline_1d_z* restrict spline = malloc (sizeof(multi_UBspline_1d_z));
  spline->spcode = MULTI_U1D;
  spline->tcode  = DOUBLE_COMPLEX;
  spline->xBC = xBC; 
  spline->num_splines = num_splines;

  // Setup internal variables
  int M = x_grid.num;
  int N;

  if (xBC.lCode == PERIODIC) {
    x_grid.delta     = (x_grid.end-x_grid.start)/(double)(x_grid.num);
    N = M+3;
  }
  else {
    x_grid.delta     = (x_grid.end-x_grid.start)/(double)(x_grid.num-1);
    N = M+2;
  }

  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;
  spline->spline_stride = N;
#ifndef HAVE_SSE2
  spline->coefs = malloc (sizeof(double)*N*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, 2*sizeof(double)*N*num_splines);
#endif
 
  init_sse_data();   
  return spline;
}

void
set_multi_UBspline_1d_z (multi_UBspline_1d_z* spline, int num, complex_double *data)
{
  int M = spline->x_grid.num;
  int N;

  complex_double *coefs = spline->coefs + spline->spline_stride * num;

  if (spline->xBC.lCode == PERIODIC)   N = M+3;
  else                                 N = M+2;

  BCtype_d xBC_r, xBC_i;
  xBC_r.lCode = spline->xBC.lCode;  xBC_r.rCode = spline->xBC.rCode;
  xBC_r.lVal  = spline->xBC.lVal_r; xBC_r.rVal  = spline->xBC.rVal_r;
  xBC_i.lCode = spline->xBC.lCode;  xBC_i.rCode = spline->xBC.rCode;
  xBC_i.lVal  = spline->xBC.lVal_i; xBC_i.rVal  = spline->xBC.rVal_i;
  // Real part
  find_coefs_1d_d (spline->x_grid, xBC_r, (double*)data, 2, 
		   (double*)coefs, 2);
  // Imaginarty part
  find_coefs_1d_d (spline->x_grid, xBC_i, ((double*)data)+1, 2, 
		   ((double*)coefs)+1, 2);
 
}


multi_UBspline_2d_z*
create_multi_UBspline_2d_z (Ugrid x_grid, Ugrid y_grid,
		      BCtype_z xBC, BCtype_z yBC, int num_splines)
{
  // Create new spline
  multi_UBspline_2d_z* restrict spline = malloc (sizeof(multi_UBspline_2d_z));
  spline->spcode = MULTI_U2D;
  spline->tcode  = DOUBLE_COMPLEX;
  spline->xBC = xBC; 
  spline->yBC = yBC;
   spline->num_splines = num_splines;

  // Setup internal variables
  int Mx = x_grid.num;
  int My = y_grid.num;
  int Nx, Ny;

  if (xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                           Nx = Mx+2;
  x_grid.delta = (x_grid.end - x_grid.start)/(double)(Nx-3);
  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC)     Ny = My+3;
  else                           Ny = My+2;
  y_grid.delta = (y_grid.end - y_grid.start)/(double)(Ny-3);
  y_grid.delta_inv = 1.0/y_grid.delta;
  spline->y_grid   = y_grid;
  spline->x_stride = Ny;
  spline->spline_stride = Nx*Ny;

#ifndef HAVE_SSE2
  spline->coefs = malloc (2*sizeof(double)*Nx*Ny*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, 2*sizeof(double)*Nx*Ny*num_splines);
#endif

  init_sse_data();
  return spline;
}


void
set_multi_UBspline_2d_z (multi_UBspline_2d_z* spline, int num,
			 complex_double *data)
{
  int Mx = spline->x_grid.num;
  int My = spline->y_grid.num;
  int Nx, Ny;

  if (spline->xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                           Nx = Mx+2;
  if (spline->yBC.lCode == PERIODIC)     Ny = My+3;
  else                           Ny = My+2;

  BCtype_d xBC_r, xBC_i, yBC_r, yBC_i;
  xBC_r.lCode = spline->xBC.lCode;  xBC_r.rCode = spline->xBC.rCode;
  xBC_r.lVal  = spline->xBC.lVal_r; xBC_r.rVal  = spline->xBC.rVal_r;
  xBC_i.lCode = spline->xBC.lCode;  xBC_i.rCode = spline->xBC.rCode;
  xBC_i.lVal  = spline->xBC.lVal_i; xBC_i.rVal  = spline->xBC.rVal_i;
  yBC_r.lCode = spline->yBC.lCode;  yBC_r.rCode = spline->yBC.rCode;
  yBC_r.lVal  = spline->yBC.lVal_r; yBC_r.rVal  = spline->yBC.rVal_r;
  yBC_i.lCode = spline->yBC.lCode;  yBC_i.rCode = spline->yBC.rCode;
  yBC_i.lVal  = spline->yBC.lVal_i; yBC_i.rVal  = spline->yBC.rVal_i;

  complex_double *coefs = spline->coefs + spline->spline_stride * num;

  // First, solve in the X-direction 
  for (int iy=0; iy<My; iy++) {
    int doffset = 2*iy;
    int coffset = 2*iy;
    // Real part
    find_coefs_1d_d (spline->x_grid, xBC_r, ((double*)data+doffset), 2*My,
		     (double*)coefs+coffset, 2*Ny);
    // Imag part
    find_coefs_1d_d (spline->x_grid, xBC_i, ((double*)data)+doffset+1, 2*My,
		     ((double*)coefs)+coffset+1, 2*Ny);
  }
  
  // Now, solve in the Y-direction
  for (int ix=0; ix<Nx; ix++) {
    int doffset = 2*ix*Ny;
    int coffset = 2*ix*Ny;
    // Real part
    find_coefs_1d_d (spline->y_grid, yBC_r, ((double*)coefs)+doffset, 2, 
		     (double*)coefs+coffset, 2);
    // Imag part
    find_coefs_1d_d (spline->y_grid, yBC_i, (double*)coefs+doffset+1, 2, 
		     ((double*)coefs)+coffset+1, 2);
  }
}



multi_UBspline_3d_z*
create_multi_UBspline_3d_z (Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
			    BCtype_z xBC, BCtype_z yBC, BCtype_z zBC,
			    int num_splines)
{
  // Create new spline
  multi_UBspline_3d_z* restrict spline = malloc (sizeof(multi_UBspline_3d_z));
  spline->spcode = MULTI_U3D;
  spline->tcode  = DOUBLE_COMPLEX;
  spline->xBC = xBC; 
  spline->yBC = yBC; 
  spline->zBC = zBC;
  spline->num_splines = num_splines;

  // Setup internal variables
  int Mx = x_grid.num;  int My = y_grid.num; int Mz = z_grid.num;
  int Nx, Ny, Nz;

  if (xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                           Nx = Mx+2;
  x_grid.delta = (x_grid.end - x_grid.start)/(double)(Nx-3);
  x_grid.delta_inv = 1.0/x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC)     Ny = My+3;
  else                           Ny = My+2;
  y_grid.delta = (y_grid.end - y_grid.start)/(double)(Ny-3);
  y_grid.delta_inv = 1.0/y_grid.delta;
  spline->y_grid   = y_grid;

  if (zBC.lCode == PERIODIC)     Nz = Mz+3;
  else                           Nz = Mz+2;
  z_grid.delta = (z_grid.end - z_grid.start)/(double)(Nz-3);
  z_grid.delta_inv = 1.0/z_grid.delta;
  spline->z_grid   = z_grid;

  spline->x_stride = Ny*Nz;
  spline->y_stride = Nz;
  spline->spline_stride = Nx*Ny*Nz;

#ifndef HAVE_SSE2
  spline->coefs      = malloc ((size_t)2*sizeof(double)*Nx*Ny*Nz*num_splines);
#else
  posix_memalign ((void**)&spline->coefs, 16, (size_t)2*sizeof(double)*Nx*Ny*Nz*num_splines);
#endif

  init_sse_data();
  return spline;
}

void
set_multi_UBspline_3d_z (multi_UBspline_3d_z* spline, int num, complex_double *data)
{
  // Setup internal variables
  int Mx = spline->x_grid.num;  
  int My = spline->y_grid.num; 
  int Mz = spline->z_grid.num;
  int Nx, Ny, Nz;

  if (spline->xBC.lCode == PERIODIC)     Nx = Mx+3;
  else                                   Nx = Mx+2;
  if (spline->yBC.lCode == PERIODIC)     Ny = My+3;
  else                                   Ny = My+2;
  if (spline->zBC.lCode == PERIODIC)     Nz = Mz+3;
  else                                   Nz = Mz+2;

  BCtype_d xBC_r, xBC_i, yBC_r, yBC_i, zBC_r, zBC_i;
  xBC_r.lCode = spline->xBC.lCode;  xBC_r.rCode = spline->xBC.rCode;
  xBC_r.lVal  = spline->xBC.lVal_r; xBC_r.rVal  = spline->xBC.rVal_r;
  xBC_i.lCode = spline->xBC.lCode;  xBC_i.rCode = spline->xBC.rCode;
  xBC_i.lVal  = spline->xBC.lVal_i; xBC_i.rVal  = spline->xBC.rVal_i;
  yBC_r.lCode = spline->yBC.lCode;  yBC_r.rCode = spline->yBC.rCode;
  yBC_r.lVal  = spline->yBC.lVal_r; yBC_r.rVal  = spline->yBC.rVal_r;
  yBC_i.lCode = spline->yBC.lCode;  yBC_i.rCode = spline->yBC.rCode;
  yBC_i.lVal  = spline->yBC.lVal_i; yBC_i.rVal  = spline->yBC.rVal_i;
  zBC_r.lCode = spline->zBC.lCode;  zBC_r.rCode = spline->zBC.rCode;
  zBC_r.lVal  = spline->zBC.lVal_r; zBC_r.rVal  = spline->zBC.rVal_r;
  zBC_i.lCode = spline->zBC.lCode;  zBC_i.rCode = spline->zBC.rCode;
  zBC_i.lVal  = spline->zBC.lVal_i; zBC_i.rVal  = spline->zBC.rVal_i;

  complex_double *coefs = spline->coefs + spline->spline_stride * num;

  // First, solve in the X-direction 
  for (int iy=0; iy<My; iy++) 
    for (int iz=0; iz<Mz; iz++) {
      int doffset = 2*(iy*Mz+iz);
      int coffset = 2*(iy*Nz+iz);
      // Real part
      find_coefs_1d_d (spline->x_grid, xBC_r, ((double*)data)+doffset, 2*My*Mz,
		       ((double*)coefs)+coffset, 2*Ny*Nz);
      // Imag part
      find_coefs_1d_d (spline->x_grid, xBC_i, ((double*)data)+doffset+1, 2*My*Mz,
		       ((double*)coefs)+coffset+1, 2*Ny*Nz);
    }
  
  // Now, solve in the Y-direction
  for (int ix=0; ix<Nx; ix++) 
    for (int iz=0; iz<Nz; iz++) {
      int doffset = 2*(ix*Ny*Nz + iz);
      int coffset = 2*(ix*Ny*Nz + iz);
      // Real part
      find_coefs_1d_d (spline->y_grid, yBC_r, ((double*)coefs)+doffset, 2*Nz, 
		       ((double*)coefs)+coffset, 2*Nz);
      // Imag part
      find_coefs_1d_d (spline->y_grid, yBC_i, ((double*)coefs)+doffset+1, 2*Nz, 
		       ((double*)coefs)+coffset+1, 2*Nz);
    }

  // Now, solve in the Z-direction
  for (int ix=0; ix<Nx; ix++) 
    for (int iy=0; iy<Ny; iy++) {
      int doffset = 2*((ix*Ny+iy)*Nz);
      int coffset = 2*((ix*Ny+iy)*Nz);
      // Real part
      find_coefs_1d_d (spline->z_grid, zBC_r, ((double*)coefs)+doffset, 2, 
		       ((double*)coefs)+coffset, 2);
      // Imag part
      find_coefs_1d_d (spline->z_grid, zBC_i, ((double*)coefs)+doffset+1, 2, 
		       ((double*)coefs)+coffset+1, 2);
    }
}


void
destroy_multi_UBspline (Bspline *spline)
{
  free (spline->coefs);
  free (spline);
}
