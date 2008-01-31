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

#ifndef MULTI_BSPLINE_EVAL_SSE_Z_H
#define MULTI_BSPLINE_EVAL_SSE_Z_H

#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <stdio.h>
#include <math.h>
extern __m128d *restrict A_d;
#include "multi_bspline_structs.h"

#ifndef _MM_DDOT4_PD
#ifdef HAVE_SSE3
#define _MM_DDOT4_PD(a0, a1, a2, a3, b0, b1, b2, b3, r)               \
do {                                                                  \
   __m128d t0 = _mm_add_pd(_mm_mul_pd (a0, b0),_mm_mul_pd (a1, b1));  \
   __m128d t1 = _mm_add_pd(_mm_mul_pd (a2, b2),_mm_mul_pd (a3, b3));  \
   r = _mm_hadd_pd (t0, t1);                                          \
 } while(0);
#define _MM_DOT4_PD(a0, a1, b0, b1, p)                                \
do {                                                                  \
  __m128d t0 = _mm_add_pd(_mm_mul_pd (a0, b0),_mm_mul_pd (a1, b1));   \
  __m128d t1 = _mm_hadd_pd (t0,t0);                                   \
  _mm_store_sd (&(p), t1);                                            \
 } while (0);
#else
#define _MM_DDOT4_PD(a0, a1, a2, a3, b0, b1, b2, b3, r)               \
do {                                                                  \
   __m128d t0 = _mm_add_pd(_mm_mul_pd (a0, b0),_mm_mul_pd (a1, b1));  \
   __m128d t1 = _mm_add_pd(_mm_mul_pd (a2, b2),_mm_mul_pd (a3, b3));  \
   r = _mm_add_pd(_mm_unpacklo_pd(t0,t1),_mm_unpackhi_pd(t0,t1));     \
 } while(0);
#define _MM_DOT4_PD(a0, a1, b0, b1, p)                                \
do {                                                                  \
  __m128d t0 = _mm_add_pd(_mm_mul_pd (a0, b0),_mm_mul_pd (a1, b1));   \
  __m128d t1 =                                                        \
      _mm_add_pd (_mm_unpacklo_pd(t0,t0), _mm_unpackhi_pd(t0,t0));    \
  _mm_store_sd (&(p), t1);                                            \
 } while (0);
#endif
#endif

/************************************************************/
/* 1D double-precision, complex evaulation functions        */
/************************************************************/


/************************************************************/
/* 2D double-precision, complex evaulation functions        */
/************************************************************/


/************************************************************/
/* 3D double-precision, complex evaulation functions        */
/************************************************************/
inline void
eval_multi_UBspline_3d_z (multi_UBspline_3d_z *spline,
			  double x, double y, double z,
			  complex_double* restrict vals)
{
  _mm_prefetch ((const char*) &A_d[0],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[1],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[2],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[3],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[4],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[5],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[6],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[7],_MM_HINT_T0);  

  x -= spline->x_grid.start;
  y -= spline->y_grid.start;  
  z -= spline->z_grid.start;
  double ux = x*spline->x_grid.delta_inv;
  double uy = y*spline->y_grid.delta_inv;
  double uz = z*spline->z_grid.delta_inv;
  ux = fmin (ux, (double)(spline->x_grid.num)-1.0e-5);
  uy = fmin (uy, (double)(spline->y_grid.num)-1.0e-5);
  uz = fmin (uz, (double)(spline->z_grid.num)-1.0e-5);
  double ipartx, iparty, ipartz, tx, ty, tz;
  tx = modf (ux, &ipartx);  int ix = (int) ipartx;
  ty = modf (uy, &iparty);  int iy = (int) iparty;
  tz = modf (uz, &ipartz);  int iz = (int) ipartz;
  
  int xs = spline->x_stride;
  int ys = spline->y_stride;
  int ss = spline->spline_stride;
  complex_double* restrict coefs = spline->coefs;
  complex_double* restrict next_coefs = coefs + ss;

  // This macro is used to give the pointer to coefficient data.
  // i and j should be in the range [0,3].  Coefficients are read four
  // at a time, so no k value is needed.
#define P(i,j,k) (const double*)(coefs+(ix+(i))*xs+(iy+(j))*ys+(iz+k))
  // Prefetch the data from main memory into cache so it's available
  // when we need to use it.
  _mm_prefetch ((const char*)P(0,0,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(0,0,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(0,1,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(0,1,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(0,2,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(0,2,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(0,3,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(0,3,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(1,0,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(1,0,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(1,1,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(1,1,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(1,2,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(1,2,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(1,3,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(1,3,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(2,0,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(2,0,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(2,1,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(2,1,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(2,2,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(2,2,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(2,3,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(2,3,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(3,0,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(3,0,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(3,1,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(3,1,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(3,2,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(3,2,2), _MM_HINT_T0);
  _mm_prefetch ((const char*)P(3,3,0), _MM_HINT_T0);  _mm_prefetch ((const char*)P(3,3,2), _MM_HINT_T0);

  // Now compute the vectors:
  // tpx = [t_x^3 t_x^2 t_x 1]
  // tpy = [t_y^3 t_y^2 t_y 1]
  // tpz = [t_z^3 t_z^2 t_z 1]

  // a  =  A * tpx,   b =  A * tpy,   c =  A * tpz
  // A is 4x4 matrix given by the rows A0, A1, A2, A3
  __m128d tpx01, tpx23, tpy01, tpy23, tpz01, tpz23,
    a01, b01, c01, cPr[8], dcPr[8], bcP01r, bcP23r, 
    a23, b23, c23, cPi[8], dcPi[8], bcP01i, bcP23i,  
    tmp0, tmp1, tmp2, tmp3, r0, r1, r2, r3, i0, i1, i2, i3;
  
  tpx01 = _mm_set_pd (tx*tx*tx, tx*tx);
  tpx23 = _mm_set_pd (tx, 1.0);
  tpy01 = _mm_set_pd (ty*ty*ty, ty*ty);
  tpy23 = _mm_set_pd (ty, 1.0);
  tpz01 = _mm_set_pd (tz*tz*tz, tz*tz);
  tpz23 = _mm_set_pd (tz, 1.0);

  // x-dependent vectors
  _MM_DDOT4_PD (A_d[0], A_d[1], A_d[2], A_d[3], tpx01, tpx23, tpx01, tpx23,   a01);
  _MM_DDOT4_PD (A_d[4], A_d[5], A_d[6], A_d[7], tpx01, tpx23, tpx01, tpx23,   a23);
  // y-dependent vectors
  _MM_DDOT4_PD (A_d[0], A_d[1], A_d[2], A_d[3], tpy01, tpy23, tpy01, tpy23,   b01);
  _MM_DDOT4_PD (A_d[4], A_d[5], A_d[6], A_d[7], tpy01, tpy23, tpy01, tpy23,   b23);
  // z-dependent vectors
  _MM_DDOT4_PD (A_d[0], A_d[1], A_d[2], A_d[3], tpz01, tpz23, tpz01, tpz23,   c01);
  _MM_DDOT4_PD (A_d[4], A_d[5], A_d[6], A_d[7], tpz01, tpz23, tpz01, tpz23,   c23);

  // Now compute tensor product of a, b, and c
  __m128d abc[32];
  __m128d b0, b1, b2, b3, a0, a1, a2, a3;
  
  abc[0] = _mm_mul_pd (b01,c01);
  abc[1] = _mm_mul_pd (

  // Compute cP, dcP, and d2cP products 1/8 at a time to maximize
  // register reuse and avoid rerereading from memory or cache.
  // Complex values are read in, then shuffled such that 4 registers
  // hold the read parts and 4 register hold the imaginary parts.
  // 1st eighth
  tmp0 = _mm_load_pd (P(0,0,0));  tmp1 = _mm_load_pd (P(0,0,1));
  r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(0,0,2));  tmp1 = _mm_load_pd (P(0,0,3));
  r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(0,1,0));  tmp1 = _mm_load_pd (P(0,1,1));
  r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(0,1,2));  tmp1 = _mm_load_pd (P(0,1,3));
  r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[0]);
  _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[0]);
  
  // 2nd eighth
  tmp0 = _mm_load_pd (P(0,2,0));  tmp1 = _mm_load_pd (P(0,2,1));
  r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(0,2,2));  tmp1 = _mm_load_pd (P(0,2,3));
  r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(0,3,0));  tmp1 = _mm_load_pd (P(0,3,1));
  r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(0,3,2));  tmp1 = _mm_load_pd (P(0,3,3));
  r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[1]);
  _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[1]);

  // 3rd eighth
  tmp0 = _mm_load_pd (P(1,0,0));  tmp1 = _mm_load_pd (P(1,0,1));
  r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(1,0,2));  tmp1 = _mm_load_pd (P(1,0,3));
  r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(1,1,0));  tmp1 = _mm_load_pd (P(1,1,1));
  r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(1,1,2));  tmp1 = _mm_load_pd (P(1,1,3));
  r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[2]);
  _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[2]);

  // 4th eighth
  tmp0 = _mm_load_pd (P(1,2,0));  tmp1 = _mm_load_pd (P(1,2,1));
  r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(1,2,2));  tmp1 = _mm_load_pd (P(1,2,3));
  r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(1,3,0));  tmp1 = _mm_load_pd (P(1,3,1));
  r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(1,3,2));  tmp1 = _mm_load_pd (P(1,3,3));
  r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[3]);
  _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[3]);

  // 5th eighth
  tmp0 = _mm_load_pd (P(2,0,0));  tmp1 = _mm_load_pd (P(2,0,1));
  r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(2,0,2));  tmp1 = _mm_load_pd (P(2,0,3));
  r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(2,1,0));  tmp1 = _mm_load_pd (P(2,1,1));
  r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(2,1,2));  tmp1 = _mm_load_pd (P(2,1,3));
  r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[4]);
  _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[4]);

  // 6th eighth
  tmp0 = _mm_load_pd (P(2,2,0));  tmp1 = _mm_load_pd (P(2,2,1));
  r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(2,2,2));  tmp1 = _mm_load_pd (P(2,2,3));
  r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(2,3,0));  tmp1 = _mm_load_pd (P(2,3,1));
  r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(2,3,2));  tmp1 = _mm_load_pd (P(2,3,3));
  r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[5]);
  _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[5]);

  // 7th eighth
  tmp0 = _mm_load_pd (P(3,0,0));  tmp1 = _mm_load_pd (P(3,0,1));
  r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(3,0,2));  tmp1 = _mm_load_pd (P(3,0,3));
  r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(3,1,0));  tmp1 = _mm_load_pd (P(3,1,1));
  r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(3,1,2));  tmp1 = _mm_load_pd (P(3,1,3));
  r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[6]);
  _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[6]);

  // 8th eighth
  tmp0 = _mm_load_pd (P(3,2,0));  tmp1 = _mm_load_pd (P(3,2,1));
  r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(3,2,2));  tmp1 = _mm_load_pd (P(3,2,3));
  r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(3,3,0));  tmp1 = _mm_load_pd (P(3,3,1));
  r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  tmp0 = _mm_load_pd (P(3,3,2));  tmp1 = _mm_load_pd (P(3,3,3));
  r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
  i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
  _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[7]);
  _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[7]);
  
  // Now compute bcP, dbcP, bdcP, d2bcP, bd2cP, and dbdc products
  _MM_DDOT4_PD (b01, b23, b01, b23, cPr[0], cPr[1], cPr[2], cPr[3], bcP01r);
  _MM_DDOT4_PD (b01, b23, b01, b23, cPi[0], cPi[1], cPi[2], cPi[3], bcP01i);
  _MM_DDOT4_PD (b01, b23, b01, b23, cPr[4], cPr[5], cPr[6], cPr[7], bcP23r);
  _MM_DDOT4_PD (b01, b23, b01, b23, cPi[4], cPi[5], cPi[6], cPi[7], bcP23i);

  // Compute value
  _MM_DOT4_PD (a01, a23, bcP01r, bcP23r, *((double*)val+0));
  _MM_DOT4_PD (a01, a23, bcP01i, bcP23i, *((double*)val+1));
#undef P
}
