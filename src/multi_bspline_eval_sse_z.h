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
  _mm_store_d (&(p), t1);                                            \
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
  complex_double* restrict coefs = spline->coefs + ix*xs + iy*ys + iz;
  complex_double* restrict next_coefs = coefs + ss;

  // This macro is used to give the pointer to coefficient data.
  // i and j should be in the range [0,3].  Coefficients are read four
  // at a time, so no k value is needed.
#define P(i,j,k) (const double*)(coefs+(i)*xs+(j)*ys+(k))
  // Prefetch the data from main memory into cache so it's available
  // when we need to use it.  This is for the first spline only.
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
    a01, b01, c01, a23, b23, c23,  
    tmp0, tmp1, r0, r1, i0, i1, val_r, val_i;
  
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
  
  //b0     = __m128d_mm_movedup_pd(b01);
  a0     = _mm_unpacklo_pd(a01,a01);
  a1     = _mm_unpackhi_pd(a01,a01);
  a2     = _mm_unpacklo_pd(a23,a23);
  a3     = _mm_unpackhi_pd(a23,a23);

  b0     = _mm_unpacklo_pd(b01,b01);
  b1     = _mm_unpackhi_pd(b01,b01);
  b2     = _mm_unpacklo_pd(b23,b23);
  b3     = _mm_unpackhi_pd(b23,b23);

  abc[ 0] = _mm_mul_pd(_mm_mul_pd (b0, c01), a0);
  abc[ 1] = _mm_mul_pd(_mm_mul_pd (b0, c23), a0);
  abc[ 2] = _mm_mul_pd(_mm_mul_pd (b1, c01), a0);
  abc[ 3] = _mm_mul_pd(_mm_mul_pd (b1, c23), a0);
  abc[ 4] = _mm_mul_pd(_mm_mul_pd (b2, c01), a0);
  abc[ 5] = _mm_mul_pd(_mm_mul_pd (b2, c23), a0);
  abc[ 6] = _mm_mul_pd(_mm_mul_pd (b3, c01), a0);
  abc[ 7] = _mm_mul_pd(_mm_mul_pd (b3, c23), a0);

  abc[ 8] = _mm_mul_pd(_mm_mul_pd (b0, c01), a1);
  abc[ 9] = _mm_mul_pd(_mm_mul_pd (b0, c23), a1);
  abc[10] = _mm_mul_pd(_mm_mul_pd (b1, c01), a1);
  abc[11] = _mm_mul_pd(_mm_mul_pd (b1, c23), a1);
  abc[12] = _mm_mul_pd(_mm_mul_pd (b2, c01), a1);
  abc[13] = _mm_mul_pd(_mm_mul_pd (b2, c23), a1);
  abc[14] = _mm_mul_pd(_mm_mul_pd (b3, c01), a1);
  abc[15] = _mm_mul_pd(_mm_mul_pd (b3, c23), a1);

  abc[16] = _mm_mul_pd(_mm_mul_pd (b0, c01), a2);
  abc[17] = _mm_mul_pd(_mm_mul_pd (b0, c23), a2);
  abc[18] = _mm_mul_pd(_mm_mul_pd (b1, c01), a2);
  abc[19] = _mm_mul_pd(_mm_mul_pd (b1, c23), a2);
  abc[20] = _mm_mul_pd(_mm_mul_pd (b2, c01), a2);
  abc[21] = _mm_mul_pd(_mm_mul_pd (b2, c23), a2);
  abc[22] = _mm_mul_pd(_mm_mul_pd (b3, c01), a2);
  abc[23] = _mm_mul_pd(_mm_mul_pd (b3, c23), a2);

  abc[24] = _mm_mul_pd(_mm_mul_pd (b0, c01), a3);
  abc[25] = _mm_mul_pd(_mm_mul_pd (b0, c23), a3);
  abc[26] = _mm_mul_pd(_mm_mul_pd (b1, c01), a3);
  abc[27] = _mm_mul_pd(_mm_mul_pd (b1, c23), a3);
  abc[28] = _mm_mul_pd(_mm_mul_pd (b2, c01), a3);
  abc[29] = _mm_mul_pd(_mm_mul_pd (b2, c23), a3);
  abc[30] = _mm_mul_pd(_mm_mul_pd (b3, c01), a3);
  abc[31] = _mm_mul_pd(_mm_mul_pd (b3, c23), a3);

  // Now loop over all splines
#define nextP(i,j,k) (const double*)(next_coefs+(i)*xs+(j)*ys+(k))
  for (int si=0; si<spline->num_splines; si++) {
    // Prefetch next set of coefficients
    next_coefs = coefs + 1*spline->spline_stride;
    _mm_prefetch((const char*)nextP(0,0,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(0,0,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(0,1,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(0,1,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(0,2,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(0,2,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(0,3,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(0,3,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(1,0,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(1,0,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(1,1,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(1,1,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(1,2,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(1,2,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(1,3,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(1,3,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(2,0,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(2,0,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(2,1,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(2,1,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(2,2,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(2,2,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(2,3,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(2,3,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(3,0,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(3,0,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(3,1,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(3,1,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(3,2,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(3,2,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(3,3,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(3,3,2), _MM_HINT_T0);

    // The fence appears to slow things down considerably
    // asm volatile ("mfence");
    // Now compute value
    int index = 0;
    val_r = _mm_sub_pd (val_r, val_r);
    val_i = _mm_sub_pd (val_i, val_i);
    for (int i=0; i<4; i++) 
      for (int j=0; j<4; j++) {
	tmp0 = _mm_load_pd (P(i,j,0));  tmp1 = _mm_load_pd (P(i,j,1));
	r0 = _mm_unpacklo_pd (tmp0, tmp1);
	i0 = _mm_unpackhi_pd (tmp0, tmp1);
	tmp0 = _mm_load_pd (P(i,j,2));  tmp1 = _mm_load_pd (P(i,j,3));	
	r1 = _mm_unpacklo_pd (tmp0, tmp1);
	i1 = _mm_unpackhi_pd (tmp0, tmp1);
	//r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
	//i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
	//r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
	//i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
	
    	val_r = _mm_add_pd (val_r, _mm_mul_pd (r0, abc[index+0]));
	val_i = _mm_add_pd (val_i, _mm_mul_pd (i0, abc[index+0]));
    	val_r = _mm_add_pd (val_r, _mm_mul_pd (r1, abc[index+1]));
	val_i = _mm_add_pd (val_i, _mm_mul_pd (i1, abc[index+1]));
	
	index += 2;
      }
    
    //_mm_storeu_pd((double*)(vals+si),_mm_hadd_pd(val_r, val_i));
    _mm_store_pd((double*)(vals+si),_mm_hadd_pd(val_r, val_i));
    
    coefs += spline->spline_stride;
  }


  // Compute cP, dcP, and d2cP products 1/8 at a time to maximize
  // register reuse and avoid rerereading from memory or cache.
  // Complex values are read in, then shuffled such that 4 registers
  // hold the read parts and 4 register hold the imaginary parts.
  // 1st eighth
//   tmp0 = _mm_load_pd (P(0,0,0));  tmp1 = _mm_load_pd (P(0,0,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,0,2));  tmp1 = _mm_load_pd (P(0,0,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,1,0));  tmp1 = _mm_load_pd (P(0,1,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,1,2));  tmp1 = _mm_load_pd (P(0,1,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[0]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[0]);
  
//   // 2nd eighth
//   tmp0 = _mm_load_pd (P(0,2,0));  tmp1 = _mm_load_pd (P(0,2,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,2,2));  tmp1 = _mm_load_pd (P(0,2,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,3,0));  tmp1 = _mm_load_pd (P(0,3,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,3,2));  tmp1 = _mm_load_pd (P(0,3,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[1]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[1]);

//   // 3rd eighth
//   tmp0 = _mm_load_pd (P(1,0,0));  tmp1 = _mm_load_pd (P(1,0,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,0,2));  tmp1 = _mm_load_pd (P(1,0,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,1,0));  tmp1 = _mm_load_pd (P(1,1,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,1,2));  tmp1 = _mm_load_pd (P(1,1,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[2]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[2]);

//   // 4th eighth
//   tmp0 = _mm_load_pd (P(1,2,0));  tmp1 = _mm_load_pd (P(1,2,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,2,2));  tmp1 = _mm_load_pd (P(1,2,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,3,0));  tmp1 = _mm_load_pd (P(1,3,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,3,2));  tmp1 = _mm_load_pd (P(1,3,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[3]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[3]);

//   // 5th eighth
//   tmp0 = _mm_load_pd (P(2,0,0));  tmp1 = _mm_load_pd (P(2,0,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,0,2));  tmp1 = _mm_load_pd (P(2,0,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,1,0));  tmp1 = _mm_load_pd (P(2,1,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,1,2));  tmp1 = _mm_load_pd (P(2,1,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[4]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[4]);

//   // 6th eighth
//   tmp0 = _mm_load_pd (P(2,2,0));  tmp1 = _mm_load_pd (P(2,2,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,2,2));  tmp1 = _mm_load_pd (P(2,2,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,3,0));  tmp1 = _mm_load_pd (P(2,3,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,3,2));  tmp1 = _mm_load_pd (P(2,3,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[5]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[5]);

//   // 7th eighth
//   tmp0 = _mm_load_pd (P(3,0,0));  tmp1 = _mm_load_pd (P(3,0,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,0,2));  tmp1 = _mm_load_pd (P(3,0,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,1,0));  tmp1 = _mm_load_pd (P(3,1,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,1,2));  tmp1 = _mm_load_pd (P(3,1,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[6]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[6]);

//   // 8th eighth
//   tmp0 = _mm_load_pd (P(3,2,0));  tmp1 = _mm_load_pd (P(3,2,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,2,2));  tmp1 = _mm_load_pd (P(3,2,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,3,0));  tmp1 = _mm_load_pd (P(3,3,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,3,2));  tmp1 = _mm_load_pd (P(3,3,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[7]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[7]);
  
//   // Now compute bcP, dbcP, bdcP, d2bcP, bd2cP, and dbdc products
//   _MM_DDOT4_PD (b01, b23, b01, b23, cPr[0], cPr[1], cPr[2], cPr[3], bcP01r);
//   _MM_DDOT4_PD (b01, b23, b01, b23, cPi[0], cPi[1], cPi[2], cPi[3], bcP01i);
//   _MM_DDOT4_PD (b01, b23, b01, b23, cPr[4], cPr[5], cPr[6], cPr[7], bcP23r);
//   _MM_DDOT4_PD (b01, b23, b01, b23, cPi[4], cPi[5], cPi[6], cPi[7], bcP23i);

//   // Compute value
//   _MM_DOT4_PD (a01, a23, bcP01r, bcP23r, *((double*)val+0));
//  _MM_DOT4_PD (a01, a23, bcP01i, bcP23i, *((double*)val+1));
#undef P
#undef nextP
}


inline void
eval_multi_UBspline_3d_z_vgh (multi_UBspline_3d_z *spline,
			      double x, double y, double z,
			      complex_double* restrict vals,
			      complex_double* restrict grads,
			      complex_double* restrict hess)
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
  complex_double* restrict coefs = spline->coefs + ix*xs + iy*ys + iz;
  complex_double* restrict next_coefs = coefs + ss;

  // This macro is used to give the pointer to coefficient data.
  // i and j should be in the range [0,3].  Coefficients are read four
  // at a time, so no k value is needed.
#define P(i,j,k) (const double*)(coefs+(i)*xs+(j)*ys+(k))
  // Prefetch the data from main memory into cache so it's available
  // when we need to use it.  This is for the first spline only.
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
    a01, b01, c01, a23, b23, c23, 
    da01, db01, dc01, da23, db23, dc23,
    d2a01, d2b01, d2c01, d2a23, d2b23, d2c23,
    tmp0, tmp1, r0, r1, i0, i1;
  
  tpx01 = _mm_set_pd (tx*tx*tx, tx*tx);
  tpx23 = _mm_set_pd (tx, 1.0);
  tpy01 = _mm_set_pd (ty*ty*ty, ty*ty);
  tpy23 = _mm_set_pd (ty, 1.0);
  tpz01 = _mm_set_pd (tz*tz*tz, tz*tz);
  tpz23 = _mm_set_pd (tz, 1.0);


  // x-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpx01, tpx23, tpx01, tpx23,   a01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpx01, tpx23, tpx01, tpx23,   a23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpx01, tpx23, tpx01, tpx23,  da01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpx01, tpx23, tpx01, tpx23,  da23);
  _MM_DDOT4_PD (A_d[16], A_d[17], A_d[18], A_d[19], tpx01, tpx23, tpx01, tpx23, d2a01);
  _MM_DDOT4_PD (A_d[20], A_d[21], A_d[22], A_d[23], tpx01, tpx23, tpx01, tpx23, d2a23);

  // y-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpy01, tpy23, tpy01, tpy23,   b01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpy01, tpy23, tpy01, tpy23,   b23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpy01, tpy23, tpy01, tpy23,  db01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpy01, tpy23, tpy01, tpy23,  db23);
  _MM_DDOT4_PD (A_d[16], A_d[17], A_d[18], A_d[19], tpy01, tpy23, tpy01, tpy23, d2b01);
  _MM_DDOT4_PD (A_d[20], A_d[21], A_d[22], A_d[23], tpy01, tpy23, tpy01, tpy23, d2b23);


  // z-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpz01, tpz23, tpz01, tpz23,   c01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpz01, tpz23, tpz01, tpz23,   c23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpz01, tpz23, tpz01, tpz23,  dc01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpz01, tpz23, tpz01, tpz23,  dc23);
  _MM_DDOT4_PD (A_d[16], A_d[17], A_d[18], A_d[19], tpz01, tpz23, tpz01, tpz23, d2c01);
  _MM_DDOT4_PD (A_d[20], A_d[21], A_d[22], A_d[23], tpz01, tpz23, tpz01, tpz23, d2c23);

  // Now compute tensor product of a, b, and c and derivatives
  // Components of abc are:
  // 20*i +  0:   a   b   c01      20*i + 10:   a   b   c23
  // 20*i +  1:  da   b   c01 	   20*i + 11:  da   b   c23
  // 20*i +  2:   a  db   c01	   20*i + 12:   a  db   c23
  // 20*i +  3:   a   b  dc01	   20*i + 13:   a   b  dc23
  // 20*i +  4: d2a   b   c01	   20*i + 14: d2a   b   c23
  // 20*i +  5:  da  db   c01	   20*i + 15:  da  db   c23
  // 20*i +  6:  da   b  dc01	   20*i + 16:  da   b  dc23
  // 20*i +  7:   a d2b   c01	   20*i + 17:   a d2b   c23
  // 20*i +  8:   a  db  dc01 	   20*i + 18:   a  db  dc23
  // 20*i +  9:   a   b d2c01	   20*i + 19:   a   b d2c23
  __m128d abc[320];
  __m128d   a0,   a1,   a2,   a3,   b0,   b1,   b2,   b3;
  __m128d  da0,  da1,  da2,  da3,  db0,  db1,  db2,  db3;
  __m128d d2a0, d2a1, d2a2, d2a3, d2b0, d2b1, d2b2, d2b3;
  
  //b0     = __m128d_mm_movedup_pd(b01);
  a0 = _mm_unpacklo_pd(a01,a01); da0 = _mm_unpacklo_pd(da01,da01); d2a0 = _mm_unpacklo_pd(d2a01,d2a01);
  a1 = _mm_unpackhi_pd(a01,a01); da1 = _mm_unpackhi_pd(da01,da01); d2a1 = _mm_unpackhi_pd(d2a01,d2a01);
  a2 = _mm_unpacklo_pd(a23,a23); da2 = _mm_unpacklo_pd(da23,da23); d2a2 = _mm_unpacklo_pd(d2a23,d2a23);
  a3 = _mm_unpackhi_pd(a23,a23); da3 = _mm_unpackhi_pd(da23,da23); d2a3 = _mm_unpackhi_pd(d2a23,d2a23);

  b0 = _mm_unpacklo_pd(b01,b01); db0 = _mm_unpacklo_pd(db01,db01); d2b0 = _mm_unpacklo_pd(d2b01,d2b01);
  b1 = _mm_unpackhi_pd(b01,b01); db1 = _mm_unpackhi_pd(db01,db01); d2b1 = _mm_unpackhi_pd(d2b01,d2b01);
  b2 = _mm_unpacklo_pd(b23,b23); db2 = _mm_unpacklo_pd(db23,db23); d2b2 = _mm_unpacklo_pd(d2b23,d2b23);
  b3 = _mm_unpackhi_pd(b23,b23); db3 = _mm_unpackhi_pd(db23,db23); d2b3 = _mm_unpackhi_pd(d2b23,d2b23);

  abc[  0] = _mm_mul_pd(_mm_mul_pd (  a0,   b0),   c01);
  abc[  1] = _mm_mul_pd(_mm_mul_pd ( da0,   b0),   c01);
  abc[  2] = _mm_mul_pd(_mm_mul_pd (  a0,  db0),   c01);
  abc[  3] = _mm_mul_pd(_mm_mul_pd (  a0,   b0),  dc01);
  abc[  4] = _mm_mul_pd(_mm_mul_pd (d2a0,   b0),   c01);
  abc[  5] = _mm_mul_pd(_mm_mul_pd ( da0,  db0),   c01);
  abc[  6] = _mm_mul_pd(_mm_mul_pd ( da0,   b0),  dc01);
  abc[  7] = _mm_mul_pd(_mm_mul_pd (  a0, d2b0),   c01);
  abc[  8] = _mm_mul_pd(_mm_mul_pd (  a0,  db0),  dc01);
  abc[  9] = _mm_mul_pd(_mm_mul_pd (  a0,   b0), d2c01);
  abc[ 10] = _mm_mul_pd(_mm_mul_pd (  a0,   b0),   c23);
  abc[ 11] = _mm_mul_pd(_mm_mul_pd ( da0,   b0),   c23);
  abc[ 12] = _mm_mul_pd(_mm_mul_pd (  a0,  db0),   c23);
  abc[ 13] = _mm_mul_pd(_mm_mul_pd (  a0,   b0),  dc23);
  abc[ 14] = _mm_mul_pd(_mm_mul_pd (d2a0,   b0),   c23);
  abc[ 15] = _mm_mul_pd(_mm_mul_pd ( da0,  db0),   c23);
  abc[ 16] = _mm_mul_pd(_mm_mul_pd ( da0,   b0),  dc23);
  abc[ 17] = _mm_mul_pd(_mm_mul_pd (  a0, d2b0),   c23);
  abc[ 18] = _mm_mul_pd(_mm_mul_pd (  a0,  db0),  dc23);
  abc[ 19] = _mm_mul_pd(_mm_mul_pd (  a0,   b0), d2c23);

  abc[ 20] = _mm_mul_pd(_mm_mul_pd (  a0,   b1),   c01);
  abc[ 21] = _mm_mul_pd(_mm_mul_pd ( da0,   b1),   c01);
  abc[ 22] = _mm_mul_pd(_mm_mul_pd (  a0,  db1),   c01);
  abc[ 23] = _mm_mul_pd(_mm_mul_pd (  a0,   b1),  dc01);
  abc[ 24] = _mm_mul_pd(_mm_mul_pd (d2a0,   b1),   c01);
  abc[ 25] = _mm_mul_pd(_mm_mul_pd ( da0,  db1),   c01);
  abc[ 26] = _mm_mul_pd(_mm_mul_pd ( da0,   b1),  dc01);
  abc[ 27] = _mm_mul_pd(_mm_mul_pd (  a0, d2b1),   c01);
  abc[ 28] = _mm_mul_pd(_mm_mul_pd (  a0,  db1),  dc01);
  abc[ 29] = _mm_mul_pd(_mm_mul_pd (  a0,   b1), d2c01);
  abc[ 30] = _mm_mul_pd(_mm_mul_pd (  a0,   b1),   c23);
  abc[ 31] = _mm_mul_pd(_mm_mul_pd ( da0,   b1),   c23);
  abc[ 32] = _mm_mul_pd(_mm_mul_pd (  a0,  db1),   c23);
  abc[ 33] = _mm_mul_pd(_mm_mul_pd (  a0,   b1),  dc23);
  abc[ 34] = _mm_mul_pd(_mm_mul_pd (d2a0,   b1),   c23);
  abc[ 35] = _mm_mul_pd(_mm_mul_pd ( da0,  db1),   c23);
  abc[ 36] = _mm_mul_pd(_mm_mul_pd ( da0,   b1),  dc23);
  abc[ 37] = _mm_mul_pd(_mm_mul_pd (  a0, d2b1),   c23);
  abc[ 38] = _mm_mul_pd(_mm_mul_pd (  a0,  db1),  dc23);
  abc[ 39] = _mm_mul_pd(_mm_mul_pd (  a0,   b1), d2c23);

  abc[ 40] = _mm_mul_pd(_mm_mul_pd (  a0,   b2),   c01);
  abc[ 41] = _mm_mul_pd(_mm_mul_pd ( da0,   b2),   c01);
  abc[ 42] = _mm_mul_pd(_mm_mul_pd (  a0,  db2),   c01);
  abc[ 43] = _mm_mul_pd(_mm_mul_pd (  a0,   b2),  dc01);
  abc[ 44] = _mm_mul_pd(_mm_mul_pd (d2a0,   b2),   c01);
  abc[ 45] = _mm_mul_pd(_mm_mul_pd ( da0,  db2),   c01);
  abc[ 46] = _mm_mul_pd(_mm_mul_pd ( da0,   b2),  dc01);
  abc[ 47] = _mm_mul_pd(_mm_mul_pd (  a0, d2b2),   c01);
  abc[ 48] = _mm_mul_pd(_mm_mul_pd (  a0,  db2),  dc01);
  abc[ 49] = _mm_mul_pd(_mm_mul_pd (  a0,   b2), d2c01);
  abc[ 50] = _mm_mul_pd(_mm_mul_pd (  a0,   b2),   c23);
  abc[ 51] = _mm_mul_pd(_mm_mul_pd ( da0,   b2),   c23);
  abc[ 52] = _mm_mul_pd(_mm_mul_pd (  a0,  db2),   c23);
  abc[ 53] = _mm_mul_pd(_mm_mul_pd (  a0,   b2),  dc23);
  abc[ 54] = _mm_mul_pd(_mm_mul_pd (d2a0,   b2),   c23);
  abc[ 55] = _mm_mul_pd(_mm_mul_pd ( da0,  db2),   c23);
  abc[ 56] = _mm_mul_pd(_mm_mul_pd ( da0,   b2),  dc23);
  abc[ 57] = _mm_mul_pd(_mm_mul_pd (  a0, d2b2),   c23);
  abc[ 58] = _mm_mul_pd(_mm_mul_pd (  a0,  db2),  dc23);
  abc[ 59] = _mm_mul_pd(_mm_mul_pd (  a0,   b2), d2c23);

  abc[ 60] = _mm_mul_pd(_mm_mul_pd (  a0,   b3),   c01);
  abc[ 61] = _mm_mul_pd(_mm_mul_pd ( da0,   b3),   c01);
  abc[ 62] = _mm_mul_pd(_mm_mul_pd (  a0,  db3),   c01);
  abc[ 63] = _mm_mul_pd(_mm_mul_pd (  a0,   b3),  dc01);
  abc[ 64] = _mm_mul_pd(_mm_mul_pd (d2a0,   b3),   c01);
  abc[ 65] = _mm_mul_pd(_mm_mul_pd ( da0,  db3),   c01);
  abc[ 66] = _mm_mul_pd(_mm_mul_pd ( da0,   b3),  dc01);
  abc[ 67] = _mm_mul_pd(_mm_mul_pd (  a0, d2b3),   c01);
  abc[ 68] = _mm_mul_pd(_mm_mul_pd (  a0,  db3),  dc01);
  abc[ 69] = _mm_mul_pd(_mm_mul_pd (  a0,   b3), d2c01);
  abc[ 70] = _mm_mul_pd(_mm_mul_pd (  a0,   b3),   c23);
  abc[ 71] = _mm_mul_pd(_mm_mul_pd ( da0,   b3),   c23);
  abc[ 72] = _mm_mul_pd(_mm_mul_pd (  a0,  db3),   c23);
  abc[ 73] = _mm_mul_pd(_mm_mul_pd (  a0,   b3),  dc23);
  abc[ 74] = _mm_mul_pd(_mm_mul_pd (d2a0,   b3),   c23);
  abc[ 75] = _mm_mul_pd(_mm_mul_pd ( da0,  db3),   c23);
  abc[ 76] = _mm_mul_pd(_mm_mul_pd ( da0,   b3),  dc23);
  abc[ 77] = _mm_mul_pd(_mm_mul_pd (  a0, d2b3),   c23);
  abc[ 78] = _mm_mul_pd(_mm_mul_pd (  a0,  db3),  dc23);
  abc[ 79] = _mm_mul_pd(_mm_mul_pd (  a0,   b3), d2c23);

  abc[ 80] = _mm_mul_pd(_mm_mul_pd (  a1,   b0),   c01);
  abc[ 81] = _mm_mul_pd(_mm_mul_pd ( da1,   b0),   c01);
  abc[ 82] = _mm_mul_pd(_mm_mul_pd (  a1,  db0),   c01);
  abc[ 83] = _mm_mul_pd(_mm_mul_pd (  a1,   b0),  dc01);
  abc[ 84] = _mm_mul_pd(_mm_mul_pd (d2a1,   b0),   c01);
  abc[ 85] = _mm_mul_pd(_mm_mul_pd ( da1,  db0),   c01);
  abc[ 86] = _mm_mul_pd(_mm_mul_pd ( da1,   b0),  dc01);
  abc[ 87] = _mm_mul_pd(_mm_mul_pd (  a1, d2b0),   c01);
  abc[ 88] = _mm_mul_pd(_mm_mul_pd (  a1,  db0),  dc01);
  abc[ 89] = _mm_mul_pd(_mm_mul_pd (  a1,   b0), d2c01);
  abc[ 90] = _mm_mul_pd(_mm_mul_pd (  a1,   b0),   c23);
  abc[ 91] = _mm_mul_pd(_mm_mul_pd ( da1,   b0),   c23);
  abc[ 92] = _mm_mul_pd(_mm_mul_pd (  a1,  db0),   c23);
  abc[ 93] = _mm_mul_pd(_mm_mul_pd (  a1,   b0),  dc23);
  abc[ 94] = _mm_mul_pd(_mm_mul_pd (d2a1,   b0),   c23);
  abc[ 95] = _mm_mul_pd(_mm_mul_pd ( da1,  db0),   c23);
  abc[ 96] = _mm_mul_pd(_mm_mul_pd ( da1,   b0),  dc23);
  abc[ 97] = _mm_mul_pd(_mm_mul_pd (  a1, d2b0),   c23);
  abc[ 98] = _mm_mul_pd(_mm_mul_pd (  a1,  db0),  dc23);
  abc[ 99] = _mm_mul_pd(_mm_mul_pd (  a1,   b0), d2c23);

  abc[100] = _mm_mul_pd(_mm_mul_pd (  a1,   b1),   c01);
  abc[101] = _mm_mul_pd(_mm_mul_pd ( da1,   b1),   c01);
  abc[102] = _mm_mul_pd(_mm_mul_pd (  a1,  db1),   c01);
  abc[103] = _mm_mul_pd(_mm_mul_pd (  a1,   b1),  dc01);
  abc[104] = _mm_mul_pd(_mm_mul_pd (d2a1,   b1),   c01);
  abc[105] = _mm_mul_pd(_mm_mul_pd ( da1,  db1),   c01);
  abc[106] = _mm_mul_pd(_mm_mul_pd ( da1,   b1),  dc01);
  abc[107] = _mm_mul_pd(_mm_mul_pd (  a1, d2b1),   c01);
  abc[108] = _mm_mul_pd(_mm_mul_pd (  a1,  db1),  dc01);
  abc[109] = _mm_mul_pd(_mm_mul_pd (  a1,   b1), d2c01);
  abc[110] = _mm_mul_pd(_mm_mul_pd (  a1,   b1),   c23);
  abc[111] = _mm_mul_pd(_mm_mul_pd ( da1,   b1),   c23);
  abc[112] = _mm_mul_pd(_mm_mul_pd (  a1,  db1),   c23);
  abc[113] = _mm_mul_pd(_mm_mul_pd (  a1,   b1),  dc23);
  abc[114] = _mm_mul_pd(_mm_mul_pd (d2a1,   b1),   c23);
  abc[115] = _mm_mul_pd(_mm_mul_pd ( da1,  db1),   c23);
  abc[116] = _mm_mul_pd(_mm_mul_pd ( da1,   b1),  dc23);
  abc[117] = _mm_mul_pd(_mm_mul_pd (  a1, d2b1),   c23);
  abc[118] = _mm_mul_pd(_mm_mul_pd (  a1,  db1),  dc23);
  abc[119] = _mm_mul_pd(_mm_mul_pd (  a1,   b1), d2c23);

  abc[120] = _mm_mul_pd(_mm_mul_pd (  a1,   b2),   c01);
  abc[121] = _mm_mul_pd(_mm_mul_pd ( da1,   b2),   c01);
  abc[122] = _mm_mul_pd(_mm_mul_pd (  a1,  db2),   c01);
  abc[123] = _mm_mul_pd(_mm_mul_pd (  a1,   b2),  dc01);
  abc[124] = _mm_mul_pd(_mm_mul_pd (d2a1,   b2),   c01);
  abc[125] = _mm_mul_pd(_mm_mul_pd ( da1,  db2),   c01);
  abc[126] = _mm_mul_pd(_mm_mul_pd ( da1,   b2),  dc01);
  abc[127] = _mm_mul_pd(_mm_mul_pd (  a1, d2b2),   c01);
  abc[128] = _mm_mul_pd(_mm_mul_pd (  a1,  db2),  dc01);
  abc[129] = _mm_mul_pd(_mm_mul_pd (  a1,   b2), d2c01);
  abc[130] = _mm_mul_pd(_mm_mul_pd (  a1,   b2),   c23);
  abc[131] = _mm_mul_pd(_mm_mul_pd ( da1,   b2),   c23);
  abc[132] = _mm_mul_pd(_mm_mul_pd (  a1,  db2),   c23);
  abc[133] = _mm_mul_pd(_mm_mul_pd (  a1,   b2),  dc23);
  abc[134] = _mm_mul_pd(_mm_mul_pd (d2a1,   b2),   c23);
  abc[135] = _mm_mul_pd(_mm_mul_pd ( da1,  db2),   c23);
  abc[136] = _mm_mul_pd(_mm_mul_pd ( da1,   b2),  dc23);
  abc[137] = _mm_mul_pd(_mm_mul_pd (  a1, d2b2),   c23);
  abc[138] = _mm_mul_pd(_mm_mul_pd (  a1,  db2),  dc23);
  abc[139] = _mm_mul_pd(_mm_mul_pd (  a1,   b2), d2c23);

  abc[140] = _mm_mul_pd(_mm_mul_pd (  a1,   b3),   c01);
  abc[141] = _mm_mul_pd(_mm_mul_pd ( da1,   b3),   c01);
  abc[142] = _mm_mul_pd(_mm_mul_pd (  a1,  db3),   c01);
  abc[143] = _mm_mul_pd(_mm_mul_pd (  a1,   b3),  dc01);
  abc[144] = _mm_mul_pd(_mm_mul_pd (d2a1,   b3),   c01);
  abc[145] = _mm_mul_pd(_mm_mul_pd ( da1,  db3),   c01);
  abc[146] = _mm_mul_pd(_mm_mul_pd ( da1,   b3),  dc01);
  abc[147] = _mm_mul_pd(_mm_mul_pd (  a1, d2b3),   c01);
  abc[148] = _mm_mul_pd(_mm_mul_pd (  a1,  db3),  dc01);
  abc[149] = _mm_mul_pd(_mm_mul_pd (  a1,   b3), d2c01);
  abc[150] = _mm_mul_pd(_mm_mul_pd (  a1,   b3),   c23);
  abc[151] = _mm_mul_pd(_mm_mul_pd ( da1,   b3),   c23);
  abc[152] = _mm_mul_pd(_mm_mul_pd (  a1,  db3),   c23);
  abc[153] = _mm_mul_pd(_mm_mul_pd (  a1,   b3),  dc23);
  abc[154] = _mm_mul_pd(_mm_mul_pd (d2a1,   b3),   c23);
  abc[155] = _mm_mul_pd(_mm_mul_pd ( da1,  db3),   c23);
  abc[156] = _mm_mul_pd(_mm_mul_pd ( da1,   b3),  dc23);
  abc[157] = _mm_mul_pd(_mm_mul_pd (  a1, d2b3),   c23);
  abc[158] = _mm_mul_pd(_mm_mul_pd (  a1,  db3),  dc23);
  abc[159] = _mm_mul_pd(_mm_mul_pd (  a1,   b3), d2c23);

  abc[160] = _mm_mul_pd(_mm_mul_pd (  a2,   b0),   c01);
  abc[161] = _mm_mul_pd(_mm_mul_pd ( da2,   b0),   c01);
  abc[162] = _mm_mul_pd(_mm_mul_pd (  a2,  db0),   c01);
  abc[163] = _mm_mul_pd(_mm_mul_pd (  a2,   b0),  dc01);
  abc[164] = _mm_mul_pd(_mm_mul_pd (d2a2,   b0),   c01);
  abc[165] = _mm_mul_pd(_mm_mul_pd ( da2,  db0),   c01);
  abc[166] = _mm_mul_pd(_mm_mul_pd ( da2,   b0),  dc01);
  abc[167] = _mm_mul_pd(_mm_mul_pd (  a2, d2b0),   c01);
  abc[168] = _mm_mul_pd(_mm_mul_pd (  a2,  db0),  dc01);
  abc[169] = _mm_mul_pd(_mm_mul_pd (  a2,   b0), d2c01);
  abc[170] = _mm_mul_pd(_mm_mul_pd (  a2,   b0),   c23);
  abc[171] = _mm_mul_pd(_mm_mul_pd ( da2,   b0),   c23);
  abc[172] = _mm_mul_pd(_mm_mul_pd (  a2,  db0),   c23);
  abc[173] = _mm_mul_pd(_mm_mul_pd (  a2,   b0),  dc23);
  abc[174] = _mm_mul_pd(_mm_mul_pd (d2a2,   b0),   c23);
  abc[175] = _mm_mul_pd(_mm_mul_pd ( da2,  db0),   c23);
  abc[176] = _mm_mul_pd(_mm_mul_pd ( da2,   b0),  dc23);
  abc[177] = _mm_mul_pd(_mm_mul_pd (  a2, d2b0),   c23);
  abc[178] = _mm_mul_pd(_mm_mul_pd (  a2,  db0),  dc23);
  abc[179] = _mm_mul_pd(_mm_mul_pd (  a2,   b0), d2c23);

  abc[180] = _mm_mul_pd(_mm_mul_pd (  a2,   b1),   c01);
  abc[181] = _mm_mul_pd(_mm_mul_pd ( da2,   b1),   c01);
  abc[182] = _mm_mul_pd(_mm_mul_pd (  a2,  db1),   c01);
  abc[183] = _mm_mul_pd(_mm_mul_pd (  a2,   b1),  dc01);
  abc[184] = _mm_mul_pd(_mm_mul_pd (d2a2,   b1),   c01);
  abc[185] = _mm_mul_pd(_mm_mul_pd ( da2,  db1),   c01);
  abc[186] = _mm_mul_pd(_mm_mul_pd ( da2,   b1),  dc01);
  abc[187] = _mm_mul_pd(_mm_mul_pd (  a2, d2b1),   c01);
  abc[188] = _mm_mul_pd(_mm_mul_pd (  a2,  db1),  dc01);
  abc[189] = _mm_mul_pd(_mm_mul_pd (  a2,   b1), d2c01);
  abc[190] = _mm_mul_pd(_mm_mul_pd (  a2,   b1),   c23);
  abc[191] = _mm_mul_pd(_mm_mul_pd ( da2,   b1),   c23);
  abc[192] = _mm_mul_pd(_mm_mul_pd (  a2,  db1),   c23);
  abc[193] = _mm_mul_pd(_mm_mul_pd (  a2,   b1),  dc23);
  abc[194] = _mm_mul_pd(_mm_mul_pd (d2a2,   b1),   c23);
  abc[195] = _mm_mul_pd(_mm_mul_pd ( da2,  db1),   c23);
  abc[196] = _mm_mul_pd(_mm_mul_pd ( da2,   b1),  dc23);
  abc[197] = _mm_mul_pd(_mm_mul_pd (  a2, d2b1),   c23);
  abc[198] = _mm_mul_pd(_mm_mul_pd (  a2,  db1),  dc23);
  abc[199] = _mm_mul_pd(_mm_mul_pd (  a2,   b1), d2c23);

  abc[200] = _mm_mul_pd(_mm_mul_pd (  a2,   b2),   c01);
  abc[201] = _mm_mul_pd(_mm_mul_pd ( da2,   b2),   c01);
  abc[202] = _mm_mul_pd(_mm_mul_pd (  a2,  db2),   c01);
  abc[203] = _mm_mul_pd(_mm_mul_pd (  a2,   b2),  dc01);
  abc[204] = _mm_mul_pd(_mm_mul_pd (d2a2,   b2),   c01);
  abc[205] = _mm_mul_pd(_mm_mul_pd ( da2,  db2),   c01);
  abc[206] = _mm_mul_pd(_mm_mul_pd ( da2,   b2),  dc01);
  abc[207] = _mm_mul_pd(_mm_mul_pd (  a2, d2b2),   c01);
  abc[208] = _mm_mul_pd(_mm_mul_pd (  a2,  db2),  dc01);
  abc[209] = _mm_mul_pd(_mm_mul_pd (  a2,   b2), d2c01);
  abc[210] = _mm_mul_pd(_mm_mul_pd (  a2,   b2),   c23);
  abc[211] = _mm_mul_pd(_mm_mul_pd ( da2,   b2),   c23);
  abc[212] = _mm_mul_pd(_mm_mul_pd (  a2,  db2),   c23);
  abc[213] = _mm_mul_pd(_mm_mul_pd (  a2,   b2),  dc23);
  abc[214] = _mm_mul_pd(_mm_mul_pd (d2a2,   b2),   c23);
  abc[215] = _mm_mul_pd(_mm_mul_pd ( da2,  db2),   c23);
  abc[216] = _mm_mul_pd(_mm_mul_pd ( da2,   b2),  dc23);
  abc[217] = _mm_mul_pd(_mm_mul_pd (  a2, d2b2),   c23);
  abc[218] = _mm_mul_pd(_mm_mul_pd (  a2,  db2),  dc23);
  abc[219] = _mm_mul_pd(_mm_mul_pd (  a2,   b2), d2c23);

  abc[220] = _mm_mul_pd(_mm_mul_pd (  a2,   b3),   c01);
  abc[221] = _mm_mul_pd(_mm_mul_pd ( da2,   b3),   c01);
  abc[222] = _mm_mul_pd(_mm_mul_pd (  a2,  db3),   c01);
  abc[223] = _mm_mul_pd(_mm_mul_pd (  a2,   b3),  dc01);
  abc[224] = _mm_mul_pd(_mm_mul_pd (d2a2,   b3),   c01);
  abc[225] = _mm_mul_pd(_mm_mul_pd ( da2,  db3),   c01);
  abc[226] = _mm_mul_pd(_mm_mul_pd ( da2,   b3),  dc01);
  abc[227] = _mm_mul_pd(_mm_mul_pd (  a2, d2b3),   c01);
  abc[228] = _mm_mul_pd(_mm_mul_pd (  a2,  db3),  dc01);
  abc[229] = _mm_mul_pd(_mm_mul_pd (  a2,   b3), d2c01);
  abc[230] = _mm_mul_pd(_mm_mul_pd (  a2,   b3),   c23);
  abc[231] = _mm_mul_pd(_mm_mul_pd ( da2,   b3),   c23);
  abc[232] = _mm_mul_pd(_mm_mul_pd (  a2,  db3),   c23);
  abc[233] = _mm_mul_pd(_mm_mul_pd (  a2,   b3),  dc23);
  abc[234] = _mm_mul_pd(_mm_mul_pd (d2a2,   b3),   c23);
  abc[235] = _mm_mul_pd(_mm_mul_pd ( da2,  db3),   c23);
  abc[236] = _mm_mul_pd(_mm_mul_pd ( da2,   b3),  dc23);
  abc[237] = _mm_mul_pd(_mm_mul_pd (  a2, d2b3),   c23);
  abc[238] = _mm_mul_pd(_mm_mul_pd (  a2,  db3),  dc23);
  abc[239] = _mm_mul_pd(_mm_mul_pd (  a2,   b3), d2c23);

  abc[240] = _mm_mul_pd(_mm_mul_pd (  a3,   b0),   c01);
  abc[241] = _mm_mul_pd(_mm_mul_pd ( da3,   b0),   c01);
  abc[242] = _mm_mul_pd(_mm_mul_pd (  a3,  db0),   c01);
  abc[243] = _mm_mul_pd(_mm_mul_pd (  a3,   b0),  dc01);
  abc[244] = _mm_mul_pd(_mm_mul_pd (d2a3,   b0),   c01);
  abc[245] = _mm_mul_pd(_mm_mul_pd ( da3,  db0),   c01);
  abc[246] = _mm_mul_pd(_mm_mul_pd ( da3,   b0),  dc01);
  abc[247] = _mm_mul_pd(_mm_mul_pd (  a3, d2b0),   c01);
  abc[248] = _mm_mul_pd(_mm_mul_pd (  a3,  db0),  dc01);
  abc[249] = _mm_mul_pd(_mm_mul_pd (  a3,   b0), d2c01);
  abc[250] = _mm_mul_pd(_mm_mul_pd (  a3,   b0),   c23);
  abc[251] = _mm_mul_pd(_mm_mul_pd ( da3,   b0),   c23);
  abc[252] = _mm_mul_pd(_mm_mul_pd (  a3,  db0),   c23);
  abc[253] = _mm_mul_pd(_mm_mul_pd (  a3,   b0),  dc23);
  abc[254] = _mm_mul_pd(_mm_mul_pd (d2a3,   b0),   c23);
  abc[255] = _mm_mul_pd(_mm_mul_pd ( da3,  db0),   c23);
  abc[256] = _mm_mul_pd(_mm_mul_pd ( da3,   b0),  dc23);
  abc[257] = _mm_mul_pd(_mm_mul_pd (  a3, d2b0),   c23);
  abc[258] = _mm_mul_pd(_mm_mul_pd (  a3,  db0),  dc23);
  abc[259] = _mm_mul_pd(_mm_mul_pd (  a3,   b0), d2c23);

  abc[260] = _mm_mul_pd(_mm_mul_pd (  a3,   b1),   c01);
  abc[261] = _mm_mul_pd(_mm_mul_pd ( da3,   b1),   c01);
  abc[262] = _mm_mul_pd(_mm_mul_pd (  a3,  db1),   c01);
  abc[263] = _mm_mul_pd(_mm_mul_pd (  a3,   b1),  dc01);
  abc[264] = _mm_mul_pd(_mm_mul_pd (d2a3,   b1),   c01);
  abc[265] = _mm_mul_pd(_mm_mul_pd ( da3,  db1),   c01);
  abc[266] = _mm_mul_pd(_mm_mul_pd ( da3,   b1),  dc01);
  abc[267] = _mm_mul_pd(_mm_mul_pd (  a3, d2b1),   c01);
  abc[268] = _mm_mul_pd(_mm_mul_pd (  a3,  db1),  dc01);
  abc[269] = _mm_mul_pd(_mm_mul_pd (  a3,   b1), d2c01);
  abc[270] = _mm_mul_pd(_mm_mul_pd (  a3,   b1),   c23);
  abc[271] = _mm_mul_pd(_mm_mul_pd ( da3,   b1),   c23);
  abc[272] = _mm_mul_pd(_mm_mul_pd (  a3,  db1),   c23);
  abc[273] = _mm_mul_pd(_mm_mul_pd (  a3,   b1),  dc23);
  abc[274] = _mm_mul_pd(_mm_mul_pd (d2a3,   b1),   c23);
  abc[275] = _mm_mul_pd(_mm_mul_pd ( da3,  db1),   c23);
  abc[276] = _mm_mul_pd(_mm_mul_pd ( da3,   b1),  dc23);
  abc[277] = _mm_mul_pd(_mm_mul_pd (  a3, d2b1),   c23);
  abc[278] = _mm_mul_pd(_mm_mul_pd (  a3,  db1),  dc23);
  abc[279] = _mm_mul_pd(_mm_mul_pd (  a3,   b1), d2c23);

  abc[280] = _mm_mul_pd(_mm_mul_pd (  a3,   b2),   c01);
  abc[281] = _mm_mul_pd(_mm_mul_pd ( da3,   b2),   c01);
  abc[282] = _mm_mul_pd(_mm_mul_pd (  a3,  db2),   c01);
  abc[283] = _mm_mul_pd(_mm_mul_pd (  a3,   b2),  dc01);
  abc[284] = _mm_mul_pd(_mm_mul_pd (d2a3,   b2),   c01);
  abc[285] = _mm_mul_pd(_mm_mul_pd ( da3,  db2),   c01);
  abc[286] = _mm_mul_pd(_mm_mul_pd ( da3,   b2),  dc01);
  abc[287] = _mm_mul_pd(_mm_mul_pd (  a3, d2b2),   c01);
  abc[288] = _mm_mul_pd(_mm_mul_pd (  a3,  db2),  dc01);
  abc[289] = _mm_mul_pd(_mm_mul_pd (  a3,   b2), d2c01);
  abc[290] = _mm_mul_pd(_mm_mul_pd (  a3,   b2),   c23);
  abc[291] = _mm_mul_pd(_mm_mul_pd ( da3,   b2),   c23);
  abc[292] = _mm_mul_pd(_mm_mul_pd (  a3,  db2),   c23);
  abc[293] = _mm_mul_pd(_mm_mul_pd (  a3,   b2),  dc23);
  abc[294] = _mm_mul_pd(_mm_mul_pd (d2a3,   b2),   c23);
  abc[295] = _mm_mul_pd(_mm_mul_pd ( da3,  db2),   c23);
  abc[296] = _mm_mul_pd(_mm_mul_pd ( da3,   b2),  dc23);
  abc[297] = _mm_mul_pd(_mm_mul_pd (  a3, d2b2),   c23);
  abc[298] = _mm_mul_pd(_mm_mul_pd (  a3,  db2),  dc23);
  abc[299] = _mm_mul_pd(_mm_mul_pd (  a3,   b2), d2c23);

  abc[300] = _mm_mul_pd(_mm_mul_pd (  a3,   b3),   c01);
  abc[301] = _mm_mul_pd(_mm_mul_pd ( da3,   b3),   c01);
  abc[302] = _mm_mul_pd(_mm_mul_pd (  a3,  db3),   c01);
  abc[303] = _mm_mul_pd(_mm_mul_pd (  a3,   b3),  dc01);
  abc[304] = _mm_mul_pd(_mm_mul_pd (d2a3,   b3),   c01);
  abc[305] = _mm_mul_pd(_mm_mul_pd ( da3,  db3),   c01);
  abc[306] = _mm_mul_pd(_mm_mul_pd ( da3,   b3),  dc01);
  abc[307] = _mm_mul_pd(_mm_mul_pd (  a3, d2b3),   c01);
  abc[308] = _mm_mul_pd(_mm_mul_pd (  a3,  db3),  dc01);
  abc[309] = _mm_mul_pd(_mm_mul_pd (  a3,   b3), d2c01);
  abc[310] = _mm_mul_pd(_mm_mul_pd (  a3,   b3),   c23);
  abc[311] = _mm_mul_pd(_mm_mul_pd ( da3,   b3),   c23);
  abc[312] = _mm_mul_pd(_mm_mul_pd (  a3,  db3),   c23);
  abc[313] = _mm_mul_pd(_mm_mul_pd (  a3,   b3),  dc23);
  abc[314] = _mm_mul_pd(_mm_mul_pd (d2a3,   b3),   c23);
  abc[315] = _mm_mul_pd(_mm_mul_pd ( da3,  db3),   c23);
  abc[316] = _mm_mul_pd(_mm_mul_pd ( da3,   b3),  dc23);
  abc[317] = _mm_mul_pd(_mm_mul_pd (  a3, d2b3),   c23);
  abc[318] = _mm_mul_pd(_mm_mul_pd (  a3,  db3),  dc23);
  abc[319] = _mm_mul_pd(_mm_mul_pd (  a3,   b3), d2c23);

  // Now loop over all splines
#define nextP(i,j,k) (const double*)(next_coefs+(i)*xs+(j)*ys+(k))
  __m128d val_r, val_i, grad_r[3], grad_i[3], hess_r[9], hess_i[9];


  for (int si=0; si<spline->num_splines; si++) {
    // Prefetch next set of coefficients
    next_coefs = coefs + 2*spline->spline_stride;
    _mm_prefetch((const char*)nextP(0,0,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(0,0,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(0,1,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(0,1,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(0,2,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(0,2,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(0,3,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(0,3,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(1,0,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(1,0,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(1,1,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(1,1,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(1,2,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(1,2,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(1,3,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(1,3,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(2,0,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(2,0,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(2,1,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(2,1,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(2,2,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(2,2,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(2,3,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(2,3,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(3,0,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(3,0,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(3,1,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(3,1,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(3,2,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(3,2,2), _MM_HINT_T0);
    _mm_prefetch((const char*)nextP(3,3,0), _MM_HINT_T0); _mm_prefetch((const char*)nextP(3,3,2), _MM_HINT_T0);

    // Zero out temp variables
    val_r = _mm_sub_pd (val_r, val_r);
    val_i = val_r;
    grad_r[0] = grad_r[1] = grad_r[2] = val_r;
    grad_i[0] = grad_i[1] = grad_i[2] = val_r;
    for (int i=0; i<9; i++) { hess_r[i] = val_r;  hess_i[i] = val_r; }

    // Now compute value, grad, and hessian
    int index = 0;
    for (int i=0; i<4; i++) 
      for (int j=0; j<4; j++) {
	tmp0 = _mm_load_pd (P(i,j,0));  tmp1 = _mm_load_pd (P(i,j,1));
	r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
	i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));

	val_r     = _mm_add_pd(val_r    , _mm_mul_pd(r0, abc[index+0]));
	grad_r[0] = _mm_add_pd(grad_r[0], _mm_mul_pd(r0, abc[index+1]));
	grad_r[1] = _mm_add_pd(grad_r[1], _mm_mul_pd(r0, abc[index+2]));
	grad_r[2] = _mm_add_pd(grad_r[2], _mm_mul_pd(r0, abc[index+3]));
	hess_r[0] = _mm_add_pd(hess_r[0], _mm_mul_pd(r0, abc[index+4]));
	hess_r[1] = _mm_add_pd(hess_r[1], _mm_mul_pd(r0, abc[index+5]));
	hess_r[2] = _mm_add_pd(hess_r[2], _mm_mul_pd(r0, abc[index+6]));
	hess_r[4] = _mm_add_pd(hess_r[4], _mm_mul_pd(r0, abc[index+7]));
	hess_r[5] = _mm_add_pd(hess_r[5], _mm_mul_pd(r0, abc[index+8]));
	hess_r[8] = _mm_add_pd(hess_r[8], _mm_mul_pd(r0, abc[index+9]));

	val_i     = _mm_add_pd(val_i,     _mm_mul_pd(i0, abc[index+0]));
	grad_i[0] = _mm_add_pd(grad_i[0], _mm_mul_pd(i0, abc[index+1]));
	grad_i[1] = _mm_add_pd(grad_i[1], _mm_mul_pd(i0, abc[index+2]));
	grad_i[2] = _mm_add_pd(grad_i[2], _mm_mul_pd(i0, abc[index+3]));
	hess_i[0] = _mm_add_pd(hess_i[0], _mm_mul_pd(i0, abc[index+4]));
	hess_i[1] = _mm_add_pd(hess_i[1], _mm_mul_pd(i0, abc[index+5]));
	hess_i[2] = _mm_add_pd(hess_i[2], _mm_mul_pd(i0, abc[index+6]));
	hess_i[4] = _mm_add_pd(hess_i[4], _mm_mul_pd(i0, abc[index+7]));
	hess_i[5] = _mm_add_pd(hess_i[5], _mm_mul_pd(i0, abc[index+8]));
	hess_i[8] = _mm_add_pd(hess_i[8], _mm_mul_pd(i0, abc[index+9]));

	tmp0 = _mm_load_pd (P(i,j,2));  tmp1 = _mm_load_pd (P(i,j,3));	
	r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
	i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
	
	val_r     = _mm_add_pd(val_r    , _mm_mul_pd(r1, abc[index+10]));
	grad_r[0] = _mm_add_pd(grad_r[0], _mm_mul_pd(r1, abc[index+11]));
	grad_r[1] = _mm_add_pd(grad_r[1], _mm_mul_pd(r1, abc[index+12]));
	grad_r[2] = _mm_add_pd(grad_r[2], _mm_mul_pd(r1, abc[index+13]));
	hess_r[0] = _mm_add_pd(hess_r[0], _mm_mul_pd(r1, abc[index+14]));
	hess_r[1] = _mm_add_pd(hess_r[1], _mm_mul_pd(r1, abc[index+15]));
	hess_r[2] = _mm_add_pd(hess_r[2], _mm_mul_pd(r1, abc[index+16]));
	hess_r[4] = _mm_add_pd(hess_r[4], _mm_mul_pd(r1, abc[index+17]));
	hess_r[5] = _mm_add_pd(hess_r[5], _mm_mul_pd(r1, abc[index+18]));
	hess_r[8] = _mm_add_pd(hess_r[8], _mm_mul_pd(r1, abc[index+19]));

	val_i     = _mm_add_pd(val_i    , _mm_mul_pd(i1, abc[index+10]));
	grad_i[0] = _mm_add_pd(grad_i[0], _mm_mul_pd(i1, abc[index+11]));
	grad_i[1] = _mm_add_pd(grad_i[1], _mm_mul_pd(i1, abc[index+12]));
	grad_i[2] = _mm_add_pd(grad_i[2], _mm_mul_pd(i1, abc[index+13]));
	hess_i[0] = _mm_add_pd(hess_i[0], _mm_mul_pd(i1, abc[index+14]));
	hess_i[1] = _mm_add_pd(hess_i[1], _mm_mul_pd(i1, abc[index+15]));
	hess_i[2] = _mm_add_pd(hess_i[2], _mm_mul_pd(i1, abc[index+16]));
	hess_i[4] = _mm_add_pd(hess_i[4], _mm_mul_pd(i1, abc[index+17]));
	hess_i[5] = _mm_add_pd(hess_i[5], _mm_mul_pd(i1, abc[index+18]));
	hess_i[8] = _mm_add_pd(hess_i[8], _mm_mul_pd(i1, abc[index+19]));
	index += 20;
      }    
    //_mm_storeu_pd((double*)(vals+si),_mm_hadd_pd(val_r, val_i));
    _mm_store_pd((double*)(vals+si)     , _mm_hadd_pd(val_r, val_i));
    _mm_store_pd((double*)(grads+3*si+0), _mm_hadd_pd(grad_r[0], grad_i[0]));
    _mm_store_pd((double*)(grads+3*si+1), _mm_hadd_pd(grad_r[1], grad_i[1]));
    _mm_store_pd((double*)(grads+3*si+2), _mm_hadd_pd(grad_r[2], grad_i[2]));
    _mm_store_pd((double*)(hess +3*si+0), _mm_hadd_pd(hess_r[0], hess_i[0]));    
    _mm_store_pd((double*)(hess +3*si+1), _mm_hadd_pd(hess_r[0], hess_i[1]));    
    _mm_store_pd((double*)(hess +3*si+2), _mm_hadd_pd(hess_r[0], hess_i[2]));    
    _mm_store_pd((double*)(hess +3*si+3), _mm_hadd_pd(hess_r[1], hess_i[1]));    
    _mm_store_pd((double*)(hess +3*si+4), _mm_hadd_pd(hess_r[4], hess_i[4]));    
    _mm_store_pd((double*)(hess +3*si+5), _mm_hadd_pd(hess_r[5], hess_i[5]));    
    _mm_store_pd((double*)(hess +3*si+6), _mm_hadd_pd(hess_r[2], hess_i[2]));    
    _mm_store_pd((double*)(hess +3*si+7), _mm_hadd_pd(hess_r[5], hess_i[5]));    
    _mm_store_pd((double*)(hess +3*si+8), _mm_hadd_pd(hess_r[8], hess_i[8]));    

    coefs += spline->spline_stride;
  }


  // Compute cP, dcP, and d2cP products 1/8 at a time to maximize
  // register reuse and avoid rerereading from memory or cache.
  // Complex values are read in, then shuffled such that 4 registers
  // hold the read parts and 4 register hold the imaginary parts.
  // 1st eighth
//   tmp0 = _mm_load_pd (P(0,0,0));  tmp1 = _mm_load_pd (P(0,0,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,0,2));  tmp1 = _mm_load_pd (P(0,0,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,1,0));  tmp1 = _mm_load_pd (P(0,1,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,1,2));  tmp1 = _mm_load_pd (P(0,1,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[0]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[0]);
  
//   // 2nd eighth
//   tmp0 = _mm_load_pd (P(0,2,0));  tmp1 = _mm_load_pd (P(0,2,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,2,2));  tmp1 = _mm_load_pd (P(0,2,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,3,0));  tmp1 = _mm_load_pd (P(0,3,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(0,3,2));  tmp1 = _mm_load_pd (P(0,3,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[1]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[1]);

//   // 3rd eighth
//   tmp0 = _mm_load_pd (P(1,0,0));  tmp1 = _mm_load_pd (P(1,0,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,0,2));  tmp1 = _mm_load_pd (P(1,0,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,1,0));  tmp1 = _mm_load_pd (P(1,1,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,1,2));  tmp1 = _mm_load_pd (P(1,1,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[2]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[2]);

//   // 4th eighth
//   tmp0 = _mm_load_pd (P(1,2,0));  tmp1 = _mm_load_pd (P(1,2,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,2,2));  tmp1 = _mm_load_pd (P(1,2,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,3,0));  tmp1 = _mm_load_pd (P(1,3,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(1,3,2));  tmp1 = _mm_load_pd (P(1,3,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[3]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[3]);

//   // 5th eighth
//   tmp0 = _mm_load_pd (P(2,0,0));  tmp1 = _mm_load_pd (P(2,0,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,0,2));  tmp1 = _mm_load_pd (P(2,0,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,1,0));  tmp1 = _mm_load_pd (P(2,1,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,1,2));  tmp1 = _mm_load_pd (P(2,1,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[4]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[4]);

//   // 6th eighth
//   tmp0 = _mm_load_pd (P(2,2,0));  tmp1 = _mm_load_pd (P(2,2,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,2,2));  tmp1 = _mm_load_pd (P(2,2,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,3,0));  tmp1 = _mm_load_pd (P(2,3,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(2,3,2));  tmp1 = _mm_load_pd (P(2,3,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[5]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[5]);

//   // 7th eighth
//   tmp0 = _mm_load_pd (P(3,0,0));  tmp1 = _mm_load_pd (P(3,0,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,0,2));  tmp1 = _mm_load_pd (P(3,0,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,1,0));  tmp1 = _mm_load_pd (P(3,1,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,1,2));  tmp1 = _mm_load_pd (P(3,1,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[6]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[6]);

//   // 8th eighth
//   tmp0 = _mm_load_pd (P(3,2,0));  tmp1 = _mm_load_pd (P(3,2,1));
//   r0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i0 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,2,2));  tmp1 = _mm_load_pd (P(3,2,3));
//   r1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i1 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,3,0));  tmp1 = _mm_load_pd (P(3,3,1));
//   r2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i2 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   tmp0 = _mm_load_pd (P(3,3,2));  tmp1 = _mm_load_pd (P(3,3,3));
//   r3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(0, 0));
//   i3 = _mm_shuffle_pd (tmp0, tmp1, _MM_SHUFFLE2(1, 1));
//   _MM_DDOT4_PD(r0, r1, r2, r3,   c01,  c23,  c01,  c23,  cPr[7]);
//   _MM_DDOT4_PD(i0, i1, i2, i3,   c01,  c23,  c01,  c23,  cPi[7]);
  
//   // Now compute bcP, dbcP, bdcP, d2bcP, bd2cP, and dbdc products
//   _MM_DDOT4_PD (b01, b23, b01, b23, cPr[0], cPr[1], cPr[2], cPr[3], bcP01r);
//   _MM_DDOT4_PD (b01, b23, b01, b23, cPi[0], cPi[1], cPi[2], cPi[3], bcP01i);
//   _MM_DDOT4_PD (b01, b23, b01, b23, cPr[4], cPr[5], cPr[6], cPr[7], bcP23r);
//   _MM_DDOT4_PD (b01, b23, b01, b23, cPi[4], cPi[5], cPi[6], cPi[7], bcP23i);

//   // Compute value
//   _MM_DOT4_PD (a01, a23, bcP01r, bcP23r, *((double*)val+0));
//  _MM_DOT4_PD (a01, a23, bcP01i, bcP23i, *((double*)val+1));
#undef P
#undef nextP
}



#endif
